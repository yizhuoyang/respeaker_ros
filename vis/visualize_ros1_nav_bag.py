#!/usr/bin/env python3
"""Offline visualization for ROS1 navigation bags.

For each sampled timestamp this script saves:
  - synchronized /camera/color/image_raw frame
  - a 2-D map image with Livox LiDAR, fusion markers, selected goal, odometry
    history/current pose, and the current navigation plan.
"""

import argparse
import bisect
import csv
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import rosbag
    import rospy
except Exception as exc:  # pragma: no cover - depends on sourced ROS1 env.
    rosbag = None
    rospy = None
    ROS_IMPORT_ERROR = exc
else:
    ROS_IMPORT_ERROR = None


DEFAULT_IMAGE_TOPIC = "/camera/color/image_raw"
DEFAULT_LIDAR_TOPIC = "/livox/lidar"
DEFAULT_ODOM_TOPIC = "/Odometry"
DEFAULT_MARKERS_TOPIC = "/audio_visual_goal_fusion/markers"
DEFAULT_GOAL_TOPIC = "/audio_visual_goal_fusion/goal"
DEFAULT_PLAN_TOPIC = "/move_base/DWAPlannerROS/local_plan"


@dataclass
class StampedMsg:
    stamp_ns: int
    msg: object


@dataclass
class OdomPose:
    stamp_ns: int
    x: float
    y: float
    z: float
    yaw: float


def main():
    args = parse_args()
    require_ros1_python()

    output_root = Path(args.output).expanduser()
    output_root.mkdir(parents=True, exist_ok=True)

    bags = discover_bags(args.input, args.recursive)
    if not bags:
        raise RuntimeError(f"No ROS1 .bag files found under: {args.input}")

    print(f"Found {len(bags)} bag(s).")
    for index, bag_path in enumerate(bags, start=1):
        print(f"[{index}/{len(bags)}] {bag_path}")
        visualize_bag(bag_path, output_root, args)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Save synchronized camera and 2-D navigation map images from ROS1 bags."
    )
    parser.add_argument("--input", required=True, help=".bag file, bag directory, or parent directory.")
    parser.add_argument("--output", default="vis/ros1_output", help="Output root directory.")
    parser.add_argument("--recursive", action="store_true", help="Recursively discover .bag files.")
    parser.add_argument("--rate", type=float, default=5.0, help="Output visualization rate in Hz.")
    parser.add_argument("--start-sec", type=float, default=None, help="Start offset from bag start.")
    parser.add_argument("--end-sec", type=float, default=None, help="End offset from bag start.")

    parser.add_argument("--image-topic", default=DEFAULT_IMAGE_TOPIC)
    parser.add_argument("--lidar-topic", default=DEFAULT_LIDAR_TOPIC)
    parser.add_argument("--odom-topic", default=DEFAULT_ODOM_TOPIC)
    parser.add_argument("--markers-topic", default=DEFAULT_MARKERS_TOPIC)
    parser.add_argument("--goal-topic", default=DEFAULT_GOAL_TOPIC)
    parser.add_argument("--plan-topic", default=DEFAULT_PLAN_TOPIC)
    parser.add_argument(
        "--plan-topic-candidates",
        default="/move_base/DWAPlannerROS/local_plan,/move_base/GlobalPlanner/plan,/plan,/global_plan,/local_plan",
        help="Comma-separated fallback nav_msgs/Path topics.",
    )
    parser.add_argument("--max-sync-diff-sec", type=float, default=0.25)

    parser.add_argument("--lidar-window-sec", type=float, default=0.0, help="0 means use all past LiDAR.")
    parser.add_argument("--lidar-crop-m", type=float, default=4.0, help="Keep LiDAR points within +/- this many meters of current odom x/y.")
    parser.add_argument("--global-lidar-crop-m", type=float, default=4.0, help="Global fixed map uses odom trajectory bounds plus this padding.")
    parser.add_argument("--lidar-max-z", type=float, default=1.9, help="Drop raw LiDAR points above this z before XY projection.")
    parser.add_argument("--lidar-front-exclusion-m", type=float, default=0.5, help="Drop raw LiDAR points with 0 <= x <= this value before accumulation.")
    parser.add_argument("--lidar-density-cell-size", type=float, default=0.20, help="Grid size for removing sparse LiDAR points; <=0 disables filtering.")
    parser.add_argument("--lidar-min-cell-points", type=int, default=2, help="Remove LiDAR points whose density cell has fewer points than this.")
    parser.add_argument(
        "--lidar-map-mode",
        choices=("final", "accumulated"),
        default="final",
        help="final uses the whole bag's accumulated LiDAR map for every frame; accumulated uses points up to the frame time.",
    )
    parser.add_argument("--lidar-stride", type=int, default=8, help="Keep every Nth point from each LiDAR msg.")
    parser.add_argument("--max-lidar-points", type=int, default=80000)
    parser.add_argument("--lidar-point-size", type=float, default=1.0)
    parser.add_argument("--sensor-yaw-deg", type=float, default=0.0)
    parser.add_argument("--sensor-offset-x", type=float, default=0.0)
    parser.add_argument("--sensor-offset-y", type=float, default=0.0)
    parser.add_argument(
        "--map-frame",
        choices=("global", "local"),
        default="global",
        help="global keeps one fixed map for every frame; local centers and rotates around current odom.",
    )

    parser.add_argument("--fig-size", default="8,8", help="Map figure size in inches, e.g. 8,8.")
    parser.add_argument("--dpi", type=int, default=140)
    parser.add_argument("--map-padding", type=float, default=1.5)
    parser.add_argument("--marker-alpha-threshold", type=float, default=0.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def require_ros1_python():
    if ROS_IMPORT_ERROR is not None:
        raise RuntimeError(
            "Cannot import ROS1 Python modules. Please source ROS1 first, e.g.\n"
            "  source /opt/ros/<distro>/setup.bash\n"
            "  source <catkin_ws>/devel/setup.bash\n"
            f"Original import error: {type(ROS_IMPORT_ERROR).__name__}: {ROS_IMPORT_ERROR}"
        )


def discover_bags(input_path, recursive=False):
    path = Path(input_path).expanduser()
    if path.is_file():
        return [path] if path.suffix == ".bag" else []
    if not path.is_dir():
        return []
    patterns = ["**/*.bag"] if recursive else ["*.bag", "*/*.bag"]
    bags = []
    seen = set()
    for pattern in patterns:
        for bag in sorted(path.glob(pattern)):
            key = bag.resolve()
            if key not in seen:
                seen.add(key)
                bags.append(bag)
    return bags


def visualize_bag(bag_path, output_root, args):
    with rosbag.Bag(str(bag_path), "r") as bag:
        available = set(bag.get_type_and_topic_info().topics)
        topics = resolve_topics(available, args)
        if topics["image"] is None:
            raise RuntimeError(f"Missing image topic {args.image_topic} in {bag_path}")
        if topics["odom"] is None:
            raise RuntimeError(f"Missing odom topic {args.odom_topic} in {bag_path}")

        seq_output = output_root / bag_path.stem
        camera_dir = seq_output / "camera"
        map_dir = seq_output / "map"
        if seq_output.exists() and not args.overwrite:
            raise RuntimeError(f"Output exists, pass --overwrite: {seq_output}")
        camera_dir.mkdir(parents=True, exist_ok=True)
        map_dir.mkdir(parents=True, exist_ok=True)

        print("  topics:")
        for label, topic in topics.items():
            print(f"    {label}: {topic or 'missing'}")

        odom = read_small_topic(bag, topics["odom"])
        if not odom:
            raise RuntimeError(f"No odom messages in {bag_path}")
        odom_poses = [odom_to_pose(item) for item in odom]

        goals = read_small_topic(bag, topics["goal"]) if topics["goal"] else []
        markers = read_small_topic(bag, topics["markers"]) if topics["markers"] else []
        plans = read_small_topic(bag, topics["plan"]) if topics["plan"] else []
        lidar_points = (
            read_lidar_points(bag, topics["lidar"], odom_poses, args)
            if topics["lidar"]
            else np.zeros((0, 3), dtype=np.float32)
        )
        lidar_points = filter_sparse_lidar_points(
            lidar_points,
            args.lidar_density_cell_size,
            args.lidar_min_cell_points,
        )
        fixed_map_limits = compute_fixed_map_limits(
            lidar_points,
            odom_poses,
            args.map_padding,
            args.global_lidar_crop_m,
        )

        start_ns = max(time_to_ns(bag.get_start_time()), odom_poses[0].stamp_ns)
        end_ns = min(time_to_ns(bag.get_end_time()), odom_poses[-1].stamp_ns)
        if args.start_sec is not None:
            start_ns = max(start_ns, time_to_ns(bag.get_start_time()) + sec_to_ns(args.start_sec))
        if args.end_sec is not None:
            end_ns = min(end_ns, time_to_ns(bag.get_start_time()) + sec_to_ns(args.end_sec))
        if end_ns <= start_ns:
            raise RuntimeError("No time range to visualize after applying start/end filters.")

        sample_stamps = list(range(start_ns, end_ns + 1, sec_to_ns(1.0 / args.rate)))
        fig_size = parse_pair(args.fig_size)
        max_sync_diff_ns = sec_to_ns(args.max_sync_diff_sec)

        index_path = seq_output / "index.csv"
        with index_path.open("w", newline="", encoding="utf-8") as index_file:
            writer = csv.writer(index_file)
            writer.writerow([
                "frame",
                "target_stamp_ns",
                "camera_file",
                "map_file",
                "image_stamp_ns",
                "odom_stamp_ns",
                "goal_stamp_ns",
                "markers_stamp_ns",
                "plan_stamp_ns",
            ])

            image_iter = bag.read_messages(topics=[topics["image"]])
            image_buffer = None
            saved_count = 0
            for frame_index, target_ns in enumerate(sample_stamps):
                image_item, image_buffer = nearest_image_from_stream(
                    image_iter,
                    image_buffer,
                    target_ns,
                    max_sync_diff_ns,
                )
                current_pose = nearest_pose(odom_poses, target_ns, max_sync_diff_ns)
                if image_item is None or current_pose is None:
                    continue

                goal_item = latest_at_or_before(goals, target_ns, max_sync_diff_ns)
                markers_item = latest_at_or_before(markers, target_ns, max_sync_diff_ns)
                plan_item = latest_at_or_before(plans, target_ns, max_sync_diff_ns)

                stem = f"{frame_index:06d}_{target_ns}"
                camera_file = camera_dir / f"{stem}.png"
                map_file = map_dir / f"{stem}.png"

                image = ros_image_to_cv2(image_item.msg)
                cv2.imwrite(str(camera_file), image)
                render_map(
                    map_file,
                    target_ns,
                    odom_poses,
                    current_pose,
                    lidar_points,
                    markers_item.msg if markers_item else None,
                    goal_item.msg if goal_item else None,
                    plan_item.msg if plan_item else None,
                    args,
                    fig_size,
                    fixed_map_limits,
                )

                writer.writerow([
                    frame_index,
                    target_ns,
                    str(camera_file.relative_to(seq_output)),
                    str(map_file.relative_to(seq_output)),
                    image_item.stamp_ns,
                    current_pose.stamp_ns,
                    goal_item.stamp_ns if goal_item else "",
                    markers_item.stamp_ns if markers_item else "",
                    plan_item.stamp_ns if plan_item else "",
                ])
                saved_count += 1
                if saved_count % max(int(args.rate) * 5, 1) == 0:
                    print(f"  saved {saved_count} synchronized frame(s)")

        write_manifest(seq_output, bag_path, topics, args)
        print(f"  saved {saved_count} frame(s) -> {seq_output}")


def resolve_topics(available, args):
    return {
        "image": first_available(available, [args.image_topic, f"{args.image_topic}/compressed"]),
        "lidar": first_available(available, [args.lidar_topic]),
        "odom": first_available(available, [args.odom_topic, "/Odometry", "/lio/odom", "/lio/robo/odom"]),
        "markers": first_available(available, [args.markers_topic]),
        "goal": first_available(available, [args.goal_topic]),
        "plan": first_available(available, [args.plan_topic] + split_csv(args.plan_topic_candidates)),
    }


def first_available(available, candidates):
    for topic in candidates:
        if topic and topic in available:
            return topic
    return None


def read_small_topic(bag, topic):
    return [StampedMsg(message_stamp_ns(msg, t), msg) for _, msg, t in bag.read_messages(topics=[topic])]


def read_lidar_points(bag, topic, odom_poses, args):
    stride = max(int(args.lidar_stride), 1)
    sensor_yaw = math.radians(args.sensor_yaw_deg)
    blocks = []
    count = 0
    for _, msg, t in bag.read_messages(topics=[topic]):
        stamp_ns = message_stamp_ns(msg, t)
        pose = nearest_pose(odom_poses, stamp_ns, None)
        if pose is None:
            continue
        local_xy = livox_xy(msg, stride, args.lidar_max_z)
        if not local_xy.size:
            continue
        world_xy = transform_lidar_xy(
            local_xy,
            pose,
            sensor_yaw,
            args.sensor_offset_x,
            args.sensor_offset_y,
            args.lidar_front_exclusion_m,
        )
        if not world_xy.size:
            continue
        stamps = np.full((world_xy.shape[0], 1), stamp_ns, dtype=np.int64)
        blocks.append(np.hstack([stamps, world_xy.astype(np.float32)]))
        count += 1
        if count % 100 == 0:
            print(f"    lidar frames decoded: {count}")
    if not blocks:
        return np.zeros((0, 3), dtype=np.float32)
    return np.vstack(blocks)


def nearest_image_from_stream(image_iter, image_buffer, target_ns, max_diff_ns):
    previous = image_buffer
    current = None
    while True:
        if image_buffer is None:
            try:
                _, msg, t = next(image_iter)
            except StopIteration:
                break
            image_buffer = StampedMsg(message_stamp_ns(msg, t), msg)

        if image_buffer.stamp_ns < target_ns:
            previous = image_buffer
            image_buffer = None
            continue

        current = image_buffer
        break

    candidates = [item for item in (previous, current) if item is not None]
    if not candidates:
        return None, image_buffer
    best = min(candidates, key=lambda item: abs(item.stamp_ns - target_ns))
    if abs(best.stamp_ns - target_ns) > max_diff_ns:
        return None, image_buffer
    return best, image_buffer


def render_map(
    output_path,
    stamp_ns,
    odom_poses,
    current_pose,
    lidar_points,
    marker_array,
    goal,
    plan,
    args,
    fig_size,
    fixed_map_limits,
):
    fig, ax = plt.subplots(figsize=fig_size)
    fig.patch.set_alpha(0.0)
    ax.set_facecolor((1.0, 1.0, 1.0, 0.0))
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()

    visible_lidar = select_lidar(
        lidar_points,
        stamp_ns,
        current_pose,
        args.lidar_window_sec,
        args.lidar_crop_m,
        args.lidar_map_mode,
        args.map_frame,
        fixed_map_limits,
        args.max_lidar_points,
    )
    if visible_lidar.size:
        ax.scatter(
            visible_lidar[:, 0],
            visible_lidar[:, 1],
            s=args.lidar_point_size,
            c="#111111",
            alpha=1.0,
            linewidths=0,
            label="LiDAR",
        )

    history = np.array(
        [[pose.x, pose.y] for pose in odom_poses if pose.stamp_ns <= current_pose.stamp_ns],
        dtype=np.float32,
    )
    if args.map_frame == "local":
        history = world_to_local_xy(history, current_pose)
    if history.size:
        ax.plot(history[:, 0], history[:, 1], color="#1f77b4", linewidth=1.6, label="odom history")

    draw_marker_array(ax, marker_array, args.marker_alpha_threshold, current_pose, args.map_frame)
    draw_robot(ax, current_pose, args.map_frame)
    set_view_limits(ax, current_pose, args.lidar_crop_m, args.map_frame, fixed_map_limits)

    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.savefig(str(output_path), dpi=args.dpi, pad_inches=0, transparent=True)
    plt.close(fig)


def select_lidar(
    lidar_points,
    stamp_ns,
    current_pose,
    window_sec,
    crop_m,
    map_mode,
    map_frame,
    fixed_map_limits,
    max_points,
):
    if lidar_points.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    stamps = lidar_points[:, 0].astype(np.int64)
    if map_mode == "final":
        start = 0
        end = len(lidar_points)
    else:
        end = bisect.bisect_right(stamps, stamp_ns)
        start = 0
    if map_mode == "accumulated" and window_sec and window_sec > 0.0:
        start = bisect.bisect_left(stamps, stamp_ns - sec_to_ns(window_sec))
    xy = lidar_points[start:end, 1:3]
    if map_frame == "local":
        xy = world_to_local_xy(xy, current_pose)
    if map_frame == "local" and crop_m and crop_m > 0.0 and xy.size:
        crop_m = float(crop_m)
        keep = (np.abs(xy[:, 0]) <= crop_m) & (np.abs(xy[:, 1]) <= crop_m)
        xy = xy[keep]
    elif map_frame == "global" and fixed_map_limits is not None and xy.size:
        xmin, xmax, ymin, ymax = fixed_map_limits
        keep = (xy[:, 0] >= xmin) & (xy[:, 0] <= xmax) & (xy[:, 1] >= ymin) & (xy[:, 1] <= ymax)
        xy = xy[keep]
    if xy.shape[0] > max_points:
        step = int(math.ceil(xy.shape[0] / float(max_points)))
        xy = xy[::step]
    return xy


def draw_path(ax, path_msg):
    if path_msg is None or not hasattr(path_msg, "poses"):
        return
    points = [(float(p.pose.position.x), float(p.pose.position.y)) for p in path_msg.poses]
    if len(points) < 2:
        return
    xy = np.asarray(points, dtype=np.float32)
    ax.plot(xy[:, 0], xy[:, 1], color="#ff7f0e", linewidth=2.0, label="current plan")


def draw_marker_array(ax, marker_array, alpha_threshold, current_pose, map_frame):
    if marker_array is None or not hasattr(marker_array, "markers"):
        return
    marker_labeled = False
    for marker in marker_array.markers:
        ns = getattr(marker, "ns", "")
        marker_type = int(getattr(marker, "type", -1))
        if marker_type == 6 and ns.startswith("audio_visual"):
            xs, ys, colors = [], [], []
            for idx, point in enumerate(marker.points):
                color = marker.colors[idx] if idx < len(marker.colors) else marker.color
                if color.a < alpha_threshold:
                    continue
                xy = np.asarray([[float(point.x), float(point.y)]], dtype=np.float32)
                if map_frame == "local":
                    xy = world_to_local_xy(xy, current_pose)
                xs.append(float(xy[0, 0]))
                ys.append(float(xy[0, 1]))
                colors.append((color.r, color.g, color.b, max(color.a, 0.05)))
            if xs:
                ax.scatter(xs, ys, s=9, c=colors, marker="s", linewidths=0)
        elif marker_type in (1, 2, 3) and ns == "audio_visual_goal":
            p = marker.pose.position
            c = marker.color
            xy = np.asarray([[float(p.x), float(p.y)]], dtype=np.float32)
            if map_frame == "local":
                xy = world_to_local_xy(xy, current_pose)
            ax.scatter(
                [float(xy[0, 0])],
                [float(xy[0, 1])],
                s=85,
                c=[(c.r, c.g, c.b, max(c.a, 0.25))],
                edgecolors="black",
                linewidths=0.5,
                label=(marker.ns or "fusion marker") if not marker_labeled else None,
            )
            marker_labeled = True


def draw_goal(ax, goal_msg):
    if goal_msg is None:
        return
    if hasattr(goal_msg, "point"):
        x, y = float(goal_msg.point.x), float(goal_msg.point.y)
    elif hasattr(goal_msg, "pose"):
        x, y = float(goal_msg.pose.position.x), float(goal_msg.pose.position.y)
    else:
        return
    ax.scatter([x], [y], s=140, c="#ffd400", edgecolors="black", marker="*", label="selected goal")


def draw_robot(ax, pose, map_frame):
    length = 0.45
    if map_frame == "local":
        x, y, yaw = 0.0, 0.0, 0.0
    else:
        x, y, yaw = pose.x, pose.y, pose.yaw
    ax.scatter([x], [y], s=70, c="#00a6d6", edgecolors="black", label="current odom")
    ax.arrow(
        x,
        y,
        math.cos(yaw) * length,
        math.sin(yaw) * length,
        width=0.03,
        head_width=0.18,
        head_length=0.18,
        color="#00a6d6",
        length_includes_head=True,
    )


def set_view_limits(ax, current_pose, crop_m, map_frame, fixed_map_limits):
    if map_frame == "global" and fixed_map_limits is not None:
        ax.set_xlim(fixed_map_limits[0], fixed_map_limits[1])
        ax.set_ylim(fixed_map_limits[2], fixed_map_limits[3])
        return
    radius = float(crop_m) if crop_m and crop_m > 0.0 else 4.0
    ax.set_xlim(-radius, radius)
    ax.set_ylim(-radius, radius)


def filter_sparse_lidar_points(lidar_points, cell_size, min_cell_points):
    if lidar_points.size == 0:
        return lidar_points
    if cell_size is None or cell_size <= 0.0 or min_cell_points <= 1:
        return lidar_points

    xy = lidar_points[:, 1:3]
    finite = np.isfinite(xy).all(axis=1)
    cells = np.floor(xy[finite] / float(cell_size)).astype(np.int64)
    if not cells.size:
        return lidar_points[:0]

    _, inverse, counts = np.unique(cells, axis=0, return_inverse=True, return_counts=True)
    keep_finite = counts[inverse] >= int(min_cell_points)
    keep = np.zeros((lidar_points.shape[0],), dtype=bool)
    keep[np.flatnonzero(finite)] = keep_finite
    filtered = lidar_points[keep]
    print(
        "    lidar density filter: kept %d/%d point(s), cell_size=%.3f min_cell_points=%d"
        % (filtered.shape[0], lidar_points.shape[0], float(cell_size), int(min_cell_points))
    )
    return filtered


def compute_fixed_map_limits(lidar_points, odom_poses, padding, global_lidar_crop_m):
    chunks = []
    if odom_poses:
        chunks.append(np.asarray([[pose.x, pose.y] for pose in odom_poses], dtype=np.float32))
    elif lidar_points.size:
        chunks.append(lidar_points[:, 1:3])
    if not chunks:
        return None

    xy = np.vstack(chunks)
    finite = np.isfinite(xy).all(axis=1)
    xy = xy[finite]
    if not xy.size:
        return None

    padding = float(global_lidar_crop_m if global_lidar_crop_m is not None else padding)
    xmin = float(np.min(xy[:, 0])) - padding
    xmax = float(np.max(xy[:, 0])) + padding
    ymin = float(np.min(xy[:, 1])) - padding
    ymax = float(np.max(xy[:, 1])) + padding

    if xmax - xmin < 4.0:
        center = 0.5 * (xmin + xmax)
        xmin, xmax = center - 2.0, center + 2.0
    if ymax - ymin < 4.0:
        center = 0.5 * (ymin + ymax)
        ymin, ymax = center - 2.0, center + 2.0
    return xmin, xmax, ymin, ymax


def ros_image_to_cv2(msg):
    if msg._type == "sensor_msgs/CompressedImage":
        data = np.frombuffer(msg.data, dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError("Failed to decode compressed image")
        return image

    encoding = msg.encoding.lower()
    dtype, channels = image_encoding_info(encoding)
    itemsize = np.dtype(dtype).itemsize
    row_bytes = int(msg.width) * channels * itemsize
    raw = np.frombuffer(msg.data, dtype=np.uint8)
    rows = raw.reshape(int(msg.height), int(msg.step))
    rows = rows[:, :row_bytes]
    image = rows.reshape(int(msg.height), int(msg.width), channels, itemsize).copy()
    image = image.view(dtype).reshape(int(msg.height), int(msg.width), channels)
    if msg.is_bigendian:
        image = image.byteswap()
    if channels == 1:
        image = image[:, :, 0]

    if encoding == "rgb8":
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    elif encoding == "rgba8":
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)
    return image


def image_encoding_info(encoding):
    if encoding in ("rgb8", "bgr8"):
        return np.uint8, 3
    if encoding in ("rgba8", "bgra8"):
        return np.uint8, 4
    if encoding in ("mono8", "8uc1"):
        return np.uint8, 1
    if encoding in ("mono16", "16uc1"):
        return np.uint16, 1
    raise RuntimeError(f"Unsupported image encoding: {encoding}")


def livox_xy(msg, stride, max_z):
    points = getattr(msg, "points", [])[::stride]
    if not points:
        return np.zeros((0, 2), dtype=np.float32)
    xyz = np.asarray([[float(p.x), float(p.y), float(p.z)] for p in points], dtype=np.float32)
    finite = np.isfinite(xyz).all(axis=1)
    if max_z is not None:
        finite &= xyz[:, 2] < float(max_z)
    return xyz[finite, :2]


def world_to_local_xy(world_xy, pose):
    xy = np.asarray(world_xy, dtype=np.float32)
    if xy.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    shifted = xy.copy()
    shifted[:, 0] -= pose.x
    shifted[:, 1] -= pose.y
    c = math.cos(-pose.yaw)
    s = math.sin(-pose.yaw)
    rot = np.array([[c, -s], [s, c]], dtype=np.float32)
    return shifted @ rot.T


def transform_lidar_xy(
    local_xy,
    pose,
    sensor_yaw,
    sensor_offset_x,
    sensor_offset_y,
    front_exclusion_m,
):
    sensor_c = math.cos(sensor_yaw)
    sensor_s = math.sin(sensor_yaw)
    sensor_rot = np.array([[sensor_c, -sensor_s], [sensor_s, sensor_c]], dtype=np.float32)
    xy = local_xy @ sensor_rot.T
    xy[:, 0] += sensor_offset_x
    xy[:, 1] += sensor_offset_y
    if front_exclusion_m and front_exclusion_m > 0.0 and xy.size:
        keep = ~((xy[:, 0] >= 0.0) & (xy[:, 0] <= float(front_exclusion_m)))
        xy = xy[keep]
        if not xy.size:
            return np.zeros((0, 2), dtype=np.float32)

    odom_c = math.cos(pose.yaw)
    odom_s = math.sin(pose.yaw)
    odom_rot = np.array([[odom_c, -odom_s], [odom_s, odom_c]], dtype=np.float32)
    xy = xy @ odom_rot.T
    xy[:, 0] += pose.x
    xy[:, 1] += pose.y
    return xy


def odom_to_pose(item):
    msg = item.msg
    p = msg.pose.pose.position
    q = msg.pose.pose.orientation
    return OdomPose(item.stamp_ns, float(p.x), float(p.y), float(p.z), quaternion_yaw(q))


def quaternion_yaw(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def latest_at_or_before(items, stamp_ns, max_age_ns):
    if not items:
        return None
    stamps = [item.stamp_ns for item in items]
    idx = bisect.bisect_right(stamps, stamp_ns) - 1
    if idx < 0:
        return None
    item = items[idx]
    if max_age_ns is not None and stamp_ns - item.stamp_ns > max_age_ns:
        return None
    return item


def nearest_pose(poses, stamp_ns, max_diff_ns):
    if not poses:
        return None
    stamps = [pose.stamp_ns for pose in poses]
    idx = bisect.bisect_left(stamps, stamp_ns)
    candidates = []
    if idx < len(poses):
        candidates.append(poses[idx])
    if idx > 0:
        candidates.append(poses[idx - 1])
    best = min(candidates, key=lambda pose: abs(pose.stamp_ns - stamp_ns))
    if max_diff_ns is not None and abs(best.stamp_ns - stamp_ns) > max_diff_ns:
        return None
    return best


def message_stamp_ns(msg, bag_time):
    header = getattr(msg, "header", None)
    stamp = getattr(header, "stamp", None)
    if stamp is None or stamp.to_sec() <= 0.0:
        return time_to_ns(bag_time)
    return time_to_ns(stamp)


def time_to_ns(value):
    if hasattr(value, "to_nsec"):
        return int(value.to_nsec())
    return int(round(float(value) * 1_000_000_000))


def sec_to_ns(value):
    return int(round(float(value) * 1_000_000_000))


def parse_pair(text):
    parts = [float(part.strip()) for part in text.split(",")]
    if len(parts) != 2:
        raise ValueError("--fig-size must be two comma-separated numbers")
    return parts[0], parts[1]


def split_csv(text):
    return [item.strip() for item in str(text).split(",") if item.strip()]


def write_manifest(output_dir, bag_path, topics, args):
    manifest = output_dir / "manifest.txt"
    with manifest.open("w", encoding="utf-8") as f:
        f.write(f"bag={bag_path}\n")
        f.write(f"rate={args.rate}\n")
        for label, topic in topics.items():
            f.write(f"{label}_topic={topic or ''}\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)

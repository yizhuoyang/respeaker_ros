#!/usr/bin/env python3
"""Offline visualization for navigation ROS2 bags.

The script reads rosbag2 sqlite bags directly, samples them at a fixed rate,
saves synchronized camera frames, and renders a 2-D map view containing LiDAR,
fusion markers, selected goal, odometry history, current odometry, and plan path.
"""

import argparse
import bisect
import csv
import math
import os
import sqlite3
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
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
except Exception as exc:  # pragma: no cover - depends on sourced ROS2 env.
    deserialize_message = None
    get_message = None
    ROS_IMPORT_ERROR = exc
else:
    ROS_IMPORT_ERROR = None


DEFAULT_IMAGE_TOPIC = "/camera/color/image_raw"
DEFAULT_LIDAR_TOPIC = "/livox/lidar"
DEFAULT_ODOM_TOPIC = "/lio/odom"
DEFAULT_MARKERS_TOPIC = "/audio_visual_goal_fusion/markers"
DEFAULT_GOAL_TOPIC = "/audio_visual_goal_fusion/goal"
DEFAULT_PLAN_TOPIC = "/plan"


@dataclass
class Topic:
    topic_id: int
    name: str
    type_name: str
    msg_type: object


@dataclass
class TimedMessage:
    bag_stamp_ns: int
    msg_stamp_ns: int
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
    require_ros2_python()

    output_root = Path(args.output).expanduser()
    output_root.mkdir(parents=True, exist_ok=True)

    bags = discover_bags(args.input, args.recursive)
    if not bags:
        raise RuntimeError(f"No .db3 rosbag2 files found under: {args.input}")

    print(f"Found {len(bags)} bag(s).")
    for index, bag_db in enumerate(bags, start=1):
        print(f"[{index}/{len(bags)}] {bag_db}")
        visualize_bag(bag_db, output_root, args)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Save synchronized camera and 2-D navigation map visualizations from ROS2 bags."
    )
    parser.add_argument("--input", required=True, help="Bag .db3, bag directory, or parent directory.")
    parser.add_argument("--output", default="vis/output", help="Output root directory.")
    parser.add_argument("--recursive", action="store_true", help="Recursively discover .db3 bags.")
    parser.add_argument("--rate", type=float, default=5.0, help="Output visualization rate in Hz.")
    parser.add_argument("--start-sec", type=float, default=None, help="Start time offset from bag start.")
    parser.add_argument("--end-sec", type=float, default=None, help="End time offset from bag start.")

    parser.add_argument("--image-topic", default=DEFAULT_IMAGE_TOPIC)
    parser.add_argument("--lidar-topic", default=DEFAULT_LIDAR_TOPIC)
    parser.add_argument("--odom-topic", default=DEFAULT_ODOM_TOPIC)
    parser.add_argument("--markers-topic", default=DEFAULT_MARKERS_TOPIC)
    parser.add_argument("--goal-topic", default=DEFAULT_GOAL_TOPIC)
    parser.add_argument("--plan-topic", default=DEFAULT_PLAN_TOPIC)
    parser.add_argument(
        "--plan-topic-candidates",
        default="/plan,/move_base/NavfnROS/plan,/move_base/GlobalPlanner/plan,/move_base/DWAPlannerROS/local_plan,/global_plan,/local_plan",
        help="Comma-separated fallback Path topics used if --plan-topic is absent.",
    )
    parser.add_argument("--max-sync-diff-sec", type=float, default=0.25)

    parser.add_argument("--lidar-window-sec", type=float, default=0.0, help="0 means use all past LiDAR.")
    parser.add_argument("--lidar-stride", type=int, default=8, help="Keep every Nth point from each LiDAR frame.")
    parser.add_argument("--max-lidar-points", type=int, default=80000)
    parser.add_argument("--sensor-yaw-deg", type=float, default=0.0)
    parser.add_argument("--sensor-offset-x", type=float, default=0.0)
    parser.add_argument("--sensor-offset-y", type=float, default=0.0)

    parser.add_argument("--fig-size", default="8,8", help="Map figure size in inches, e.g. 8,8.")
    parser.add_argument("--map-padding", type=float, default=1.5)
    parser.add_argument("--dpi", type=int, default=140)
    parser.add_argument("--marker-threshold", type=float, default=0.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def require_ros2_python():
    if ROS_IMPORT_ERROR is not None:
        raise RuntimeError(
            "Cannot import ROS2 Python modules. Please source your ROS2 workspace first, e.g.\n"
            "  source /opt/ros/<distro>/setup.bash\n"
            "  source <your_ws>/install/setup.bash\n"
            f"Original import error: {type(ROS_IMPORT_ERROR).__name__}: {ROS_IMPORT_ERROR}"
        )


def discover_bags(input_path, recursive=False):
    path = Path(input_path).expanduser()
    if path.is_file():
        return [path] if path.suffix == ".db3" else []
    if not path.is_dir():
        return []
    patterns = ["**/*.db3"] if recursive else ["*.db3", "*/*.db3"]
    bags = []
    seen = set()
    for pattern in patterns:
        for db in sorted(path.glob(pattern)):
            key = db.resolve()
            if key not in seen:
                seen.add(key)
                bags.append(db)
    return bags


def visualize_bag(bag_db, output_root, args):
    with sqlite3.connect(str(bag_db)) as conn:
        topics = load_topics(conn)
        image_topic = resolve_image_topic(topics, args.image_topic)
        plan_topic = resolve_first_topic(
            topics,
            [args.plan_topic] + split_csv(args.plan_topic_candidates),
        )
        selected = {
            "image": image_topic,
            "lidar": resolve_first_topic(topics, [args.lidar_topic]),
            "odom": resolve_first_topic(topics, [args.odom_topic, "/Odometry", "/lio/robo/odom"]),
            "markers": resolve_first_topic(topics, [args.markers_topic]),
            "goal": resolve_first_topic(topics, [args.goal_topic]),
            "plan": plan_topic,
        }
        if selected["image"] is None:
            raise RuntimeError(f"Missing image topic {args.image_topic} in {bag_db}")
        if selected["odom"] is None:
            raise RuntimeError(f"Missing odom topic {args.odom_topic} in {bag_db}")

        seq_name = bag_name_from_db(bag_db)
        seq_output = output_root / seq_name
        camera_dir = seq_output / "camera"
        map_dir = seq_output / "map"
        if seq_output.exists() and not args.overwrite:
            raise RuntimeError(f"Output exists, pass --overwrite to replace/add frames: {seq_output}")
        camera_dir.mkdir(parents=True, exist_ok=True)
        map_dir.mkdir(parents=True, exist_ok=True)

        print("  topics:")
        for label, topic in selected.items():
            print(f"    {label}: {topic.name if topic else 'missing'}")

        odom = load_timed_messages(conn, selected["odom"])
        if not odom:
            raise RuntimeError(f"No odom messages in {bag_db}")
        odom_poses = [odom_to_pose(item) for item in odom]

        images = load_timed_messages(conn, selected["image"])
        markers = load_timed_messages(conn, selected["markers"]) if selected["markers"] else []
        goals = load_timed_messages(conn, selected["goal"]) if selected["goal"] else []
        plans = load_timed_messages(conn, selected["plan"]) if selected["plan"] else []

        lidar_points = np.zeros((0, 3), dtype=np.float32)
        if selected["lidar"] is not None:
            lidar_points = precompute_lidar_points(
                conn,
                selected["lidar"],
                odom_poses,
                args.lidar_stride,
                math.radians(args.sensor_yaw_deg),
                args.sensor_offset_x,
                args.sensor_offset_y,
            )

        if not images:
            raise RuntimeError(f"No image messages in {bag_db}")

        start_ns = max(min_stamp_ns(images), min_stamp_ns(odom))
        end_ns = min(max_stamp_ns(images), max_stamp_ns(odom))
        if args.start_sec is not None:
            start_ns = max(start_ns, min_stamp_ns(odom) + sec_to_ns(args.start_sec))
        if args.end_sec is not None:
            end_ns = min(end_ns, min_stamp_ns(odom) + sec_to_ns(args.end_sec))
        if end_ns <= start_ns:
            raise RuntimeError("No overlapping image/odom time range to visualize.")

        dt_ns = sec_to_ns(1.0 / args.rate)
        sample_stamps = list(range(start_ns, end_ns + 1, dt_ns))
        fig_size = parse_pair(args.fig_size)
        max_sync_diff_ns = sec_to_ns(args.max_sync_diff_sec)

        index_path = seq_output / "index.csv"
        with index_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
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

            for frame_index, stamp_ns in enumerate(sample_stamps):
                image_item = nearest(images, stamp_ns, max_sync_diff_ns)
                odom_pose = nearest_pose(odom_poses, stamp_ns, max_sync_diff_ns)
                if image_item is None or odom_pose is None:
                    continue

                goal_item = latest_at_or_before(goals, stamp_ns, max_sync_diff_ns)
                markers_item = latest_at_or_before(markers, stamp_ns, max_sync_diff_ns)
                plan_item = latest_at_or_before(plans, stamp_ns, max_sync_diff_ns)

                stem = f"{frame_index:06d}_{stamp_ns}"
                camera_file = camera_dir / f"{stem}.png"
                map_file = map_dir / f"{stem}.png"
                image = image_to_cv2(image_item.msg)
                cv2.imwrite(str(camera_file), image)

                render_map(
                    map_file,
                    stamp_ns,
                    odom_poses,
                    odom_pose,
                    lidar_points,
                    markers_item.msg if markers_item else None,
                    goal_item.msg if goal_item else None,
                    plan_item.msg if plan_item else None,
                    args,
                    fig_size,
                )

                writer.writerow([
                    frame_index,
                    stamp_ns,
                    str(camera_file.relative_to(seq_output)),
                    str(map_file.relative_to(seq_output)),
                    image_item.msg_stamp_ns,
                    odom_pose.stamp_ns,
                    goal_item.msg_stamp_ns if goal_item else "",
                    markers_item.msg_stamp_ns if markers_item else "",
                    plan_item.msg_stamp_ns if plan_item else "",
                ])

                if frame_index % max(int(args.rate) * 5, 1) == 0:
                    print(f"  frame {frame_index}/{len(sample_stamps)}")

        write_manifest(seq_output, bag_db, selected, args)
        print(f"  saved -> {seq_output}")


def load_topics(conn):
    rows = conn.execute("SELECT id, name, type FROM topics ORDER BY id").fetchall()
    topics = {}
    for topic_id, name, type_name in rows:
        try:
            msg_type = get_message(type_name)
        except Exception as exc:
            print(f"  skip type import: {name} ({type_name}): {exc}")
            msg_type = None
        topics[name] = Topic(topic_id, name, type_name, msg_type)
    return topics


def resolve_image_topic(topics, base_topic):
    return resolve_first_topic(
        topics,
        [base_topic, f"{base_topic}/compressed", f"{base_topic}/compressedDepth"],
    )


def resolve_first_topic(topics, candidates):
    for candidate in candidates:
        if candidate and candidate in topics and topics[candidate].msg_type is not None:
            return topics[candidate]
    return None


def load_timed_messages(conn, topic):
    if topic is None:
        return []
    rows = conn.execute(
        "SELECT timestamp, data FROM messages WHERE topic_id = ? ORDER BY timestamp",
        (topic.topic_id,),
    )
    messages = []
    for bag_stamp_ns, serialized in rows:
        msg = deserialize_message(serialized, topic.msg_type)
        messages.append(
            TimedMessage(
                int(bag_stamp_ns),
                message_stamp_ns(msg, int(bag_stamp_ns)),
                msg,
            )
        )
    return messages


def precompute_lidar_points(conn, topic, odom_poses, stride, sensor_yaw, sensor_offset_x, sensor_offset_y):
    stride = max(int(stride), 1)
    rows = conn.execute(
        "SELECT timestamp, data FROM messages WHERE topic_id = ? ORDER BY timestamp",
        (topic.topic_id,),
    )
    output = []
    count = 0
    for bag_stamp_ns, serialized in rows:
        msg = deserialize_message(serialized, topic.msg_type)
        stamp_ns = message_stamp_ns(msg, int(bag_stamp_ns))
        pose = nearest_pose(odom_poses, stamp_ns, None)
        if pose is None:
            continue
        local_xy = lidar_xy(msg, stride)
        if local_xy.size == 0:
            continue
        world_xy = transform_lidar_xy(
            local_xy,
            pose,
            sensor_yaw,
            sensor_offset_x,
            sensor_offset_y,
        )
        stamps = np.full((world_xy.shape[0], 1), stamp_ns, dtype=np.int64)
        output.append(np.hstack([stamps, world_xy.astype(np.float32)]))
        count += 1
        if count % 100 == 0:
            print(f"    lidar frames decoded: {count}")
    if not output:
        return np.zeros((0, 3), dtype=np.float32)
    return np.vstack(output)


def lidar_xy(msg, stride):
    if hasattr(msg, "points"):
        points = getattr(msg, "points")[::stride]
        if not points:
            return np.zeros((0, 2), dtype=np.float32)
        xy = np.array([[float(p.x), float(p.y)] for p in points], dtype=np.float32)
        finite = np.isfinite(xy).all(axis=1)
        return xy[finite]
    return np.zeros((0, 2), dtype=np.float32)


def transform_lidar_xy(local_xy, pose, sensor_yaw, sensor_offset_x, sensor_offset_y):
    c_sensor = math.cos(sensor_yaw)
    s_sensor = math.sin(sensor_yaw)
    sensor_rot = np.array([[c_sensor, -s_sensor], [s_sensor, c_sensor]], dtype=np.float32)
    xy = local_xy @ sensor_rot.T
    xy[:, 0] += sensor_offset_x
    xy[:, 1] += sensor_offset_y

    c = math.cos(pose.yaw)
    s = math.sin(pose.yaw)
    odom_rot = np.array([[c, -s], [s, c]], dtype=np.float32)
    xy = xy @ odom_rot.T
    xy[:, 0] += pose.x
    xy[:, 1] += pose.y
    return xy


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
):
    fig, ax = plt.subplots(figsize=fig_size)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="#dddddd", linewidth=0.5)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(f"{stamp_ns / 1e9:.3f}s")

    visible_lidar = select_lidar(lidar_points, stamp_ns, args.lidar_window_sec, args.max_lidar_points)
    if visible_lidar.size:
        ax.scatter(
            visible_lidar[:, 0],
            visible_lidar[:, 1],
            s=0.25,
            c="#2b2b2b",
            alpha=0.22,
            linewidths=0,
            label="LiDAR",
        )

    history = np.array(
        [[pose.x, pose.y] for pose in odom_poses if pose.stamp_ns <= current_pose.stamp_ns],
        dtype=np.float32,
    )
    if history.size:
        ax.plot(history[:, 0], history[:, 1], color="#1f77b4", linewidth=1.6, label="odom history")

    draw_plan(ax, plan)
    draw_markers(ax, marker_array, args.marker_threshold)
    draw_goal(ax, goal)
    draw_robot(ax, current_pose)

    set_view_limits(ax, visible_lidar, history, current_pose, goal, plan, args.map_padding)
    ax.legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=args.dpi)
    plt.close(fig)


def select_lidar(lidar_points, stamp_ns, window_sec, max_points):
    if lidar_points.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    stamps = lidar_points[:, 0].astype(np.int64)
    end = bisect.bisect_right(stamps, stamp_ns)
    if window_sec and window_sec > 0.0:
        start_stamp = stamp_ns - sec_to_ns(window_sec)
        start = bisect.bisect_left(stamps, start_stamp)
    else:
        start = 0
    xy = lidar_points[start:end, 1:3]
    if xy.shape[0] > max_points:
        step = int(math.ceil(xy.shape[0] / float(max_points)))
        xy = xy[::step]
    return xy


def draw_plan(ax, plan):
    if plan is None or not hasattr(plan, "poses"):
        return
    points = []
    for pose_stamped in plan.poses:
        p = pose_stamped.pose.position
        points.append((float(p.x), float(p.y)))
    if len(points) < 2:
        return
    xy = np.array(points, dtype=np.float32)
    ax.plot(xy[:, 0], xy[:, 1], color="#ff7f0e", linewidth=2.0, label="plan")


def draw_markers(ax, marker_array, threshold):
    if marker_array is None or not hasattr(marker_array, "markers"):
        return
    heatmap_drawn = False
    marker_drawn = False
    for marker in marker_array.markers:
        ns = getattr(marker, "ns", "")
        marker_type = int(getattr(marker, "type", -1))
        if marker_type == 6 and hasattr(marker, "points"):
            xs, ys, colors = [], [], []
            for idx, point in enumerate(marker.points):
                alpha = marker.colors[idx].a if idx < len(marker.colors) else marker.color.a
                if alpha < threshold:
                    continue
                xs.append(float(point.x))
                ys.append(float(point.y))
                if idx < len(marker.colors):
                    c = marker.colors[idx]
                    colors.append((c.r, c.g, c.b, max(c.a, 0.05)))
                else:
                    c = marker.color
                    colors.append((c.r, c.g, c.b, max(c.a, 0.05)))
            if xs:
                ax.scatter(
                    xs,
                    ys,
                    s=9,
                    c=colors,
                    marker="s",
                    linewidths=0,
                    label="fusion markers" if not heatmap_drawn else None,
                )
                heatmap_drawn = True
            continue

        if marker_type in (2, 3, 1):
            p = marker.pose.position
            color = marker.color
            label = ns or "marker"
            ax.scatter(
                [float(p.x)],
                [float(p.y)],
                s=80,
                c=[(color.r, color.g, color.b, max(color.a, 0.2))],
                edgecolors="black",
                linewidths=0.5,
                label=label if not marker_drawn else None,
            )
            marker_drawn = True


def draw_goal(ax, goal):
    if goal is None:
        return
    if hasattr(goal, "point"):
        x, y = float(goal.point.x), float(goal.point.y)
    elif hasattr(goal, "pose"):
        x, y = float(goal.pose.position.x), float(goal.pose.position.y)
    else:
        return
    ax.scatter([x], [y], s=130, c="#ffd400", edgecolors="black", marker="*", label="goal")


def draw_robot(ax, pose):
    length = 0.45
    dx = math.cos(pose.yaw) * length
    dy = math.sin(pose.yaw) * length
    ax.scatter([pose.x], [pose.y], s=70, c="#00a6d6", edgecolors="black", label="current odom")
    ax.arrow(
        pose.x,
        pose.y,
        dx,
        dy,
        width=0.03,
        head_width=0.18,
        head_length=0.18,
        color="#00a6d6",
        length_includes_head=True,
    )


def set_view_limits(ax, lidar_xy, history, current_pose, goal, plan, padding):
    xs = [current_pose.x]
    ys = [current_pose.y]
    if lidar_xy.size:
        xs.extend(lidar_xy[:, 0].tolist())
        ys.extend(lidar_xy[:, 1].tolist())
    if history.size:
        xs.extend(history[:, 0].tolist())
        ys.extend(history[:, 1].tolist())
    if goal is not None and hasattr(goal, "point"):
        xs.append(float(goal.point.x))
        ys.append(float(goal.point.y))
    if plan is not None and hasattr(plan, "poses"):
        for pose_stamped in plan.poses:
            xs.append(float(pose_stamped.pose.position.x))
            ys.append(float(pose_stamped.pose.position.y))

    xmin, xmax = min(xs) - padding, max(xs) + padding
    ymin, ymax = min(ys) - padding, max(ys) + padding
    if xmax - xmin < 4.0:
        center = 0.5 * (xmin + xmax)
        xmin, xmax = center - 2.0, center + 2.0
    if ymax - ymin < 4.0:
        center = 0.5 * (ymin + ymax)
        ymin, ymax = center - 2.0, center + 2.0
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)


def image_to_cv2(msg):
    type_name = type(msg).__name__.lower()
    if "compressed" in type_name or hasattr(msg, "format") and not hasattr(msg, "encoding"):
        data = bytes(msg.data)
        image_bytes = extract_embedded_image_bytes(data) or data
        decoded = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if decoded is None:
            raise RuntimeError("Failed to decode compressed image.")
        return decoded

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


def extract_embedded_image_bytes(data):
    png_magic = b"\x89PNG\r\n\x1a\n"
    png_pos = data.find(png_magic)
    if png_pos >= 0:
        return data[png_pos:]
    jpeg_magic = b"\xff\xd8"
    jpeg_pos = data.find(jpeg_magic)
    if jpeg_pos >= 0:
        return data[jpeg_pos:]
    return None


def odom_to_pose(item):
    msg = item.msg
    p = msg.pose.pose.position
    q = msg.pose.pose.orientation
    return OdomPose(item.msg_stamp_ns, float(p.x), float(p.y), float(p.z), quaternion_yaw(q))


def quaternion_yaw(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def message_stamp_ns(msg, fallback_ns):
    header = getattr(msg, "header", None)
    stamp = getattr(header, "stamp", None)
    if stamp is None:
        return fallback_ns
    sec = int(getattr(stamp, "sec", 0))
    nanosec = int(getattr(stamp, "nanosec", getattr(stamp, "nsec", 0)))
    value = sec * 1_000_000_000 + nanosec
    return value if value > 0 else fallback_ns


def nearest(items, stamp_ns, max_diff_ns):
    if not items:
        return None
    stamps = [item.msg_stamp_ns for item in items]
    idx = bisect.bisect_left(stamps, stamp_ns)
    candidates = []
    if idx < len(items):
        candidates.append(items[idx])
    if idx > 0:
        candidates.append(items[idx - 1])
    best = min(candidates, key=lambda item: abs(item.msg_stamp_ns - stamp_ns))
    if max_diff_ns is not None and abs(best.msg_stamp_ns - stamp_ns) > max_diff_ns:
        return None
    return best


def latest_at_or_before(items, stamp_ns, max_age_ns):
    if not items:
        return None
    stamps = [item.msg_stamp_ns for item in items]
    idx = bisect.bisect_right(stamps, stamp_ns) - 1
    if idx < 0:
        return None
    item = items[idx]
    if max_age_ns is not None and stamp_ns - item.msg_stamp_ns > max_age_ns:
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


def min_stamp_ns(items):
    return min(item.msg_stamp_ns for item in items)


def max_stamp_ns(items):
    return max(item.msg_stamp_ns for item in items)


def sec_to_ns(value):
    return int(round(float(value) * 1_000_000_000))


def parse_pair(text):
    parts = [float(part.strip()) for part in text.split(",")]
    if len(parts) != 2:
        raise ValueError("--fig-size must be two comma-separated numbers")
    return parts[0], parts[1]


def split_csv(text):
    return [item.strip() for item in str(text).split(",") if item.strip()]


def bag_name_from_db(db_path):
    db_path = Path(db_path)
    if db_path.parent.name and db_path.parent.name != ".":
        return db_path.parent.name
    return db_path.stem


def write_manifest(output_dir, bag_db, selected, args):
    manifest = output_dir / "manifest.txt"
    with manifest.open("w", encoding="utf-8") as f:
        f.write(f"bag={bag_db}\n")
        f.write(f"rate={args.rate}\n")
        for label, topic in selected.items():
            if topic is None:
                f.write(f"{label}_topic=\n")
            else:
                f.write(f"{label}_topic={topic.name}\n")
                f.write(f"{label}_type={topic.type_name}\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)

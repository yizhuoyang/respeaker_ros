#!/usr/bin/env python3
"""Fuse audio and visual occupancy maps and publish the weighted argmax goal."""

import json
import threading
import time

import numpy as np


def grid_to_array(msg):
    data = np.asarray(msg.data, dtype=np.float32).reshape(
        int(msg.info.height), int(msg.info.width)
    )
    data[data < 0.0] = 0.0
    return np.clip(data / 100.0, 0.0, 1.0)


def normalize_map(values):
    values = np.asarray(values, dtype=np.float32)
    maximum = float(np.max(values)) if values.size else 0.0
    if maximum <= 1e-12:
        return np.zeros_like(values, dtype=np.float32), maximum
    return values / maximum, maximum


def same_geometry(a, b, tolerance=1e-6):
    return (
        a.header.frame_id == b.header.frame_id
        and int(a.info.width) == int(b.info.width)
        and int(a.info.height) == int(b.info.height)
        and abs(float(a.info.resolution) - float(b.info.resolution)) <= tolerance
        and abs(float(a.info.origin.position.x) - float(b.info.origin.position.x)) <= tolerance
        and abs(float(a.info.origin.position.y) - float(b.info.origin.position.y)) <= tolerance
    )


def same_frame(a, b):
    return a.header.frame_id == b.header.frame_id


def cell_center(msg, row, column):
    resolution = float(msg.info.resolution)
    return (
        float(msg.info.origin.position.x) + (float(column) + 0.5) * resolution,
        float(msg.info.origin.position.y) + (float(row) + 0.5) * resolution,
    )


def audio_heatmap_cell_point(msg, row, column):
    """Draw like audio_map_heatmap: use the audio map grid-point coordinates."""
    resolution = float(msg.info.resolution)
    return (
        float(msg.info.origin.position.x) + (float(column) + 0.5) * resolution,
        float(msg.info.origin.position.y) + (float(row) + 0.5) * resolution,
    )


def geometry_summary(msg):
    return {
        "frame_id": msg.header.frame_id,
        "width": int(msg.info.width),
        "height": int(msg.info.height),
        "resolution": float(msg.info.resolution),
        "origin_x": float(msg.info.origin.position.x),
        "origin_y": float(msg.info.origin.position.y),
    }


def copy_grid_info(source_info):
    from geometry_msgs.msg import Pose
    from nav_msgs.msg import MapMetaData

    info = MapMetaData()
    info.map_load_time = source_info.map_load_time
    info.resolution = source_info.resolution
    info.width = source_info.width
    info.height = source_info.height
    info.origin = Pose()
    info.origin.position.x = source_info.origin.position.x
    info.origin.position.y = source_info.origin.position.y
    info.origin.position.z = source_info.origin.position.z
    info.origin.orientation.x = source_info.origin.orientation.x
    info.origin.orientation.y = source_info.origin.orientation.y
    info.origin.orientation.z = source_info.origin.orientation.z
    info.origin.orientation.w = source_info.origin.orientation.w
    return info


def resample_to_reference(source_msg, source_values, reference_msg):
    src_resolution = float(source_msg.info.resolution)
    ref_resolution = float(reference_msg.info.resolution)
    src_origin_x = float(source_msg.info.origin.position.x)
    src_origin_y = float(source_msg.info.origin.position.y)
    ref_origin_x = float(reference_msg.info.origin.position.x)
    ref_origin_y = float(reference_msg.info.origin.position.y)
    ref_height = int(reference_msg.info.height)
    ref_width = int(reference_msg.info.width)

    ref_rows, ref_columns = np.indices((ref_height, ref_width), dtype=np.float32)
    world_x = ref_origin_x + (ref_columns + 0.5) * ref_resolution
    world_y = ref_origin_y + (ref_rows + 0.5) * ref_resolution
    src_columns = np.floor((world_x - src_origin_x) / src_resolution).astype(np.int32)
    src_rows = np.floor((world_y - src_origin_y) / src_resolution).astype(np.int32)
    valid = (
        (src_columns >= 0)
        & (src_columns < int(source_msg.info.width))
        & (src_rows >= 0)
        & (src_rows < int(source_msg.info.height))
    )
    output = np.zeros((ref_height, ref_width), dtype=np.float32)
    output[valid] = source_values[src_rows[valid], src_columns[valid]]
    return output


def map_world_bounds(msg):
    resolution = float(msg.info.resolution)
    origin_x = float(msg.info.origin.position.x)
    origin_y = float(msg.info.origin.position.y)
    return (
        origin_x,
        origin_y,
        origin_x + int(msg.info.width) * resolution,
        origin_y + int(msg.info.height) * resolution,
    )


def map_nonzero_world_points(msg, values):
    rows, columns = np.nonzero(values > 0.0)
    if rows.size == 0:
        return (
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )
    resolution = float(msg.info.resolution)
    origin_x = float(msg.info.origin.position.x)
    origin_y = float(msg.info.origin.position.y)
    xs = origin_x + (columns.astype(np.float32) + 0.5) * resolution
    ys = origin_y + (rows.astype(np.float32) + 0.5) * resolution
    return xs, ys, values[rows, columns].astype(np.float32)


class AudioVisualGoalFusionNode:
    def __init__(self):
        import rospy
        from geometry_msgs.msg import PointStamped, PoseStamped
        from nav_msgs.msg import OccupancyGrid
        from std_msgs.msg import String
        from visualization_msgs.msg import Marker, MarkerArray

        self.rospy = rospy
        self.lock = threading.Lock()
        self.audio_map = None
        self.visual_map = None
        self.last_status = "initializing"
        self.last_goal = None
        self.last_audio_max = 0.0
        self.last_visual_max = 0.0
        self.last_fused_max = 0.0
        self.last_fusion_mode = "none"
        self.last_reference_geometry = None
        self.last_published_goal_pose = None

        self.audio_map_topic = rospy.get_param("~audio_map_topic", "/sslnet_audio_map/map")
        self.visual_map_topic = rospy.get_param("~visual_map_topic", "/yoloe_visual_map/map")
        self.audio_weight = float(rospy.get_param("~audio_weight", 1.0))
        self.visual_weight = float(rospy.get_param("~visual_weight", 1.0))
        self.weight_epsilon = float(rospy.get_param("~weight_epsilon", 1e-9))
        self.global_frame_id = str(rospy.get_param("~global_frame_id", "camera_init")).strip()
        self.fusion_mode = str(rospy.get_param("~fusion_mode", "global_grid_sum")).strip().lower()
        if self.fusion_mode not in ("global_grid_sum", "world_overlay", "grid_resample"):
            raise ValueError(
                "~fusion_mode must be 'global_grid_sum', 'world_overlay', or 'grid_resample'."
            )
        self.normalize_inputs = bool(rospy.get_param("~normalize_inputs", False))
        self.allow_resample = bool(rospy.get_param("~allow_resample", True))
        self.reference_map = str(rospy.get_param("~reference_map", "audio")).strip().lower()
        if self.reference_map not in ("audio", "visual"):
            raise ValueError("~reference_map must be 'audio' or 'visual'.")
        self.output_stamp_mode = str(
            rospy.get_param("~output_stamp_mode", "source")
        ).strip().lower()
        if self.output_stamp_mode not in ("source", "current", "zero"):
            raise ValueError("~output_stamp_mode must be 'source', 'current', or 'zero'.")
        self.min_fused_value = float(rospy.get_param("~min_fused_value", 0.0))
        self.goal_z = float(rospy.get_param("~goal_z", 0.0))
        self.goal_pose_topic = rospy.get_param(
            "~goal_pose_topic", "/move_based_simple/goal_raw"
        )
        self.goal_pose_yaw_rad = float(rospy.get_param("~goal_pose_yaw_rad", 0.0))
        self.goal_pose_publish_period_sec = float(
            rospy.get_param("~goal_pose_publish_period_sec", 5.0)
        )
        self.goal_pose_initial_delay_sec = float(
            rospy.get_param("~goal_pose_initial_delay_sec", 8.0)
        )
        self._goal_pose_start_wall_time = time.monotonic()
        self._last_goal_pose_publish_wall_time = None
        self.marker_height = float(rospy.get_param("~marker_height", 0.18))
        self.overlay_resolution = float(rospy.get_param("~overlay_resolution", 0.05))
        self.publish_heatmap_marker = bool(rospy.get_param("~publish_heatmap_marker", True))
        self.heatmap_marker_threshold = float(
            np.clip(rospy.get_param("~heatmap_marker_threshold", 0.10), 0.0, 1.0)
        )
        self.heatmap_marker_max_cells = max(
            int(rospy.get_param("~heatmap_marker_max_cells", 6000)), 1
        )
        self.heatmap_marker_height = float(rospy.get_param("~heatmap_marker_height", 0.03))

        self.map_pub = rospy.Publisher("~map", OccupancyGrid, queue_size=1, latch=True)
        self.goal_pub = rospy.Publisher("~goal", PointStamped, queue_size=2, latch=True)
        self.goal_pose_pub = rospy.Publisher(
            self.goal_pose_topic, PoseStamped, queue_size=2, latch=True
        )
        self.goal_json_pub = rospy.Publisher("~goal_json", String, queue_size=2, latch=True)
        self.marker_pub = rospy.Publisher("~marker", Marker, queue_size=1, latch=True)
        self.markers_pub = rospy.Publisher("~markers", MarkerArray, queue_size=1, latch=True)
        self.status_pub = rospy.Publisher("~status", String, queue_size=1, latch=True)

        self.audio_sub = rospy.Subscriber(
            self.audio_map_topic, OccupancyGrid, self.audio_callback, queue_size=1
        )
        self.visual_sub = rospy.Subscriber(
            self.visual_map_topic, OccupancyGrid, self.visual_callback, queue_size=1
        )
        self.status_timer = rospy.Timer(rospy.Duration(1.0), self.status_callback)

        rospy.loginfo(
            "Audio-visual goal fusion started: audio=%s visual=%s weights=(%.3f, %.3f) "
            "fusion_mode=%s normalize_inputs=%s allow_resample=%s reference_map=%s output_stamp_mode=%s",
            self.audio_map_topic,
            self.visual_map_topic,
            self.audio_weight,
            self.visual_weight,
            self.fusion_mode,
            self.normalize_inputs,
            self.allow_resample,
            self.reference_map,
            self.output_stamp_mode,
        )
        rospy.loginfo(
            "Audio-visual fusion outputs: map=%s goal=%s goal_pose=%s marker=%s markers=%s status=%s",
            rospy.resolve_name("~map"),
            rospy.resolve_name("~goal"),
            self.goal_pose_topic,
            rospy.resolve_name("~marker"),
            rospy.resolve_name("~markers"),
            rospy.resolve_name("~status"),
        )

    def audio_callback(self, msg):
        with self.lock:
            self.audio_map = msg
        self.try_publish()

    def visual_callback(self, msg):
        with self.lock:
            self.visual_map = msg
        self.try_publish()

    def try_publish(self):
        with self.lock:
            audio_msg = self.audio_map
            visual_msg = self.visual_map

        if self.audio_weight > self.weight_epsilon and self.visual_weight <= self.weight_epsilon:
            if audio_msg is None:
                self.last_status = "waiting_for_maps"
                return
            if not self.accept_global_frame(audio_msg, "audio"):
                return
            self.publish_single_map_goal(audio_msg, "audio")
            return

        if self.visual_weight > self.weight_epsilon and self.audio_weight <= self.weight_epsilon:
            if visual_msg is None:
                self.last_status = "waiting_for_maps"
                return
            if not self.accept_global_frame(visual_msg, "visual"):
                return
            self.publish_single_map_goal(visual_msg, "visual")
            return

        if audio_msg is None or visual_msg is None:
            self.last_status = "waiting_for_maps"
            return
        if not self.accept_global_frame(audio_msg, "audio"):
            return
        if not self.accept_global_frame(visual_msg, "visual"):
            return

        if not same_frame(audio_msg, visual_msg):
            self.last_status = "map_geometry_mismatch"
            self.rospy.logwarn_throttle(
                5.0,
                "Cannot fuse maps because frames differ: audio frame=%s visual frame=%s",
                audio_msg.header.frame_id,
                visual_msg.header.frame_id,
            )
            return

        if self.fusion_mode in ("global_grid_sum", "world_overlay", "grid_resample"):
            self.publish_global_grid_sum_goal(audio_msg, visual_msg)
            return

    def accept_global_frame(self, msg, source_name):
        if not self.global_frame_id:
            return True
        if msg.header.frame_id == self.global_frame_id:
            return True
        self.last_status = "%s_frame_mismatch" % source_name
        self.rospy.logwarn_throttle(
            5.0,
            "Ignoring %s map in frame '%s'; expected global odom frame '%s'.",
            source_name,
            msg.header.frame_id,
            self.global_frame_id,
        )
        return False

    def publish_global_grid_sum_goal(self, audio_msg, visual_msg):
        geometry_matches = same_geometry(audio_msg, visual_msg)
        if not geometry_matches and not self.allow_resample:
            self.last_status = "map_geometry_mismatch"
            self.rospy.logwarn_throttle(
                5.0,
                "Cannot fuse maps because geometry differs and ~allow_resample is false: "
                "audio size=%dx%d res=%.3f origin=(%.3f, %.3f), "
                "visual size=%dx%d res=%.3f origin=(%.3f, %.3f)",
                audio_msg.info.width,
                audio_msg.info.height,
                audio_msg.info.resolution,
                audio_msg.info.origin.position.x,
                audio_msg.info.origin.position.y,
                visual_msg.info.width,
                visual_msg.info.height,
                visual_msg.info.resolution,
                visual_msg.info.origin.position.x,
                visual_msg.info.origin.position.y,
            )
            return

        reference_msg = audio_msg
        if self.reference_map != "audio":
            self.rospy.logwarn_throttle(
                5.0,
                "For RViz consistency, fused heatmap uses audio_map_heatmap geometry. "
                "Ignoring ~reference_map=%s for two-map fusion.",
                self.reference_map,
            )

        audio = grid_to_array(audio_msg)
        visual = grid_to_array(visual_msg)
        resampled = False

        if not geometry_matches:
            visual = resample_to_reference(visual_msg, visual, reference_msg)
            resampled = True
            self.rospy.logwarn_throttle(
                5.0,
                "Map geometry differs; adding maps on the audio heatmap grid in frame %s.",
                reference_msg.header.frame_id,
            )

        if self.normalize_inputs:
            audio_used, audio_max = normalize_map(audio)
            visual_used, visual_max = normalize_map(visual)
        else:
            audio_used, audio_max = audio, float(np.max(audio)) if audio.size else 0.0
            visual_used, visual_max = visual, float(np.max(visual)) if visual.size else 0.0

        fused = self.audio_weight * audio_used + self.visual_weight * visual_used
        fused_max = float(np.max(fused)) if fused.size else 0.0
        if fused_max <= self.min_fused_value:
            self.last_status = "fused_map_below_threshold"
            self.update_status(None, audio_max, visual_max, fused_max)
            return

        row, column = np.unravel_index(int(np.argmax(fused)), fused.shape)
        x, y = cell_center(reference_msg, row, column)
        stamp = self.output_stamp(reference_msg)
        frame_id = reference_msg.header.frame_id
        self.update_status(
            (x, y, row, column, frame_id, geometry_summary(reference_msg)),
            audio_max,
            visual_max,
            fused_max,
        )
        self.publish_outputs(
            reference_msg,
            fused,
            frame_id,
            stamp,
            x,
            y,
            row,
            column,
            audio_max,
            visual_max,
            resampled,
            fusion_mode="global_grid_sum",
            marker_reference_msg=audio_msg,
        )
        self.last_status = "running"

    def publish_world_overlay_goal(self, audio_msg, visual_msg):
        audio = grid_to_array(audio_msg)
        visual = grid_to_array(visual_msg)
        if self.normalize_inputs:
            audio_used, audio_max = normalize_map(audio)
            visual_used, visual_max = normalize_map(visual)
        else:
            audio_used, audio_max = audio, float(np.max(audio)) if audio.size else 0.0
            visual_used, visual_max = visual, float(np.max(visual)) if visual.size else 0.0

        scores = {}
        self.accumulate_world_scores(scores, audio_msg, audio_used, self.audio_weight)
        self.accumulate_world_scores(scores, visual_msg, visual_used, self.visual_weight)
        if not scores:
            self.last_status = "world_overlay_empty"
            self.update_status(None, audio_max, visual_max, 0.0)
            return

        best_key, best_score = max(scores.items(), key=lambda item: item[1])
        if best_score <= self.min_fused_value:
            self.last_status = "fused_map_below_threshold"
            self.update_status(None, audio_max, visual_max, best_score)
            return

        x = best_key[0] * self.overlay_resolution
        y = best_key[1] * self.overlay_resolution
        reference_msg = audio_msg if self.reference_map == "audio" else visual_msg
        stamp = self.output_stamp(reference_msg)
        frame_id = audio_msg.header.frame_id
        fused_grid, row, column = self.world_scores_to_grid(scores, reference_msg, best_key)
        geometry = geometry_summary(fused_grid)
        self.update_status((x, y, row, column, frame_id, geometry), audio_max, visual_max, best_score)
        self.publish_outputs(
            fused_grid,
            grid_to_array(fused_grid),
            frame_id,
            stamp,
            x,
            y,
            row,
            column,
            audio_max,
            visual_max,
            False,
            preserve_grid_data=fused_grid.data,
            fusion_mode="world_overlay",
        )
        self.last_status = "running"

    def accumulate_world_scores(self, scores, msg, values, weight):
        if weight <= self.weight_epsilon:
            return
        xs, ys, vals = map_nonzero_world_points(msg, values)
        if vals.size == 0:
            return
        key_x = np.rint(xs / self.overlay_resolution).astype(np.int64)
        key_y = np.rint(ys / self.overlay_resolution).astype(np.int64)
        weighted = vals * float(weight)
        for kx, ky, value in zip(key_x.tolist(), key_y.tolist(), weighted.tolist()):
            key = (int(kx), int(ky))
            scores[key] = scores.get(key, 0.0) + float(value)

    def world_scores_to_grid(self, scores, reference_msg, best_key):
        from nav_msgs.msg import OccupancyGrid

        resolution = float(reference_msg.info.resolution)
        width = int(reference_msg.info.width)
        height = int(reference_msg.info.height)
        origin_x = float(reference_msg.info.origin.position.x)
        origin_y = float(reference_msg.info.origin.position.y)
        grid_values = np.zeros((height, width), dtype=np.float32)

        max_score = max(scores.values()) if scores else 0.0
        for (kx, ky), value in scores.items():
            x = kx * self.overlay_resolution
            y = ky * self.overlay_resolution
            column = int(np.floor((x - origin_x) / resolution))
            row = int(np.floor((y - origin_y) / resolution))
            if 0 <= row < height and 0 <= column < width:
                normalized = float(value) / max_score if max_score > 1e-12 else 0.0
                grid_values[row, column] = max(grid_values[row, column], normalized)

        best_x = best_key[0] * self.overlay_resolution
        best_y = best_key[1] * self.overlay_resolution
        best_column = int(np.floor((best_x - origin_x) / resolution))
        best_row = int(np.floor((best_y - origin_y) / resolution))

        grid = OccupancyGrid()
        grid.header.frame_id = reference_msg.header.frame_id
        grid.header.stamp = self.output_stamp(reference_msg)
        grid.info = copy_grid_info(reference_msg.info)
        grid.data = np.rint(100.0 * grid_values).astype(np.int8).reshape(-1).tolist()
        return grid, best_row, best_column

    def publish_single_map_goal(self, source_msg, source_name):
        values = grid_to_array(source_msg)
        source_max = float(np.max(values)) if values.size else 0.0
        if source_max <= self.min_fused_value:
            self.last_status = "%s_map_below_threshold" % source_name
            if source_name == "audio":
                self.update_status(None, source_max, 0.0, source_max)
            else:
                self.update_status(None, 0.0, source_max, source_max)
            return

        row, column = np.unravel_index(int(np.argmax(values)), values.shape)
        x, y = cell_center(source_msg, row, column)
        stamp = self.output_stamp(source_msg)
        frame_id = source_msg.header.frame_id
        audio_max = source_max if source_name == "audio" else 0.0
        visual_max = source_max if source_name == "visual" else 0.0
        self.update_status(
            (x, y, row, column, frame_id, geometry_summary(source_msg)),
            audio_max,
            visual_max,
            source_max,
        )
        self.publish_outputs(
            source_msg,
            values,
            frame_id,
            stamp,
            x,
            y,
            row,
            column,
            audio_max,
            visual_max,
            False,
            preserve_grid_data=source_msg.data,
            fusion_mode="%s_only_passthrough" % source_name,
        )
        self.last_status = "%s_only_passthrough" % source_name

    def output_stamp(self, reference_msg):
        if self.output_stamp_mode == "zero":
            return self.rospy.Time(0)
        if self.output_stamp_mode == "current":
            return self.rospy.Time.now()
        if reference_msg.header.stamp.to_sec() > 0.0:
            return reference_msg.header.stamp
        return self.rospy.Time.now()

    def publish_outputs(
        self,
        reference_msg,
        fused,
        frame_id,
        stamp,
        x,
        y,
        row,
        column,
        audio_max,
        visual_max,
        resampled,
        preserve_grid_data=None,
        fusion_mode="weighted_sum",
        marker_reference_msg=None,
    ):
        from geometry_msgs.msg import Point, PointStamped, PoseStamped
        from nav_msgs.msg import OccupancyGrid
        from std_msgs.msg import String
        from visualization_msgs.msg import Marker, MarkerArray

        maximum = float(np.max(fused)) if fused.size else 0.0
        if maximum > 1e-12:
            fused_for_grid = np.clip(fused / maximum, 0.0, 1.0)
        else:
            fused_for_grid = np.zeros_like(fused, dtype=np.float32)

        grid = OccupancyGrid()
        grid.header.frame_id = frame_id
        grid.header.stamp = stamp
        grid.info = copy_grid_info(reference_msg.info)
        grid.info.origin.orientation.x = 0.0
        grid.info.origin.orientation.y = 0.0
        grid.info.origin.orientation.z = 0.0
        grid.info.origin.orientation.w = 1.0
        if preserve_grid_data is None:
            grid.data = np.rint(100.0 * fused_for_grid).astype(np.int8).reshape(-1).tolist()
        else:
            grid.data = list(preserve_grid_data)
        self.map_pub.publish(grid)

        goal = PointStamped()
        goal.header.frame_id = frame_id
        goal.header.stamp = stamp
        goal.point.x = float(x)
        goal.point.y = float(y)
        goal.point.z = self.goal_z
        self.goal_pub.publish(goal)
        if self.should_publish_goal_pose():
            goal_pose = self.make_goal_pose(frame_id, stamp, x, y)
            self.goal_pose_pub.publish(goal_pose)
            with self.lock:
                self.last_published_goal_pose = goal_pose

        marker_header = (marker_reference_msg or reference_msg).header
        marker = Marker()
        marker.header = marker_header
        marker.ns = "audio_visual_goal"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position = Point(float(x), float(y), self.marker_height)
        marker.pose.orientation.w = 1.0
        marker.scale.x = marker.scale.y = marker.scale.z = 0.30
        marker.color.r = 1.0
        marker.color.g = 0.85
        marker.color.b = 0.05
        marker.color.a = 0.95
        self.marker_pub.publish(marker)
        self.markers_pub.publish(
            self.make_markers(fused, marker_reference_msg or reference_msg, marker)
        )

        payload = {
            "frame_id": frame_id,
            "stamp": stamp.to_sec(),
            "x": float(x),
            "y": float(y),
            "z": self.goal_z,
            "row": int(row),
            "column": int(column),
            "audio_weight": self.audio_weight,
            "visual_weight": self.visual_weight,
            "fusion_mode": fusion_mode,
            "normalize_inputs": self.normalize_inputs,
            "reference_map": self.reference_map,
            "output_stamp_mode": self.output_stamp_mode,
            "resampled": bool(resampled),
            "audio_max": float(audio_max),
            "visual_max": float(visual_max),
            "fused_max": float(maximum),
        }
        self.goal_json_pub.publish(String(data=json.dumps(payload, ensure_ascii=True)))
        self.set_fusion_mode(fusion_mode)

    def should_publish_goal_pose(self):
        if self.goal_pose_publish_period_sec <= 0.0:
            return True
        now = time.monotonic()
        if self._last_goal_pose_publish_wall_time is None:
            if now - self._goal_pose_start_wall_time < self.goal_pose_initial_delay_sec:
                return False
            self._last_goal_pose_publish_wall_time = now
            return True
        if now - self._last_goal_pose_publish_wall_time < self.goal_pose_publish_period_sec:
            return False
        self._last_goal_pose_publish_wall_time = now
        return True

    def make_goal_pose(self, frame_id, stamp, x, y):
        from geometry_msgs.msg import PoseStamped

        pose = PoseStamped()
        pose.header.frame_id = frame_id
        pose.header.stamp = stamp
        pose.pose.position.x = float(x)
        pose.pose.position.y = float(y)
        pose.pose.position.z = self.goal_z
        half_yaw = 0.5 * self.goal_pose_yaw_rad
        pose.pose.orientation.z = float(np.sin(half_yaw))
        pose.pose.orientation.w = float(np.cos(half_yaw))
        return pose

    def make_markers(self, fused, heatmap_reference_msg, goal_marker):
        from geometry_msgs.msg import Point
        from std_msgs.msg import ColorRGBA
        from visualization_msgs.msg import Marker, MarkerArray

        with self.lock:
            last_published_goal_pose = self.last_published_goal_pose

        markers = []
        if self.publish_heatmap_marker:
            probability = np.clip(np.asarray(fused, dtype=np.float32), 0.0, None)
            maximum = float(probability.max()) if probability.size else 0.0
            if maximum > 0.0:
                normalized = probability / maximum
            else:
                normalized = np.zeros_like(probability, dtype=np.float32)

            heatmap = Marker()
            heatmap.header = heatmap_reference_msg.header
            heatmap.ns = "audio_visual_fused_heatmap"
            heatmap.id = 0
            heatmap.type = Marker.CUBE_LIST
            heatmap.action = Marker.ADD
            heatmap.pose.orientation.w = 1.0
            heatmap.scale.x = float(heatmap_reference_msg.info.resolution)
            heatmap.scale.y = float(heatmap_reference_msg.info.resolution)
            heatmap.scale.z = max(float(heatmap_reference_msg.info.resolution) * 0.10, 0.01)
            heatmap.color.a = 1.0

            selected = np.flatnonzero(
                normalized.reshape(-1) >= self.heatmap_marker_threshold
            )
            if selected.size > self.heatmap_marker_max_cells:
                values = normalized.reshape(-1)[selected]
                top = np.argpartition(values, -self.heatmap_marker_max_cells)[
                    -self.heatmap_marker_max_cells:
                ]
                selected = selected[top]

            if selected.size:
                for flat_index in selected.tolist():
                    row, column = np.unravel_index(flat_index, normalized.shape)
                    value = float(normalized[row, column])
                    x, y = audio_heatmap_cell_point(heatmap_reference_msg, row, column)
                    heatmap.points.append(Point(float(x), float(y), self.heatmap_marker_height))
                    heatmap.colors.append(
                        ColorRGBA(
                            r=min(1.0, 2.0 * value),
                            g=min(1.0, 2.0 * (1.0 - value)),
                            b=0.05,
                            a=0.15 + 0.70 * value,
                        )
                    )
            else:
                heatmap.action = Marker.DELETE
            markers.append(heatmap)

        goal_copy = Marker()
        goal_copy.header = heatmap_reference_msg.header
        goal_copy.ns = "audio_visual_goal"
        goal_copy.id = 1
        goal_copy.type = goal_marker.type
        goal_copy.action = goal_marker.action
        goal_copy.pose = goal_marker.pose
        goal_copy.scale = goal_marker.scale
        goal_copy.color = goal_marker.color
        markers.append(goal_copy)

        label = Marker()
        label.header = heatmap_reference_msg.header
        label.ns = "audio_visual_goal"
        label.id = 2
        label.type = Marker.TEXT_VIEW_FACING
        label.action = Marker.ADD
        label.pose.position = Point(
            float(goal_marker.pose.position.x),
            float(goal_marker.pose.position.y),
            float(goal_marker.pose.position.z) + 0.35,
        )
        label.pose.orientation.w = 1.0
        label.scale.z = 0.20
        label.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
        label.text = "fusion goal\n(%.2f, %.2f)" % (
            goal_marker.pose.position.x,
            goal_marker.pose.position.y,
        )
        markers.append(label)

        if last_published_goal_pose is not None:
            planner_goal = Marker()
            planner_goal.header = heatmap_reference_msg.header
            planner_goal.ns = "audio_visual_planner_goal"
            planner_goal.id = 0
            planner_goal.type = Marker.SPHERE
            planner_goal.action = Marker.ADD
            planner_goal.pose.position.x = last_published_goal_pose.pose.position.x
            planner_goal.pose.position.y = last_published_goal_pose.pose.position.y
            planner_goal.pose.position.z = self.marker_height
            planner_goal.pose.orientation.w = 1.0
            planner_goal.scale.x = planner_goal.scale.y = planner_goal.scale.z = 0.30
            planner_goal.color = ColorRGBA(r=1.0, g=0.05, b=0.95, a=0.98)
            markers.append(planner_goal)

            planner_label = Marker()
            planner_label.header = heatmap_reference_msg.header
            planner_label.ns = "audio_visual_planner_goal"
            planner_label.id = 1
            planner_label.type = Marker.TEXT_VIEW_FACING
            planner_label.action = Marker.ADD
            planner_label.pose.position = Point(
                float(last_published_goal_pose.pose.position.x),
                float(last_published_goal_pose.pose.position.y),
                self.marker_height + 0.35,
            )
            planner_label.pose.orientation.w = 1.0
            planner_label.scale.z = 0.20
            planner_label.color = ColorRGBA(r=1.0, g=0.05, b=0.95, a=1.0)
            planner_label.text = "planner goal\n(%.2f, %.2f)" % (
                last_published_goal_pose.pose.position.x,
                last_published_goal_pose.pose.position.y,
            )
            markers.append(planner_label)
        return MarkerArray(markers=markers)

    def update_status(self, goal, audio_max, visual_max, fused_max):
        with self.lock:
            self.last_audio_max = float(audio_max)
            self.last_visual_max = float(visual_max)
            self.last_fused_max = float(fused_max)
            self.last_goal = goal
            if goal is not None:
                self.last_reference_geometry = goal[5] if len(goal) > 5 else None

    def set_fusion_mode(self, mode):
        with self.lock:
            self.last_fusion_mode = str(mode)

    def status_callback(self, _event):
        from std_msgs.msg import String

        with self.lock:
            audio_ready = self.audio_map is not None
            visual_ready = self.visual_map is not None
            goal = self.last_goal
            reference_geometry = self.last_reference_geometry
            payload = {
                "status": self.last_status,
                "audio_map_topic": self.audio_map_topic,
                "visual_map_topic": self.visual_map_topic,
                "audio_ready": audio_ready,
                "visual_ready": visual_ready,
                "audio_weight": self.audio_weight,
                "visual_weight": self.visual_weight,
                "fusion_mode": self.last_fusion_mode,
                "configured_fusion_mode": self.fusion_mode,
                "overlay_resolution": self.overlay_resolution,
                "global_frame_id": self.global_frame_id,
                "normalize_inputs": self.normalize_inputs,
                "allow_resample": self.allow_resample,
                "reference_map": self.reference_map,
                "output_stamp_mode": self.output_stamp_mode,
                "audio_max": self.last_audio_max,
                "visual_max": self.last_visual_max,
                "fused_max": self.last_fused_max,
                "reference_geometry": reference_geometry,
                "goal": None
                if goal is None
                else {
                    "x": float(goal[0]),
                    "y": float(goal[1]),
                    "row": int(goal[2]),
                    "column": int(goal[3]),
                    "frame_id": goal[4],
                },
            }
        self.status_pub.publish(String(data=json.dumps(payload, ensure_ascii=True)))


def main():
    import rospy

    rospy.init_node("audio_visual_goal_fusion")
    AudioVisualGoalFusionNode()
    rospy.spin()


if __name__ == "__main__":
    main()

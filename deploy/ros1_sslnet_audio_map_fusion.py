#!/usr/bin/env python3
"""Fuse streaming SSLNet observations and odometry into a global RViz audio map."""

from collections import deque
import json
import math
from pathlib import Path
import sys
import threading

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utlis.prob_update_doa import StreamingSourceMapFusion  # noqa: E402


def quaternion_to_yaw(quaternion):
    """Return ROS planar yaw: 0 is +x and positive rotation points toward +y."""
    x = float(quaternion.x)
    y = float(quaternion.y)
    z = float(quaternion.z)
    w = float(quaternion.w)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def ros_yaw_to_fusion_heading(yaw):
    """Adapt ROS x/y/yaw coordinates to StreamingSourceMapFusion x/z/heading."""
    return -float(yaw) - math.pi / 2.0


def ssl_doa_to_fusion_distribution(probability):
    """Convert 0=front, 90=left SSL bins to the fusion class angular convention."""
    probability = np.asarray(probability, dtype=np.float32).reshape(-1)
    if probability.size == 0:
        return probability
    quarter_turn = int(round(probability.size / 4.0))
    return np.roll(probability, -quarter_turn)


class SSLNetAudioMapFusionNode:
    def __init__(self):
        import rospy
        from nav_msgs.msg import OccupancyGrid, Odometry
        from std_msgs.msg import Float32MultiArray, String
        from visualization_msgs.msg import MarkerArray
        from geometry_msgs.msg import PointStamped

        self.rospy = rospy
        self.lock = threading.Lock()
        self.fusion_lock = threading.Lock()
        self.odom_history = deque(maxlen=max(int(rospy.get_param("~odom_buffer_size", 200)), 1))
        self.robot_path = deque(maxlen=max(int(rospy.get_param("~robot_path_length", 1000)), 1))
        self.max_odom_diff_sec = float(rospy.get_param("~max_odom_diff_sec", 0.20))
        self.latest_summary = None
        self.latest_doa = None
        self.latest_distance = None
        self.doa_version = 0
        self.distance_version = 0
        self.processed_doa_version = 0
        self.processed_distance_version = 0
        self.processed_stamp = None
        self.published_frame_count = 0

        resolution = float(rospy.get_param("~resolution", 0.05))
        map_center_x = rospy.get_param("~map_center_x", 0.0)
        map_center_y = rospy.get_param("~map_center_y", 0.0)
        if (map_center_x is None) != (map_center_y is None):
            raise ValueError("Set both ~map_center_x and ~map_center_y, or neither.")
        self.map_center_pose = (
            (float(map_center_x), float(map_center_y))
            if map_center_x is not None
            else None
        )
        self.min_confidence = float(rospy.get_param("~min_confidence", 0.2))
        self.fusion = StreamingSourceMapFusion(
            map_size_m=float(rospy.get_param("~map_size_m", 10.0)),
            res=resolution,
            node_res=float(rospy.get_param("~argmax_resolution", resolution)),
            sigma_Q_cells=float(rospy.get_param("~sigma_Q_cells", 1.5)),
            beta_r=float(rospy.get_param("~beta_r", 0.1)),
            use_entropy_weight=bool(rospy.get_param("~use_entropy_weight", True)),
            w_min=float(rospy.get_param("~w_min", 0.2)),
            r_max=float(rospy.get_param("~max_distance_m", 6.0)),
            use_softmax=False,
            intensity_zero_eps=self.min_confidence,
        )
        self.max_audio_input_mean_abs = float(
            rospy.get_param("~max_audio_input_mean_abs", 0.06)
        )
        self.min_signal_prob = float(rospy.get_param("~min_signal_prob", 0.5))
        self.frame_id_override = rospy.get_param("~frame_id", "")
        self.marker_height = float(rospy.get_param("~marker_height", 0.12))
        self.map_alpha_threshold = float(rospy.get_param("~map_alpha_threshold", 0.0))
        self.publish_heatmap_marker = bool(rospy.get_param("~publish_heatmap_marker", True))
        self.heatmap_marker_threshold = float(
            rospy.get_param("~heatmap_marker_threshold", 0.08)
        )
        self.heatmap_marker_max_cells = max(
            int(rospy.get_param("~heatmap_marker_max_cells", 6000)), 1
        )
        self.heatmap_marker_height = float(rospy.get_param("~heatmap_marker_height", 0.02))
        self.broadcast_odom_tf = bool(rospy.get_param("~broadcast_odom_tf", False))
        self.sensor_frame_id = rospy.get_param("~sensor_frame_id", "")
        self.warned_frame_override = False
        self.tf_broadcaster = None
        if self.broadcast_odom_tf:
            import tf2_ros

            self.tf_broadcaster = tf2_ros.TransformBroadcaster()

        odom_topic = rospy.get_param("~odom_topic", "/Odometry")
        prediction_topic = rospy.get_param(
            "~prediction_topic", "/sslnet_audio_inference/prediction_json"
        )
        doa_topic = rospy.get_param(
            "~doa_topic", "/sslnet_audio_inference/doa_distribution"
        )
        distance_topic = rospy.get_param(
            "~distance_topic", "/sslnet_audio_inference/distance_distribution"
        )

        self.map_pub = rospy.Publisher("~map", OccupancyGrid, queue_size=1, latch=True)
        self.argmax_pub = rospy.Publisher("~argmax", PointStamped, queue_size=2, latch=True)
        self.argmax_json_pub = rospy.Publisher("~argmax_json", String, queue_size=2, latch=True)
        self.markers_pub = rospy.Publisher("~markers", MarkerArray, queue_size=2, latch=True)
        self.status_pub = rospy.Publisher("~status", String, queue_size=1, latch=True)
        self.odom_sub = rospy.Subscriber(odom_topic, Odometry, self.odom_callback, queue_size=20)
        self.summary_sub = rospy.Subscriber(prediction_topic, String, self.summary_callback, queue_size=10)
        self.doa_sub = rospy.Subscriber(doa_topic, Float32MultiArray, self.doa_callback, queue_size=10)
        self.distance_sub = rospy.Subscriber(
            distance_topic, Float32MultiArray, self.distance_callback, queue_size=10
        )
        from std_srvs.srv import Empty

        self.reset_service = rospy.Service("~reset", Empty, self.reset_callback)
        self.status_timer = rospy.Timer(rospy.Duration(1.0), self.status_callback)

        rospy.loginfo(
            "SSLNet audio map fusion started: odom=%s prediction=%s map_size=%.1fm resolution=%.2fm",
            odom_topic,
            prediction_topic,
            self.fusion.map_size_m,
            self.fusion.res,
        )
        rospy.loginfo(
            "Audio map output topics: map=%s argmax=%s markers=%s status=%s",
            rospy.resolve_name("~map"),
            rospy.resolve_name("~argmax"),
            rospy.resolve_name("~markers"),
            rospy.resolve_name("~status"),
        )

    def odom_callback(self, msg):
        stamp_sec = msg.header.stamp.to_sec()
        if stamp_sec <= 0.0:
            stamp_sec = self.rospy.Time.now().to_sec()
        with self.lock:
            self.odom_history.append((stamp_sec, msg))
        if self.tf_broadcaster is not None:
            self.publish_odom_tf(msg)
        self.try_fuse()

    def publish_odom_tf(self, odom):
        from geometry_msgs.msg import TransformStamped

        parent_frame = self.global_frame_id(odom)
        child_frame = self.sensor_frame_id or odom.child_frame_id or "livox_frame"
        if parent_frame == child_frame:
            return
        transform = TransformStamped()
        transform.header.stamp = odom.header.stamp
        transform.header.frame_id = parent_frame
        transform.child_frame_id = child_frame
        transform.transform.translation.x = odom.pose.pose.position.x
        transform.transform.translation.y = odom.pose.pose.position.y
        transform.transform.translation.z = odom.pose.pose.position.z
        transform.transform.rotation = odom.pose.pose.orientation
        self.tf_broadcaster.sendTransform(transform)

    def global_frame_id(self, odom):
        odom_frame = odom.header.frame_id
        if (
            odom_frame
            and self.frame_id_override
            and self.frame_id_override != odom_frame
            and not self.warned_frame_override
        ):
            self.rospy.logwarn(
                "Ignoring ~frame_id=%s because odom is expressed in global frame %s. "
                "Relabeling an accumulated audio map as a sensor frame would move the heatmap.",
                self.frame_id_override,
                odom_frame,
            )
            self.warned_frame_override = True
        return odom_frame or self.frame_id_override or "odom"

    def reset_callback(self, _request):
        from std_srvs.srv import EmptyResponse

        with self.fusion_lock:
            self.fusion.reset(new_center_pose=self.map_center_pose, clear_bins=False)
            with self.lock:
                self.robot_path.clear()
                self.processed_stamp = None
                self.processed_doa_version = self.doa_version
                self.processed_distance_version = self.distance_version
        self.rospy.loginfo("Reset SSLNet global audio map")
        return EmptyResponse()

    def status_callback(self, _event):
        from std_msgs.msg import String

        with self.lock:
            has_odom = bool(self.odom_history)
            has_summary = self.latest_summary is not None
            has_doa = self.latest_doa is not None
            has_distance = self.latest_distance is not None
            last_odom_stamp = self.odom_history[-1][0] if has_odom else None
            summary_stamp = (
                float(self.latest_summary["stamp"]) if has_summary else None
            )
            nearest_odom_diff = (
                min(abs(item[0] - summary_stamp) for item in self.odom_history)
                if has_odom and has_summary
                else None
            )
            published_frame_count = self.published_frame_count
            doa_version = self.doa_version
            distance_version = self.distance_version
            processed_doa_version = self.processed_doa_version
            processed_distance_version = self.processed_distance_version

        if not has_odom:
            waiting_for = "odom"
        elif not has_summary:
            waiting_for = "prediction_json"
        elif not has_doa:
            waiting_for = "doa_distribution"
        elif not has_distance:
            waiting_for = "distance_distribution"
        elif last_odom_stamp < summary_stamp:
            waiting_for = "odom_at_or_after_prediction_stamp"
        elif nearest_odom_diff > self.max_odom_diff_sec:
            waiting_for = "odom_timestamp_difference_too_large"
        elif (
            doa_version <= processed_doa_version
            or distance_version <= processed_distance_version
        ):
            waiting_for = "new_prediction_distribution"
        else:
            waiting_for = "processing_or_timestamp_match"

        payload = {
            "published_frame_count": int(published_frame_count),
            "waiting_for": waiting_for,
            "has_odom": has_odom,
            "has_prediction_json": has_summary,
            "has_doa_distribution": has_doa,
            "has_distance_distribution": has_distance,
            "last_odom_stamp": last_odom_stamp,
            "prediction_stamp": summary_stamp,
            "nearest_odom_diff_sec": nearest_odom_diff,
            "max_odom_diff_sec": self.max_odom_diff_sec,
        }
        self.status_pub.publish(String(data=json.dumps(payload, ensure_ascii=True)))
        if published_frame_count == 0:
            self.rospy.logwarn_throttle(
                5.0,
                "Audio map has not published markers yet; waiting_for=%s "
                "(odom=%s json=%s doa=%s distance=%s)",
                waiting_for,
                has_odom,
                has_summary,
                has_doa,
                has_distance,
            )

    def summary_callback(self, msg):
        try:
            summary = json.loads(msg.data)
            stamp = float(summary["stamp"])
        except (KeyError, TypeError, ValueError):
            self.rospy.logwarn_throttle(5.0, "Cannot parse SSLNet prediction_json for audio map")
            return
        with self.lock:
            self.latest_summary = summary
        self.try_fuse()

    def doa_callback(self, msg):
        values = np.asarray(msg.data, dtype=np.float32)
        if values.size == 0:
            return
        with self.lock:
            self.latest_doa = values
            self.doa_version += 1
        self.try_fuse()

    def distance_callback(self, msg):
        values = np.asarray(msg.data, dtype=np.float32)
        if values.size == 0:
            return
        with self.lock:
            self.latest_distance = values
            self.distance_version += 1
        self.try_fuse()

    def try_fuse(self):
        with self.fusion_lock:
            with self.lock:
                if (
                    not self.odom_history
                    or self.latest_summary is None
                    or self.latest_doa is None
                    or self.latest_distance is None
                ):
                    return
                stamp = float(self.latest_summary["stamp"])
                if stamp == self.processed_stamp:
                    return
                if (
                    self.doa_version <= self.processed_doa_version
                    or self.distance_version <= self.processed_distance_version
                ):
                    return
                summary = dict(self.latest_summary)
                if self.odom_history[-1][0] < stamp:
                    return
                odom_stamp, odom = min(
                    self.odom_history,
                    key=lambda item: abs(item[0] - stamp),
                )
                if abs(odom_stamp - stamp) > self.max_odom_diff_sec:
                    self.rospy.logwarn_throttle(
                        5.0,
                        "No odom within %.3fs of prediction stamp %.6f; nearest diff=%.3fs",
                        self.max_odom_diff_sec,
                        stamp,
                        abs(odom_stamp - stamp),
                    )
                    return
                doa = self.latest_doa.copy()
                distance = self.latest_distance.copy()
                doa_version = self.doa_version
                distance_version = self.distance_version

            pose = odom.pose.pose.position
            yaw = quaternion_to_yaw(odom.pose.pose.orientation)
            confidence = float(summary.get("doa_confidence", 1.0))
            signal_prob = float(summary.get("signal_prob", 1.0))
            audio_input_mean_abs = summary.get("audio_input_mean_abs")
            audio_too_loud = (
                audio_input_mean_abs is not None
                and self.max_audio_input_mean_abs > 0.0
                and float(audio_input_mean_abs) > self.max_audio_input_mean_abs
            )
            no_signal = signal_prob < self.min_signal_prob
            update_intensity = 0.0 if (audio_too_loud or no_signal) else confidence
            if audio_too_loud:
                self.rospy.logwarn_throttle(
                    2.0,
                    "Skip audio map update because network input mean abs %.3f > %.3f",
                    float(audio_input_mean_abs),
                    self.max_audio_input_mean_abs,
                )
            if no_signal:
                self.rospy.logwarn_throttle(
                    2.0,
                    "Skip audio map update because signal_prob %.3f < %.3f",
                    signal_prob,
                    self.min_signal_prob,
                )
            if not self.fusion.inited and self.map_center_pose is not None:
                self.fusion.reset(new_center_pose=self.map_center_pose, clear_bins=False)
            output = self.fusion.update_frame(
                pred_theta=ssl_doa_to_fusion_distribution(doa),
                pred_r=distance,
                pose=(float(pose.x), float(pose.y)),
                heading=ros_yaw_to_fusion_heading(yaw),
                audio_intensity=update_intensity,
            )
            frame_id = self.global_frame_id(odom)
            source_stamp = self.rospy.Time.from_sec(float(summary["stamp"]))
            with self.lock:
                self.robot_path.append((float(pose.x), float(pose.y)))
            self.publish_outputs(
                output,
                frame_id,
                source_stamp,
                pose,
                yaw,
                confidence,
                audio_input_mean_abs=audio_input_mean_abs,
                audio_too_loud=audio_too_loud,
                signal_prob=signal_prob,
                no_signal=no_signal,
            )

            with self.lock:
                self.published_frame_count += 1
                self.processed_stamp = stamp
                self.processed_doa_version = doa_version
                self.processed_distance_version = distance_version

    def publish_outputs(
        self,
        output,
        frame_id,
        stamp,
        robot_position,
        robot_yaw,
        confidence,
        audio_input_mean_abs=None,
        audio_too_loud=False,
        signal_prob=1.0,
        no_signal=False,
    ):
        self.map_pub.publish(self.make_map(output, frame_id, stamp))
        self.argmax_pub.publish(self.make_argmax_point(output, frame_id, stamp))
        self.markers_pub.publish(
            self.make_markers(output, frame_id, stamp, robot_position, robot_yaw, confidence)
        )
        estimate = output["map_argmax_world"]
        payload = {
            "x": float(estimate[0]),
            "y": float(estimate[1]),
            "confidence": confidence,
            "did_update": bool(output["do_update"]),
            "audio_input_mean_abs": (
                None if audio_input_mean_abs is None else float(audio_input_mean_abs)
            ),
            "audio_too_loud": bool(audio_too_loud),
            "signal_prob": float(signal_prob),
            "no_signal": bool(no_signal),
            "min_signal_prob": float(self.min_signal_prob),
            "frame_id": frame_id,
            "frame_count": int(output["t"]),
            "stamp": stamp.to_sec(),
        }
        from std_msgs.msg import String

        self.argmax_json_pub.publish(String(data=json.dumps(payload, ensure_ascii=True)))
        self.rospy.loginfo(
            "AudioMap: source=(%.2f, %.2f) frame=%s update=%s confidence=%.3f signal=%.3f",
            estimate[0],
            estimate[1],
            frame_id,
            output["do_update"],
            confidence,
            signal_prob,
        )

    def make_map(self, output, frame_id, stamp):
        from nav_msgs.msg import OccupancyGrid

        probability = np.asarray(output["P"], dtype=np.float32)
        normalized = np.clip(probability, 0.0, 1.0)
        if self.map_alpha_threshold > 0.0:
            cells = np.where(
                normalized >= self.map_alpha_threshold,
                np.rint(100.0 * normalized),
                -1,
            )
        else:
            cells = np.rint(100.0 * normalized)
        grid = OccupancyGrid()
        grid.header.frame_id = frame_id
        grid.header.stamp = stamp
        grid.info.resolution = self.fusion.res
        grid.info.width = self.fusion.W
        grid.info.height = self.fusion.H
        grid.info.origin.position.x = float(self.fusion.x_min - 0.5 * self.fusion.res)
        grid.info.origin.position.y = float(self.fusion.z_min - 0.5 * self.fusion.res)
        grid.info.origin.orientation.w = 1.0
        grid.data = cells.astype(np.int8).reshape(-1).tolist()
        return grid

    def make_argmax_point(self, output, frame_id, stamp):
        from geometry_msgs.msg import PointStamped

        estimate = output["map_argmax_world"]
        point = PointStamped()
        point.header.frame_id = frame_id
        point.header.stamp = stamp
        point.point.x = float(estimate[0])
        point.point.y = float(estimate[1])
        point.point.z = self.marker_height
        return point

    def make_heatmap_marker(self, output, header):
        from geometry_msgs.msg import Point
        from std_msgs.msg import ColorRGBA
        from visualization_msgs.msg import Marker

        marker = Marker()
        marker.header = header
        marker.ns = "audio_map_heatmap"
        marker.id = 0
        marker.type = Marker.CUBE_LIST
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = self.fusion.res
        marker.scale.y = self.fusion.res
        marker.scale.z = max(self.fusion.res * 0.10, 0.01)

        probability = np.clip(np.asarray(output["P"], dtype=np.float32), 0.0, None)
        maximum = float(probability.max()) if probability.size else 0.0
        if maximum <= 0.0:
            return marker

        normalized = probability / maximum
        threshold = float(np.clip(self.heatmap_marker_threshold, 0.0, 1.0))
        selected = np.flatnonzero(normalized.reshape(-1) >= threshold)
        if selected.size > self.heatmap_marker_max_cells:
            selected_values = normalized.reshape(-1)[selected]
            top_indices = np.argpartition(
                selected_values, -self.heatmap_marker_max_cells
            )[-self.heatmap_marker_max_cells:]
            selected = selected[top_indices]

        for flat_index in selected.tolist():
            row, col = np.unravel_index(flat_index, normalized.shape)
            value = float(normalized[row, col])
            marker.points.append(
                Point(
                    float(self.fusion.Xg[row, col]),
                    float(self.fusion.Zg[row, col]),
                    self.heatmap_marker_height,
                )
            )
            marker.colors.append(
                ColorRGBA(
                    r=min(1.0, 2.0 * value),
                    g=min(1.0, 2.0 * (1.0 - value)),
                    b=0.05,
                    a=0.15 + 0.70 * value,
                )
            )
        return marker

    def make_markers(self, output, frame_id, stamp, robot_position, robot_yaw, confidence):
        from geometry_msgs.msg import Point
        from visualization_msgs.msg import Marker, MarkerArray

        estimate = output["map_argmax_world"]
        with self.lock:
            robot_path = list(self.robot_path)
        source = Marker()
        source.header.frame_id = frame_id
        source.header.stamp = stamp
        source.ns = "audio_map_argmax"
        source.id = 0
        source.type = Marker.SPHERE
        source.action = Marker.ADD
        source.pose.position = Point(float(estimate[0]), float(estimate[1]), self.marker_height)
        source.pose.orientation.w = 1.0
        source.scale.x = source.scale.y = source.scale.z = 0.28
        source.color.r = 1.0
        source.color.g = 0.05
        source.color.b = 0.05
        source.color.a = 0.95

        text = Marker()
        text.header = source.header
        text.ns = "audio_map_argmax"
        text.id = 1
        text.type = Marker.TEXT_VIEW_FACING
        text.action = Marker.ADD
        text.pose.position = Point(float(estimate[0]), float(estimate[1]), self.marker_height + 0.35)
        text.pose.orientation.w = 1.0
        text.scale.z = 0.20
        text.color.r = text.color.g = text.color.b = text.color.a = 1.0
        text.text = "audio map pred\n(%.2f, %.2f) p=%.3f" % (
            estimate[0],
            estimate[1],
            confidence,
        )

        robot = Marker()
        robot.header = source.header
        robot.ns = "audio_map_robot"
        robot.id = 0
        robot.type = Marker.ARROW
        robot.action = Marker.ADD
        robot.pose.position = Point(
            float(robot_position.x), float(robot_position.y), self.marker_height
        )
        robot.pose.orientation.z = math.sin(robot_yaw / 2.0)
        robot.pose.orientation.w = math.cos(robot_yaw / 2.0)
        robot.scale.x = 0.60
        robot.scale.y = 0.10
        robot.scale.z = 0.10
        robot.color.r = 0.05
        robot.color.g = 0.80
        robot.color.b = 1.0
        robot.color.a = 0.95

        trajectory = Marker()
        trajectory.header = source.header
        trajectory.ns = "audio_map_robot"
        trajectory.id = 1
        trajectory.type = Marker.LINE_STRIP
        trajectory.action = Marker.ADD
        trajectory.pose.orientation.w = 1.0
        trajectory.scale.x = 0.045
        trajectory.color.r = 0.05
        trajectory.color.g = 0.80
        trajectory.color.b = 1.0
        trajectory.color.a = 0.80
        trajectory.points = [
            Point(float(x), float(y), self.marker_height * 0.5) for x, y in robot_path
        ]
        markers = [source, text, robot, trajectory]
        if self.publish_heatmap_marker:
            markers.insert(0, self.make_heatmap_marker(output, source.header))
        return MarkerArray(markers=markers)


def main():
    import rospy

    rospy.init_node("sslnet_audio_map")
    SSLNetAudioMapFusionNode()
    rospy.spin()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Publish SSLNet DOA/distance predictions as RViz markers for LiDAR overlay."""

import json
import math
import threading

import numpy as np


def prediction_xy(doa_deg, distance_m, origin_x=0.0, origin_y=0.0, yaw_offset_deg=0.0):
    """Return the predicted planar point in the configured marker frame."""
    theta = math.radians(float(doa_deg) + float(yaw_offset_deg))
    return (
        float(origin_x) + float(distance_m) * math.cos(theta),
        float(origin_y) + float(distance_m) * math.sin(theta),
    )


def circle_xy(radius, origin_x=0.0, origin_y=0.0, samples=72):
    """Create a closed XY circle as numeric points."""
    angles = np.linspace(0.0, 2.0 * np.pi, int(samples) + 1)
    x = float(origin_x) + float(radius) * np.cos(angles)
    y = float(origin_y) + float(radius) * np.sin(angles)
    return zip(x.tolist(), y.tolist())


class SSLNetRvizMarkers:
    def __init__(self):
        import rospy
        from std_msgs.msg import Float32MultiArray, String
        from visualization_msgs.msg import MarkerArray

        self.rospy = rospy
        self.lock = threading.Lock()
        self.latest_summary = None
        self.latest_doa = np.array([], dtype=np.float32)
        self.latest_distance = np.array([], dtype=np.float32)

        self.frame_id = rospy.get_param("~frame_id", "livox_frame")
        self.origin_x = float(rospy.get_param("~origin_x", 0.0))
        self.origin_y = float(rospy.get_param("~origin_y", 0.0))
        self.origin_z = float(rospy.get_param("~origin_z", 0.0))
        self.yaw_offset_deg = float(rospy.get_param("~yaw_offset_deg", 0.0))
        self.max_distance_m = float(rospy.get_param("~max_distance_m", 6.0))
        self.doa_distribution_radius_m = float(rospy.get_param("~doa_distribution_radius_m", 1.2))
        self.distance_ring_stride = max(int(rospy.get_param("~distance_ring_stride", 4)), 1)
        self.show_distributions = bool(rospy.get_param("~show_distributions", True))
        self.lifetime_sec = float(rospy.get_param("~lifetime_sec", 1.5))

        prediction_topic = rospy.get_param(
            "~prediction_topic", "/sslnet_audio_inference/prediction_json"
        )
        doa_topic = rospy.get_param(
            "~doa_topic", "/sslnet_audio_inference/doa_distribution"
        )
        distance_topic = rospy.get_param(
            "~distance_topic", "/sslnet_audio_inference/distance_distribution"
        )
        self.publisher = rospy.Publisher("~markers", MarkerArray, queue_size=2, latch=True)
        self.summary_sub = rospy.Subscriber(prediction_topic, String, self.summary_callback, queue_size=5)
        self.doa_sub = rospy.Subscriber(doa_topic, Float32MultiArray, self.doa_callback, queue_size=5)
        self.distance_sub = rospy.Subscriber(
            distance_topic, Float32MultiArray, self.distance_callback, queue_size=5
        )
        rospy.loginfo(
            "SSLNet RViz markers started: markers=%s frame=%s origin=(%.2f, %.2f, %.2f) yaw_offset=%.1f deg",
            rospy.resolve_name("~markers"),
            self.frame_id,
            self.origin_x,
            self.origin_y,
            self.origin_z,
            self.yaw_offset_deg,
        )

    def doa_callback(self, msg):
        with self.lock:
            self.latest_doa = np.asarray(msg.data, dtype=np.float32)
        self.publish_latest()

    def distance_callback(self, msg):
        with self.lock:
            self.latest_distance = np.asarray(msg.data, dtype=np.float32)
        self.publish_latest()

    def summary_callback(self, msg):
        try:
            summary = json.loads(msg.data)
            float(summary["doa_deg"])
            float(summary["distance_m"])
        except (KeyError, TypeError, ValueError):
            self.rospy.logwarn_throttle(5.0, "Cannot parse SSLNet prediction_json for RViz")
            return
        with self.lock:
            self.latest_summary = summary
        self.publish_latest()

    def publish_latest(self):
        with self.lock:
            if self.latest_summary is None:
                return
            summary = dict(self.latest_summary)
            doa = self.latest_doa.copy()
            distance = self.latest_distance.copy()
        doa_deg = float(summary["doa_deg"])
        distance_m = float(summary["distance_m"])
        self.publish_markers(summary, doa_deg, distance_m, doa, distance)

    def new_marker(self, marker_id, namespace, marker_type, stamp):
        from visualization_msgs.msg import Marker

        marker = Marker()
        marker.header.frame_id = self.frame_id
        marker.header.stamp = stamp
        marker.ns = namespace
        marker.id = marker_id
        marker.type = marker_type
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.lifetime = self.rospy.Duration(self.lifetime_sec)
        return marker

    @staticmethod
    def point(x, y, z):
        from geometry_msgs.msg import Point

        return Point(x=float(x), y=float(y), z=float(z))

    @staticmethod
    def color(red, green, blue, alpha):
        from std_msgs.msg import ColorRGBA

        return ColorRGBA(r=float(red), g=float(green), b=float(blue), a=float(alpha))

    def publish_markers(self, summary, doa_deg, distance_m, doa, distance):
        from visualization_msgs.msg import Marker, MarkerArray

        stamp = self.rospy.Time.now()
        x, y = prediction_xy(
            doa_deg,
            distance_m,
            self.origin_x,
            self.origin_y,
            self.yaw_offset_deg,
        )
        origin = self.point(self.origin_x, self.origin_y, self.origin_z)
        target = self.point(x, y, self.origin_z)
        markers = []

        arrow = self.new_marker(0, "sslnet_prediction", Marker.ARROW, stamp)
        arrow.points = [origin, target]
        arrow.scale.x = 0.06
        arrow.scale.y = 0.16
        arrow.scale.z = 0.16
        arrow.color = self.color(1.0, 0.15, 0.05, 0.95)
        markers.append(arrow)

        target_point = self.new_marker(1, "sslnet_prediction", Marker.SPHERE, stamp)
        target_point.pose.position = target
        target_point.scale.x = 0.20
        target_point.scale.y = 0.20
        target_point.scale.z = 0.20
        target_point.color = self.color(1.0, 0.15, 0.05, 0.95)
        markers.append(target_point)

        peak_range = self.new_marker(2, "sslnet_prediction", Marker.LINE_STRIP, stamp)
        peak_range.scale.x = 0.035
        peak_range.color = self.color(0.1, 1.0, 0.25, 0.80)
        peak_range.points = [
            self.point(px, py, self.origin_z + 0.015)
            for px, py in circle_xy(distance_m, self.origin_x, self.origin_y)
        ]
        markers.append(peak_range)

        label = self.new_marker(3, "sslnet_prediction", Marker.TEXT_VIEW_FACING, stamp)
        label.pose.position = self.point(x, y, self.origin_z + 0.35)
        label.scale.z = 0.18
        label.color = self.color(1.0, 1.0, 1.0, 1.0)
        label.text = "DOA %.1f deg  D %.2f m\np=(%.3f, %.3f)" % (
            doa_deg,
            distance_m,
            float(summary.get("doa_confidence", 0.0)),
            float(summary.get("distance_confidence", 0.0)),
        )
        markers.append(label)

        if self.show_distributions and doa.size:
            markers.append(self.doa_distribution_marker(doa, stamp))
        if self.show_distributions and distance.size:
            markers.append(self.distance_distribution_marker(distance, stamp))

        self.publisher.publish(MarkerArray(markers=markers))

    def doa_distribution_marker(self, values, stamp):
        from visualization_msgs.msg import Marker

        probability = np.maximum(np.asarray(values, dtype=np.float32), 0.0)
        maximum = float(probability.max())
        normalized = probability / maximum if maximum > 0.0 else probability
        outline = self.new_marker(10, "sslnet_distribution", Marker.LINE_STRIP, stamp)
        outline.scale.x = 0.025
        outline.color = self.color(1.0, 0.80, 0.05, 0.85)
        for index, value in enumerate(normalized.tolist() + normalized[:1].tolist()):
            angle = math.radians(float(index % len(normalized)) + self.yaw_offset_deg)
            radius = self.doa_distribution_radius_m * float(value)
            outline.points.append(
                self.point(
                    self.origin_x + radius * math.cos(angle),
                    self.origin_y + radius * math.sin(angle),
                    self.origin_z + 0.035,
                )
            )
        return outline

    def distance_distribution_marker(self, values, stamp):
        from visualization_msgs.msg import Marker

        probability = np.maximum(np.asarray(values, dtype=np.float32), 0.0)
        maximum = float(probability.max())
        normalized = probability / maximum if maximum > 0.0 else probability
        rings = self.new_marker(11, "sslnet_distribution", Marker.LINE_LIST, stamp)
        rings.scale.x = 0.012
        rings.color = self.color(0.1, 0.45, 1.0, 0.30)
        for index in range(1, len(normalized), self.distance_ring_stride):
            radius = self.max_distance_m * index / max(len(normalized) - 1, 1)
            intensity = float(normalized[index])
            color = self.color(
                0.05 + 0.15 * intensity,
                0.15 + 0.75 * intensity,
                0.30 + 0.70 * intensity,
                0.30,
            )
            points = list(circle_xy(radius, self.origin_x, self.origin_y, samples=48))
            for start, end in zip(points[:-1], points[1:]):
                rings.points.append(self.point(start[0], start[1], self.origin_z + 0.01))
                rings.points.append(self.point(end[0], end[1], self.origin_z + 0.01))
                rings.colors.extend([color, color])
        return rings


def main():
    import rospy

    rospy.init_node("sslnet_rviz_markers")
    SSLNetRvizMarkers()
    rospy.spin()


if __name__ == "__main__":
    main()

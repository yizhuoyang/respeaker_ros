#!/usr/bin/env python3
"""Publish noisy synthetic SSLNet predictions from odometry and a fixed source."""

import json
import math
from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deploy.ros1_sslnet_audio_map_fusion import quaternion_to_yaw  # noqa: E402


def circular_difference_deg(values, center):
    """Return signed wrapped angular error in degrees."""
    return (np.asarray(values, dtype=np.float32) - float(center) + 180.0) % 360.0 - 180.0


def gaussian_probability(values, center, sigma, circular=False):
    """Build a normalized Gaussian probability distribution on fixed bin values."""
    values = np.asarray(values, dtype=np.float32)
    delta = circular_difference_deg(values, center) if circular else values - float(center)
    sigma = max(float(sigma), 1e-6)
    probability = np.exp(-0.5 * np.square(delta / sigma)).astype(np.float32)
    return probability / max(float(probability.sum()), 1e-12)


def mixture_with_uniform(probability, uniform_weight):
    """Mix prediction probability with uniform clutter."""
    probability = np.asarray(probability, dtype=np.float32)
    weight = float(np.clip(uniform_weight, 0.0, 1.0))
    mixed = (1.0 - weight) * probability + weight / max(probability.size, 1)
    return mixed / max(float(mixed.sum()), 1e-12)


def relative_target_prediction(source_x, source_y, robot_x, robot_y, robot_yaw):
    """Return target DOA and distance using SSLNet's front/left convention."""
    dx = float(source_x) - float(robot_x)
    dy = float(source_y) - float(robot_y)
    distance_m = math.hypot(dx, dy)
    world_angle = math.atan2(dy, dx)
    doa_deg = math.degrees(world_angle - float(robot_yaw)) % 360.0
    return doa_deg, distance_m


def source_from_npz(path):
    """Read the final odometry position from an extracted odom NPZ file."""
    odom = np.load(str(path), allow_pickle=False)
    fields = [str(field) for field in odom["fields"]]
    if "px" not in fields or "py" not in fields or len(odom["data"]) == 0:
        raise RuntimeError(f"NPZ does not contain non-empty px/py odometry: {path}")
    final = odom["data"][-1]
    return float(final[fields.index("px")]), float(final[fields.index("py")])


def source_from_ros1_bag(path, odom_topic):
    """Read the final odometry position from a ROS1 bag topic."""
    import rosbag

    last_odom = None
    with rosbag.Bag(str(path), "r") as bag:
        for _topic, msg, _stamp in bag.read_messages(topics=[odom_topic]):
            last_odom = msg
    if last_odom is None:
        raise RuntimeError(f"No odometry message found on {odom_topic} in ROS1 bag: {path}")
    return float(last_odom.pose.pose.position.x), float(last_odom.pose.pose.position.y)


class FakeSSLNetPredictionNode:
    def __init__(self):
        import rospy
        from geometry_msgs.msg import PointStamped
        from nav_msgs.msg import Odometry
        from std_msgs.msg import Float32MultiArray, String
        from visualization_msgs.msg import MarkerArray

        self.rospy = rospy
        self.odom_topic = rospy.get_param("~odom_topic", "/Odometry")
        self.source_x, self.source_y, self.source_description = self.load_source_position()
        self.max_distance_m = float(rospy.get_param("~max_distance_m", 6.0))
        self.distance_bins = int(rospy.get_param("~distance_bins", 120))
        self.doa_bins = int(rospy.get_param("~doa_bins", 360))
        self.angle_noise_std_deg = float(rospy.get_param("~angle_noise_std_deg", 8.0))
        self.distance_noise_std_m = float(rospy.get_param("~distance_noise_std_m", 0.15))
        self.doa_sigma_deg = float(rospy.get_param("~doa_sigma_deg", 6.0))
        self.distance_sigma_m = float(rospy.get_param("~distance_sigma_m", 0.12))
        self.uniform_noise_weight = float(rospy.get_param("~uniform_noise_weight", 0.02))
        self.angle_bias_deg = float(rospy.get_param("~angle_bias_deg", 0.0))
        self.distance_bias_m = float(rospy.get_param("~distance_bias_m", 0.0))
        publish_hz = float(rospy.get_param("~publish_hz", 0.0))
        self.publish_period = 1.0 / publish_hz if publish_hz > 0.0 else 0.0
        self.last_stamp_sec = None
        self.marker_height = float(rospy.get_param("~marker_height", 0.18))
        self.frame_id_override = rospy.get_param("~frame_id", "")
        seed = int(rospy.get_param("~seed", 0))
        self.rng = np.random.default_rng(seed)

        prediction_topic = rospy.get_param(
            "~prediction_topic", "/sslnet_audio_inference/prediction"
        )
        json_topic = rospy.get_param(
            "~prediction_json_topic", "/sslnet_audio_inference/prediction_json"
        )
        doa_topic = rospy.get_param(
            "~doa_topic", "/sslnet_audio_inference/doa_distribution"
        )
        distance_topic = rospy.get_param(
            "~distance_topic", "/sslnet_audio_inference/distance_distribution"
        )
        self.summary_pub = rospy.Publisher(prediction_topic, Float32MultiArray, queue_size=5)
        self.json_pub = rospy.Publisher(json_topic, String, queue_size=5)
        self.doa_pub = rospy.Publisher(doa_topic, Float32MultiArray, queue_size=5)
        self.distance_pub = rospy.Publisher(distance_topic, Float32MultiArray, queue_size=5)
        self.source_pub = rospy.Publisher("~source_ground_truth", PointStamped, queue_size=1, latch=True)
        self.marker_pub = rospy.Publisher("~markers", MarkerArray, queue_size=1, latch=True)
        self.odom_sub = rospy.Subscriber(self.odom_topic, Odometry, self.odom_callback, queue_size=30)

        rospy.loginfo(
            "Fake SSLNet prediction started: odom=%s source=(%.3f, %.3f) from %s",
            self.odom_topic,
            self.source_x,
            self.source_y,
            self.source_description,
        )
        rospy.logwarn(
            "Fake predictions publish on SSLNet inference topics. Do not run ros1_sslnet_audio_node.py simultaneously."
        )

    def load_source_position(self):
        rospy = self.rospy
        if rospy.has_param("~source_x") and rospy.has_param("~source_y"):
            return (
                float(rospy.get_param("~source_x")),
                float(rospy.get_param("~source_y")),
                "manual parameters",
            )
        if rospy.has_param("~odom_npz"):
            path = Path(rospy.get_param("~odom_npz")).expanduser()
            x, y = source_from_npz(path)
            return x, y, f"last frame of {path}"
        bag_path = rospy.get_param("~bag_path", "")
        if not bag_path:
            raise RuntimeError(
                "Set ~bag_path to a ROS1 .bag, set ~odom_npz to an extracted lio_odom.npz, "
                "or set both ~source_x and ~source_y."
            )
        bag_odom_topic = rospy.get_param("~bag_odom_topic", self.odom_topic)
        path = Path(bag_path).expanduser()
        x, y = source_from_ros1_bag(path, bag_odom_topic)
        return x, y, f"last {bag_odom_topic} frame of {path}"

    def odom_callback(self, msg):
        stamp = msg.header.stamp
        stamp_sec = stamp.to_sec()
        if stamp_sec <= 0.0:
            stamp = self.rospy.Time.now()
            stamp_sec = stamp.to_sec()
        if self.last_stamp_sec is not None:
            if stamp_sec < self.last_stamp_sec:
                self.last_stamp_sec = None
            elif self.publish_period > 0.0 and stamp_sec - self.last_stamp_sec < self.publish_period:
                return
        self.last_stamp_sec = stamp_sec

        position = msg.pose.pose.position
        yaw = quaternion_to_yaw(msg.pose.pose.orientation)
        gt_doa_deg, gt_distance_m = relative_target_prediction(
            self.source_x,
            self.source_y,
            position.x,
            position.y,
            yaw,
        )
        if gt_distance_m > self.max_distance_m:
            self.rospy.logwarn_throttle(
                5.0,
                "Fake source is %.2f m away, outside max_distance_m=%.2f; distance prediction is clipped",
                gt_distance_m,
                self.max_distance_m,
            )
        pred_doa_deg = (
            gt_doa_deg + self.angle_bias_deg + self.rng.normal(0.0, self.angle_noise_std_deg)
        ) % 360.0
        pred_distance_m = np.clip(
            gt_distance_m + self.distance_bias_m + self.rng.normal(0.0, self.distance_noise_std_m),
            0.0,
            self.max_distance_m,
        )
        doa_axis = np.linspace(0.0, 360.0, self.doa_bins, endpoint=False, dtype=np.float32)
        distance_axis = np.linspace(
            0.0, self.max_distance_m, self.distance_bins, dtype=np.float32
        )
        doa_probability = mixture_with_uniform(
            gaussian_probability(doa_axis, pred_doa_deg, self.doa_sigma_deg, circular=True),
            self.uniform_noise_weight,
        )
        distance_probability = mixture_with_uniform(
            gaussian_probability(distance_axis, pred_distance_m, self.distance_sigma_m),
            self.uniform_noise_weight,
        )
        frame_id = self.frame_id_override or msg.header.frame_id or "odom"
        self.publish_prediction(
            stamp,
            frame_id,
            gt_doa_deg,
            gt_distance_m,
            float(pred_doa_deg),
            float(pred_distance_m),
            doa_probability,
            distance_probability,
        )

    def publish_prediction(
        self,
        stamp,
        frame_id,
        gt_doa_deg,
        gt_distance_m,
        pred_doa_deg,
        pred_distance_m,
        doa_probability,
        distance_probability,
    ):
        from std_msgs.msg import Float32MultiArray, String

        doa_confidence = float(doa_probability.max())
        distance_confidence = float(distance_probability.max())
        self.summary_pub.publish(
            Float32MultiArray(
                data=[
                    pred_doa_deg,
                    pred_distance_m,
                    doa_confidence,
                    distance_confidence,
                    0.0,
                ]
            )
        )
        self.doa_pub.publish(Float32MultiArray(data=doa_probability.tolist()))
        self.distance_pub.publish(Float32MultiArray(data=distance_probability.tolist()))
        summary = {
            "doa_deg": pred_doa_deg,
            "distance_m": pred_distance_m,
            "doa_confidence": doa_confidence,
            "distance_confidence": distance_confidence,
            "inference_ms": 0.0,
            "window_seconds": 0.0,
            "stamp": stamp.to_sec(),
            "fake": True,
            "ground_truth_doa_deg": gt_doa_deg,
            "ground_truth_distance_m": gt_distance_m,
            "ground_truth_source_x": self.source_x,
            "ground_truth_source_y": self.source_y,
            "doa_abs_error_deg": abs(float(circular_difference_deg(pred_doa_deg, gt_doa_deg))),
            "distance_abs_error_m": abs(pred_distance_m - gt_distance_m),
        }
        self.json_pub.publish(String(data=json.dumps(summary, ensure_ascii=True)))
        self.publish_ground_truth(frame_id, stamp)
        self.rospy.loginfo(
            "FakeSSLNet: gt=(%.1f deg, %.2f m) pred=(%.1f deg, %.2f m) error=(%.1f deg, %.2f m) source=(%.2f, %.2f)",
            gt_doa_deg,
            gt_distance_m,
            pred_doa_deg,
            pred_distance_m,
            summary["doa_abs_error_deg"],
            summary["distance_abs_error_m"],
            self.source_x,
            self.source_y,
        )

    def publish_ground_truth(self, frame_id, stamp):
        from geometry_msgs.msg import Point, PointStamped
        from visualization_msgs.msg import Marker, MarkerArray

        point = PointStamped()
        point.header.frame_id = frame_id
        point.header.stamp = stamp
        point.point = Point(self.source_x, self.source_y, self.marker_height)
        self.source_pub.publish(point)

        marker = Marker()
        marker.header = point.header
        marker.ns = "fake_source_ground_truth"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position = point.point
        marker.pose.orientation.w = 1.0
        marker.scale.x = marker.scale.y = marker.scale.z = 0.32
        marker.color.r = 0.1
        marker.color.g = 1.0
        marker.color.b = 0.1
        marker.color.a = 0.95

        label = Marker()
        label.header = point.header
        label.ns = "fake_source_ground_truth"
        label.id = 1
        label.type = Marker.TEXT_VIEW_FACING
        label.action = Marker.ADD
        label.pose.position = Point(self.source_x, self.source_y, self.marker_height + 0.35)
        label.pose.orientation.w = 1.0
        label.scale.z = 0.20
        label.color.r = 0.1
        label.color.g = 1.0
        label.color.b = 0.1
        label.color.a = 1.0
        label.text = "fake source GT\n(%.2f, %.2f)" % (self.source_x, self.source_y)
        self.marker_pub.publish(MarkerArray(markers=[marker, label]))


def main():
    import rospy

    rospy.init_node("sslnet_fake_prediction")
    FakeSSLNetPredictionNode()
    rospy.spin()


if __name__ == "__main__":
    main()

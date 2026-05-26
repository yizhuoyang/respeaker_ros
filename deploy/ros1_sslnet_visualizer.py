#!/usr/bin/env python3
"""Live matplotlib visualization for ROS1 SSLNet DOA/distance predictions."""

import json
from pathlib import Path
import sys
import threading

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class SSLNetVisualizer:
    def __init__(self):
        import rospy
        from std_msgs.msg import Float32MultiArray, String

        self.rospy = rospy
        self.lock = threading.Lock()
        self.latest_summary = {}
        self.latest_doa = np.zeros(360, dtype=np.float32)
        self.latest_distance = np.zeros(120, dtype=np.float32)
        self.has_doa = False
        self.has_distance = False
        self.max_distance_m = float(rospy.get_param("~max_distance_m", 6.0))
        self.refresh_hz = float(rospy.get_param("~refresh_hz", 8.0))
        snapshot_dir = rospy.get_param("~snapshot_dir", "")
        self.snapshot_dir = Path(snapshot_dir) if snapshot_dir else None

        prediction_topic = rospy.get_param(
            "~prediction_topic", "/sslnet_audio_inference/prediction_json"
        )
        doa_topic = rospy.get_param(
            "~doa_topic", "/sslnet_audio_inference/doa_distribution"
        )
        distance_topic = rospy.get_param(
            "~distance_topic", "/sslnet_audio_inference/distance_distribution"
        )
        self.summary_sub = rospy.Subscriber(prediction_topic, String, self.summary_callback, queue_size=5)
        self.doa_sub = rospy.Subscriber(doa_topic, Float32MultiArray, self.doa_callback, queue_size=5)
        self.distance_sub = rospy.Subscriber(
            distance_topic, Float32MultiArray, self.distance_callback, queue_size=5
        )

        self.figure = plt.figure(figsize=(14, 4.8))
        self.polar_axis = self.figure.add_subplot(131, projection="polar")
        self.distance_axis = self.figure.add_subplot(132)
        self.xy_axis = self.figure.add_subplot(133)
        self.figure.canvas.mpl_connect("key_press_event", self.key_callback)
        self.figure.canvas.mpl_connect("close_event", self.close_callback)
        self.animation = FuncAnimation(
            self.figure,
            self.draw,
            interval=max(int(round(1000.0 / self.refresh_hz)), 1),
            cache_frame_data=False,
        )

        rospy.loginfo(
            "SSLNet visualizer started: prediction=%s doa=%s distance=%s; press s to save a snapshot",
            prediction_topic,
            doa_topic,
            distance_topic,
        )

    def summary_callback(self, msg):
        try:
            summary = json.loads(msg.data)
        except ValueError:
            self.rospy.logwarn_throttle(5.0, "Cannot parse SSLNet prediction_json message")
            return
        with self.lock:
            self.latest_summary = summary

    def doa_callback(self, msg):
        values = np.asarray(msg.data, dtype=np.float32)
        if values.size == 0:
            return
        with self.lock:
            self.latest_doa = values
            self.has_doa = True

    def distance_callback(self, msg):
        values = np.asarray(msg.data, dtype=np.float32)
        if values.size == 0:
            return
        with self.lock:
            self.latest_distance = values
            self.has_distance = True

    def current_values(self):
        with self.lock:
            return (
                dict(self.latest_summary),
                self.latest_doa.copy(),
                self.latest_distance.copy(),
                self.has_doa,
                self.has_distance,
            )

    def draw(self, _frame):
        summary, doa, distance, has_doa, has_distance = self.current_values()
        doa_deg = float(summary.get("doa_deg", 0.0))
        distance_m = float(summary.get("distance_m", 0.0))
        doa_confidence = float(summary.get("doa_confidence", 0.0))
        distance_confidence = float(summary.get("distance_confidence", 0.0))

        self.polar_axis.clear()
        theta = np.linspace(0.0, 2.0 * np.pi, len(doa), endpoint=False)
        if has_doa:
            self.polar_axis.plot(theta, doa, color="tab:blue", linewidth=1.8)
            self.polar_axis.fill_between(theta, 0.0, doa, color="tab:blue", alpha=0.15)
            peak_theta = np.deg2rad(doa_deg)
            peak_value = float(doa.max())
            self.polar_axis.plot([peak_theta, peak_theta], [0.0, peak_value], color="tab:red", linewidth=2)
        self.polar_axis.set_theta_zero_location("E")
        self.polar_axis.set_theta_direction(1)
        self.polar_axis.set_thetagrids([0, 90, 180, 270], labels=["Front +x", "Left +y", "Back", "Right"])
        self.polar_axis.set_title(f"DOA: {doa_deg:.1f} deg\nconfidence={doa_confidence:.3f}")

        self.distance_axis.clear()
        distance_grid = np.linspace(0.0, self.max_distance_m, len(distance), dtype=np.float32)
        if has_distance:
            self.distance_axis.plot(distance_grid, distance, color="tab:green", linewidth=1.8)
            self.distance_axis.fill_between(distance_grid, 0.0, distance, color="tab:green", alpha=0.15)
            self.distance_axis.axvline(distance_m, color="tab:red", linewidth=2)
        self.distance_axis.set_xlim(0.0, self.max_distance_m)
        self.distance_axis.set_xlabel("Distance (m)")
        self.distance_axis.set_ylabel("Probability")
        self.distance_axis.set_title(f"Distance: {distance_m:.2f} m\nconfidence={distance_confidence:.3f}")
        self.distance_axis.grid(True, alpha=0.3)

        self.xy_axis.clear()
        radius = self.max_distance_m
        self.xy_axis.set_xlim(-radius, radius)
        self.xy_axis.set_ylim(-radius, radius)
        self.xy_axis.set_aspect("equal", adjustable="box")
        self.xy_axis.axhline(0.0, color="0.75", linewidth=1)
        self.xy_axis.axvline(0.0, color="0.75", linewidth=1)
        self.xy_axis.scatter([0.0], [0.0], color="black", marker="s", label="robot")
        if summary:
            angle_rad = np.deg2rad(doa_deg)
            target_x = distance_m * np.cos(angle_rad)
            target_y = distance_m * np.sin(angle_rad)
            self.xy_axis.arrow(
                0.0,
                0.0,
                target_x,
                target_y,
                width=0.035,
                head_width=0.18,
                length_includes_head=True,
                color="tab:red",
            )
            self.xy_axis.scatter([target_x], [target_y], color="tab:red", label="prediction")
        self.xy_axis.set_xlabel("+x front / -x back (m)")
        self.xy_axis.set_ylabel("+y left / -y right (m)")
        inference_ms = summary.get("inference_ms")
        timing = f" | {float(inference_ms):.1f} ms" if inference_ms is not None else ""
        self.xy_axis.set_title(f"Top-down prediction{timing}")
        self.xy_axis.grid(True, alpha=0.3)
        self.xy_axis.legend(loc="upper right")
        self.figure.tight_layout()

    def key_callback(self, event):
        if event.key != "s":
            return
        if self.snapshot_dir is not None:
            self.snapshot_dir.mkdir(parents=True, exist_ok=True)
            output = self.snapshot_dir / f"sslnet_snapshot_{self.rospy.Time.now().to_nsec()}.png"
        else:
            output = Path(f"sslnet_snapshot_{self.rospy.Time.now().to_nsec()}.png")
        self.figure.savefig(output, dpi=150)
        self.rospy.loginfo("Saved SSLNet visualization snapshot: %s", output)

    def close_callback(self, _event):
        self.rospy.signal_shutdown("visualizer window closed")

    def show(self):
        plt.show()


def main():
    import rospy

    rospy.init_node("sslnet_visualizer", anonymous=True)
    visualizer = SSLNetVisualizer()
    visualizer.show()


if __name__ == "__main__":
    main()

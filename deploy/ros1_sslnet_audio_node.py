#!/usr/bin/env python3
"""ROS1 node for streaming SSLNet audio DOA and distance inference."""

import json
import os
from pathlib import Path
from queue import Empty, Full, Queue
import sys
import threading

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deploy.sslnet_realtime import SSLNetStreamingPredictor, SlidingAudioWindow  # noqa: E402


def optional_param(rospy, name):
    return rospy.get_param(name) if rospy.has_param(name) else None


def load_audio_message_type():
    """Load the recorder message from a sourced ROS environment or catkin output."""
    try:
        from respeaker_ros_recorder.msg import AudioDataStamped

        return AudioDataStamped, None
    except ModuleNotFoundError as exc:
        if exc.name != "respeaker_ros_recorder":
            raise

    candidates = []
    configured_path = os.environ.get("RESPEAKER_ROS_MSG_PATH")
    if configured_path:
        candidates.append(Path(configured_path).expanduser())
    candidates.append(Path.home() / "yyz" / "respeaker_ros" / "devel" / "lib" / "python3" / "dist-packages")

    for candidate in candidates:
        if not (candidate / "respeaker_ros_recorder" / "msg").is_dir():
            continue
        sys.path.insert(0, str(candidate))
        try:
            from respeaker_ros_recorder.msg import AudioDataStamped

            return AudioDataStamped, str(candidate)
        except ModuleNotFoundError:
            sys.path.remove(str(candidate))

    searched = ", ".join(str(path) for path in candidates)
    raise ModuleNotFoundError(
        "Cannot import respeaker_ros_recorder.msg.AudioDataStamped. "
        "Run 'source /opt/ros/noetic/setup.bash' and "
        "'source /home/kemove/yyz/respeaker_ros/devel/setup.bash' before starting this node, "
        "or set RESPEAKER_ROS_MSG_PATH to your catkin "
        "devel/lib/python3/dist-packages directory. "
        f"Searched fallback paths: {searched}"
    )


class SSLNetAudioNode:
    def __init__(self):
        import rospy
        from std_msgs.msg import Float32MultiArray, String

        self.rospy = rospy
        AudioDataStamped, fallback_path = load_audio_message_type()
        if fallback_path is not None:
            rospy.logwarn(
                "Loaded respeaker_ros_recorder messages from %s because the catkin workspace "
                "was not present on PYTHONPATH. Prefer sourcing its devel/setup.bash.",
                fallback_path,
            )
        checkpoint = rospy.get_param("~checkpoint", "")
        if not checkpoint:
            raise RuntimeError("Set the private ROS parameter ~checkpoint to a trained SSLNet checkpoint")

        self.topic = rospy.get_param("~audio_topic", "/respeaker/audio_raw")
        self.sample_rate = int(rospy.get_param("~sample_rate", 16000))
        self.window_seconds = float(rospy.get_param("~window_seconds", 1.0))
        self.hop_seconds = float(rospy.get_param("~hop_seconds", self.window_seconds))
        self.predictor = SSLNetStreamingPredictor(
            checkpoint_path=checkpoint,
            device=rospy.get_param("~device", "cuda:0"),
            audio_feat=optional_param(rospy, "~audio_feat"),
            audio_channels=optional_param(rospy, "~audio_channels"),
            ipd_pairs=optional_param(rospy, "~ipd_pairs"),
            use_compress=optional_param(rospy, "~use_compress"),
            expected_sample_rate=self.sample_rate,
            audio_bandpass_low_hz=optional_param(rospy, "~audio_bandpass_low_hz"),
            audio_bandpass_high_hz=optional_param(rospy, "~audio_bandpass_high_hz"),
            use_filter_mute_denoise=optional_param(rospy, "~use_filter_mute_denoise"),
            filter_mute_highpass_hz=optional_param(rospy, "~filter_mute_highpass_hz"),
            filter_mute_notches_hz=optional_param(rospy, "~filter_mute_notches_hz"),
            filter_mute_threshold=optional_param(rospy, "~filter_mute_threshold"),
            filter_mute_window_sec=optional_param(rospy, "~filter_mute_window_sec"),
            filter_mute_floor=optional_param(rospy, "~filter_mute_floor"),
            filter_mute_edge_smooth_ms=optional_param(rospy, "~filter_mute_edge_smooth_ms"),
        )
        self.window = SlidingAudioWindow(
            sample_rate=self.sample_rate,
            window_seconds=self.window_seconds,
            hop_seconds=self.hop_seconds,
        )
        self.last_channels = None
        self.shutdown_timeout = float(rospy.get_param("~shutdown_timeout", 0.5))
        self.inference_queue = Queue(maxsize=max(int(rospy.get_param("~inference_queue_size", 1)), 1))
        self.stop_event = threading.Event()
        self.subscriber = None
        self.worker = threading.Thread(
            target=self.inference_loop,
            name="sslnet_inference_worker",
            daemon=True,
        )

        self.summary_pub = rospy.Publisher("~prediction", Float32MultiArray, queue_size=5)
        self.json_pub = rospy.Publisher("~prediction_json", String, queue_size=5)
        self.doa_pub = rospy.Publisher("~doa_distribution", Float32MultiArray, queue_size=2)
        self.distance_pub = rospy.Publisher("~distance_distribution", Float32MultiArray, queue_size=2)
        rospy.on_shutdown(self.shutdown)
        self.subscriber = rospy.Subscriber(self.topic, AudioDataStamped, self.audio_callback, queue_size=30)
        self.worker.start()

        config = self.predictor.describe()
        rospy.loginfo("SSLNet real-time inference started: topic=%s window=%.3fs hop=%.3fs", self.topic, self.window_seconds, self.hop_seconds)
        rospy.loginfo("SSLNet inference config: %s", config)
        rospy.loginfo("DOA coordinates: 0 deg=+x/front, 90 deg=+y/left (pairs_ros1 bbox convention)")
        rospy.loginfo("Inference runs in a daemon worker so ROS shutdown does not wait indefinitely for GPU work")

    def audio_callback(self, msg):
        rospy = self.rospy
        if self.stop_event.is_set() or rospy.is_shutdown():
            return
        if int(msg.sample_rate) != self.sample_rate:
            rospy.logerr_throttle(5.0, "Received sample_rate=%s, expected %s", msg.sample_rate, self.sample_rate)
            return
        channels = int(msg.channels)
        if channels <= 0 or len(msg.data) % channels:
            rospy.logerr_throttle(5.0, "Invalid audio message length=%s channels=%s", len(msg.data), channels)
            return
        if self.last_channels is not None and channels != self.last_channels:
            self.window.reset()
            rospy.logwarn("Audio channel count changed from %s to %s; rolling window reset", self.last_channels, channels)
        self.last_channels = channels
        audio = np.asarray(msg.data, dtype=np.int16).reshape(-1, channels).astype(np.float32)
        audio /= float(np.iinfo(np.int16).max)

        for segment in self.window.append(audio):
            self.enqueue_inference(segment, msg.header.stamp.to_sec())

    def enqueue_inference(self, segment, stamp):
        """Queue the newest window without allowing inference backlog to grow."""
        task = (segment, stamp)
        try:
            self.inference_queue.put_nowait(task)
            return
        except Full:
            try:
                self.inference_queue.get_nowait()
                self.inference_queue.task_done()
            except Empty:
                pass
        try:
            self.inference_queue.put_nowait(task)
            self.rospy.logwarn_throttle(5.0, "SSLNet inference is slower than audio input; dropped an older window")
        except Full:
            pass

    def inference_loop(self):
        while not self.stop_event.is_set():
            try:
                task = self.inference_queue.get(timeout=0.1)
            except Empty:
                continue
            if task is None:
                self.inference_queue.task_done()
                return
            segment, stamp = task
            try:
                result = self.predictor.predict(segment, self.sample_rate)
                result["audio_input_mean_abs"] = self.network_audio_mean_abs(segment)
            except Exception as exc:
                self.rospy.logerr_throttle(5.0, "SSLNet inference failed: %s", exc)
            else:
                if not self.stop_event.is_set() and not self.rospy.is_shutdown():
                    self.publish_prediction(result, stamp)
            finally:
                self.inference_queue.task_done()

    def network_audio_mean_abs(self, segment):
        """Mean absolute waveform amplitude for the channels used by the SSLNet input."""
        audio_channels = getattr(self.predictor, "audio_channels", ())
        if not audio_channels:
            return float(np.mean(np.abs(segment)))
        if max(audio_channels) >= segment.shape[1]:
            return float(np.mean(np.abs(segment)))
        return float(np.mean(np.abs(segment[:, audio_channels])))

    def shutdown(self):
        """Stop ROS input immediately; never wait indefinitely for an in-flight inference."""
        if self.stop_event.is_set():
            return
        self.stop_event.set()
        if self.subscriber is not None:
            self.subscriber.unregister()
        self.window.reset()
        while True:
            try:
                self.inference_queue.get_nowait()
                self.inference_queue.task_done()
            except Empty:
                break
        try:
            self.inference_queue.put_nowait(None)
        except Full:
            pass
        if self.worker.is_alive() and threading.current_thread() is not self.worker:
            self.worker.join(timeout=self.shutdown_timeout)
        if self.worker.is_alive():
            self.rospy.logwarn(
                "Inference worker did not stop within %.2fs; leaving daemon worker during shutdown",
                self.shutdown_timeout,
            )
        else:
            self.rospy.loginfo("SSLNet inference worker stopped")

    def publish_prediction(self, result, stamp):
        from std_msgs.msg import Float32MultiArray, String

        summary = Float32MultiArray()
        summary.data = [
            result["doa_deg"],
            result["distance_m"],
            result["doa_confidence"],
            result["distance_confidence"],
            result["inference_ms"],
        ]
        doa = Float32MultiArray(data=result["doa_probability"].tolist())
        distance = Float32MultiArray(data=result["distance_probability"].tolist())
        json_result = {
            "doa_deg": result["doa_deg"],
            "distance_m": result["distance_m"],
            "doa_confidence": result["doa_confidence"],
            "distance_confidence": result["distance_confidence"],
            "audio_input_mean_abs": result.get("audio_input_mean_abs", 0.0),
            "inference_ms": result["inference_ms"],
            "window_seconds": self.window_seconds,
            "stamp": stamp,
        }
        if "class_id" in result:
            json_result["class_id"] = result["class_id"]
            json_result["class_probability"] = result["class_probability"].tolist()
        self.summary_pub.publish(summary)
        self.json_pub.publish(String(data=json.dumps(json_result, ensure_ascii=True)))
        self.doa_pub.publish(doa)
        self.distance_pub.publish(distance)
        self.rospy.loginfo(
            "SSLNet: doa=%.1f deg distance=%.2f m confidence=(%.3f, %.3f) audio_mean_abs=%.3f inference=%.1f ms",
            result["doa_deg"],
            result["distance_m"],
            result["doa_confidence"],
            result["distance_confidence"],
            result.get("audio_input_mean_abs", 0.0),
            result["inference_ms"],
        )


def main():
    import rospy

    rospy.init_node("sslnet_audio_inference")
    node = SSLNetAudioNode()
    try:
        rospy.spin()
    finally:
        node.shutdown()


if __name__ == "__main__":
    main()

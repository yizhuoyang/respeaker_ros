#!/usr/bin/env python3
"""ROS1 node for streaming SSLNet audio inference through TensorRT."""

from pathlib import Path
from queue import Queue
import sys
import threading


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deploy.ros1_sslnet_audio_node import (  # noqa: E402
    SSLNetAudioNode,
    load_audio_message_type,
    optional_param,
)
from deploy.sslnet_engine_realtime import SSLNetTensorRTPredictor  # noqa: E402
from deploy.sslnet_realtime import SlidingAudioWindow  # noqa: E402


class SSLNetAudioEngineNode(SSLNetAudioNode):
    """Keep the ROS output contract of SSLNetAudioNode while replacing PyTorch forward."""

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
        default_engine = REPO_ROOT / "weights" / "pairs_ros1_sslnet_audio" / "best_model.engine"
        engine = rospy.get_param("~engine", str(default_engine))
        metadata = rospy.get_param("~metadata", f"{engine}.json")
        if not Path(engine).is_file():
            raise RuntimeError(
                f"TensorRT engine not found: {engine}. Run deploy/export_sslnet_tensorrt.py first."
            )
        if not Path(metadata).is_file():
            raise RuntimeError(
                f"TensorRT metadata not found: {metadata}. Export the engine and metadata together."
            )

        self.topic = rospy.get_param("~audio_topic", "/respeaker/audio_raw")
        self.sample_rate = int(rospy.get_param("~sample_rate", 16000))
        self.window_seconds = float(rospy.get_param("~window_seconds", 1.0))
        self.hop_seconds = float(rospy.get_param("~hop_seconds", self.window_seconds))
        self.predictor = SSLNetTensorRTPredictor(
            engine_path=engine,
            metadata_path=metadata,
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
        exported_window_seconds = float(self.predictor.metadata.get("window_seconds", 1.0))
        if abs(self.window_seconds - exported_window_seconds) > 1e-6:
            raise ValueError(
                f"~window_seconds={self.window_seconds} does not match the engine export "
                f"window_seconds={exported_window_seconds}. Export a matching engine."
            )
        self.window = SlidingAudioWindow(
            sample_rate=self.sample_rate,
            window_seconds=self.window_seconds,
            hop_seconds=self.hop_seconds,
        )
        self.last_channels = None
        self.shutdown_timeout = float(rospy.get_param("~shutdown_timeout", 0.5))
        self.inference_queue = Queue(
            maxsize=max(int(rospy.get_param("~inference_queue_size", 1)), 1)
        )
        self.stop_event = threading.Event()
        self.subscriber = None
        self.worker = threading.Thread(
            target=self.inference_loop,
            name="sslnet_tensorrt_inference_worker",
            daemon=True,
        )

        self.summary_pub = rospy.Publisher("~prediction", Float32MultiArray, queue_size=5)
        self.json_pub = rospy.Publisher("~prediction_json", String, queue_size=5)
        self.doa_pub = rospy.Publisher("~doa_distribution", Float32MultiArray, queue_size=2)
        self.distance_pub = rospy.Publisher("~distance_distribution", Float32MultiArray, queue_size=2)
        rospy.on_shutdown(self.shutdown)
        self.subscriber = rospy.Subscriber(
            self.topic, AudioDataStamped, self.audio_callback, queue_size=30
        )
        self.worker.start()

        rospy.loginfo(
            "SSLNet TensorRT inference started: topic=%s window=%.3fs hop=%.3fs",
            self.topic,
            self.window_seconds,
            self.hop_seconds,
        )
        rospy.loginfo("SSLNet TensorRT inference config: %s", self.predictor.describe())
        rospy.loginfo(
            "Outputs remain compatible with ros1_sslnet_audio_map_fusion.py: "
            "~prediction_json, ~doa_distribution, ~distance_distribution"
        )


def main():
    import rospy

    rospy.init_node("sslnet_audio_inference")
    node = SSLNetAudioEngineNode()
    try:
        rospy.spin()
    finally:
        node.shutdown()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import numpy as np
import pyaudio
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node

from respeaker_ros2_recorder.msg import AudioDataStamped


class ReSpeakerMultiChannelNode(Node):
    def __init__(self):
        super().__init__("respeaker_multichannel_node")

        self.declare_parameter("sample_rate", 16000)
        self.declare_parameter("channels", 6)
        self.declare_parameter("chunk_size", 1600)
        self.declare_parameter("device_index", -1)
        self.declare_parameter("frame_id", "respeaker")
        self.declare_parameter("topic_name", "/respeaker/audio_raw")

        self.sample_rate = int(self.get_parameter("sample_rate").value)
        self.channels = int(self.get_parameter("channels").value)
        self.chunk_size = int(self.get_parameter("chunk_size").value)
        self.device_index = int(self.get_parameter("device_index").value)
        self.frame_id = str(self.get_parameter("frame_id").value)
        self.topic_name = str(self.get_parameter("topic_name").value)

        input_device_index = None if self.device_index < 0 else self.device_index

        self.publisher = self.create_publisher(
            AudioDataStamped,
            self.topic_name,
            20,
        )

        self.pa = pyaudio.PyAudio()
        self.stream = None

        self.get_logger().info("Opening ReSpeaker audio stream...")
        self.get_logger().info(f"sample_rate: {self.sample_rate}")
        self.get_logger().info(f"channels: {self.channels}")
        self.get_logger().info(f"chunk_size: {self.chunk_size}")
        self.get_logger().info(f"device_index: {input_device_index}")

        self.stream = self.pa.open(
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            input_device_index=input_device_index,
            frames_per_buffer=self.chunk_size,
        )

        self.chunk_duration_sec = float(self.chunk_size) / float(self.sample_rate)
        self.timer = self.create_timer(self.chunk_duration_sec, self.publish_audio_chunk)

        self.get_logger().info("ReSpeaker multi-channel recorder started.")

    def publish_audio_chunk(self):
        try:
            raw_data = self.stream.read(
                self.chunk_size,
                exception_on_overflow=False,
            )

            t_end = self.get_clock().now()
            t_start = t_end - Duration(seconds=self.chunk_duration_sec)

            audio_np = np.frombuffer(raw_data, dtype=np.int16)
            expected_size = self.chunk_size * self.channels
            if audio_np.size != expected_size:
                self.get_logger().warning(
                    f"Unexpected audio size: {audio_np.size}, expected {expected_size}"
                )
                return

            msg = AudioDataStamped()
            msg.header.stamp = t_start.to_msg()
            msg.header.frame_id = self.frame_id
            msg.sample_rate = self.sample_rate
            msg.channels = self.channels
            msg.frames = self.chunk_size
            msg.data = audio_np.tolist()

            self.publisher.publish(msg)

        except IOError as exc:
            self.get_logger().warning(f"Audio input overflow or IOError: {exc}")
        except Exception as exc:
            self.get_logger().error(f"Unexpected error: {exc}")
            raise

    def close(self):
        self.get_logger().info("Closing ReSpeaker audio stream...")

        if self.stream is not None:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except Exception:
                pass

        try:
            self.pa.terminate()
        except Exception:
            pass


def main(args=None):
    rclpy.init(args=args)
    node = ReSpeakerMultiChannelNode()

    try:
        rclpy.spin(node)
    finally:
        node.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import threading

import numpy as np
import pyaudio
import rospy

from respeaker_ros_recorder.msg import AudioDataStamped


def normalize_device_index(device_index):
    device_index = int(device_index)
    return None if device_index < 0 else device_index


class AudioCaptureWorker:
    def __init__(
        self,
        name,
        sample_rate,
        channels,
        chunk_size,
        device_index,
        frame_id,
        topic_name,
    ):
        self.name = name
        self.sample_rate = int(sample_rate)
        self.channels = int(channels)
        self.chunk_size = int(chunk_size)
        self.device_index = normalize_device_index(device_index)
        self.frame_id = frame_id
        self.topic_name = topic_name
        self.chunk_duration = float(self.chunk_size) / float(self.sample_rate)

        self.pub = rospy.Publisher(self.topic_name, AudioDataStamped, queue_size=20)
        self.pa = pyaudio.PyAudio()
        self.stream = None
        self.thread = None
        self.stop_event = threading.Event()

    def start(self):
        rospy.loginfo("[%s] opening audio stream", self.name)
        rospy.loginfo("[%s] sample_rate: %s", self.name, self.sample_rate)
        rospy.loginfo("[%s] channels: %s", self.name, self.channels)
        rospy.loginfo("[%s] chunk_size: %s", self.name, self.chunk_size)
        rospy.loginfo("[%s] device_index: %s", self.name, self.device_index)
        rospy.loginfo("[%s] topic_name: %s", self.name, self.topic_name)

        self.stream = self.pa.open(
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            input_device_index=self.device_index,
            frames_per_buffer=self.chunk_size,
        )

        self.thread = threading.Thread(target=self.run, name=self.name)
        self.thread.daemon = True
        self.thread.start()

    def run(self):
        expected_size = self.chunk_size * self.channels

        while not rospy.is_shutdown() and not self.stop_event.is_set():
            try:
                raw_data = self.stream.read(
                    self.chunk_size,
                    exception_on_overflow=False,
                )
                t_end = rospy.Time.now()
                t_start = t_end - rospy.Duration.from_sec(self.chunk_duration)

                audio_np = np.frombuffer(raw_data, dtype=np.int16)
                if audio_np.size != expected_size:
                    rospy.logwarn(
                        "[%s] unexpected audio size: %s, expected %s",
                        self.name,
                        audio_np.size,
                        expected_size,
                    )
                    continue

                msg = AudioDataStamped()
                msg.header.stamp = t_start
                msg.header.frame_id = self.frame_id
                msg.sample_rate = self.sample_rate
                msg.channels = self.channels
                msg.frames = self.chunk_size
                msg.data = audio_np.tolist()

                self.pub.publish(msg)

            except IOError as exc:
                rospy.logwarn("[%s] audio input overflow or IOError: %s", self.name, exc)
            except Exception as exc:
                rospy.logerr("[%s] unexpected error: %s", self.name, exc)
                rospy.signal_shutdown(f"{self.name} failed")
                break

    def close(self):
        rospy.loginfo("[%s] closing audio stream", self.name)
        self.stop_event.set()

        if self.thread is not None:
            self.thread.join(timeout=2.0)

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


class DualAudioCaptureNode:
    def __init__(self):
        self.workers = [
            AudioCaptureWorker(
                name="mic1",
                sample_rate=rospy.get_param("~mic1_sample_rate", 16000),
                channels=rospy.get_param("~mic1_channels", 2),
                chunk_size=rospy.get_param("~mic1_chunk_size", 1600),
                device_index=rospy.get_param("~mic1_device_index", -1),
                frame_id=rospy.get_param("~mic1_frame_id", "mic1"),
                topic_name=rospy.get_param("~mic1_topic_name", "/mic1/audio_raw"),
            ),
            AudioCaptureWorker(
                name="mic2",
                sample_rate=rospy.get_param("~mic2_sample_rate", 16000),
                channels=rospy.get_param("~mic2_channels", 6),
                chunk_size=rospy.get_param("~mic2_chunk_size", 1600),
                device_index=rospy.get_param("~mic2_device_index", -1),
                frame_id=rospy.get_param("~mic2_frame_id", "mic2"),
                topic_name=rospy.get_param("~mic2_topic_name", "/mic2/audio_raw"),
            ),
        ]

    def start(self):
        for worker in self.workers:
            worker.start()

        rospy.loginfo("Dual audio capture started.")

    def close(self):
        for worker in self.workers:
            worker.close()


if __name__ == "__main__":
    rospy.init_node("dual_audio_capture_node")

    node = DualAudioCaptureNode()

    try:
        node.start()
        rospy.spin()
    finally:
        node.close()

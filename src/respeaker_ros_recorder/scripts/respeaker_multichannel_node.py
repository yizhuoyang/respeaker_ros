#!/usr/bin/env python3
import rospy
import pyaudio
import numpy as np

from respeaker_ros_recorder.msg import AudioDataStamped


class ReSpeakerMultiChannelNode:
    def __init__(self):
        self.sample_rate = int(rospy.get_param("~sample_rate", 16000))
        self.channels = int(rospy.get_param("~channels", 6))
        self.chunk_size = int(rospy.get_param("~chunk_size", 1600))
        self.device_index = rospy.get_param("~device_index", -1)

        if self.device_index < 0:
            self.device_index = None
        else:
            self.device_index = int(self.device_index)

        self.frame_id = rospy.get_param("~frame_id", "respeaker")
        self.topic_name = rospy.get_param("~topic_name", "/respeaker/audio_raw")

        self.pub = rospy.Publisher(
            self.topic_name,
            AudioDataStamped,
            queue_size=20
        )

        self.pa = pyaudio.PyAudio()

        rospy.loginfo("Opening ReSpeaker audio stream...")
        rospy.loginfo(f"sample_rate: {self.sample_rate}")
        rospy.loginfo(f"channels: {self.channels}")
        rospy.loginfo(f"chunk_size: {self.chunk_size}")
        rospy.loginfo(f"device_index: {self.device_index}")

        self.stream = self.pa.open(
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            input_device_index=self.device_index,
            frames_per_buffer=self.chunk_size
        )

        self.chunk_duration = float(self.chunk_size) / float(self.sample_rate)

        rospy.loginfo("ReSpeaker multi-channel recorder started.")

    def run(self):
        while not rospy.is_shutdown():
            try:
                # 这里先记录 read 结束后的时间
                t_end = rospy.Time.now()

                raw_data = self.stream.read(
                    self.chunk_size,
                    exception_on_overflow=False
                )

                # 估计 chunk 起始时间
                t_start = t_end - rospy.Duration.from_sec(self.chunk_duration)

                audio_np = np.frombuffer(raw_data, dtype=np.int16)

                expected_size = self.chunk_size * self.channels
                if audio_np.size != expected_size:
                    rospy.logwarn(
                        f"Unexpected audio size: {audio_np.size}, "
                        f"expected {expected_size}"
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

            except IOError as e:
                rospy.logwarn(f"Audio input overflow or IOError: {e}")
                continue
            except Exception as e:
                rospy.logerr(f"Unexpected error: {e}")
                break

    def close(self):
        rospy.loginfo("Closing ReSpeaker audio stream...")
        try:
            self.stream.stop_stream()
            self.stream.close()
        except Exception:
            pass

        try:
            self.pa.terminate()
        except Exception:
            pass


if __name__ == "__main__":
    rospy.init_node("respeaker_multichannel_node")

    node = ReSpeakerMultiChannelNode()

    try:
        node.run()
    finally:
        node.close()

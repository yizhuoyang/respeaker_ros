#!/usr/bin/env python3
"""Publish approximate ROS CameraInfo from incoming RGB image metadata for testing."""

import math


def pinhole_intrinsics(width, height, hfov_deg=69.0, vfov_deg=None, fx=None, fy=None, cx=None, cy=None):
    """Return approximate pinhole intrinsics, allowing calibrated values to override FOV guesses."""
    width = int(width)
    height = int(height)
    if width <= 0 or height <= 0:
        raise ValueError("Image width and height must be positive.")
    if fx is None:
        hfov_rad = math.radians(float(hfov_deg))
        if hfov_rad <= 0.0 or hfov_rad >= math.pi:
            raise ValueError("~hfov_deg must be between 0 and 180 degrees.")
        fx = width / (2.0 * math.tan(hfov_rad / 2.0))
    if fy is None:
        if vfov_deg is None:
            fy = fx
        else:
            vfov_rad = math.radians(float(vfov_deg))
            if vfov_rad <= 0.0 or vfov_rad >= math.pi:
                raise ValueError("~vfov_deg must be between 0 and 180 degrees.")
            fy = height / (2.0 * math.tan(vfov_rad / 2.0))
    if cx is None:
        cx = (width - 1.0) / 2.0
    if cy is None:
        cy = (height - 1.0) / 2.0
    return float(fx), float(fy), float(cx), float(cy)


class FakeCameraInfoNode:
    def __init__(self):
        import rospy
        from sensor_msgs.msg import CameraInfo, Image

        self.rospy = rospy
        self.CameraInfo = CameraInfo
        self.rgb_topic = rospy.get_param("~rgb_topic", "/camera/color/image_raw")
        self.output_topic = rospy.get_param("~output_topic", "/camera/color/camera_info")
        self.frame_id_override = rospy.get_param("~frame_id", "")
        self.distortion_model = rospy.get_param("~distortion_model", "plumb_bob")
        self.hfov_deg = float(rospy.get_param("~hfov_deg", 69.0))
        self.vfov_deg = self.optional_float_param("~vfov_deg")
        self.fx = self.optional_float_param("~fx")
        self.fy = self.optional_float_param("~fy")
        self.cx = self.optional_float_param("~cx")
        self.cy = self.optional_float_param("~cy")
        self.last_spec = None
        self.publisher = rospy.Publisher(self.output_topic, CameraInfo, queue_size=1, latch=True)
        self.subscriber = rospy.Subscriber(self.rgb_topic, Image, self.rgb_callback, queue_size=2)
        rospy.logwarn(
            "Publishing approximate CameraInfo for testing only: input=%s output=%s hfov=%.2f deg. "
            "Replace it with calibrated CameraInfo before evaluating map accuracy.",
            self.rgb_topic,
            self.output_topic,
            self.hfov_deg,
        )

    def optional_float_param(self, name):
        if self.rospy.has_param(name):
            return float(self.rospy.get_param(name))
        return None

    def rgb_callback(self, image):
        message, spec = self.make_camera_info(image)
        self.publisher.publish(message)
        if spec != self.last_spec:
            self.rospy.loginfo(
                "Fake CameraInfo: frame=%s size=%dx%d fx=%.3f fy=%.3f cx=%.3f cy=%.3f",
                message.header.frame_id,
                message.width,
                message.height,
                message.K[0],
                message.K[4],
                message.K[2],
                message.K[5],
            )
            self.last_spec = spec

    def make_camera_info(self, image):
        fx, fy, cx, cy = pinhole_intrinsics(
            image.width,
            image.height,
            hfov_deg=self.hfov_deg,
            vfov_deg=self.vfov_deg,
            fx=self.fx,
            fy=self.fy,
            cx=self.cx,
            cy=self.cy,
        )
        message = self.CameraInfo()
        message.header.seq = image.header.seq
        message.header.stamp = image.header.stamp
        message.header.frame_id = self.frame_id_override or image.header.frame_id
        message.width = image.width
        message.height = image.height
        message.distortion_model = self.distortion_model
        message.D = [0.0, 0.0, 0.0, 0.0, 0.0]
        message.K = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
        message.R = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        message.P = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
        spec = (message.header.frame_id, message.width, message.height, fx, fy, cx, cy)
        return message, spec


def main():
    import rospy

    rospy.init_node("fake_color_camera_info")
    FakeCameraInfoNode()
    rospy.spin()


if __name__ == "__main__":
    main()

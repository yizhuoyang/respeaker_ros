#!/usr/bin/env python3
"""Convert Livox ROS1 CustomMsg point clouds into RViz-compatible PointCloud2."""

import math
import os
from pathlib import Path
import sys


def load_livox_custom_message_type():
    """Load CustomMsg from a sourced driver workspace or its generated Python path."""
    try:
        from livox_ros_driver2.msg import CustomMsg

        return CustomMsg, None
    except ModuleNotFoundError as exc:
        if exc.name != "livox_ros_driver2":
            raise

    candidates = []
    configured_path = os.environ.get("LIVOX_ROS_MSG_PATH")
    if configured_path:
        candidates.append(Path(configured_path).expanduser())
    candidates.append(Path.home() / "driver_ws" / "devel" / "lib" / "python3" / "dist-packages")

    for candidate in candidates:
        if not (candidate / "livox_ros_driver2" / "msg").is_dir():
            continue
        sys.path.insert(0, str(candidate))
        try:
            from livox_ros_driver2.msg import CustomMsg

            return CustomMsg, str(candidate)
        except ModuleNotFoundError:
            sys.path.remove(str(candidate))

    searched = ", ".join(str(path) for path in candidates)
    raise ModuleNotFoundError(
        "Cannot import livox_ros_driver2.msg.CustomMsg. "
        "Run 'source /opt/ros/noetic/setup.bash' and "
        "'source /home/kemove/driver_ws/devel/setup.bash', or set LIVOX_ROS_MSG_PATH "
        "to the driver workspace devel/lib/python3/dist-packages directory. "
        f"Searched fallback paths: {searched}"
    )


def custom_points_to_xyzi(
    points,
    stride=1,
    x_min=None,
    x_max=None,
    y_min=None,
    y_max=None,
    z_min=None,
    z_max=None,
    range_min_m=None,
    range_max_m=None,
):
    """Filter and convert Livox custom points to PointCloud2 XYZI rows."""
    stride = max(int(stride), 1)
    rows = []
    retained_index = 0
    for point in points:
        x = float(point.x)
        y = float(point.y)
        z = float(point.z)
        if not all(math.isfinite(value) for value in (x, y, z)):
            continue
        xy_range = math.hypot(x, y)
        if x_min is not None and x < x_min:
            continue
        if x_max is not None and x > x_max:
            continue
        if y_min is not None and y < y_min:
            continue
        if y_max is not None and y > y_max:
            continue
        if z_min is not None and z < z_min:
            continue
        if z_max is not None and z > z_max:
            continue
        if range_min_m is not None and xy_range < range_min_m:
            continue
        if range_max_m is not None and xy_range > range_max_m:
            continue
        if retained_index % stride == 0:
            rows.append((x, y, z, float(point.reflectivity)))
        retained_index += 1
    return rows


def optional_float_param(rospy, name):
    return float(rospy.get_param(name)) if rospy.has_param(name) else None


class LivoxCustomToPointCloud2:
    def __init__(self):
        import rospy
        from sensor_msgs.msg import PointCloud2

        self.rospy = rospy
        CustomMsg, fallback_path = load_livox_custom_message_type()
        if fallback_path is not None:
            rospy.logwarn(
                "Loaded livox_ros_driver2 messages from %s because driver_ws was not on PYTHONPATH. "
                "Prefer sourcing its devel/setup.bash.",
                fallback_path,
            )

        input_topic = rospy.get_param("~input_topic", "/livox/lidar")
        output_topic = rospy.get_param("~output_topic", "/livox/points_rviz")
        self.frame_id_override = rospy.get_param("~frame_id", "")
        self.point_stride = max(int(rospy.get_param("~point_stride", 1)), 1)
        self.stamp_mode = str(rospy.get_param("~stamp_mode", "source")).strip().lower()
        if self.stamp_mode not in ("source", "current", "zero"):
            raise ValueError("~stamp_mode must be 'source', 'current', or 'zero'.")
        self.max_message_age_sec = float(rospy.get_param("~max_message_age_sec", 0.0))
        self.bounds = {
            "x_min": optional_float_param(rospy, "~x_min"),
            "x_max": optional_float_param(rospy, "~x_max"),
            "y_min": optional_float_param(rospy, "~y_min"),
            "y_max": optional_float_param(rospy, "~y_max"),
            "z_min": optional_float_param(rospy, "~z_min"),
            "z_max": optional_float_param(rospy, "~z_max"),
            "range_min_m": optional_float_param(rospy, "~range_min_m"),
            "range_max_m": optional_float_param(rospy, "~range_max_m"),
        }
        self.publisher = rospy.Publisher(output_topic, PointCloud2, queue_size=1)
        self.subscriber = rospy.Subscriber(
            input_topic, CustomMsg, self.callback, queue_size=1, tcp_nodelay=True
        )
        rospy.loginfo(
            "Livox CustomMsg -> PointCloud2: input=%s output=%s point_stride=%s "
            "stamp_mode=%s max_message_age_sec=%.3f filters=%s",
            input_topic,
            output_topic,
            self.point_stride,
            self.stamp_mode,
            self.max_message_age_sec,
            {key: value for key, value in self.bounds.items() if value is not None} or "none",
        )

    def callback(self, msg):
        from sensor_msgs import point_cloud2
        from sensor_msgs.msg import PointField
        from std_msgs.msg import Header

        now = self.rospy.Time.now()
        source_stamp = msg.header.stamp
        if (
            self.max_message_age_sec > 0.0
            and source_stamp.to_sec() > 0.0
            and (now - source_stamp).to_sec() > self.max_message_age_sec
        ):
            self.rospy.logwarn_throttle(
                2.0,
                "Dropping stale Livox frame: age %.3fs > %.3fs",
                (now - source_stamp).to_sec(),
                self.max_message_age_sec,
            )
            return

        header = Header()
        header.seq = msg.header.seq
        header.frame_id = msg.header.frame_id
        if self.stamp_mode == "current":
            header.stamp = now
        elif self.stamp_mode == "zero":
            header.stamp = self.rospy.Time(0)
        else:
            header.stamp = source_stamp
        if self.frame_id_override:
            header.frame_id = self.frame_id_override
        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="intensity", offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        cloud = point_cloud2.create_cloud(
            header,
            fields,
            custom_points_to_xyzi(msg.points, self.point_stride, **self.bounds),
        )
        self.publisher.publish(cloud)


def main():
    import rospy

    rospy.init_node("livox_custom_to_pointcloud2")
    LivoxCustomToPointCloud2()
    rospy.spin()


if __name__ == "__main__":
    main()

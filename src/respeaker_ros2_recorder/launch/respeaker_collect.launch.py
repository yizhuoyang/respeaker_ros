from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import ExecuteProcess
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    sample_rate = LaunchConfiguration("sample_rate")
    channels = LaunchConfiguration("channels")
    chunk_size = LaunchConfiguration("chunk_size")
    device_index = LaunchConfiguration("device_index")
    frame_id = LaunchConfiguration("frame_id")
    topic_name = LaunchConfiguration("topic_name")
    record_bag = LaunchConfiguration("record_bag")
    bag_path = LaunchConfiguration("bag_path")

    return LaunchDescription([
        DeclareLaunchArgument("sample_rate", default_value="16000"),
        DeclareLaunchArgument("channels", default_value="6"),
        DeclareLaunchArgument("chunk_size", default_value="1600"),
        DeclareLaunchArgument("device_index", default_value="-1"),
        DeclareLaunchArgument("frame_id", default_value="respeaker"),
        DeclareLaunchArgument("topic_name", default_value="/respeaker/audio_raw"),
        DeclareLaunchArgument("record_bag", default_value="false"),
        DeclareLaunchArgument("bag_path", default_value="respeaker_audio"),

        Node(
            package="respeaker_ros2_recorder",
            executable="respeaker_multichannel_node.py",
            name="respeaker_multichannel_node",
            output="screen",
            parameters=[{
                "sample_rate": sample_rate,
                "channels": channels,
                "chunk_size": chunk_size,
                "device_index": device_index,
                "frame_id": frame_id,
                "topic_name": topic_name,
            }],
        ),

        ExecuteProcess(
            condition=IfCondition(record_bag),
            cmd=["ros2", "bag", "record", "-o", bag_path, topic_name],
            output="screen",
        ),
    ])

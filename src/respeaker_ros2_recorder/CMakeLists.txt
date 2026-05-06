cmake_minimum_required(VERSION 3.8)
project(respeaker_ros2_recorder)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/AudioDataStamped.msg"
  DEPENDENCIES std_msgs
)

ament_python_install_package(${PROJECT_NAME})

install(PROGRAMS
  scripts/respeaker_multichannel_node.py
  scripts/export_audio_from_bag.py
  scripts/batch_export_audio_from_bags.py
  scripts/list_audio_devices.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  launch
  msg
  DESTINATION share/${PROJECT_NAME}
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()

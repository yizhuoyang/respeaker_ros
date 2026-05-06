// generated from rosidl_generator_cpp/resource/rosidl_generator_cpp__visibility_control.hpp.in
// generated code does not contain a copyright notice

#ifndef RESPEAKER_ROS2_RECORDER__MSG__ROSIDL_GENERATOR_CPP__VISIBILITY_CONTROL_HPP_
#define RESPEAKER_ROS2_RECORDER__MSG__ROSIDL_GENERATOR_CPP__VISIBILITY_CONTROL_HPP_

#ifdef __cplusplus
extern "C"
{
#endif

// This logic was borrowed (then namespaced) from the examples on the gcc wiki:
//     https://gcc.gnu.org/wiki/Visibility

#if defined _WIN32 || defined __CYGWIN__
  #ifdef __GNUC__
    #define ROSIDL_GENERATOR_CPP_EXPORT_respeaker_ros2_recorder __attribute__ ((dllexport))
    #define ROSIDL_GENERATOR_CPP_IMPORT_respeaker_ros2_recorder __attribute__ ((dllimport))
  #else
    #define ROSIDL_GENERATOR_CPP_EXPORT_respeaker_ros2_recorder __declspec(dllexport)
    #define ROSIDL_GENERATOR_CPP_IMPORT_respeaker_ros2_recorder __declspec(dllimport)
  #endif
  #ifdef ROSIDL_GENERATOR_CPP_BUILDING_DLL_respeaker_ros2_recorder
    #define ROSIDL_GENERATOR_CPP_PUBLIC_respeaker_ros2_recorder ROSIDL_GENERATOR_CPP_EXPORT_respeaker_ros2_recorder
  #else
    #define ROSIDL_GENERATOR_CPP_PUBLIC_respeaker_ros2_recorder ROSIDL_GENERATOR_CPP_IMPORT_respeaker_ros2_recorder
  #endif
#else
  #define ROSIDL_GENERATOR_CPP_EXPORT_respeaker_ros2_recorder __attribute__ ((visibility("default")))
  #define ROSIDL_GENERATOR_CPP_IMPORT_respeaker_ros2_recorder
  #if __GNUC__ >= 4
    #define ROSIDL_GENERATOR_CPP_PUBLIC_respeaker_ros2_recorder __attribute__ ((visibility("default")))
  #else
    #define ROSIDL_GENERATOR_CPP_PUBLIC_respeaker_ros2_recorder
  #endif
#endif

#ifdef __cplusplus
}
#endif

#endif  // RESPEAKER_ROS2_RECORDER__MSG__ROSIDL_GENERATOR_CPP__VISIBILITY_CONTROL_HPP_

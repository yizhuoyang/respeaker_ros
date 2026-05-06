// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from respeaker_ros2_recorder:msg/AudioDataStamped.idl
// generated code does not contain a copyright notice

#ifndef RESPEAKER_ROS2_RECORDER__MSG__DETAIL__AUDIO_DATA_STAMPED__STRUCT_H_
#define RESPEAKER_ROS2_RECORDER__MSG__DETAIL__AUDIO_DATA_STAMPED__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'header'
#include "std_msgs/msg/detail/header__struct.h"
// Member 'data'
#include "rosidl_runtime_c/primitives_sequence.h"

// Struct defined in msg/AudioDataStamped in the package respeaker_ros2_recorder.
typedef struct respeaker_ros2_recorder__msg__AudioDataStamped
{
  std_msgs__msg__Header header;
  uint32_t sample_rate;
  uint32_t channels;
  uint32_t frames;
  rosidl_runtime_c__int16__Sequence data;
} respeaker_ros2_recorder__msg__AudioDataStamped;

// Struct for a sequence of respeaker_ros2_recorder__msg__AudioDataStamped.
typedef struct respeaker_ros2_recorder__msg__AudioDataStamped__Sequence
{
  respeaker_ros2_recorder__msg__AudioDataStamped * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} respeaker_ros2_recorder__msg__AudioDataStamped__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // RESPEAKER_ROS2_RECORDER__MSG__DETAIL__AUDIO_DATA_STAMPED__STRUCT_H_

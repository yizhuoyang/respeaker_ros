// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from respeaker_ros2_recorder:msg/AudioDataStamped.idl
// generated code does not contain a copyright notice

#ifndef RESPEAKER_ROS2_RECORDER__MSG__DETAIL__AUDIO_DATA_STAMPED__TRAITS_HPP_
#define RESPEAKER_ROS2_RECORDER__MSG__DETAIL__AUDIO_DATA_STAMPED__TRAITS_HPP_

#include "respeaker_ros2_recorder/msg/detail/audio_data_stamped__struct.hpp"
#include <rosidl_runtime_cpp/traits.hpp>
#include <stdint.h>
#include <type_traits>

// Include directives for member types
// Member 'header'
#include "std_msgs/msg/detail/header__traits.hpp"

namespace rosidl_generator_traits
{

template<>
inline const char * data_type<respeaker_ros2_recorder::msg::AudioDataStamped>()
{
  return "respeaker_ros2_recorder::msg::AudioDataStamped";
}

template<>
inline const char * name<respeaker_ros2_recorder::msg::AudioDataStamped>()
{
  return "respeaker_ros2_recorder/msg/AudioDataStamped";
}

template<>
struct has_fixed_size<respeaker_ros2_recorder::msg::AudioDataStamped>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<respeaker_ros2_recorder::msg::AudioDataStamped>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<respeaker_ros2_recorder::msg::AudioDataStamped>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // RESPEAKER_ROS2_RECORDER__MSG__DETAIL__AUDIO_DATA_STAMPED__TRAITS_HPP_

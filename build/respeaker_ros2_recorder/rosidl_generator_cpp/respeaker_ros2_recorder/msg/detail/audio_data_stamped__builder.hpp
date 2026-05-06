// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from respeaker_ros2_recorder:msg/AudioDataStamped.idl
// generated code does not contain a copyright notice

#ifndef RESPEAKER_ROS2_RECORDER__MSG__DETAIL__AUDIO_DATA_STAMPED__BUILDER_HPP_
#define RESPEAKER_ROS2_RECORDER__MSG__DETAIL__AUDIO_DATA_STAMPED__BUILDER_HPP_

#include "respeaker_ros2_recorder/msg/detail/audio_data_stamped__struct.hpp"
#include <rosidl_runtime_cpp/message_initialization.hpp>
#include <algorithm>
#include <utility>


namespace respeaker_ros2_recorder
{

namespace msg
{

namespace builder
{

class Init_AudioDataStamped_data
{
public:
  explicit Init_AudioDataStamped_data(::respeaker_ros2_recorder::msg::AudioDataStamped & msg)
  : msg_(msg)
  {}
  ::respeaker_ros2_recorder::msg::AudioDataStamped data(::respeaker_ros2_recorder::msg::AudioDataStamped::_data_type arg)
  {
    msg_.data = std::move(arg);
    return std::move(msg_);
  }

private:
  ::respeaker_ros2_recorder::msg::AudioDataStamped msg_;
};

class Init_AudioDataStamped_frames
{
public:
  explicit Init_AudioDataStamped_frames(::respeaker_ros2_recorder::msg::AudioDataStamped & msg)
  : msg_(msg)
  {}
  Init_AudioDataStamped_data frames(::respeaker_ros2_recorder::msg::AudioDataStamped::_frames_type arg)
  {
    msg_.frames = std::move(arg);
    return Init_AudioDataStamped_data(msg_);
  }

private:
  ::respeaker_ros2_recorder::msg::AudioDataStamped msg_;
};

class Init_AudioDataStamped_channels
{
public:
  explicit Init_AudioDataStamped_channels(::respeaker_ros2_recorder::msg::AudioDataStamped & msg)
  : msg_(msg)
  {}
  Init_AudioDataStamped_frames channels(::respeaker_ros2_recorder::msg::AudioDataStamped::_channels_type arg)
  {
    msg_.channels = std::move(arg);
    return Init_AudioDataStamped_frames(msg_);
  }

private:
  ::respeaker_ros2_recorder::msg::AudioDataStamped msg_;
};

class Init_AudioDataStamped_sample_rate
{
public:
  explicit Init_AudioDataStamped_sample_rate(::respeaker_ros2_recorder::msg::AudioDataStamped & msg)
  : msg_(msg)
  {}
  Init_AudioDataStamped_channels sample_rate(::respeaker_ros2_recorder::msg::AudioDataStamped::_sample_rate_type arg)
  {
    msg_.sample_rate = std::move(arg);
    return Init_AudioDataStamped_channels(msg_);
  }

private:
  ::respeaker_ros2_recorder::msg::AudioDataStamped msg_;
};

class Init_AudioDataStamped_header
{
public:
  Init_AudioDataStamped_header()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_AudioDataStamped_sample_rate header(::respeaker_ros2_recorder::msg::AudioDataStamped::_header_type arg)
  {
    msg_.header = std::move(arg);
    return Init_AudioDataStamped_sample_rate(msg_);
  }

private:
  ::respeaker_ros2_recorder::msg::AudioDataStamped msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::respeaker_ros2_recorder::msg::AudioDataStamped>()
{
  return respeaker_ros2_recorder::msg::builder::Init_AudioDataStamped_header();
}

}  // namespace respeaker_ros2_recorder

#endif  // RESPEAKER_ROS2_RECORDER__MSG__DETAIL__AUDIO_DATA_STAMPED__BUILDER_HPP_

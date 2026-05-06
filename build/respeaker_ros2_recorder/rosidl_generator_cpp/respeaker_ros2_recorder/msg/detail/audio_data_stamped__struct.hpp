// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from respeaker_ros2_recorder:msg/AudioDataStamped.idl
// generated code does not contain a copyright notice

#ifndef RESPEAKER_ROS2_RECORDER__MSG__DETAIL__AUDIO_DATA_STAMPED__STRUCT_HPP_
#define RESPEAKER_ROS2_RECORDER__MSG__DETAIL__AUDIO_DATA_STAMPED__STRUCT_HPP_

#include <rosidl_runtime_cpp/bounded_vector.hpp>
#include <rosidl_runtime_cpp/message_initialization.hpp>
#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>


// Include directives for member types
// Member 'header'
#include "std_msgs/msg/detail/header__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__respeaker_ros2_recorder__msg__AudioDataStamped __attribute__((deprecated))
#else
# define DEPRECATED__respeaker_ros2_recorder__msg__AudioDataStamped __declspec(deprecated)
#endif

namespace respeaker_ros2_recorder
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct AudioDataStamped_
{
  using Type = AudioDataStamped_<ContainerAllocator>;

  explicit AudioDataStamped_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : header(_init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->sample_rate = 0ul;
      this->channels = 0ul;
      this->frames = 0ul;
    }
  }

  explicit AudioDataStamped_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : header(_alloc, _init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->sample_rate = 0ul;
      this->channels = 0ul;
      this->frames = 0ul;
    }
  }

  // field types and members
  using _header_type =
    std_msgs::msg::Header_<ContainerAllocator>;
  _header_type header;
  using _sample_rate_type =
    uint32_t;
  _sample_rate_type sample_rate;
  using _channels_type =
    uint32_t;
  _channels_type channels;
  using _frames_type =
    uint32_t;
  _frames_type frames;
  using _data_type =
    std::vector<int16_t, typename ContainerAllocator::template rebind<int16_t>::other>;
  _data_type data;

  // setters for named parameter idiom
  Type & set__header(
    const std_msgs::msg::Header_<ContainerAllocator> & _arg)
  {
    this->header = _arg;
    return *this;
  }
  Type & set__sample_rate(
    const uint32_t & _arg)
  {
    this->sample_rate = _arg;
    return *this;
  }
  Type & set__channels(
    const uint32_t & _arg)
  {
    this->channels = _arg;
    return *this;
  }
  Type & set__frames(
    const uint32_t & _arg)
  {
    this->frames = _arg;
    return *this;
  }
  Type & set__data(
    const std::vector<int16_t, typename ContainerAllocator::template rebind<int16_t>::other> & _arg)
  {
    this->data = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    respeaker_ros2_recorder::msg::AudioDataStamped_<ContainerAllocator> *;
  using ConstRawPtr =
    const respeaker_ros2_recorder::msg::AudioDataStamped_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<respeaker_ros2_recorder::msg::AudioDataStamped_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<respeaker_ros2_recorder::msg::AudioDataStamped_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      respeaker_ros2_recorder::msg::AudioDataStamped_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<respeaker_ros2_recorder::msg::AudioDataStamped_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      respeaker_ros2_recorder::msg::AudioDataStamped_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<respeaker_ros2_recorder::msg::AudioDataStamped_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<respeaker_ros2_recorder::msg::AudioDataStamped_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<respeaker_ros2_recorder::msg::AudioDataStamped_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__respeaker_ros2_recorder__msg__AudioDataStamped
    std::shared_ptr<respeaker_ros2_recorder::msg::AudioDataStamped_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__respeaker_ros2_recorder__msg__AudioDataStamped
    std::shared_ptr<respeaker_ros2_recorder::msg::AudioDataStamped_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const AudioDataStamped_ & other) const
  {
    if (this->header != other.header) {
      return false;
    }
    if (this->sample_rate != other.sample_rate) {
      return false;
    }
    if (this->channels != other.channels) {
      return false;
    }
    if (this->frames != other.frames) {
      return false;
    }
    if (this->data != other.data) {
      return false;
    }
    return true;
  }
  bool operator!=(const AudioDataStamped_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct AudioDataStamped_

// alias to use template instance with default allocator
using AudioDataStamped =
  respeaker_ros2_recorder::msg::AudioDataStamped_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace respeaker_ros2_recorder

#endif  // RESPEAKER_ROS2_RECORDER__MSG__DETAIL__AUDIO_DATA_STAMPED__STRUCT_HPP_

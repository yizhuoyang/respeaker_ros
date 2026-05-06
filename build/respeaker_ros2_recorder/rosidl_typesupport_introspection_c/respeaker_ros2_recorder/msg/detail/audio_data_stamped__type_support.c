// generated from rosidl_typesupport_introspection_c/resource/idl__type_support.c.em
// with input from respeaker_ros2_recorder:msg/AudioDataStamped.idl
// generated code does not contain a copyright notice

#include <stddef.h>
#include "respeaker_ros2_recorder/msg/detail/audio_data_stamped__rosidl_typesupport_introspection_c.h"
#include "respeaker_ros2_recorder/msg/rosidl_typesupport_introspection_c__visibility_control.h"
#include "rosidl_typesupport_introspection_c/field_types.h"
#include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/message_introspection.h"
#include "respeaker_ros2_recorder/msg/detail/audio_data_stamped__functions.h"
#include "respeaker_ros2_recorder/msg/detail/audio_data_stamped__struct.h"


// Include directives for member types
// Member `header`
#include "std_msgs/msg/header.h"
// Member `header`
#include "std_msgs/msg/detail/header__rosidl_typesupport_introspection_c.h"
// Member `data`
#include "rosidl_runtime_c/primitives_sequence_functions.h"

#ifdef __cplusplus
extern "C"
{
#endif

void AudioDataStamped__rosidl_typesupport_introspection_c__AudioDataStamped_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  respeaker_ros2_recorder__msg__AudioDataStamped__init(message_memory);
}

void AudioDataStamped__rosidl_typesupport_introspection_c__AudioDataStamped_fini_function(void * message_memory)
{
  respeaker_ros2_recorder__msg__AudioDataStamped__fini(message_memory);
}

static rosidl_typesupport_introspection_c__MessageMember AudioDataStamped__rosidl_typesupport_introspection_c__AudioDataStamped_message_member_array[5] = {
  {
    "header",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(respeaker_ros2_recorder__msg__AudioDataStamped, header),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "sample_rate",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_UINT32,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(respeaker_ros2_recorder__msg__AudioDataStamped, sample_rate),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "channels",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_UINT32,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(respeaker_ros2_recorder__msg__AudioDataStamped, channels),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "frames",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_UINT32,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(respeaker_ros2_recorder__msg__AudioDataStamped, frames),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "data",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_INT16,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    true,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(respeaker_ros2_recorder__msg__AudioDataStamped, data),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers AudioDataStamped__rosidl_typesupport_introspection_c__AudioDataStamped_message_members = {
  "respeaker_ros2_recorder__msg",  // message namespace
  "AudioDataStamped",  // message name
  5,  // number of fields
  sizeof(respeaker_ros2_recorder__msg__AudioDataStamped),
  AudioDataStamped__rosidl_typesupport_introspection_c__AudioDataStamped_message_member_array,  // message members
  AudioDataStamped__rosidl_typesupport_introspection_c__AudioDataStamped_init_function,  // function to initialize message memory (memory has to be allocated)
  AudioDataStamped__rosidl_typesupport_introspection_c__AudioDataStamped_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t AudioDataStamped__rosidl_typesupport_introspection_c__AudioDataStamped_message_type_support_handle = {
  0,
  &AudioDataStamped__rosidl_typesupport_introspection_c__AudioDataStamped_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_respeaker_ros2_recorder
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, respeaker_ros2_recorder, msg, AudioDataStamped)() {
  AudioDataStamped__rosidl_typesupport_introspection_c__AudioDataStamped_message_member_array[0].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, std_msgs, msg, Header)();
  if (!AudioDataStamped__rosidl_typesupport_introspection_c__AudioDataStamped_message_type_support_handle.typesupport_identifier) {
    AudioDataStamped__rosidl_typesupport_introspection_c__AudioDataStamped_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &AudioDataStamped__rosidl_typesupport_introspection_c__AudioDataStamped_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif

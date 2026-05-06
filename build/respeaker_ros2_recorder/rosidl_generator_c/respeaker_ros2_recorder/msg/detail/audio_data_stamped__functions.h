// generated from rosidl_generator_c/resource/idl__functions.h.em
// with input from respeaker_ros2_recorder:msg/AudioDataStamped.idl
// generated code does not contain a copyright notice

#ifndef RESPEAKER_ROS2_RECORDER__MSG__DETAIL__AUDIO_DATA_STAMPED__FUNCTIONS_H_
#define RESPEAKER_ROS2_RECORDER__MSG__DETAIL__AUDIO_DATA_STAMPED__FUNCTIONS_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stdlib.h>

#include "rosidl_runtime_c/visibility_control.h"
#include "respeaker_ros2_recorder/msg/rosidl_generator_c__visibility_control.h"

#include "respeaker_ros2_recorder/msg/detail/audio_data_stamped__struct.h"

/// Initialize msg/AudioDataStamped message.
/**
 * If the init function is called twice for the same message without
 * calling fini inbetween previously allocated memory will be leaked.
 * \param[in,out] msg The previously allocated message pointer.
 * Fields without a default value will not be initialized by this function.
 * You might want to call memset(msg, 0, sizeof(
 * respeaker_ros2_recorder__msg__AudioDataStamped
 * )) before or use
 * respeaker_ros2_recorder__msg__AudioDataStamped__create()
 * to allocate and initialize the message.
 * \return true if initialization was successful, otherwise false
 */
ROSIDL_GENERATOR_C_PUBLIC_respeaker_ros2_recorder
bool
respeaker_ros2_recorder__msg__AudioDataStamped__init(respeaker_ros2_recorder__msg__AudioDataStamped * msg);

/// Finalize msg/AudioDataStamped message.
/**
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_respeaker_ros2_recorder
void
respeaker_ros2_recorder__msg__AudioDataStamped__fini(respeaker_ros2_recorder__msg__AudioDataStamped * msg);

/// Create msg/AudioDataStamped message.
/**
 * It allocates the memory for the message, sets the memory to zero, and
 * calls
 * respeaker_ros2_recorder__msg__AudioDataStamped__init().
 * \return The pointer to the initialized message if successful,
 * otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_respeaker_ros2_recorder
respeaker_ros2_recorder__msg__AudioDataStamped *
respeaker_ros2_recorder__msg__AudioDataStamped__create();

/// Destroy msg/AudioDataStamped message.
/**
 * It calls
 * respeaker_ros2_recorder__msg__AudioDataStamped__fini()
 * and frees the memory of the message.
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_respeaker_ros2_recorder
void
respeaker_ros2_recorder__msg__AudioDataStamped__destroy(respeaker_ros2_recorder__msg__AudioDataStamped * msg);

/// Check for msg/AudioDataStamped message equality.
/**
 * \param[in] lhs The message on the left hand size of the equality operator.
 * \param[in] rhs The message on the right hand size of the equality operator.
 * \return true if messages are equal, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_respeaker_ros2_recorder
bool
respeaker_ros2_recorder__msg__AudioDataStamped__are_equal(const respeaker_ros2_recorder__msg__AudioDataStamped * lhs, const respeaker_ros2_recorder__msg__AudioDataStamped * rhs);

/// Copy a msg/AudioDataStamped message.
/**
 * This functions performs a deep copy, as opposed to the shallow copy that
 * plain assignment yields.
 *
 * \param[in] input The source message pointer.
 * \param[out] output The target message pointer, which must
 *   have been initialized before calling this function.
 * \return true if successful, or false if either pointer is null
 *   or memory allocation fails.
 */
ROSIDL_GENERATOR_C_PUBLIC_respeaker_ros2_recorder
bool
respeaker_ros2_recorder__msg__AudioDataStamped__copy(
  const respeaker_ros2_recorder__msg__AudioDataStamped * input,
  respeaker_ros2_recorder__msg__AudioDataStamped * output);

/// Initialize array of msg/AudioDataStamped messages.
/**
 * It allocates the memory for the number of elements and calls
 * respeaker_ros2_recorder__msg__AudioDataStamped__init()
 * for each element of the array.
 * \param[in,out] array The allocated array pointer.
 * \param[in] size The size / capacity of the array.
 * \return true if initialization was successful, otherwise false
 * If the array pointer is valid and the size is zero it is guaranteed
 # to return true.
 */
ROSIDL_GENERATOR_C_PUBLIC_respeaker_ros2_recorder
bool
respeaker_ros2_recorder__msg__AudioDataStamped__Sequence__init(respeaker_ros2_recorder__msg__AudioDataStamped__Sequence * array, size_t size);

/// Finalize array of msg/AudioDataStamped messages.
/**
 * It calls
 * respeaker_ros2_recorder__msg__AudioDataStamped__fini()
 * for each element of the array and frees the memory for the number of
 * elements.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_respeaker_ros2_recorder
void
respeaker_ros2_recorder__msg__AudioDataStamped__Sequence__fini(respeaker_ros2_recorder__msg__AudioDataStamped__Sequence * array);

/// Create array of msg/AudioDataStamped messages.
/**
 * It allocates the memory for the array and calls
 * respeaker_ros2_recorder__msg__AudioDataStamped__Sequence__init().
 * \param[in] size The size / capacity of the array.
 * \return The pointer to the initialized array if successful, otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_respeaker_ros2_recorder
respeaker_ros2_recorder__msg__AudioDataStamped__Sequence *
respeaker_ros2_recorder__msg__AudioDataStamped__Sequence__create(size_t size);

/// Destroy array of msg/AudioDataStamped messages.
/**
 * It calls
 * respeaker_ros2_recorder__msg__AudioDataStamped__Sequence__fini()
 * on the array,
 * and frees the memory of the array.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_respeaker_ros2_recorder
void
respeaker_ros2_recorder__msg__AudioDataStamped__Sequence__destroy(respeaker_ros2_recorder__msg__AudioDataStamped__Sequence * array);

/// Check for msg/AudioDataStamped message array equality.
/**
 * \param[in] lhs The message array on the left hand size of the equality operator.
 * \param[in] rhs The message array on the right hand size of the equality operator.
 * \return true if message arrays are equal in size and content, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_respeaker_ros2_recorder
bool
respeaker_ros2_recorder__msg__AudioDataStamped__Sequence__are_equal(const respeaker_ros2_recorder__msg__AudioDataStamped__Sequence * lhs, const respeaker_ros2_recorder__msg__AudioDataStamped__Sequence * rhs);

/// Copy an array of msg/AudioDataStamped messages.
/**
 * This functions performs a deep copy, as opposed to the shallow copy that
 * plain assignment yields.
 *
 * \param[in] input The source array pointer.
 * \param[out] output The target array pointer, which must
 *   have been initialized before calling this function.
 * \return true if successful, or false if either pointer
 *   is null or memory allocation fails.
 */
ROSIDL_GENERATOR_C_PUBLIC_respeaker_ros2_recorder
bool
respeaker_ros2_recorder__msg__AudioDataStamped__Sequence__copy(
  const respeaker_ros2_recorder__msg__AudioDataStamped__Sequence * input,
  respeaker_ros2_recorder__msg__AudioDataStamped__Sequence * output);

#ifdef __cplusplus
}
#endif

#endif  // RESPEAKER_ROS2_RECORDER__MSG__DETAIL__AUDIO_DATA_STAMPED__FUNCTIONS_H_

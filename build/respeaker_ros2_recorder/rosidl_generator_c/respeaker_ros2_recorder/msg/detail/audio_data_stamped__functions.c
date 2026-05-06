// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from respeaker_ros2_recorder:msg/AudioDataStamped.idl
// generated code does not contain a copyright notice
#include "respeaker_ros2_recorder/msg/detail/audio_data_stamped__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


// Include directives for member types
// Member `header`
#include "std_msgs/msg/detail/header__functions.h"
// Member `data`
#include "rosidl_runtime_c/primitives_sequence_functions.h"

bool
respeaker_ros2_recorder__msg__AudioDataStamped__init(respeaker_ros2_recorder__msg__AudioDataStamped * msg)
{
  if (!msg) {
    return false;
  }
  // header
  if (!std_msgs__msg__Header__init(&msg->header)) {
    respeaker_ros2_recorder__msg__AudioDataStamped__fini(msg);
    return false;
  }
  // sample_rate
  // channels
  // frames
  // data
  if (!rosidl_runtime_c__int16__Sequence__init(&msg->data, 0)) {
    respeaker_ros2_recorder__msg__AudioDataStamped__fini(msg);
    return false;
  }
  return true;
}

void
respeaker_ros2_recorder__msg__AudioDataStamped__fini(respeaker_ros2_recorder__msg__AudioDataStamped * msg)
{
  if (!msg) {
    return;
  }
  // header
  std_msgs__msg__Header__fini(&msg->header);
  // sample_rate
  // channels
  // frames
  // data
  rosidl_runtime_c__int16__Sequence__fini(&msg->data);
}

bool
respeaker_ros2_recorder__msg__AudioDataStamped__are_equal(const respeaker_ros2_recorder__msg__AudioDataStamped * lhs, const respeaker_ros2_recorder__msg__AudioDataStamped * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // header
  if (!std_msgs__msg__Header__are_equal(
      &(lhs->header), &(rhs->header)))
  {
    return false;
  }
  // sample_rate
  if (lhs->sample_rate != rhs->sample_rate) {
    return false;
  }
  // channels
  if (lhs->channels != rhs->channels) {
    return false;
  }
  // frames
  if (lhs->frames != rhs->frames) {
    return false;
  }
  // data
  if (!rosidl_runtime_c__int16__Sequence__are_equal(
      &(lhs->data), &(rhs->data)))
  {
    return false;
  }
  return true;
}

bool
respeaker_ros2_recorder__msg__AudioDataStamped__copy(
  const respeaker_ros2_recorder__msg__AudioDataStamped * input,
  respeaker_ros2_recorder__msg__AudioDataStamped * output)
{
  if (!input || !output) {
    return false;
  }
  // header
  if (!std_msgs__msg__Header__copy(
      &(input->header), &(output->header)))
  {
    return false;
  }
  // sample_rate
  output->sample_rate = input->sample_rate;
  // channels
  output->channels = input->channels;
  // frames
  output->frames = input->frames;
  // data
  if (!rosidl_runtime_c__int16__Sequence__copy(
      &(input->data), &(output->data)))
  {
    return false;
  }
  return true;
}

respeaker_ros2_recorder__msg__AudioDataStamped *
respeaker_ros2_recorder__msg__AudioDataStamped__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  respeaker_ros2_recorder__msg__AudioDataStamped * msg = (respeaker_ros2_recorder__msg__AudioDataStamped *)allocator.allocate(sizeof(respeaker_ros2_recorder__msg__AudioDataStamped), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(respeaker_ros2_recorder__msg__AudioDataStamped));
  bool success = respeaker_ros2_recorder__msg__AudioDataStamped__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
respeaker_ros2_recorder__msg__AudioDataStamped__destroy(respeaker_ros2_recorder__msg__AudioDataStamped * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    respeaker_ros2_recorder__msg__AudioDataStamped__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
respeaker_ros2_recorder__msg__AudioDataStamped__Sequence__init(respeaker_ros2_recorder__msg__AudioDataStamped__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  respeaker_ros2_recorder__msg__AudioDataStamped * data = NULL;

  if (size) {
    data = (respeaker_ros2_recorder__msg__AudioDataStamped *)allocator.zero_allocate(size, sizeof(respeaker_ros2_recorder__msg__AudioDataStamped), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = respeaker_ros2_recorder__msg__AudioDataStamped__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        respeaker_ros2_recorder__msg__AudioDataStamped__fini(&data[i - 1]);
      }
      allocator.deallocate(data, allocator.state);
      return false;
    }
  }
  array->data = data;
  array->size = size;
  array->capacity = size;
  return true;
}

void
respeaker_ros2_recorder__msg__AudioDataStamped__Sequence__fini(respeaker_ros2_recorder__msg__AudioDataStamped__Sequence * array)
{
  if (!array) {
    return;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();

  if (array->data) {
    // ensure that data and capacity values are consistent
    assert(array->capacity > 0);
    // finalize all array elements
    for (size_t i = 0; i < array->capacity; ++i) {
      respeaker_ros2_recorder__msg__AudioDataStamped__fini(&array->data[i]);
    }
    allocator.deallocate(array->data, allocator.state);
    array->data = NULL;
    array->size = 0;
    array->capacity = 0;
  } else {
    // ensure that data, size, and capacity values are consistent
    assert(0 == array->size);
    assert(0 == array->capacity);
  }
}

respeaker_ros2_recorder__msg__AudioDataStamped__Sequence *
respeaker_ros2_recorder__msg__AudioDataStamped__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  respeaker_ros2_recorder__msg__AudioDataStamped__Sequence * array = (respeaker_ros2_recorder__msg__AudioDataStamped__Sequence *)allocator.allocate(sizeof(respeaker_ros2_recorder__msg__AudioDataStamped__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = respeaker_ros2_recorder__msg__AudioDataStamped__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
respeaker_ros2_recorder__msg__AudioDataStamped__Sequence__destroy(respeaker_ros2_recorder__msg__AudioDataStamped__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    respeaker_ros2_recorder__msg__AudioDataStamped__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
respeaker_ros2_recorder__msg__AudioDataStamped__Sequence__are_equal(const respeaker_ros2_recorder__msg__AudioDataStamped__Sequence * lhs, const respeaker_ros2_recorder__msg__AudioDataStamped__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!respeaker_ros2_recorder__msg__AudioDataStamped__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
respeaker_ros2_recorder__msg__AudioDataStamped__Sequence__copy(
  const respeaker_ros2_recorder__msg__AudioDataStamped__Sequence * input,
  respeaker_ros2_recorder__msg__AudioDataStamped__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(respeaker_ros2_recorder__msg__AudioDataStamped);
    respeaker_ros2_recorder__msg__AudioDataStamped * data =
      (respeaker_ros2_recorder__msg__AudioDataStamped *)realloc(output->data, allocation_size);
    if (!data) {
      return false;
    }
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!respeaker_ros2_recorder__msg__AudioDataStamped__init(&data[i])) {
        /* free currently allocated and return false */
        for (; i-- > output->capacity; ) {
          respeaker_ros2_recorder__msg__AudioDataStamped__fini(&data[i]);
        }
        free(data);
        return false;
      }
    }
    output->data = data;
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!respeaker_ros2_recorder__msg__AudioDataStamped__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}

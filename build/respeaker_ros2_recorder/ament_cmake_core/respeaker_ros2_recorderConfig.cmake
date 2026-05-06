# generated from ament/cmake/core/templates/nameConfig.cmake.in

# prevent multiple inclusion
if(_respeaker_ros2_recorder_CONFIG_INCLUDED)
  # ensure to keep the found flag the same
  if(NOT DEFINED respeaker_ros2_recorder_FOUND)
    # explicitly set it to FALSE, otherwise CMake will set it to TRUE
    set(respeaker_ros2_recorder_FOUND FALSE)
  elseif(NOT respeaker_ros2_recorder_FOUND)
    # use separate condition to avoid uninitialized variable warning
    set(respeaker_ros2_recorder_FOUND FALSE)
  endif()
  return()
endif()
set(_respeaker_ros2_recorder_CONFIG_INCLUDED TRUE)

# output package information
if(NOT respeaker_ros2_recorder_FIND_QUIETLY)
  message(STATUS "Found respeaker_ros2_recorder: 0.0.0 (${respeaker_ros2_recorder_DIR})")
endif()

# warn when using a deprecated package
if(NOT "" STREQUAL "")
  set(_msg "Package 'respeaker_ros2_recorder' is deprecated")
  # append custom deprecation text if available
  if(NOT "" STREQUAL "TRUE")
    set(_msg "${_msg} ()")
  endif()
  # optionally quiet the deprecation message
  if(NOT ${respeaker_ros2_recorder_DEPRECATED_QUIET})
    message(DEPRECATION "${_msg}")
  endif()
endif()

# flag package as ament-based to distinguish it after being find_package()-ed
set(respeaker_ros2_recorder_FOUND_AMENT_PACKAGE TRUE)

# include all config extra files
set(_extras "rosidl_cmake-extras.cmake;ament_cmake_export_dependencies-extras.cmake;ament_cmake_export_libraries-extras.cmake;ament_cmake_export_targets-extras.cmake;ament_cmake_export_include_directories-extras.cmake;rosidl_cmake_export_typesupport_libraries-extras.cmake;rosidl_cmake_export_typesupport_targets-extras.cmake")
foreach(_extra ${_extras})
  include("${respeaker_ros2_recorder_DIR}/${_extra}")
endforeach()

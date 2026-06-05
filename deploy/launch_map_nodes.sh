#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${ROOT_DIR}/logs/map_nodes"
mkdir -p "${LOG_DIR}"

PIDS=()
NAMES=()

start_node() {
  local name="$1"
  shift
  local log_file="${LOG_DIR}/${name}.log"

  echo "[map] starting ${name}"
  (
    cd "${ROOT_DIR}"
    exec "$@"
  ) >"${log_file}" 2>&1 &

  local pid=$!
  PIDS+=("${pid}")
  NAMES+=("${name}")
  echo "[map] ${name} pid=${pid} log=${log_file}"
}

stop_all() {
  echo
  echo "[map] stopping ${#PIDS[@]} process(es)"
  for pid in "${PIDS[@]}"; do
    if kill -0 "${pid}" >/dev/null 2>&1; then
      kill -INT "${pid}" >/dev/null 2>&1 || true
    fi
  done
  sleep 2
  for pid in "${PIDS[@]}"; do
    if kill -0 "${pid}" >/dev/null 2>&1; then
      kill -TERM "${pid}" >/dev/null 2>&1 || true
    fi
  done
  sleep 1
  for pid in "${PIDS[@]}"; do
    if kill -0 "${pid}" >/dev/null 2>&1; then
      echo "[map] force killing pid=${pid}"
      kill -KILL "${pid}" >/dev/null 2>&1 || true
    fi
  done
  wait || true
  echo "[map] stopped"
}

trap stop_all INT TERM EXIT

start_node sslnet_audio_map_fusion \
  python deploy/ros1_sslnet_audio_map_fusion.py \
    _broadcast_odom_tf:=true \
    _sensor_frame_id:=livox_frame \
    _min_signal_prob:=0.5




start_node yoloe_visual_map \
  python deploy_yolo/ros1_yoloe_visual_map.py \
    _model:=/home/kemove/yyz/audio-nav/respeaker_ros/deploy_yolo/yoloe-11s-seg.engine \
    _rgb_topic:=/camera/color/image_raw \
    _depth_topic:=/camera/depth/image_rect_raw \
    _camera_info_topic:=/camera/color/camera_info \
    _odom_topic:=/Odometry \
    _use_audio_map_geometry:=true \
    _audio_map_topic:=/sslnet_audio_map/map \
    _projection_pose_source:=odom \
    _classes:=guitar \
    _conf:=0.2 \
    _map_update_mode:=tracks \
    _track_merge_distance_m:=0.60 \
    _track_smoothing_alpha:=0.25 \
    _track_marker_radius_m:=0.20 \
    _project_mask_footprint:=false \
    _device:=0

start_node audio_visual_goal_fusion \
  python deploy/ros1_audio_visual_goal_fusion.py \
    _audio_map_topic:=/sslnet_audio_map/map \
    _visual_map_topic:=/yoloe_visual_map/map \
    _audio_weight:=1.0 \
    _visual_weight:=1.0 \
    _global_frame_id:=camera_init \
    _reference_map:=audio \
    _normalize_inputs:=false

echo
echo "[map] map nodes started. Logs: ${LOG_DIR}"
echo "[map] restart this script for each test sequence to reset maps."
echo "[map] press Ctrl+C to stop all map nodes."

while true; do
  for index in "${!PIDS[@]}"; do
    pid="${PIDS[$index]}"
    name="${NAMES[$index]}"
    if ! kill -0 "${pid}" >/dev/null 2>&1; then
      echo "[map] ${name} exited. See ${LOG_DIR}/${name}.log"
      exit 1
    fi
  done
  sleep 1
done

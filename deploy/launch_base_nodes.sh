#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${ROOT_DIR}/logs/base_nodes"
mkdir -p "${LOG_DIR}"

PIDS=()
NAMES=()

start_node() {
  local name="$1"
  shift
  local log_file="${LOG_DIR}/${name}.log"

  echo "[base] starting ${name}"
  (
    cd "${ROOT_DIR}"
    exec "$@"
  ) >"${log_file}" 2>&1 &

  local pid=$!
  PIDS+=("${pid}")
  NAMES+=("${name}")
  echo "[base] ${name} pid=${pid} log=${log_file}"
}

stop_all() {
  echo
  echo "[base] stopping ${#PIDS[@]} process(es)"
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
      echo "[base] force killing pid=${pid}"
      kill -KILL "${pid}" >/dev/null 2>&1 || true
    fi
  done
  wait || true
  echo "[base] stopped"
}

trap stop_all INT TERM EXIT

start_node sslnet_audio_engine \
  python deploy/ros1_sslnet_audio_engine_node.py \
    _engine:=weights/ssl_doa_distance_synced/last_model.engine \
    _device:=cuda:0 \
    _window_seconds:=0.5 \
    _hop_seconds:=0.25 \
    audio_topic:=/mic2/audio_raw

start_node livox_to_pointcloud2 \
  python deploy/ros1_livox_custom_to_pointcloud2.py \
    _z_max:=1.9 \
    _point_stride:=4

start_node sslnet_rviz_markers \
  python deploy/ros1_sslnet_rviz_markers.py \
    _frame_id:=livox_frame \
    _show_distributions:=false

echo
echo "[base] persistent nodes started. Logs: ${LOG_DIR}"
echo "[base] press Ctrl+C to stop all base nodes."

while true; do
  for index in "${!PIDS[@]}"; do
    pid="${PIDS[$index]}"
    name="${NAMES[$index]}"
    if ! kill -0 "${pid}" >/dev/null 2>&1; then
      echo "[base] ${name} exited. See ${LOG_DIR}/${name}.log"
      exit 1
    fi
  done
  sleep 1
done

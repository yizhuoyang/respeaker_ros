#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import numpy as np


ODOM_FIELDS = [
    "px",
    "py",
    "pz",
    "qx",
    "qy",
    "qz",
    "qw",
    "linear_x",
    "linear_y",
    "linear_z",
    "angular_x",
    "angular_y",
    "angular_z",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        required=True,
        help="Synchronized dataset directory, or a parent directory containing multiple datasets.",
    )
    parser.add_argument(
        "--odom",
        default="lio_robo_odom",
        choices=["lio_odom", "lio_robo_odom"],
        help="Which odom npz to use.",
    )
    parser.add_argument(
        "--target",
        default="final",
        help="Target position. Use 'final' or 'x,y,z'. Default: final odom position.",
    )
    parser.add_argument(
        "--output-prefix",
        default="doa",
        help="Output prefix under dataset directory.",
    )
    parser.add_argument(
        "--sample-output-dir",
        default=None,
        help="Per-sample npy directory name under each dataset. Default: <output-prefix>_<odom>.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="When --dataset is a parent directory, search datasets recursively.",
    )
    parser.add_argument(
        "--input-frame",
        default="mic",
        choices=["mic", "lidar"],
        help="Frame represented by the odom pose. Use 'lidar' to convert it to mic/head frame before DOA.",
    )
    parser.add_argument(
        "--lidar-pitch-deg",
        type=float,
        default=-23.0,
        help="LiDAR pitch relative to mic/head frame in degrees. Downward tilt is negative. Used with --input-frame lidar.",
    )
    parser.add_argument(
        "--mic-translation",
        default="0.2,0.0,0.0",
        help="Mic origin translation from LiDAR origin, expressed in LiDAR frame, meters: x,y,z. Used with --input-frame lidar.",
    )
    args = parser.parse_args()

    dataset_dirs = find_dataset_dirs(Path(args.dataset), args.odom, args.recursive)
    for dataset_dir in dataset_dirs:
        compute_doa(
            dataset_dir=dataset_dir,
            odom_name=args.odom,
            target_arg=args.target,
            output_prefix=args.output_prefix,
            sample_output_dir=args.sample_output_dir,
            input_frame=args.input_frame,
            lidar_pitch_deg=args.lidar_pitch_deg,
            mic_translation_arg=args.mic_translation,
        )


def compute_doa(
    dataset_dir,
    odom_name="lio_robo_odom",
    target_arg="final",
    output_prefix="doa",
    sample_output_dir=None,
    input_frame="mic",
    lidar_pitch_deg=-23.0,
    mic_translation_arg="0.2,0.0,0.0",
):
    odom_path = dataset_dir / f"{odom_name}.npz"
    if not odom_path.exists():
        raise RuntimeError(f"Odom npz does not exist: {odom_path}")

    odom = np.load(odom_path, allow_pickle=False)
    fields = [str(field) for field in odom["fields"]]
    data = odom["data"]
    sample_ids = odom["sample_ids"]
    timestamps_ns = odom["timestamps_ns"]

    field_index = {field: fields.index(field) for field in fields}
    positions = data[:, [
        field_index["px"],
        field_index["py"],
        field_index["pz"],
    ]]
    quaternions = data[:, [
        field_index["qx"],
        field_index["qy"],
        field_index["qz"],
        field_index["qw"],
    ]]

    mic_translation_lidar = parse_vector3(mic_translation_arg, "--mic-translation")
    lidar_to_mic_rotation = build_lidar_to_mic_rotation(lidar_pitch_deg)
    mic_positions, mic_rotations = convert_poses_to_mic_frame(
        positions=positions,
        quaternions=quaternions,
        input_frame=input_frame,
        lidar_to_mic_rotation=lidar_to_mic_rotation,
        mic_translation_lidar=mic_translation_lidar,
    )

    target = parse_target(target_arg, mic_positions)

    rows = []
    for sample_id, timestamp_ns, lidar_position, mic_position, mic_rotation in zip(
        sample_ids,
        timestamps_ns,
        positions,
        mic_positions,
        mic_rotations,
    ):
        heading_world = mic_rotation @ np.array([1.0, 0.0, 0.0], dtype=np.float64)
        target_vector_3d = target - mic_position
        target_vector_xy = target[:2] - mic_position[:2]
        heading_xy = heading_world[:2]
        distance_xy = float(np.linalg.norm(target_vector_xy))
        distance_3d = float(np.linalg.norm(target_vector_3d))
        robot_heading_world = horizontal_angle_deg(heading_xy)
        target_azimuth_world = horizontal_angle_deg(target_vector_xy)
        signed_yaw = signed_horizontal_angle_deg(heading_xy, target_vector_xy)

        rows.append({
            "sample_id": format_sample_id(sample_id),
            "timestamp_ns": int(timestamp_ns),
            "robot_x": float(mic_position[0]),
            "robot_y": float(mic_position[1]),
            "source_odom_x": float(lidar_position[0]),
            "source_odom_y": float(lidar_position[1]),
            "target_x": float(target[0]),
            "target_y": float(target[1]),
            "target_vector_world_x": float(target_vector_xy[0]),
            "target_vector_world_y": float(target_vector_xy[1]),
            "heading_world_x": float(heading_world[0]),
            "heading_world_y": float(heading_world[1]),
            "distance_xy": distance_xy,
            "distance_3d": distance_3d,
            "robot_heading_world_deg": robot_heading_world,
            "target_azimuth_world_deg": target_azimuth_world,
            "heading_target_yaw_signed_deg": signed_yaw,
            "heading_target_yaw_abs_deg": abs(signed_yaw),
        })

    csv_path = dataset_dir / f"{output_prefix}_{odom_name}.csv"
    npz_path = dataset_dir / f"{output_prefix}_{odom_name}.npz"
    sample_dir = dataset_dir / (sample_output_dir or f"{output_prefix}_{odom_name}")
    distance_csv_path = dataset_dir / f"distance_{odom_name}.csv"
    distance_npz_path = dataset_dir / f"distance_{odom_name}.npz"
    distance_sample_dir = dataset_dir / f"distance_{odom_name}"
    write_csv(csv_path, rows)
    write_npz(npz_path, rows)
    write_sample_npy(sample_dir, rows)
    write_distance_csv(distance_csv_path, rows)
    write_distance_npz(distance_npz_path, rows)
    write_distance_sample_npy(distance_sample_dir, rows)

    print(f"Wrote {csv_path}")
    print(f"Wrote {npz_path}")
    print(f"Wrote per-sample npy files under {sample_dir}")
    print(f"Wrote {distance_csv_path}")
    print(f"Wrote {distance_npz_path}")
    print(f"Wrote per-sample distance npy files under {distance_sample_dir}")
    print(f"Target position: {target.tolist()}")
    if input_frame == "lidar":
        print(f"Applied LiDAR->mic transform: lidar_pitch_deg={lidar_pitch_deg}, mic_translation={mic_translation_lidar.tolist()}")


def find_dataset_dirs(dataset_path, odom_name, recursive=False):
    odom_filename = f"{odom_name}.npz"
    if (dataset_path / odom_filename).exists():
        return [dataset_path]

    pattern = f"**/{odom_filename}" if recursive else f"*/{odom_filename}"
    dataset_dirs = sorted({path.parent for path in dataset_path.glob(pattern)})
    if not dataset_dirs:
        raise RuntimeError(f"No dataset containing {odom_filename} found under: {dataset_path}")

    print(f"Found {len(dataset_dirs)} datasets under {dataset_path}")
    return dataset_dirs


def parse_target(target_arg, positions):
    if target_arg == "final":
        return positions[-1].astype(np.float64)

    return parse_vector3(target_arg, "--target")


def format_sample_id(sample_id):
    text = str(sample_id)
    if text.isdigit():
        return text.zfill(6)
    return text


def parse_vector3(value, argument_name):
    values = [float(item.strip()) for item in value.split(",")]
    if len(values) != 3:
        raise RuntimeError(f"{argument_name} must be 'x,y,z'")
    return np.array(values, dtype=np.float64)


def convert_poses_to_mic_frame(
    positions,
    quaternions,
    input_frame,
    lidar_to_mic_rotation,
    mic_translation_lidar,
):
    mic_positions = []
    mic_rotations = []
    for position, quaternion in zip(positions, quaternions):
        source_rotation = quaternion_to_rotation_matrix(quaternion)

        if input_frame == "lidar":
            mic_position = position + source_rotation @ mic_translation_lidar
            mic_rotation = source_rotation @ lidar_to_mic_rotation
        else:
            mic_position = position
            mic_rotation = source_rotation

        mic_positions.append(mic_position)
        mic_rotations.append(mic_rotation)

    return np.array(mic_positions, dtype=np.float64), np.array(mic_rotations, dtype=np.float64)


def build_lidar_to_mic_rotation(lidar_pitch_deg):
    lidar_rotation_in_mic = rotation_y(np.radians(lidar_pitch_deg))
    return lidar_rotation_in_mic.T


def rotation_y(angle_rad):
    cosine = np.cos(angle_rad)
    sine = np.sin(angle_rad)
    return np.array([
        [cosine, 0.0, sine],
        [0.0, 1.0, 0.0],
        [-sine, 0.0, cosine],
    ], dtype=np.float64)


def horizontal_angle_deg(vector_xy):
    if np.linalg.norm(vector_xy) == 0:
        return 0.0

    return float(np.degrees(np.arctan2(vector_xy[1], vector_xy[0])))


def signed_horizontal_angle_deg(heading_xy, target_xy):
    heading_norm = np.linalg.norm(heading_xy)
    target_norm = np.linalg.norm(target_xy)
    if heading_norm == 0 or target_norm == 0:
        return 0.0

    heading_xy = heading_xy / heading_norm
    target_xy = target_xy / target_norm
    cross_z = heading_xy[0] * target_xy[1] - heading_xy[1] * target_xy[0]
    dot = np.dot(heading_xy, target_xy)
    return float(np.degrees(np.arctan2(cross_z, dot)))


def quaternion_to_rotation_matrix(quaternion_xyzw):
    x, y, z, w = quaternion_xyzw
    norm = np.linalg.norm([x, y, z, w])
    if norm == 0:
        return np.eye(3)

    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def write_csv(csv_path, rows):
    if not rows:
        return

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_npz(npz_path, rows):
    if not rows:
        return

    sample_ids = np.array([row["sample_id"] for row in rows])
    timestamps_ns = np.array([row["timestamp_ns"] for row in rows], dtype=np.int64)
    fields = doa_fields()
    data = rows_to_array(rows, fields)

    np.savez(
        npz_path,
        sample_ids=sample_ids,
        timestamps_ns=timestamps_ns,
        fields=fields,
        data=data,
    )


def write_sample_npy(sample_dir, rows):
    if not rows:
        return

    sample_dir.mkdir(parents=True, exist_ok=True)
    fields = doa_fields()
    for row in rows:
        sample_id = str(row["sample_id"])
        np.save(sample_dir / f"{sample_id}.npy", rows_to_array([row], fields)[0])


def write_distance_csv(csv_path, rows):
    if not rows:
        return

    fields = list(distance_fields())
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_id", "timestamp_ns", *fields])
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "sample_id": row["sample_id"],
                "timestamp_ns": row["timestamp_ns"],
                **{field: row[field] for field in fields},
            })


def write_distance_npz(npz_path, rows):
    if not rows:
        return

    fields = distance_fields()
    sample_ids = np.array([row["sample_id"] for row in rows])
    timestamps_ns = np.array([row["timestamp_ns"] for row in rows], dtype=np.int64)
    data = rows_to_array(rows, fields)

    np.savez(
        npz_path,
        sample_ids=sample_ids,
        timestamps_ns=timestamps_ns,
        fields=fields,
        data=data,
    )


def write_distance_sample_npy(sample_dir, rows):
    if not rows:
        return

    sample_dir.mkdir(parents=True, exist_ok=True)
    fields = distance_fields()
    for row in rows:
        sample_id = str(row["sample_id"])
        np.save(sample_dir / f"{sample_id}.npy", rows_to_array([row], fields)[0])


def doa_fields():
    return np.array([
        "robot_x",
        "robot_y",
        "source_odom_x",
        "source_odom_y",
        "target_x",
        "target_y",
        "target_vector_world_x",
        "target_vector_world_y",
        "heading_world_x",
        "heading_world_y",
        "distance_xy",
        "distance_3d",
        "robot_heading_world_deg",
        "target_azimuth_world_deg",
        "heading_target_yaw_signed_deg",
        "heading_target_yaw_abs_deg",
    ])


def distance_fields():
    return np.array([
        "distance_xy",
        "distance_3d",
    ])


def rows_to_array(rows, fields):
    return np.array([[row[str(field)] for field in fields] for row in rows], dtype=np.float64)


if __name__ == "__main__":
    main()

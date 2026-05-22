#!/usr/bin/env python3
import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np


def quaternion_xyzw_to_rotation(qx, qy, qz, qw):
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm <= 0.0:
        raise ValueError("Invalid zero-length quaternion")

    qx /= norm
    qy /= norm
    qz /= norm
    qw /= norm

    xx = qx * qx
    yy = qy * qy
    zz = qz * qz
    xy = qx * qy
    xz = qx * qz
    yz = qy * qz
    wx = qw * qx
    wy = qw * qy
    wz = qw * qz

    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def load_odom(path):
    odom = np.load(path)
    if odom.shape[0] < 8:
        raise ValueError(f"Odom file must contain at least 8 values: {path}")

    stamp = float(odom[0])
    position = odom[1:4].astype(np.float64)
    rotation = quaternion_xyzw_to_rotation(
        float(odom[4]),
        float(odom[5]),
        float(odom[6]),
        float(odom[7]),
    )
    return stamp, position, rotation


def compute_doa(source_world, lidar_position, lidar_rotation, mic_offset_lidar):
    mic_world = lidar_position + lidar_rotation @ mic_offset_lidar
    source_vector_world = source_world - mic_world

    distance = float(np.linalg.norm(source_vector_world))
    if distance <= 0.0:
        raise ValueError("Source and microphone positions are identical")

    source_vector_mic = lidar_rotation.T @ source_vector_world
    unit_mic = source_vector_mic / distance

    azimuth = math.atan2(unit_mic[1], unit_mic[0])
    elevation = math.atan2(unit_mic[2], math.sqrt(unit_mic[0] ** 2 + unit_mic[1] ** 2))

    return {
        "mic_world": mic_world,
        "vector_mic": source_vector_mic,
        "unit_mic": unit_mic,
        "distance": distance,
        "azimuth": azimuth,
        "elevation": elevation,
    }


def write_csv(path, rows):
    fieldnames = [
        "index",
        "odom_file",
        "stamp",
        "source_x",
        "source_y",
        "source_z",
        "mic_x",
        "mic_y",
        "mic_z",
        "distance",
        "azimuth_rad",
        "azimuth_deg",
        "elevation_rad",
        "elevation_deg",
        "unit_x",
        "unit_y",
        "unit_z",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_config(path, args):
    path.write_text(json.dumps(vars(args), indent=2, sort_keys=True) + "\n")


def process_bag_dir(bag_dir, args):
    odom_dir = bag_dir / args.odom_dir_name
    if not odom_dir.is_dir():
        raise FileNotFoundError(f"Missing odom directory: {odom_dir}")

    odom_files = sorted(odom_dir.glob("*.npy"))
    if not odom_files:
        raise RuntimeError(f"No odom npy files found in: {odom_dir}")

    _, source_world, _ = load_odom(odom_files[-1])
    doa_dir = bag_dir / args.doa_dir_name
    doa_dir.mkdir(parents=True, exist_ok=True)

    mic_offset_lidar = np.array(
        [args.mic_offset_x, args.mic_offset_y, args.mic_offset_z],
        dtype=np.float64,
    )

    rows = []
    for index, odom_path in enumerate(odom_files, start=1):
        stamp, lidar_position, lidar_rotation = load_odom(odom_path)
        doa = compute_doa(
            source_world=source_world,
            lidar_position=lidar_position,
            lidar_rotation=lidar_rotation,
            mic_offset_lidar=mic_offset_lidar,
        )

        label = np.array(
            [
                doa["azimuth"],
                doa["elevation"],
                doa["unit_mic"][0],
                doa["unit_mic"][1],
                doa["unit_mic"][2],
                doa["distance"],
            ],
            dtype=np.float32,
        )
        np.save(doa_dir / odom_path.name, label)

        rows.append(
            {
                "index": index,
                "odom_file": str(odom_path),
                "stamp": f"{stamp:.9f}",
                "source_x": f"{source_world[0]:.9f}",
                "source_y": f"{source_world[1]:.9f}",
                "source_z": f"{source_world[2]:.9f}",
                "mic_x": f"{doa['mic_world'][0]:.9f}",
                "mic_y": f"{doa['mic_world'][1]:.9f}",
                "mic_z": f"{doa['mic_world'][2]:.9f}",
                "distance": f"{doa['distance']:.9f}",
                "azimuth_rad": f"{doa['azimuth']:.9f}",
                "azimuth_deg": f"{math.degrees(doa['azimuth']):.9f}",
                "elevation_rad": f"{doa['elevation']:.9f}",
                "elevation_deg": f"{math.degrees(doa['elevation']):.9f}",
                "unit_x": f"{doa['unit_mic'][0]:.9f}",
                "unit_y": f"{doa['unit_mic'][1]:.9f}",
                "unit_z": f"{doa['unit_mic'][2]:.9f}",
            }
        )

    write_csv(bag_dir / "doa_labels.csv", rows)
    return len(odom_files)


def iter_bag_dirs(data_root, recursive=False):
    if recursive:
        candidates = sorted(path for path in data_root.rglob("*") if path.is_dir())
    else:
        candidates = sorted(path for path in data_root.iterdir() if path.is_dir())

    return [path for path in candidates if (path / "odom").is_dir()]


def main():
    parser = argparse.ArgumentParser(
        description="Generate microphone-relative DOA labels from synchronized odom npy files."
    )
    parser.add_argument("--data-root", required=True, help="Dataset root containing bag_name/odom/*.npy")
    parser.add_argument("--odom-dir-name", default="odom")
    parser.add_argument("--doa-dir-name", default="doa")
    parser.add_argument(
        "--mic-offset-x",
        type=float,
        default=0.48,
        help="Microphone x offset from lidar in lidar/body frame, meters. Positive x is forward.",
    )
    parser.add_argument("--mic-offset-y", type=float, default=0.0)
    parser.add_argument("--mic-offset-z", type=float, default=0.0)
    parser.add_argument("--recursive", action="store_true", help="Search bag directories recursively")
    parser.add_argument("--keep-going", action="store_true", help="Continue if one bag directory fails")

    args = parser.parse_args()

    data_root = Path(args.data_root).expanduser().resolve()
    if not data_root.is_dir():
        raise NotADirectoryError(f"--data-root must be a directory: {data_root}")

    bag_dirs = iter_bag_dirs(data_root, recursive=args.recursive)
    if not bag_dirs:
        raise RuntimeError(f"No bag directories with odom data found in: {data_root}")

    save_config(data_root / "doa_config.json", args)

    ok_count = 0
    fail_count = 0

    print(f"Found {len(bag_dirs)} bag directory/directories.")
    print("DOA label format: [azimuth_rad, elevation_rad, unit_x, unit_y, unit_z, distance_m]")

    for bag_dir in bag_dirs:
        print("")
        print(f"Processing: {bag_dir}")
        try:
            count = process_bag_dir(bag_dir, args)
            ok_count += 1
            print(f"Saved {count} DOA label(s).")
        except Exception as exc:
            fail_count += 1
            print(f"Failed: {exc}", file=sys.stderr)
            if not args.keep_going:
                raise

    print("")
    print("Done.")
    print(f"Processed bag dirs: {ok_count}")
    print(f"Failed bag dirs: {fail_count}")

    return 1 if fail_count else 0


if __name__ == "__main__":
    sys.exit(main())

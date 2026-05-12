#!/usr/bin/env python3
import argparse
import json
import shutil
from pathlib import Path

import numpy as np


DEFAULT_MODALITY_DIRS = [
    "audio",
    "color",
    "depth",
    "lio_odom",
    "lio_robo_odom",
    "metadata",
    "doa_lio_odom",
    "distance_lio_odom",
    "doa_lio_robo_odom",
    "distance_lio_robo_odom",
]


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Find and process synchronized samples by distance, e.g. remove samples "
            "closer/farther than a threshold or samples whose distance maps to a bin range."
        )
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="One synced dataset directory, or a parent directory containing clock*/seq* datasets.",
    )
    parser.add_argument(
        "--odom",
        default="lio_odom",
        choices=["lio_odom", "lio_robo_odom"],
        help="Which distance_<odom> directory to inspect.",
    )
    parser.add_argument(
        "--bin",
        type=int,
        default=0,
        help="Exact distance bin to process when --max-distance and --max-bin are not set. Default: 0.",
    )
    parser.add_argument(
        "--max-bin",
        type=int,
        default=None,
        help="Process samples whose distance bin is <= this value.",
    )
    parser.add_argument(
        "--max-distance",
        type=float,
        default=None,
        help="Process samples whose distance_xy is <= this threshold in meters. This is the clearest option for removing near samples.",
    )
    parser.add_argument(
        "--min-distance",
        type=float,
        default=None,
        help="Process samples whose distance_xy is >= this threshold in meters. Use this to remove far samples.",
    )
    parser.add_argument(
        "--outside-range",
        action="store_true",
        help=(
            "When both --min-distance and --max-distance are set, process samples "
            "outside the range instead of inside it."
        ),
    )
    parser.add_argument("--num-bins", type=int, default=120, help="Distance bins used by the dataloader.")
    parser.add_argument("--r-min", type=float, default=0.0, help="Minimum distance used by the dataloader.")
    parser.add_argument("--r-max", type=float, default=6.0, help="Maximum distance used by the dataloader.")
    parser.add_argument(
        "--action",
        default="report",
        choices=["report", "move", "delete"],
        help="report only, move matched files aside, or delete matched files.",
    )
    parser.add_argument(
        "--trash-dir",
        default="_filtered_dist_bin",
        help="Directory name used with --action move. Created inside each dataset dir.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search for dataset directories under --dataset.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be changed without moving/deleting files.",
    )
    parser.add_argument(
        "--write-report",
        default=None,
        help="Optional JSON report path. Parent directories are created automatically.",
    )
    args = parser.parse_args()
    if args.outside_range and (args.min_distance is None or args.max_distance is None):
        parser.error("--outside-range requires both --min-distance and --max-distance")
    if (
        args.min_distance is not None
        and args.max_distance is not None
        and args.min_distance > args.max_distance
    ):
        parser.error("--min-distance must be <= --max-distance")

    dataset_dirs = find_dataset_dirs(Path(args.dataset), args.odom, recursive=args.recursive)
    report = {
        "dataset": str(Path(args.dataset)),
        "odom": args.odom,
        "bin": args.bin,
        "max_bin": args.max_bin,
        "max_distance": args.max_distance,
        "min_distance": args.min_distance,
        "outside_range": args.outside_range,
        "num_bins": args.num_bins,
        "r_min": args.r_min,
        "r_max": args.r_max,
        "action": args.action,
        "dry_run": args.dry_run,
        "datasets": [],
    }

    total = 0
    for dataset_dir in dataset_dirs:
        result = process_dataset(
            dataset_dir=dataset_dir,
            odom_name=args.odom,
            target_bin=args.bin,
            max_bin=args.max_bin,
            max_distance=args.max_distance,
            min_distance=args.min_distance,
            outside_range=args.outside_range,
            num_bins=args.num_bins,
            r_min=args.r_min,
            r_max=args.r_max,
            action=args.action,
            trash_dir_name=args.trash_dir,
            dry_run=args.dry_run,
        )
        total += len(result["matched_samples"])
        report["datasets"].append(result)

    print(f"\nTotal matched samples: {total}")
    if args.action == "report" or args.dry_run:
        print("No files were modified.")

    if args.write_report:
        report_path = Path(args.write_report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Wrote report: {report_path}")


def find_dataset_dirs(dataset_path, odom_name, recursive=False):
    distance_dir_name = f"distance_{odom_name}"
    if (dataset_path / distance_dir_name).exists():
        return [dataset_path]

    pattern = f"**/{distance_dir_name}" if recursive else f"*/{distance_dir_name}"
    dataset_dirs = sorted({path.parent for path in dataset_path.glob(pattern)})
    if not dataset_dirs:
        raise RuntimeError(f"No dataset containing {distance_dir_name} found under: {dataset_path}")
    return dataset_dirs


def process_dataset(
    dataset_dir,
    odom_name,
    target_bin,
    max_bin,
    max_distance,
    min_distance,
    outside_range,
    num_bins,
    r_min,
    r_max,
    action,
    trash_dir_name,
    dry_run,
):
    distance_dir = dataset_dir / f"distance_{odom_name}"
    if not distance_dir.exists():
        raise RuntimeError(f"Missing distance directory: {distance_dir}")

    matched = []
    for path in sorted(distance_dir.glob("*.npy"), key=lambda p: numeric_stem(p.stem)):
        distance = load_distance_xy(path)
        dist_bin = distance_to_bin(distance, num_bins=num_bins, r_min=r_min, r_max=r_max)
        if matches_filter(
            distance,
            dist_bin,
            target_bin,
            max_bin,
            max_distance,
            min_distance,
            outside_range,
        ):
            matched.append({
                "sample_id": path.stem,
                "distance_xy": distance,
                "dist_bin": dist_bin,
            })

    print(
        f"\n{dataset_dir.name}: matched {len(matched)} sample(s) with "
        f"{describe_filter(target_bin, max_bin, max_distance, min_distance, outside_range)}"
    )
    for row in matched[:20]:
        print(f"  {row['sample_id']}: distance_xy={row['distance_xy']:.6f}, bin={row['dist_bin']}")
    if len(matched) > 20:
        print(f"  ... {len(matched) - 20} more")

    if action in ("move", "delete") and not dry_run:
        for row in matched:
            process_sample_files(
                dataset_dir=dataset_dir,
                sample_id=row["sample_id"],
                action=action,
                trash_dir_name=trash_dir_name,
            )

    return {
        "dataset_dir": str(dataset_dir),
        "matched_samples": matched,
    }


def load_distance_xy(path):
    values = np.load(path).astype(np.float64)
    if values.size == 0:
        raise RuntimeError(f"Empty distance file: {path}")
    return float(values[0])


def distance_to_bin(distance, num_bins=120, r_min=0.0, r_max=6.0):
    axis = np.linspace(r_min, r_max, num_bins, dtype=np.float64)
    return int(np.argmin(np.abs(axis - distance)))


def matches_filter(distance, dist_bin, target_bin, max_bin, max_distance, min_distance, outside_range):
    if max_distance is not None or min_distance is not None:
        if outside_range:
            if min_distance is None or max_distance is None:
                raise RuntimeError("--outside-range requires both --min-distance and --max-distance")
            return distance < min_distance or distance > max_distance
        if max_distance is not None and distance > max_distance:
            return False
        if min_distance is not None and distance < min_distance:
            return False
        return True
    if max_bin is not None:
        return dist_bin <= max_bin
    return dist_bin == target_bin


def describe_filter(target_bin, max_bin, max_distance, min_distance, outside_range):
    if outside_range:
        return f"distance_xy < {min_distance:.6f} m or distance_xy > {max_distance:.6f} m"
    if max_distance is not None and min_distance is not None:
        return f"{min_distance:.6f} m <= distance_xy <= {max_distance:.6f} m"
    if max_distance is not None:
        return f"distance_xy <= {max_distance:.6f} m"
    if min_distance is not None:
        return f"distance_xy >= {min_distance:.6f} m"
    if max_bin is not None:
        return f"dist_bin <= {max_bin}"
    return f"dist_bin == {target_bin}"


def process_sample_files(dataset_dir, sample_id, action, trash_dir_name):
    for modality_dir_name in DEFAULT_MODALITY_DIRS:
        modality_dir = dataset_dir / modality_dir_name
        if not modality_dir.exists():
            continue

        for source in modality_dir.glob(f"{sample_id}.*"):
            if action == "delete":
                source.unlink()
            elif action == "move":
                target_dir = dataset_dir / trash_dir_name / modality_dir_name
                target_dir.mkdir(parents=True, exist_ok=True)
                shutil.move(str(source), str(target_dir / source.name))


def numeric_stem(stem):
    try:
        return int(stem)
    except ValueError:
        return stem


if __name__ == "__main__":
    main()

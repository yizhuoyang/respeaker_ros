#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np


MODALITY_DIRS = [
    "audio",
    "color",
    "depth",
    "lio_odom",
    "lio_robo_odom",
    "metadata",
]

NPZ_FILES = [
    "lio_odom.npz",
    "lio_robo_odom.npz",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        required=True,
        help="Synchronized dataset directory, e.g. synced_dataset/bag_001.",
    )
    parser.add_argument(
        "--keep-through",
        type=int,
        required=True,
        help="Keep samples with index <= this value. Later samples are deleted.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be deleted without modifying files.",
    )
    args = parser.parse_args()

    truncate_dataset(
        dataset_dir=Path(args.dataset),
        keep_through=args.keep_through,
        dry_run=args.dry_run,
    )


def truncate_dataset(dataset_dir, keep_through, dry_run=False):
    if keep_through < 0:
        raise RuntimeError("--keep-through must be >= 0")

    if not dataset_dir.is_dir():
        raise RuntimeError(f"Dataset directory does not exist: {dataset_dir}")

    print(f"Dataset: {dataset_dir}")
    print(f"Keeping sample indices <= {keep_through:06d}")
    print(f"Dry run: {dry_run}")

    delete_modality_files(dataset_dir, keep_through, dry_run)
    truncate_manifest(dataset_dir / "dataset_manifest.json", keep_through, dry_run)
    for npz_name in NPZ_FILES:
        truncate_npz(dataset_dir / npz_name, keep_through, dry_run)


def delete_modality_files(dataset_dir, keep_through, dry_run=False):
    for modality in MODALITY_DIRS:
        modality_dir = dataset_dir / modality
        if not modality_dir.is_dir():
            continue

        files = sorted(path for path in modality_dir.iterdir() if path.is_file())
        delete_count = 0
        for path in files:
            sample_index = parse_sample_index(path)
            if sample_index is None:
                continue
            if sample_index > keep_through:
                delete_count += 1
                print(f"delete {path}")
                if not dry_run:
                    path.unlink()

        print(f"{modality}: deleted {delete_count} file(s)")


def parse_sample_index(path):
    stem = path.stem
    if not stem.isdigit():
        return None
    return int(stem)


def truncate_manifest(manifest_path, keep_through, dry_run=False):
    if not manifest_path.exists():
        return

    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    samples = data.get("samples", [])
    kept = [
        sample for sample in samples
        if int(str(sample.get("id", "-1"))) <= keep_through
    ]

    print(f"dataset_manifest.json: {len(samples)} -> {len(kept)} sample(s)")
    if dry_run:
        return

    data["samples"] = kept
    manifest_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def truncate_npz(npz_path, keep_through, dry_run=False):
    if not npz_path.exists():
        return

    loaded = np.load(npz_path, allow_pickle=False)
    sample_ids = loaded["sample_ids"]
    keep_mask = np.array([int(str(sample_id)) <= keep_through for sample_id in sample_ids])
    old_count = len(sample_ids)
    new_count = int(keep_mask.sum())

    print(f"{npz_path.name}: {old_count} -> {new_count} row(s)")
    if dry_run:
        loaded.close()
        return

    arrays = {}
    for key in loaded.files:
        value = loaded[key]
        if key in ("sample_ids", "timestamps_ns", "diff_ns", "data"):
            arrays[key] = value[keep_mask]
        else:
            arrays[key] = value

    loaded.close()
    np.savez(npz_path, **arrays)


if __name__ == "__main__":
    main()

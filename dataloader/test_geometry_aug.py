#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataloader.data_loader_new import SyncedDeepMusicDataset, circular_abs_diff_deg


def main():
    args = parse_args()
    dataset_no_aug = SyncedDeepMusicDataset(
        root=args.data_root,
        split=args.split,
        object_names=args.object_name,
        odom_name=args.odom,
        geometry_aug=False,
    )
    dataset_aug = SyncedDeepMusicDataset(
        root=args.data_root,
        split=args.split,
        object_names=args.object_name,
        odom_name=args.odom,
        geometry_aug=True,
        geometry_aug_step_deg=args.geometry_aug_step_deg,
    )

    _, base_doa, base_sv, _ = dataset_no_aug[args.index]
    base_doa_value = float(base_doa.item())
    print(f"Dataset split={args.split}, samples={len(dataset_aug)}, index={args.index}")
    print(f"Base DOA without augmentation: {base_doa_value:.2f} deg")
    print(f"geometry_aug enabled in dataset: {dataset_aug.geometry_aug}")
    print(f"geometry_aug_step_deg: {dataset_aug.geometry_aug_step_deg}")
    print("")
    print("Repeated reads of the same sample:")

    seen = []
    for repeat_idx in range(args.repeats):
        _, aug_doa, aug_sv, _ = dataset_aug[args.index]
        aug_doa_value = float(aug_doa.item())
        inferred_rotation = (aug_doa_value - base_doa_value) % 360.0
        sv_delta = torch.mean(torch.abs(aug_sv - base_sv)).item()
        seen.append(round(inferred_rotation, 6))
        print(
            f"  repeat={repeat_idx:02d} "
            f"aug_doa={aug_doa_value:8.2f} deg "
            f"inferred_rotation={inferred_rotation:8.2f} deg "
            f"sv_mean_abs_delta={sv_delta:.6e}"
        )

    unique_rotations = sorted(set(seen))
    print("")
    print(f"Unique inferred rotations: {len(unique_rotations)} / {args.repeats}")
    if len(unique_rotations) > 1:
        print("PASS: geometry augmentation changes the DOA label and steering vector.")
    else:
        print("WARN: only one rotation observed. Increase --repeats or check split/geometry settings.")

    if args.geometry_aug_step_deg > 0:
        max_step_error = max(
            circular_abs_diff_deg(rotation, round(rotation / args.geometry_aug_step_deg) * args.geometry_aug_step_deg)
            for rotation in unique_rotations
        )
        print(f"Max step quantization error: {max_step_error:.6f} deg")


def parse_args():
    parser = argparse.ArgumentParser(description="Check whether SyncedDeepMusicDataset geometry augmentation is active.")
    parser.add_argument("--data-root", default="synced_dataset")
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument("--object-name", default="clock")
    parser.add_argument("--odom", default="lio_odom", choices=["lio_odom", "lio_robo_odom"])
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--geometry-aug-step-deg", type=float, default=1.0)
    return parser.parse_args()


if __name__ == "__main__":
    main()

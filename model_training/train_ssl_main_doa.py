#!/usr/bin/env python3
"""Convenience launcher for the original main_doa.py SSLNet training path.

This script does not define a new model. It only builds a recommended
main_doa.py command for the current synced_dataset layout and runs it.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the original SSLNet DOA/distance model through main_doa.py."
    )

    parser.add_argument("--data-root", default="synced_dataset")
    parser.add_argument("--object-name", default=None, help="Only load one object prefix, e.g. clock/person/dryer/guitar.")
    parser.add_argument("--run-name", default="ssl_main_doa", help="Used to create default weights/runs directories.")
    parser.add_argument("--save-dir", default=None)
    parser.add_argument("--log-dir", default=None)
    parser.add_argument("--checkpoint", default=None, help="Optional checkpoint for resume or finetune.")

    parser.add_argument("--model", default="audio", choices=["audio", "audio_depth"])
    parser.add_argument("--odom", default="lio_odom", choices=["lio_odom", "lio_robo_odom"])
    parser.add_argument("--audio-feat", default="ipd", choices=["ipd", "spec", "phase", "both", "gcc_phat_complex"])
    parser.add_argument("--audio-channels", default="1,2,3,4")
    parser.add_argument("--ipd-pairs", default="0-1,0-2,0-3,1-2,1-3,2-3")
    parser.add_argument("--min-distance", type=float, default=None, help="Only load samples with distance_xy >= this value.")
    parser.add_argument("--max-distance", type=float, default=None, help="Only load samples with distance_xy <= this value.")
    parser.add_argument("--use-classification", action="store_true")
    parser.add_argument("--classification-only", action="store_true")
    parser.add_argument("--freeze-classifier", action="store_true")
    parser.add_argument("--gate-doa-by-pred-class", action="store_true")
    parser.add_argument("--classification-weight", type=float, default=1.0)
    parser.add_argument("--distance-weight", type=float, default=0.5)
    parser.add_argument("--max-signal-abs", type=float, default=0.06)
    parser.add_argument("--class-balanced-sampler", action="store_true")
    parser.add_argument("--no-class-loss-weights", action="store_true")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--use-compress", action="store_true")
    parser.add_argument("--require-depth", action="store_true", help="By default missing depth is allowed for audio-only runs.")
    parser.add_argument("--no-pretrained-depth", action="store_true")
    parser.add_argument("--freeze-depth", action="store_true")

    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--balanced-sampler", action="store_true")

    parser.add_argument("--use-filter-mute-denoise", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--filter-mute-highpass-hz", type=float, default=120.0)
    parser.add_argument(
        "--filter-mute-notches-hz",
        default="48.8,66.4,179.7,341.8,867.2,1271.5,1845.7,2533.2,3783.2",
    )
    parser.add_argument("--filter-mute-threshold", type=float, default=0.06)
    parser.add_argument("--filter-mute-window-sec", type=float, default=0.05)
    parser.add_argument("--filter-mute-floor", type=float, default=0.02)
    parser.add_argument("--filter-mute-edge-smooth-ms", type=float, default=5.0)

    parser.add_argument("--use-time-mask", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--time-mask-prob", type=float, default=0.5)
    parser.add_argument("--time-mask-num", type=int, default=1)
    parser.add_argument("--time-mask-max-width", type=int, default=12)
    parser.add_argument("--time-mask-fill", default="zero", choices=["zero", "mean"])

    parser.add_argument("--use-spectral-denoise", action="store_true", help="Also enable the older spectral-gating denoiser.")
    parser.add_argument("--denoise-stationary-noise", action="append", default=None)
    parser.add_argument("--denoise-motion-noise", action="append", default=None)
    parser.add_argument("--denoise-highpass-hz", type=float, default=90.0)
    parser.add_argument("--denoise-notches-hz", default="48,180,342,1845")
    parser.add_argument("--denoise-spectral-strength", type=float, default=0.4)
    parser.add_argument("--denoise-gain-floor", type=float, default=0.5)
    parser.add_argument("--denoise-motion-strength", type=float, default=0.3)
    parser.add_argument("--denoise-motion-gain-floor", type=float, default=0.6)

    parser.add_argument("--dry-run", action="store_true", help="Only print the generated main_doa.py command.")
    return parser.parse_args()


def add_flag(cmd, enabled, flag):
    if enabled:
        cmd.append(flag)


def add_optional(cmd, flag, value):
    if value is not None:
        cmd.extend([flag, str(value)])


def main():
    args = parse_args()

    save_dir = args.save_dir or str(Path("weights") / args.run_name)
    log_dir = args.log_dir or str(Path("runs") / args.run_name)

    cmd = [
        sys.executable,
        str(REPO_ROOT / "main_doa.py"),
        "--data-root",
        args.data_root,
        "--odom",
        args.odom,
        "--model",
        args.model,
        "--audio-feat",
        args.audio_feat,
        "--audio-channels",
        args.audio_channels,
        "--ipd-pairs",
        args.ipd_pairs,
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
        "--weight-decay",
        str(args.weight_decay),
        "--num-workers",
        str(args.num_workers),
        "--device",
        args.device,
        "--seed",
        str(args.seed),
        "--image-size",
        str(args.image_size),
        "--save-dir",
        save_dir,
        "--log-dir",
        log_dir,
    ]

    add_optional(cmd, "--object-name", args.object_name)
    add_optional(cmd, "--checkpoint", args.checkpoint)
    add_optional(cmd, "--min-distance", args.min_distance)
    add_optional(cmd, "--max-distance", args.max_distance)
    if args.use_classification or args.classification_only or args.freeze_classifier or args.gate_doa_by_pred_class:
        if args.classification_only:
            cmd.append("--classification-only")
        if args.freeze_classifier:
            cmd.append("--freeze-classifier")
        if args.gate_doa_by_pred_class:
            cmd.append("--gate-doa-by-pred-class")
        cmd.extend([
            "--use-classification",
            "--classification-weight",
            str(args.classification_weight),
            "--distance-weight",
            str(args.distance_weight),
            "--max-signal-abs",
            str(args.max_signal_abs),
        ])
    add_flag(cmd, args.class_balanced_sampler, "--class-balanced-sampler")
    add_flag(cmd, args.no_class_loss_weights, "--no-class-loss-weights")

    add_flag(cmd, not args.require_depth, "--allow-missing-depth")
    add_flag(cmd, args.use_compress, "--use-compress")
    add_flag(cmd, args.no_pretrained_depth, "--no-pretrained-depth")
    add_flag(cmd, args.freeze_depth, "--freeze-depth")
    add_flag(cmd, args.balanced_sampler, "--balanced-sampler")

    if args.use_filter_mute_denoise:
        cmd.extend([
            "--use-filter-mute-denoise",
            "--filter-mute-highpass-hz",
            str(args.filter_mute_highpass_hz),
            "--filter-mute-notches-hz",
            args.filter_mute_notches_hz,
            "--filter-mute-threshold",
            str(args.filter_mute_threshold),
            "--filter-mute-window-sec",
            str(args.filter_mute_window_sec),
            "--filter-mute-floor",
            str(args.filter_mute_floor),
            "--filter-mute-edge-smooth-ms",
            str(args.filter_mute_edge_smooth_ms),
        ])

    if args.use_time_mask:
        cmd.extend([
            "--use-time-mask",
            "--time-mask-prob",
            str(args.time_mask_prob),
            "--time-mask-num",
            str(args.time_mask_num),
            "--time-mask-max-width",
            str(args.time_mask_max_width),
            "--time-mask-fill",
            args.time_mask_fill,
        ])

    if args.use_spectral_denoise:
        cmd.append("--use-denoise")
        for noise_path in args.denoise_stationary_noise or []:
            cmd.extend(["--denoise-stationary-noise", noise_path])
        for noise_path in args.denoise_motion_noise or []:
            cmd.extend(["--denoise-motion-noise", noise_path])
        cmd.extend([
            "--denoise-highpass-hz",
            str(args.denoise_highpass_hz),
            "--denoise-notches-hz",
            args.denoise_notches_hz,
            "--denoise-spectral-strength",
            str(args.denoise_spectral_strength),
            "--denoise-gain-floor",
            str(args.denoise_gain_floor),
            "--denoise-motion-strength",
            str(args.denoise_motion_strength),
            "--denoise-motion-gain-floor",
            str(args.denoise_motion_gain_floor),
        ])

    print("Generated command:")
    print(" ".join(cmd))
    if args.dry_run:
        return

    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")
    subprocess.run(cmd, cwd=REPO_ROOT, check=True, env=env)


if __name__ == "__main__":
    main()

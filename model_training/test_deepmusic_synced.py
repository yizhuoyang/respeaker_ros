#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_processing.utlis import generate_music_gt
from dataloader.data_loader_new import SyncedDeepMusicDataset
from network.audionet.DeepMusic_auto import DeepMusic_plus


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    audio_channels = parse_int_tuple(args.audio_channels)

    dataset = SyncedDeepMusicDataset(
        root=args.data_root,
        split=args.split,
        odom_name=args.odom,
        object_names=args.object_name,
        audio_channels=audio_channels,
        mic_geometry=args.mic_geometry,
        mic_radius=args.mic_radius,
        mic_rotation_deg=args.mic_rotation_deg,
        mic_channel_order=args.mic_channel_order,
        geometry_aug=False,
        noise_aug=False,
        time_mask=False,
        use_filter_mute=args.use_filter_mute_denoise,
        filter_mute_threshold=args.filter_mute_threshold,
        filter_mute_window_sec=args.filter_mute_window_sec,
        filter_mute_floor=args.filter_mute_floor,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    model = DeepMusic_plus(
        N=len(audio_channels),
        T=64,
        M=1,
        device=device,
        attention=not args.no_attention,
        input_channel=len(audio_channels) * 2,
    ).to(device)
    load_checkpoint(model, args.checkpoint, device)
    model.eval()

    rows = []
    total_loss = 0.0
    all_soft_errors = []
    all_peak_errors = []
    mse_loss = torch.nn.MSELoss()

    with torch.no_grad():
        sample_offset = 0
        for spectrogram, target_deg, steering_vector, correlation in loader:
            spectrogram = spectrogram.to(device).float()
            target_deg = target_deg.to(device).float()
            steering_vector = steering_vector.to(device)
            correlation = correlation.to(device)

            pred_soft_deg, pred_spectrum = model(spectrogram, steering_vector, correlation)
            gt_spectrum = generate_music_gt(target_deg, sigma=args.gt_sigma)
            loss = mse_loss(pred_spectrum, gt_spectrum)
            total_loss += loss.item() * spectrogram.shape[0]

            pred_peak_deg = torch.argmax(pred_spectrum, dim=1).float()
            soft_error = circular_abs_error_deg(pred_soft_deg.view(-1), target_deg.view(-1))
            peak_error = circular_abs_error_deg(pred_peak_deg, target_deg.view(-1))
            all_soft_errors.append(soft_error.cpu())
            all_peak_errors.append(peak_error.cpu())

            for batch_i in range(spectrogram.shape[0]):
                item = dataset.samples[sample_offset + batch_i]
                row = {
                    "sequence": item["sequence"],
                    "sample_id": item["sample_id"],
                    "target_deg": float(target_deg[batch_i].item()),
                    "pred_soft_deg": float(pred_soft_deg[batch_i].item()),
                    "pred_peak_deg": float(pred_peak_deg[batch_i].item()),
                    "soft_abs_error_deg": float(soft_error[batch_i].item()),
                    "peak_abs_error_deg": float(peak_error[batch_i].item()),
                }
                if args.print_samples:
                    print_sample_result(sample_offset + batch_i, row)
                if args.save_csv:
                    rows.append(row)
            sample_offset += spectrogram.shape[0]

    soft_errors = torch.cat(all_soft_errors)
    peak_errors = torch.cat(all_peak_errors)
    avg_loss = total_loss / len(dataset)

    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Split: {args.split}, samples: {len(dataset)}")
    print(f"Spectrum MSE: {avg_loss:.6f}")
    print_metric("Soft-argmax angle error", soft_errors)
    print_metric("Peak-bin angle error", peak_errors)

    if args.save_csv:
        save_csv(args.save_csv, rows)
        print(f"Wrote CSV: {args.save_csv}")


def load_checkpoint(model, checkpoint, device):
    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.exists():
        raise RuntimeError(f"Checkpoint does not exist: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt.get("model", ckpt.get("state_dict", ckpt)) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state_dict, strict=False)


def circular_abs_error_deg(pred, target):
    diff = (pred - target + 180.0) % 360.0 - 180.0
    return torch.abs(diff)


def print_metric(name, values):
    print(
        f"{name}: mean={values.mean().item():.3f} deg, "
        f"median={values.median().item():.3f} deg, "
        f"p90={torch.quantile(values, 0.9).item():.3f} deg"
    )


def save_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate DeepMusic_plus on this repo's synced_dataset.")
    parser.add_argument("--data-root", default="synced_dataset")
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--object-name", default=None)
    parser.add_argument("--odom", default="lio_odom", choices=["lio_odom", "lio_robo_odom"])
    parser.add_argument("--checkpoint", default="weights/deepmusic_synced/best_model")
    parser.add_argument("--audio-channels", default="1,2,3,4")
    parser.add_argument("--mic-geometry", default="respeaker_v3", choices=["respeaker_v3", "circular"])
    parser.add_argument("--mic-radius", type=float, default=0.032)
    parser.add_argument("--mic-rotation-deg", type=float, default=0.0)
    parser.add_argument("--mic-channel-order", default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no-attention", action="store_true")
    parser.add_argument("--gt-sigma", type=float, default=10.0)
    parser.add_argument("--use-filter-mute-denoise", action="store_true")
    parser.add_argument("--filter-mute-threshold", type=float, default=0.06)
    parser.add_argument("--filter-mute-window-sec", type=float, default=0.05)
    parser.add_argument("--filter-mute-floor", type=float, default=0.02)
    parser.add_argument("--print-samples", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-csv", default=None)
    return parser.parse_args()


def parse_int_tuple(value):
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def print_sample_result(index, row):
    print(
        f"[{index:06d}] seq={row['sequence']} sample={row['sample_id']} "
        f"gt={row['target_deg']:.2f} "
        f"pred_soft={row['pred_soft_deg']:.2f} err_soft={row['soft_abs_error_deg']:.2f} "
        f"pred_peak={row['pred_peak_deg']:.2f} err_peak={row['peak_abs_error_deg']:.2f}"
    )


if __name__ == "__main__":
    main()

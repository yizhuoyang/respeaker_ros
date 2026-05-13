#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataloader.data_loader_new import SyncedDeepMusicDataset
from model_training import ModelTrainer
from network.audionet.DeepMusic_auto import DeepMusic_plus


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    audio_channels = parse_int_tuple(args.audio_channels)

    train_dataset = SyncedDeepMusicDataset(
        root=args.data_root,
        split="train",
        odom_name=args.odom,
        object_names=args.object_name,
        audio_channels=audio_channels,
        min_freq_hz=args.min_freq_hz,
        max_audio_abs=args.max_audio_abs,
        min_distance=args.min_distance,
        max_distance=args.max_distance,
        mic_geometry=args.mic_geometry,
        mic_radius=args.mic_radius,
        mic_rotation_deg=args.mic_rotation_deg,
        mic_channel_order=args.mic_channel_order,
        geometry_aug=args.geometry_aug,
        geometry_aug_step_deg=args.geometry_aug_step_deg,
        noise_aug=args.noise_aug,
        snr_min_db=args.snr_min_db,
        snr_max_db=args.snr_max_db,
        time_mask=args.time_mask,
        time_mask_prob=args.time_mask_prob,
        time_mask_max_width=args.time_mask_max_width,
        use_filter_mute=args.use_filter_mute_denoise,
        filter_mute_threshold=args.filter_mute_threshold,
        filter_mute_window_sec=args.filter_mute_window_sec,
        filter_mute_floor=args.filter_mute_floor,
    )
    val_dataset = SyncedDeepMusicDataset(
        root=args.data_root,
        split="test",
        odom_name=args.odom,
        object_names=args.object_name,
        audio_channels=audio_channels,
        min_freq_hz=args.min_freq_hz,
        max_audio_abs=args.max_audio_abs,
        min_distance=args.min_distance,
        max_distance=args.max_distance,
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

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    input_channel = len(audio_channels) * 2
    model = DeepMusic_plus(
        N=len(audio_channels),
        T=64,
        M=1,
        device=device,
        attention=not args.no_attention,
        input_channel=input_channel,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.05,
    )

    print(f"Device: {device}")
    print(f"Train samples: {len(train_dataset)}, Test samples: {len(val_dataset)}")
    print(f"Input channels: {input_channel}, audio_channels={audio_channels}")
    print(f"Save dir: {args.save_dir}")

    trainer = ModelTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=torch.nn.MSELoss(),
        optimizer=optimizer,
        epoch=args.epochs,
        model_path=args.save_dir,
        device=device,
        lr_scheduler=scheduler,
        save_best=True,
    )
    trainer.train()


def parse_args():
    parser = argparse.ArgumentParser(description="Train DeepMusic_plus on this repo's synced_dataset.")
    parser.add_argument("--data-root", default="synced_dataset")
    parser.add_argument("--object-name", default=None, help="Optional object filter, e.g. clock,dryer,person,guitar.")
    parser.add_argument("--odom", default="lio_odom", choices=["lio_odom", "lio_robo_odom"])
    parser.add_argument("--audio-channels", default="1,2,3,4")
    parser.add_argument("--min-freq-hz", type=float, default=2000.0, help="Use only STFT bins at or above this frequency.")
    parser.add_argument("--max-audio-abs", type=float, default=0.06, help="Skip samples whose selected-channel max abs amplitude is above this value. Use <=0 to disable.")
    parser.add_argument("--min-distance", type=float, default=None, help="Only load samples with distance_xy >= this value.")
    parser.add_argument("--max-distance", type=float, default=None, help="Only load samples with distance_xy <= this value.")
    parser.add_argument("--mic-geometry", default="respeaker_v3", choices=["respeaker_v3", "circular"])
    parser.add_argument(
        "--mic-radius",
        type=float,
        default=0.032,
        help=(
            "Circular mic radius in meters. ReSpeaker Mic Array v3.0 board diameter is 70mm; "
            "0.032m is a practical mic-center estimate and can be calibrated."
        ),
    )
    parser.add_argument("--mic-rotation-deg", type=float, default=0.0, help="Rotate mic geometry relative to robot/front frame.")
    parser.add_argument("--mic-channel-order", default=None, help="Optional permutation of selected mic channels, e.g. 0,1,2,3 or 0,3,2,1.")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--save-dir", default="weights/deepmusic_synced")
    parser.add_argument("--no-attention", action="store_true")
    parser.add_argument("--geometry-aug", action="store_true", help="Rotate array steering vectors and labels during training.")
    parser.add_argument("--geometry-aug-step-deg", type=float, default=1.0)
    parser.add_argument("--noise-aug", action="store_true")
    parser.add_argument("--snr-min-db", type=float, default=0.0)
    parser.add_argument("--snr-max-db", type=float, default=25.0)
    parser.add_argument("--time-mask", action="store_true")
    parser.add_argument("--time-mask-prob", type=float, default=0.5)
    parser.add_argument("--time-mask-max-width", type=int, default=8)
    parser.add_argument("--use-filter-mute-denoise", action="store_true")
    parser.add_argument("--filter-mute-threshold", type=float, default=0.06)
    parser.add_argument("--filter-mute-window-sec", type=float, default=0.05)
    parser.add_argument("--filter-mute-floor", type=float, default=0.02)
    return parser.parse_args()


def parse_int_tuple(value):
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


if __name__ == "__main__":
    main()

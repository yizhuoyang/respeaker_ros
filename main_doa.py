import argparse
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler, random_split
from torch.utils.tensorboard import SummaryWriter

from dataloader.ssl_dataset import SingleStepDataset
from network.audionet.ssl_net import SSLNet_DOA, SSLNet_depth_DOA
from model_training.train_doa import train_one_epoch, validate


def parse_args():
    parser = argparse.ArgumentParser(description="Train DOA + distance model on synchronized ROS2 dataset.")
    parser.add_argument("--data-root", default="synced_dataset", help="Dataset root or a single synchronized dataset directory.")
    parser.add_argument("--train-root", default=None, help="Optional train dataset root. Overrides --data-root split.")
    parser.add_argument("--val-root", default=None, help="Optional val dataset root. Overrides --data-root split.")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation ratio when using --data-root only.")
    parser.add_argument("--odom", default="lio_odom", choices=["lio_odom", "lio_robo_odom"])
    parser.add_argument("--model", default="audio_depth", choices=["audio", "audio_depth"])
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--log-dir", default="runs/ssl_doa_distance_synced")
    parser.add_argument("--save-dir", default="weights/ssl_doa_distance_synced")
    parser.add_argument("--checkpoint", default=None, help="Optional checkpoint to initialize from.")
    parser.add_argument("--audio-feat", default="ipd", choices=["ipd", "spec", "phase", "both", "gcc_phat_complex"])
    parser.add_argument("--audio-channels", default="0,1,2,3", help="Comma-separated wav channels to use.")
    parser.add_argument("--ipd-pairs", default="0-1,0-2,0-3,1-2,1-3,2-3")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--use-compress", action="store_true")
    parser.add_argument(
        "--allow-missing-depth",
        action="store_true",
        help="Allow samples without depth images. Useful for audio-only training.",
    )
    parser.add_argument("--no-pretrained-depth", action="store_true", help="Avoid downloading/using ImageNet depth backbone weights.")
    parser.add_argument("--freeze-depth", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--balanced-sampler", action="store_true", help="Balance training samples by DOA angle bins.")
    return parser.parse_args()


def parse_int_tuple(value):
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def infer_audio_in_channels(audio_feat, audio_channels, ipd_pairs):
    if audio_feat == "spec":
        return len(audio_channels)
    if audio_feat == "phase":
        return len(audio_channels) * 2
    if audio_feat in ("ipd", "gcc_phat_complex"):
        return len(ipd_pairs) * 2
    if audio_feat == "both":
        return len(audio_channels) * 2 + len(ipd_pairs) * 2
    raise ValueError(audio_feat)


def build_dataset(root, args):
    return SingleStepDataset(
        root_dir=root,
        odom_name=args.odom,
        use_compress=args.use_compress,
        audio_feat=args.audio_feat,
        audio_channels=parse_int_tuple(args.audio_channels),
        ipd_pairs=args.ipd_pairs,
        image_size=(args.image_size, args.image_size),
        require_depth=not args.allow_missing_depth,
    )


def build_loaders(args):
    if args.train_root and args.val_root:
        train_dataset = build_dataset(args.train_root, args)
        val_dataset = build_dataset(args.val_root, args)
    else:
        full_dataset = build_dataset(args.data_root, args)
        val_size = max(1, int(len(full_dataset) * args.val_ratio))
        train_size = len(full_dataset) - val_size
        if train_size <= 0:
            raise RuntimeError("Dataset is too small for the requested val split")
        generator = torch.Generator().manual_seed(args.seed)
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)

    sampler = make_balanced_sampler(train_dataset) if args.balanced_sampler else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
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
    )
    return train_dataset, val_dataset, train_loader, val_loader


def make_balanced_sampler(dataset, num_bins=12):
    yaws = []
    for index in range(len(dataset)):
        if hasattr(dataset, "dataset"):
            yaw = dataset.dataset.get_yaw_deg(dataset.indices[index])
        else:
            yaw = dataset.get_yaw_deg(index)
        yaws.append((yaw + 180.0) % 360.0)

    bin_ids = torch.clamp((torch.tensor(yaws) / (360.0 / num_bins)).long(), 0, num_bins - 1)
    counts = torch.bincount(bin_ids, minlength=num_bins).float()
    weights = 1.0 / counts[bin_ids].clamp_min(1.0)
    return WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)


def build_model(args, audio_in_channels):
    if args.model == "audio":
        return SSLNet_DOA(
            use_compress=args.use_compress,
            audio_in_channels=audio_in_channels,
        )
    return SSLNet_depth_DOA(
        use_compress=args.use_compress,
        audio_in_channels=audio_in_channels,
        pretrained_depth_encoder=not args.no_pretrained_depth,
        freeze_depth_encoder=args.freeze_depth,
        drop_depth_prob=0.1,
    )


def load_checkpoint_if_needed(model, checkpoint, device):
    if not checkpoint:
        return
    if not os.path.exists(checkpoint):
        print(f"[WARN] checkpoint not found: {checkpoint}")
        return
    ckpt = torch.load(checkpoint, map_location=device)
    state_dict = ckpt.get("model", ckpt.get("state_dict", ckpt)) if isinstance(ckpt, dict) else ckpt
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"[INFO] loaded checkpoint: {checkpoint}")
    if missing:
        print(f"[WARN] missing keys: {len(missing)}")
    if unexpected:
        print(f"[WARN] unexpected keys: {len(unexpected)}")


def main():
    args = parse_args()
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    print(f"Using device: {device}")

    audio_channels = parse_int_tuple(args.audio_channels)
    from dataloader.utils import parse_channel_pairs
    ipd_pairs = parse_channel_pairs(args.ipd_pairs)
    audio_in_channels = infer_audio_in_channels(args.audio_feat, audio_channels, ipd_pairs)
    print(f"Audio feature: {args.audio_feat}, input channels: {audio_in_channels}")

    train_dataset, val_dataset, train_loader, val_loader = build_loaders(args)
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    model = build_model(args, audio_in_channels).to(device)
    load_checkpoint_if_needed(model, args.checkpoint, device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.05)
    writer = SummaryWriter(args.log_dir)

    best_val_loss = float("inf")
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs} | lr={optimizer.param_groups[0]['lr']:.3e}")
        train_loss, global_step = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            writer=writer,
            global_step=global_step,
        )
        val_loss = validate(
            model=model,
            val_loader=val_loader,
            device=device,
            epoch=epoch,
            writer=writer,
        )
        # scheduler.step()
        print(f"Train loss: {train_loss:.6f} | Val loss: {val_loss:.6f}")

        state = {
            "model": model.state_dict(),
            "epoch": epoch,
            "args": vars(args),
            "audio_in_channels": audio_in_channels,
        }
        torch.save(state, Path(args.save_dir) / "last_model.pth")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(state, Path(args.save_dir) / "best_model.pth")
            print(f"Saved best model, val_loss={best_val_loss:.6f}")

    writer.close()
    print("Training finished.")


if __name__ == "__main__":
    main()

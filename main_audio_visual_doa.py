import argparse
import math
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter

from dataloader import AudioVisualDoaDataset
from dataloader.audio_visual_doa_dataset import parse_channel_pairs
from model_training.train_audio_visual_doa import train_one_epoch, validate
from network.audionet import AudioVisualDOANet


def parse_args():
    parser = argparse.ArgumentParser(description="Train audio/visual DOA + distance model.")
    parser.add_argument("--data-root", default="data", help="Dataset root or a single bag dataset directory.")
    parser.add_argument("--train-root", default=None, help="Optional train dataset root. Overrides --data-root split.")
    parser.add_argument("--val-root", default=None, help="Optional val dataset root. Overrides --data-root split.")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--model", default="audio_depth", choices=["audio", "audio_depth", "audio_image", "audio_image_depth"])
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--log-dir", default="runs/audio_visual_doa")
    parser.add_argument("--save-dir", default="weights/audio_visual_doa")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--audio-feat", default="ipd", choices=["ipd", "spec", "phase", "both", "gcc_phat_complex"])
    parser.add_argument("--audio-channels", default="1234", help="WAV channels to use, e.g. 1234 or 1,2,3,4.")
    parser.add_argument("--ipd-pairs", default="0-1,0-2,0-3,1-2,1-3,2-3")
    parser.add_argument("--ipd-freq-min", type=float, default=None, help="Min frequency in Hz for IPD/phase features.")
    parser.add_argument("--ipd-freq-max", type=float, default=None, help="Max frequency in Hz for IPD/phase features.")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--depth-max-m", type=float, default=10.0, help="Clip invalid depth above this range and normalize depth to 0..1.")
    parser.add_argument("--sample-stride", type=int, default=1, help="Use every Nth synchronized sample in each sequence.")
    parser.add_argument("--doa-num-bins", type=int, default=360)
    parser.add_argument("--doa-sigma-deg", type=float, default=10.0)
    parser.add_argument("--distance-num-bins", type=int, default=120)
    parser.add_argument("--distance-min", type=float, default=0.0)
    parser.add_argument("--distance-max", type=float, default=6.0)
    parser.add_argument("--distance-sigma", type=float, default=0.15)
    parser.add_argument("--min-distance", type=float, default=0.3, help="Drop samples whose source distance is below this value.")
    parser.add_argument("--distance-weight", "--dist-loss-weight", dest="distance_weight", type=float, default=0.5)
    parser.add_argument("--dropout", type=float, default=0.0, help="Model dropout. Use 0.0 for overfit sanity checks.")
    parser.add_argument("--audio-feat-dim", type=int, default=256)
    parser.add_argument("--depth-feat-dim", type=int, default=64)
    parser.add_argument("--fusion-dim", type=int, default=256)
    parser.add_argument("--drop-depth-prob", type=float, default=0.1)
    parser.add_argument("--freeze-depth-encoder", action="store_true")
    parser.add_argument("--spec-augment", action="store_true", help="Apply frequency/time masks on training spectrogram features.")
    parser.add_argument("--freq-mask-param", type=int, default=8)
    parser.add_argument("--time-mask-param", type=int, default=12)
    parser.add_argument("--num-freq-masks", type=int, default=1)
    parser.add_argument("--num-time-masks", type=int, default=1)
    parser.add_argument("--feature-noise-std", type=float, default=0.0, help="Gaussian noise std added to training audio features.")
    parser.add_argument("--allow-missing-image", action="store_true")
    parser.add_argument("--allow-missing-depth", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--balanced-sampler", action="store_true")
    return parser.parse_args()


def parse_audio_channels(value):
    text = value.strip()
    if "," in text:
        return tuple(int(item.strip()) for item in text.split(",") if item.strip())
    return tuple(int(item) for item in text)


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


class SpectrogramAugment:
    def __init__(
        self,
        freq_mask_param=8,
        time_mask_param=12,
        num_freq_masks=1,
        num_time_masks=1,
        noise_std=0.0,
    ):
        self.freq_mask_param = max(int(freq_mask_param), 0)
        self.time_mask_param = max(int(time_mask_param), 0)
        self.num_freq_masks = max(int(num_freq_masks), 0)
        self.num_time_masks = max(int(num_time_masks), 0)
        self.noise_std = max(float(noise_std), 0.0)

    def __call__(self, sample):
        spectrogram = sample.get("spectrogram")
        if spectrogram is None:
            return sample
        spectrogram = spectrogram.clone()
        _, num_freq, num_time = spectrogram.shape
        for _ in range(self.num_freq_masks):
            self._mask_axis(spectrogram, axis=1, axis_size=num_freq, max_width=self.freq_mask_param)
        for _ in range(self.num_time_masks):
            self._mask_axis(spectrogram, axis=2, axis_size=num_time, max_width=self.time_mask_param)
        if self.noise_std > 0.0:
            spectrogram = spectrogram + torch.randn_like(spectrogram) * self.noise_std
        sample["spectrogram"] = spectrogram
        return sample

    @staticmethod
    def _mask_axis(tensor, axis, axis_size, max_width):
        if max_width <= 0 or axis_size <= 1:
            return
        width = int(torch.randint(0, min(max_width, axis_size) + 1, (1,)).item())
        if width <= 0:
            return
        start = int(torch.randint(0, axis_size - width + 1, (1,)).item())
        slices = [slice(None)] * tensor.ndim
        slices[axis] = slice(start, start + width)
        tensor[tuple(slices)] = 0.0


def make_train_transform(args):
    if not args.spec_augment and args.feature_noise_std <= 0.0:
        return None
    return SpectrogramAugment(
        freq_mask_param=args.freq_mask_param if args.spec_augment else 0,
        time_mask_param=args.time_mask_param if args.spec_augment else 0,
        num_freq_masks=args.num_freq_masks if args.spec_augment else 0,
        num_time_masks=args.num_time_masks if args.spec_augment else 0,
        noise_std=args.feature_noise_std,
    )


def build_dataset(root, args, transform=None):
    return AudioVisualDoaDataset(
        root_dir=root,
        transform=transform,
        audio_channels=args.audio_channels,
        image_size=(args.image_size, args.image_size),
        require_image=not args.allow_missing_image,
        require_depth=not args.allow_missing_depth,
        audio_feat=args.audio_feat,
        ipd_pairs=args.ipd_pairs,
        ipd_freq_min=args.ipd_freq_min,
        ipd_freq_max=args.ipd_freq_max,
        depth_max_m=args.depth_max_m,
        doa_num_bins=args.doa_num_bins,
        doa_sigma_deg=args.doa_sigma_deg,
        distance_num_bins=args.distance_num_bins,
        distance_min=args.distance_min,
        distance_max=args.distance_max,
        distance_sigma=args.distance_sigma,
        min_distance=args.min_distance,
        sample_stride=args.sample_stride,
    )


def build_loaders(args):
    if args.train_root and args.val_root:
        train_dataset = build_dataset(args.train_root, args, transform=make_train_transform(args))
        val_dataset = build_dataset(args.val_root, args)
    else:
        full_train_dataset = build_dataset(args.data_root, args, transform=make_train_transform(args))
        full_val_dataset = build_dataset(args.data_root, args)
        dataset_size = len(full_val_dataset)
        val_size = max(1, int(dataset_size * args.val_ratio))
        train_size = dataset_size - val_size
        if train_size <= 0:
            raise RuntimeError("Dataset is too small for the requested val split")
        generator = torch.Generator().manual_seed(args.seed)
        indices = torch.randperm(dataset_size, generator=generator).tolist()
        train_dataset = Subset(full_train_dataset, indices[:train_size])
        val_dataset = Subset(full_val_dataset, indices[train_size:])

    sampler = make_balanced_sampler(train_dataset, args.doa_num_bins) if args.balanced_sampler else None
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


def make_balanced_sampler(dataset, num_bins):
    bin_ids = []
    for index in range(len(dataset)):
        if hasattr(dataset, "dataset"):
            azimuth = dataset.dataset.get_azimuth_rad(dataset.indices[index])
        else:
            azimuth = dataset.get_azimuth_rad(index)
        angle = (azimuth + math.pi) % (2.0 * math.pi)
        bin_id = min(int(angle / (2.0 * math.pi) * num_bins), num_bins - 1)
        bin_ids.append(bin_id)

    bin_ids = torch.tensor(bin_ids, dtype=torch.long)
    counts = torch.bincount(bin_ids, minlength=num_bins).float()
    weights = 1.0 / counts[bin_ids].clamp_min(1.0)
    return WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)


def build_model(args, audio_in_channels):
    return AudioVisualDOANet(
        audio_in_channels=audio_in_channels,
        model_type=args.model,
        audio_feat_dim=args.audio_feat_dim,
        depth_feat_dim=args.depth_feat_dim,
        fusion_dim=args.fusion_dim,
        num_doa_bins=args.doa_num_bins,
        num_distance_bins=args.distance_num_bins,
        dropout=args.dropout,
        freeze_depth_encoder=args.freeze_depth_encoder,
        drop_depth_prob=args.drop_depth_prob,
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

    audio_channels = parse_audio_channels(args.audio_channels)
    ipd_pairs = parse_channel_pairs(args.ipd_pairs)
    audio_in_channels = infer_audio_in_channels(args.audio_feat, audio_channels, ipd_pairs)
    print(f"Audio feature: {args.audio_feat}, model audio input channels: {audio_in_channels}")

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
        print(f"\\nEpoch {epoch}/{args.epochs} | lr={optimizer.param_groups[0]['lr']:.3e}")
        train_metrics, global_step = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            writer=writer,
            global_step=global_step,
            distance_weight=args.distance_weight,
        )
        val_metrics = validate(
            model=model,
            val_loader=val_loader,
            device=device,
            epoch=epoch,
            writer=writer,
            distance_weight=args.distance_weight,
        )
        scheduler.step()

        print(
            f"Train loss: {train_metrics['loss']:.6f} | "
            f"Train doa bin err: {train_metrics['doa_bin_error']:.2f} | "
            f"Train dist bin err: {train_metrics['distance_bin_error']:.2f} | "
            f"Val loss: {val_metrics['loss']:.6f} | "
            f"Val doa bin err: {val_metrics['doa_bin_error']:.2f} | "
            f"Val dist bin err: {val_metrics['distance_bin_error']:.2f}"
        )

        state = {
            "model": model.state_dict(),
            "epoch": epoch,
            "args": vars(args),
            "audio_in_channels": audio_in_channels,
        }
        torch.save(state, Path(args.save_dir) / "last_model.pth")
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save(state, Path(args.save_dir) / "best_model.pth")
            print(f"Saved best model, val_loss={best_val_loss:.6f}")

    writer.close()
    print("Training finished.")


if __name__ == "__main__":
    main()

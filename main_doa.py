import argparse
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler, random_split
from torch.utils.tensorboard import SummaryWriter

from dataloader.ssl_dataset import CLASS_NAMES, SingleStepDataset, object_prefix
from network.audionet.ssl_net import SSLNet_DOA, SSLNet_depth_DOA
from model_training.train_doa import train_one_epoch, validate


def parse_args():
    parser = argparse.ArgumentParser(description="Train DOA + distance model on synced odom or pairs_ros1 labelCloud data.")
    parser.add_argument(
        "--data-root",
        default="synced_dataset",
        help="Dataset root or sequence directory; supports synced_dataset and pairs_ros1 train/test layouts.",
    )
    parser.add_argument("--train-root", default=None, help="Optional train dataset root. Overrides --data-root split.")
    parser.add_argument("--val-root", default=None, help="Optional val dataset root. Overrides --data-root split.")
    parser.add_argument(
        "--objects",
        "--object-name",
        dest="object_name",
        default=None,
        help=(
            "Optional comma-separated object category filter, e.g. chair,person,guitar. "
            "It matches folders with the same prefix, such as chair31/person11."
        ),
    )
    parser.add_argument(
        "--val-clocks",
        default=None,
        help="Comma-separated clock folder names to hold out for validation/testing, e.g. clock2.",
    )
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation ratio when using --data-root only.")
    parser.add_argument("--odom", default="lio_odom", choices=["lio_odom", "lio_robo_odom"])
    parser.add_argument("--model", default="audio_depth", choices=["audio", "audio_depth"])
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument(
        "--lr-scheduler",
        default="cosine",
        choices=["cosine", "step", "none"],
        help="Learning-rate decay strategy. cosine decays smoothly across all epochs.",
    )
    parser.add_argument(
        "--min-lr-ratio",
        type=float,
        default=0.05,
        help="Minimum LR as a fraction of --lr for cosine decay.",
    )
    parser.add_argument("--lr-step-size", type=int, default=20, help="Epoch interval for step LR decay.")
    parser.add_argument("--lr-gamma", type=float, default=0.5, help="Multiplicative LR decay for --lr-scheduler step.")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--log-dir", default="runs/ssl_doa_distance_synced")
    parser.add_argument("--save-dir", default="weights/ssl_doa_distance_synced")
    parser.add_argument("--checkpoint", default=None, help="Optional checkpoint to initialize from.")
    parser.add_argument("--audio-feat", default="ipd", choices=["ipd", "spec", "phase", "both", "gcc_phat_complex"])
    parser.add_argument("--audio-channels", default="1,2,3,4", help="Comma-separated wav channels to use.")
    parser.add_argument("--ipd-pairs", default="0-1,0-2,0-3,1-2,1-3,2-3")
    parser.add_argument("--audio-bandpass-low-hz", type=float, default=0.0, help="Apply a train/test audio bandpass before feature extraction.")
    parser.add_argument("--audio-bandpass-high-hz", type=float, default=0.0, help="Apply a train/test audio bandpass before feature extraction.")
    parser.add_argument("--min-distance", type=float, default=None, help="Only load samples with distance_xy >= this value.")
    parser.add_argument("--max-distance", type=float, default=None, help="Only load samples with distance_xy <= this value.")
    parser.add_argument("--use-classification", action="store_true", help="Enable 3-class robot_noise/moving_sound/signal_static branch.")
    parser.add_argument("--classification-only", action="store_true", help="Train only the 3-class classification branch.")
    parser.add_argument("--freeze-classifier", action="store_true", help="Freeze classifier/backbone modules loaded from --checkpoint.")
    parser.add_argument("--gate-doa-by-pred-class", action="store_true", help="Only compute DOA/distance loss when predicted class is signal_static.")
    parser.add_argument("--classification-weight", type=float, default=1.0)
    parser.add_argument("--distance-weight", type=float, default=0.5)
    parser.add_argument("--max-signal-abs", type=float, default=0.06, help="With --use-classification, skip signal samples with selected-channel max abs above this value. Use <=0 to disable.")
    parser.add_argument("--class-balanced-sampler", action="store_true", help="Balance training samples by classification label.")
    parser.add_argument("--no-class-loss-weights", action="store_true", help="Disable inverse-frequency classification loss weights.")
    parser.add_argument("--use-denoise", action="store_true", help="Enable default dataloader denoising.")
    parser.add_argument("--denoise-noise", action="append", default=None, help="Backward-compatible stationary noise wav override. Can be repeated.")
    parser.add_argument("--denoise-stationary-noise", action="append", default=None, help="Always-present noise wavs, e.g. robot/lidar noise. Can be repeated.")
    parser.add_argument("--denoise-motion-noise", action="append", default=None, help="Motion-only noise wavs used with frame gating. Can be repeated.")
    parser.add_argument("--denoise-highpass-hz", type=float, default=90.0)
    parser.add_argument("--denoise-notches-hz", default="48,180,342,1845")
    parser.add_argument("--denoise-spectral-strength", type=float, default=0.4)
    parser.add_argument("--denoise-gain-floor", type=float, default=0.5)
    parser.add_argument("--denoise-motion-strength", type=float, default=0.3)
    parser.add_argument("--denoise-motion-gain-floor", type=float, default=0.6)
    parser.add_argument("--denoise-motion-gate-threshold", type=float, default=1.6)
    parser.add_argument("--denoise-motion-gate-smooth-frames", type=int, default=5)
    parser.add_argument("--denoise-transient-attenuation", type=float, default=0.35)
    parser.add_argument("--denoise-transient-threshold", type=float, default=3.0)
    parser.add_argument("--denoise-transient-frame-ms", type=float, default=20.0)
    parser.add_argument("--denoise-transient-hop-ms", type=float, default=5.0)
    parser.add_argument("--denoise-transient-smooth-frames", type=int, default=5)
    parser.add_argument("--use-filter-mute-denoise", action="store_true", help="Use fixed robot filters plus threshold-based motion-impact muting.")
    parser.add_argument("--filter-mute-highpass-hz", type=float, default=120.0)
    parser.add_argument("--filter-mute-notches-hz", default="48.8,66.4,179.7,341.8,867.2,1271.5,1845.7,2533.2,3783.2")
    parser.add_argument("--filter-mute-threshold", type=float, default=0.06)
    parser.add_argument("--filter-mute-window-sec", type=float, default=0.05)
    parser.add_argument("--filter-mute-floor", type=float, default=0.02, help="0 means hard zero; small positive values keep STFT phase more stable.")
    parser.add_argument("--filter-mute-edge-smooth-ms", type=float, default=5.0)
    parser.add_argument("--use-time-mask", action="store_true", help="Enable train-only time masking on audio features.")
    parser.add_argument("--time-mask-prob", type=float, default=0.5)
    parser.add_argument("--time-mask-num", type=int, default=1)
    parser.add_argument("--time-mask-max-width", type=int, default=12, help="Maximum masked feature frames per mask.")
    parser.add_argument("--time-mask-fill", default="zero", choices=["zero", "mean"])
    parser.add_argument("--noise-aug", action="store_true", help="Enable train-only real noise mixing augmentation before feature extraction.")
    parser.add_argument( 
        "--noise-aug-root",
        default=None,
        help="Directory containing noise wav files. Defaults to <data-root>/noise when it exists.",
    )
    parser.add_argument(
        "--noise-aug-path",
        action="append",
        default=None,
        help="Specific noise wav file or directory. Can be repeated.",
    )
    parser.add_argument("--noise-aug-prob", type=float, default=1.0, help="Probability of applying noise augmentation to each training sample.")
    parser.add_argument("--snr-min-db", type=float, default=0.0, help="Minimum random SNR for noise augmentation.")
    parser.add_argument("--snr-max-db", type=float, default=25.0, help="Maximum random SNR for noise augmentation.")
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


def parse_name_set(value):
    if not value:
        return set()
    return {item.strip() for item in value.split(",") if item.strip()}


def format_object_filter(value):
    names = sorted(parse_name_set(value))
    return ",".join(names) if names else "all"


def get_stationary_noise_paths(args):
    if not args.use_denoise:
        return []
    if args.denoise_stationary_noise:
        return args.denoise_stationary_noise
    if args.denoise_noise:
        return args.denoise_noise
    return ["audio_data/noise/robot_noise.wav"]


def get_motion_noise_paths(args):
    if not args.use_denoise:
        return []
    if args.denoise_motion_noise:
        return args.denoise_motion_noise
    return ["audio_data/noise/moving_sound.wav"]


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


def build_dataset(root, args, is_train=False):
    return SingleStepDataset(
        root_dir=root,
        odom_name=args.odom,
        use_compress=args.use_compress,
        audio_feat=args.audio_feat,
        audio_channels=parse_int_tuple(args.audio_channels),
        ipd_pairs=args.ipd_pairs,
        image_size=(args.image_size, args.image_size),
        require_depth=not args.allow_missing_depth,
        denoise_stationary_noise_paths=get_stationary_noise_paths(args),
        denoise_motion_noise_paths=get_motion_noise_paths(args),
        denoise_highpass_hz=args.denoise_highpass_hz if args.use_denoise else 0.0,
        denoise_notches_hz=args.denoise_notches_hz if args.use_denoise else "",
        denoise_spectral_strength=args.denoise_spectral_strength if args.use_denoise else 0.0,
        denoise_gain_floor=args.denoise_gain_floor,
        denoise_motion_strength=args.denoise_motion_strength if args.use_denoise else 0.0,
        denoise_motion_gain_floor=args.denoise_motion_gain_floor,
        denoise_motion_gate_threshold=args.denoise_motion_gate_threshold,
        denoise_motion_gate_smooth_frames=args.denoise_motion_gate_smooth_frames,
        denoise_transient_attenuation=args.denoise_transient_attenuation if args.use_denoise else 0.0,
        denoise_transient_threshold=args.denoise_transient_threshold,
        denoise_transient_frame_ms=args.denoise_transient_frame_ms,
        denoise_transient_hop_ms=args.denoise_transient_hop_ms,
        denoise_transient_smooth_frames=args.denoise_transient_smooth_frames,
        filter_mute_enabled=args.use_filter_mute_denoise,
        filter_mute_highpass_hz=args.filter_mute_highpass_hz,
        filter_mute_notches_hz=args.filter_mute_notches_hz,
        filter_mute_threshold=args.filter_mute_threshold,
        filter_mute_window_sec=args.filter_mute_window_sec,
        filter_mute_floor=args.filter_mute_floor,
        filter_mute_edge_smooth_ms=args.filter_mute_edge_smooth_ms,
        time_mask_enabled=is_train and args.use_time_mask,
        time_mask_prob=args.time_mask_prob,
        time_mask_num=args.time_mask_num,
        time_mask_max_width=args.time_mask_max_width,
        time_mask_fill=args.time_mask_fill,
        noise_aug_enabled=is_train and args.noise_aug,
        noise_aug_paths=args.noise_aug_path,
        noise_aug_root=args.noise_aug_root,
        noise_aug_prob=args.noise_aug_prob,
        noise_aug_snr_min_db=args.snr_min_db,
        noise_aug_snr_max_db=args.snr_max_db,
        object_names=args.object_name,
        min_distance=args.min_distance,
        max_distance=args.max_distance,
        use_classification=args.use_classification,
        max_signal_abs=args.max_signal_abs if args.use_classification else None,
        audio_bandpass_low_hz=args.audio_bandpass_low_hz,
        audio_bandpass_high_hz=args.audio_bandpass_high_hz,
    )


def build_loaders(args):
    if args.train_root and args.val_root:
        train_dataset = build_dataset(args.train_root, args, is_train=True)
        val_dataset = build_dataset(args.val_root, args, is_train=False)
    elif has_explicit_train_test_split(args.data_root):
        train_dataset = build_dataset_with_split(args.data_root, args, split="train", is_train=True)
        val_dataset = build_dataset_with_split(args.data_root, args, split="test", is_train=False)
    else:
        full_dataset = build_dataset(args.data_root, args, is_train=False)
        has_train_only_aug = args.use_time_mask or args.noise_aug
        train_source = build_dataset(args.data_root, args, is_train=True) if has_train_only_aug else full_dataset
        if args.val_clocks:
            train_indices, val_indices = split_indices_by_clocks(full_dataset, args.val_clocks)
            train_dataset = Subset(train_source, train_indices)
            val_dataset = Subset(full_dataset, val_indices)
        else:
            val_size = max(1, int(len(full_dataset) * args.val_ratio))
            train_size = len(full_dataset) - val_size
            if train_size <= 0:
                raise RuntimeError("Dataset is too small for the requested val split")
            generator = torch.Generator().manual_seed(args.seed)
            indices = torch.randperm(len(full_dataset), generator=generator).tolist()
            train_indices = indices[:train_size]
            val_indices = indices[train_size:]
            train_dataset = Subset(train_source, train_indices)
            val_dataset = Subset(full_dataset, val_indices)

    if args.use_classification and args.class_balanced_sampler:
        sampler = make_class_balanced_sampler(train_dataset)
    else:
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


def has_explicit_train_test_split(data_root):
    root = Path(data_root)
    return (root / "train").is_dir() and (root / "test").is_dir()


def build_dataset_with_split(root, args, split, is_train=False):
    return SingleStepDataset(
        root_dir=root,
        split=split,
        odom_name=args.odom,
        use_compress=args.use_compress,
        audio_feat=args.audio_feat,
        audio_channels=parse_int_tuple(args.audio_channels),
        ipd_pairs=args.ipd_pairs,
        image_size=(args.image_size, args.image_size),
        require_depth=not args.allow_missing_depth,
        denoise_stationary_noise_paths=get_stationary_noise_paths(args),
        denoise_motion_noise_paths=get_motion_noise_paths(args),
        denoise_highpass_hz=args.denoise_highpass_hz if args.use_denoise else 0.0,
        denoise_notches_hz=args.denoise_notches_hz if args.use_denoise else "",
        denoise_spectral_strength=args.denoise_spectral_strength if args.use_denoise else 0.0,
        denoise_gain_floor=args.denoise_gain_floor,
        denoise_motion_strength=args.denoise_motion_strength if args.use_denoise else 0.0,
        denoise_motion_gain_floor=args.denoise_motion_gain_floor,
        denoise_motion_gate_threshold=args.denoise_motion_gate_threshold,
        denoise_motion_gate_smooth_frames=args.denoise_motion_gate_smooth_frames,
        denoise_transient_attenuation=args.denoise_transient_attenuation if args.use_denoise else 0.0,
        denoise_transient_threshold=args.denoise_transient_threshold,
        denoise_transient_frame_ms=args.denoise_transient_frame_ms,
        denoise_transient_hop_ms=args.denoise_transient_hop_ms,
        denoise_transient_smooth_frames=args.denoise_transient_smooth_frames,
        filter_mute_enabled=args.use_filter_mute_denoise,
        filter_mute_highpass_hz=args.filter_mute_highpass_hz,
        filter_mute_notches_hz=args.filter_mute_notches_hz,
        filter_mute_threshold=args.filter_mute_threshold,
        filter_mute_window_sec=args.filter_mute_window_sec,
        filter_mute_floor=args.filter_mute_floor,
        filter_mute_edge_smooth_ms=args.filter_mute_edge_smooth_ms,
        time_mask_enabled=is_train and args.use_time_mask,
        time_mask_prob=args.time_mask_prob,
        time_mask_num=args.time_mask_num,
        time_mask_max_width=args.time_mask_max_width,
        time_mask_fill=args.time_mask_fill,
        noise_aug_enabled=is_train and args.noise_aug,
        noise_aug_paths=args.noise_aug_path,
        noise_aug_root=args.noise_aug_root,
        noise_aug_prob=args.noise_aug_prob,
        noise_aug_snr_min_db=args.snr_min_db,
        noise_aug_snr_max_db=args.snr_max_db,
        object_names=args.object_name,
        min_distance=args.min_distance,
        max_distance=args.max_distance,
        use_classification=args.use_classification,
        max_signal_abs=args.max_signal_abs if args.use_classification else None,
        audio_bandpass_low_hz=args.audio_bandpass_low_hz,
        audio_bandpass_high_hz=args.audio_bandpass_high_hz,
    )


def split_dataset_by_clocks(dataset, val_clocks):
    train_indices, val_indices = split_indices_by_clocks(dataset, val_clocks)
    return Subset(dataset, train_indices), Subset(dataset, val_indices)


def split_indices_by_clocks(dataset, val_clocks):
    val_clock_names = parse_name_set(val_clocks)
    train_indices = []
    val_indices = []

    for index, item in enumerate(dataset.file_list):
        clock_name = item["dataset_dir"].name
        if clock_name in val_clock_names:
            val_indices.append(index)
        else:
            train_indices.append(index)

    if not train_indices:
        raise RuntimeError(f"No train samples left after holding out clocks: {sorted(val_clock_names)}")
    if not val_indices:
        raise RuntimeError(f"No validation samples found for clocks: {sorted(val_clock_names)}")

    print(f"Clock split: val_clocks={sorted(val_clock_names)}")
    return train_indices, val_indices


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


def get_dataset_item_label(dataset, index):
    if hasattr(dataset, "dataset"):
        return dataset.dataset.get_class_label(dataset.indices[index])
    return dataset.get_class_label(index)


def make_class_balanced_sampler(dataset, num_classes=3):
    labels = [get_dataset_item_label(dataset, index) for index in range(len(dataset))]
    label_tensor = torch.tensor(labels, dtype=torch.long)
    counts = torch.bincount(label_tensor, minlength=num_classes).float()
    weights = 1.0 / counts[label_tensor].clamp_min(1.0)
    return WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)


def compute_class_weights(dataset, num_classes=3):
    labels = [get_dataset_item_label(dataset, index) for index in range(len(dataset))]
    counts = torch.bincount(torch.tensor(labels, dtype=torch.long), minlength=num_classes).float()
    weights = counts.sum() / counts.clamp_min(1.0)
    weights = weights / weights.mean().clamp_min(1e-8)
    print("Class counts:", {CLASS_NAMES[i]: int(counts[i].item()) for i in range(num_classes)})
    print("Class weights:", {CLASS_NAMES[i]: float(weights[i].item()) for i in range(num_classes)})
    return weights


def iter_base_dataset_items(dataset):
    if hasattr(dataset, "datasets"):
        for child in dataset.datasets:
            yield from iter_base_dataset_items(child)
        return
    if hasattr(dataset, "dataset") and hasattr(dataset, "indices"):
        base = dataset.dataset
        for index in dataset.indices:
            yield base.file_list[int(index)]
        return
    if hasattr(dataset, "file_list"):
        yield from dataset.file_list


def object_counts(dataset):
    counts = {}
    for item in iter_base_dataset_items(dataset):
        name = object_prefix(item["dataset_dir"].name)
        counts[name] = counts.get(name, 0) + 1
    return dict(sorted(counts.items(), key=lambda pair: pair[0]))


def print_object_counts(title, dataset):
    counts = object_counts(dataset)
    text = ", ".join(f"{name}:{count}" for name, count in counts.items()) if counts else "none"
    print(f"{title} object counts: {text}")


def freeze_classifier_modules(model):
    module_names = [
        "spec_encoder",
        "depth_encoder",
        "film_gamma",
        "film_beta",
        "fusion_fc",
        "class_head",
    ]
    frozen_params = 0
    for name in module_names:
        module = getattr(model, name, None)
        if module is None:
            continue
        module.eval()
        for param in module.parameters():
            param.requires_grad = False
            frozen_params += param.numel()
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    print(f"Frozen classifier/gate params: {frozen_params}")
    print(f"Trainable params after freeze: {trainable_params}")


def build_model(args, audio_in_channels):
    if args.model == "audio":
        return SSLNet_DOA(
            use_compress=args.use_compress,
            audio_in_channels=audio_in_channels,
            num_classes=3 if args.use_classification else 0,
        )
    return SSLNet_depth_DOA(
        use_compress=args.use_compress,
        audio_in_channels=audio_in_channels,
        pretrained_depth_encoder=not args.no_pretrained_depth,
        freeze_depth_encoder=args.freeze_depth,
        drop_depth_prob=0.1,
        num_classes=3 if args.use_classification else 0,
    )


def build_scheduler(optimizer, args):
    if args.lr_scheduler == "none":
        return None
    if args.lr_scheduler == "cosine":
        if not 0.0 <= args.min_lr_ratio <= 1.0:
            raise ValueError("--min-lr-ratio must be between 0 and 1")
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(args.epochs, 1),
            eta_min=args.lr * args.min_lr_ratio,
        )
    if args.lr_step_size <= 0:
        raise ValueError("--lr-step-size must be positive")
    if not 0.0 < args.lr_gamma <= 1.0:
        raise ValueError("--lr-gamma must be in (0, 1]")
    return torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=args.lr_step_size,
        gamma=args.lr_gamma,
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
    if args.classification_only or args.freeze_classifier or args.gate_doa_by_pred_class:
        args.use_classification = True
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    print(f"Using device: {device}")

    audio_channels = parse_int_tuple(args.audio_channels)
    from dataloader.utils import parse_channel_pairs
    ipd_pairs = parse_channel_pairs(args.ipd_pairs)
    audio_in_channels = infer_audio_in_channels(args.audio_feat, audio_channels, ipd_pairs)
    print(f"Audio feature: {args.audio_feat}, input channels: {audio_in_channels}")
    print(f"Object filter: {format_object_filter(args.object_name)}")
    if args.noise_aug:
        noise_root = args.noise_aug_root or str(Path(args.data_root) / "noise")
        print(
            "Noise augmentation: "
            f"prob={args.noise_aug_prob:g}, snr=[{args.snr_min_db:g}, {args.snr_max_db:g}] dB, "
            f"root={noise_root}, extra_paths={args.noise_aug_path or []}"
        )

    train_dataset, val_dataset, train_loader, val_loader = build_loaders(args)
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    print_object_counts("Train", train_dataset)
    print_object_counts("Val", val_dataset)
    class_weights = None
    if args.use_classification and not args.no_class_loss_weights:
        class_weights = compute_class_weights(train_dataset)

    model = build_model(args, audio_in_channels).to(device)
    load_checkpoint_if_needed(model, args.checkpoint, device)
    if args.freeze_classifier:
        freeze_classifier_modules(model)

    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = build_scheduler(optimizer, args)
    writer = SummaryWriter(args.log_dir)
    print(f"LR scheduler: {args.lr_scheduler}")

    best_val_loss = float("inf")
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"\nEpoch {epoch}/{args.epochs} | lr={current_lr:.3e}")
        writer.add_scalar("LR/epoch", current_lr, epoch)
        train_loss, global_step = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            writer=writer,
            global_step=global_step,
            distance_weight=args.distance_weight,
            classification_weight=args.classification_weight if args.use_classification else 0.0,
            class_weights=class_weights,
            classification_only=args.classification_only,
            gate_doa_by_pred_class=args.gate_doa_by_pred_class,
            signal_class_id=2,
            freeze_classifier_eval=args.freeze_classifier,
        )
        val_loss = validate(
            model=model,
            val_loader=val_loader,
            device=device,
            epoch=epoch,
            writer=writer,
            distance_weight=args.distance_weight,
            classification_weight=args.classification_weight if args.use_classification else 0.0,
            class_weights=class_weights,
            classification_only=args.classification_only,
            gate_doa_by_pred_class=args.gate_doa_by_pred_class,
            signal_class_id=2,
        )
        if scheduler is not None:
            scheduler.step()
        next_lr = optimizer.param_groups[0]["lr"]
        print(f"Train loss: {train_loss:.6f} | Val loss: {val_loss:.6f}")
        if scheduler is not None:
            print(f"Next lr: {next_lr:.3e}")

        state = {
            "model": model.state_dict(),
            "epoch": epoch,
            "args": vars(args),
            "audio_in_channels": audio_in_channels,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "trained_lr": current_lr,
            "lr": next_lr,
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

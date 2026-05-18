import argparse
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Subset, random_split

from dataloader.ssl_dataset import CLASS_NAMES, SingleStepDataset
from dataloader.utils import parse_channel_pairs
from model_training.train_doa import combined_loss
from network.audionet.ssl_net import SSLNet_DOA, SSLNet_depth_DOA
from main_doa import has_explicit_train_test_split, infer_audio_in_channels, parse_int_tuple, parse_name_set


def parse_args():
    parser = argparse.ArgumentParser(description="Test and visualize DOA + distance model.")
    parser.add_argument("--data-root", default="synced_dataset")
    parser.add_argument("--train-root", default=None, help="Train root used during training.")
    parser.add_argument("--val-root", default=None, help="Val root used during training.")
    parser.add_argument("--eval-split", default="val", choices=["val", "train", "all"])
    parser.add_argument(
        "--object-name",
        default=None,
        help=(
            "Optional object name filter, e.g. dryer, person, guiter. "
            "It matches folders with the same prefix, such as dryer1/dryer2."
        ),
    )
    parser.add_argument("--val-clocks", default=None, help="Clock folder names held out during training, e.g. clock2.")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Must match the value used by main_doa.py.")
    parser.add_argument("--seed", type=int, default=7, help="Must match the value used by main_doa.py.")
    parser.add_argument("--odom", default="lio_odom", choices=["lio_odom", "lio_robo_odom"])
    parser.add_argument("--checkpoint", default="weights/ssl_doa_distance_synced/best_model.pth")
    parser.add_argument("--model", default="audio_depth", choices=["audio", "audio_depth"])
    parser.add_argument("--indices", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--print-samples", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-vis", action="store_true", help="Save DOA and distance visualization images.")
    parser.add_argument("--vis-dir", default="vis_result_doa")
    parser.add_argument("--vis-dist-dir", default="vis_result_dist")
    parser.add_argument("--doa-vis", default="curve", choices=["curve", "polar"], help="Save one DOA visualization type.")
    parser.add_argument("--audio-feat", default="ipd", choices=["ipd", "spec", "phase", "both", "gcc_phat_complex"])
    parser.add_argument("--audio-channels", default="1,2,3,4")
    parser.add_argument("--ipd-pairs", default="0-1,0-2,0-3,1-2,1-3,2-3")
    parser.add_argument("--audio-bandpass-low-hz", type=float, default=0.0)
    parser.add_argument("--audio-bandpass-high-hz", type=float, default=0.0)
    parser.add_argument("--min-distance", type=float, default=None, help="Only load samples with distance_xy >= this value.")
    parser.add_argument("--max-distance", type=float, default=None, help="Only load samples with distance_xy <= this value.")
    parser.add_argument("--use-classification", action="store_true", help="Enable 3-class classification branch while testing.")
    parser.add_argument("--classification-only", action="store_true", help="Evaluate only the 3-class classification branch.")
    parser.add_argument("--gate-doa-by-pred-class", action="store_true", help="Only compute DOA/distance loss when predicted class is signal_static.")
    parser.add_argument("--classification-weight", type=float, default=1.0)
    parser.add_argument("--distance-weight", type=float, default=0.5)
    parser.add_argument("--max-signal-abs", type=float, default=0.06)
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
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--use-compress", action="store_true")
    parser.add_argument(
        "--allow-missing-depth",
        action="store_true",
        help="Allow samples without depth images. Useful for audio-only testing.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no-pretrained-depth", action="store_true")
    return parser.parse_args()


def parse_indices(value):
    if value == "all":
        return None
    return [int(item.strip()) for item in value.split(",") if item.strip()]


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
        freeze_depth_encoder=False,
        drop_depth_prob=0.0,
        num_classes=3 if args.use_classification else 0,
    )


def load_checkpoint(model, checkpoint, device):
    if not checkpoint or not os.path.exists(checkpoint):
        print(f"[WARN] checkpoint not found: {checkpoint}")
        return
    ckpt = torch.load(checkpoint, map_location=device)
    state_dict = ckpt.get("model", ckpt.get("state_dict", ckpt)) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state_dict, strict=False)
    print(f"Loaded checkpoint: {checkpoint}")


def build_dataset(root, args):
    audio_channels = parse_int_tuple(args.audio_channels)
    ipd_pairs = parse_channel_pairs(args.ipd_pairs)
    return SingleStepDataset(
        root_dir=root,
        odom_name=args.odom,
        use_compress=args.use_compress,
        audio_feat=args.audio_feat,
        audio_channels=audio_channels,
        ipd_pairs=ipd_pairs,
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
        object_names=args.object_name,
        min_distance=args.min_distance,
        max_distance=args.max_distance,
        use_classification=args.use_classification,
        max_signal_abs=args.max_signal_abs if args.use_classification else None,
        audio_bandpass_low_hz=args.audio_bandpass_low_hz,
        audio_bandpass_high_hz=args.audio_bandpass_high_hz,
    )


def build_dataset_with_split(root, args, split):
    audio_channels = parse_int_tuple(args.audio_channels)
    ipd_pairs = parse_channel_pairs(args.ipd_pairs)
    return SingleStepDataset(
        root_dir=root,
        split=split,
        odom_name=args.odom,
        use_compress=args.use_compress,
        audio_feat=args.audio_feat,
        audio_channels=audio_channels,
        ipd_pairs=ipd_pairs,
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
        object_names=args.object_name,
        min_distance=args.min_distance,
        max_distance=args.max_distance,
        use_classification=args.use_classification,
        max_signal_abs=args.max_signal_abs if args.use_classification else None,
        audio_bandpass_low_hz=args.audio_bandpass_low_hz,
        audio_bandpass_high_hz=args.audio_bandpass_high_hz,
    )


def build_eval_dataset(args):
    if args.train_root and args.val_root:
        if args.eval_split == "train":
            dataset = build_dataset(args.train_root, args)
            return dataset, "train"
        if args.eval_split == "val":
            dataset = build_dataset(args.val_root, args)
            return dataset, "val"

        train_dataset = build_dataset(args.train_root, args)
        val_dataset = build_dataset(args.val_root, args)
        return torch.utils.data.ConcatDataset([train_dataset, val_dataset]), "all"

    if has_explicit_train_test_split(args.data_root):
        if args.eval_split == "train":
            return build_dataset_with_split(args.data_root, args, "train"), "train"
        if args.eval_split == "val":
            return build_dataset_with_split(args.data_root, args, "test"), "test"

        train_dataset = build_dataset_with_split(args.data_root, args, "train")
        test_dataset = build_dataset_with_split(args.data_root, args, "test")
        return torch.utils.data.ConcatDataset([train_dataset, test_dataset]), "all"

    full_dataset = build_dataset(args.data_root, args)
    if args.eval_split == "all":
        return full_dataset, "all"
    if args.val_clocks:
        train_dataset, val_dataset = split_dataset_by_clocks(full_dataset, args.val_clocks)
        return (train_dataset, "train") if args.eval_split == "train" else (val_dataset, "val")

    val_size = max(1, int(len(full_dataset) * args.val_ratio))
    train_size = len(full_dataset) - val_size
    if train_size <= 0:
        raise RuntimeError("Dataset is too small for the requested val split")

    generator = torch.Generator().manual_seed(args.seed)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)
    return (train_dataset, "train") if args.eval_split == "train" else (val_dataset, "val")


def split_dataset_by_clocks(dataset, val_clocks):
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
    return Subset(dataset, train_indices), Subset(dataset, val_indices)


def visualize_distribution(pred, gt, idx, save_path, title, xlabel):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 3))
    plt.plot(gt, label="GT", linewidth=2)
    plt.plot(pred, label="Pred", linewidth=2, linestyle="--")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("value")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def visualize_doa_polar(pred, gt, idx, save_path):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    bins = len(gt)
    theta = np.linspace(0.0, 2.0 * np.pi, bins, endpoint=False)
    pred_idx = int(np.argmax(pred))
    gt_idx = int(np.argmax(gt))

    plt.figure(figsize=(6, 6))
    ax = plt.subplot(111, projection="polar")
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.plot(theta, gt, label="GT", linewidth=2)
    ax.plot(theta, pred, label="Pred", linewidth=2, linestyle="--")
    ax.scatter([theta[gt_idx]], [gt[gt_idx]], c="C0", label=f"GT {gt_idx} deg")
    ax.scatter([theta[pred_idx]], [pred[pred_idx]], c="C1", marker="x", label=f"Pred {pred_idx} deg")
    ax.set_thetagrids([0, 90, 180, 270], labels=["Right", "Front", "Left", "Back"])
    ax.set_title(f"DOA sample {idx}")
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.2), fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main():
    args = parse_args()
    if args.classification_only or args.gate_doa_by_pred_class:
        args.use_classification = True
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    print(f"Using device: {device}")

    audio_channels = parse_int_tuple(args.audio_channels)
    ipd_pairs = parse_channel_pairs(args.ipd_pairs)
    audio_in_channels = infer_audio_in_channels(args.audio_feat, audio_channels, ipd_pairs)

    dataset, split_name = build_eval_dataset(args)
    print(f"Eval split: {split_name}, samples: {len(dataset)}, audio_in_channels={audio_in_channels}")

    model = build_model(args, audio_in_channels).to(device)
    load_checkpoint(model, args.checkpoint, device)
    model.eval()

    indices = parse_indices(args.indices)
    if indices is None:
        indices = list(range(len(dataset)))

    total_loss = 0.0
    n_samples = 0
    class_correct = 0
    class_total = 0
    class_counts = torch.zeros(len(CLASS_NAMES), dtype=torch.long)
    class_correct_counts = torch.zeros(len(CLASS_NAMES), dtype=torch.long)
    with torch.no_grad():
        for idx in indices:
            if idx < 0 or idx >= len(dataset):
                print(f"[WARN] index {idx} out of range, skip")
                continue
            sample = dataset[idx]
            depth = sample["depth"].unsqueeze(0).to(device)
            spectrogram = sample["spectrogram"].unsqueeze(0).to(device)
            gt_doa = sample["doa_map"].unsqueeze(0).to(device)
            gt_dist = sample["distant_map"].unsqueeze(0).to(device)
            class_target = sample.get("class_label")
            has_doa = sample.get("has_doa")
            if class_target is not None:
                class_target = class_target.unsqueeze(0).to(device)
            if has_doa is not None:
                has_doa = has_doa.unsqueeze(0).to(device)

            outputs = model(spectrogram, depth)
            logits_doa, logits_dist = outputs[:2]
            class_logits = outputs[2] if len(outputs) > 2 else None
            loss_mask = has_doa
            if args.gate_doa_by_pred_class and class_logits is not None:
                pred_signal = class_logits.argmax(dim=1) == 2
                loss_mask = pred_signal if loss_mask is None else (loss_mask.bool() & pred_signal)
            pred_doa = torch.softmax(logits_doa, dim=1)
            pred_dist = torch.softmax(logits_dist, dim=1)
            loss, loss_doa, loss_dist, loss_cls = combined_loss(
                logits_doa,
                logits_dist,
                gt_doa,
                gt_dist,
                distance_weight=args.distance_weight,
                class_logits=class_logits,
                class_target=class_target,
                classification_weight=args.classification_weight if args.use_classification else 0.0,
                has_doa=loss_mask,
                classification_only=args.classification_only,
            )
            total_loss += loss.item()
            n_samples += 1

            pred_doa_np = pred_doa.squeeze(0).cpu().numpy()
            gt_doa_np = gt_doa.squeeze(0).cpu().numpy()
            pred_dist_np = pred_dist.squeeze(0).cpu().numpy()
            gt_dist_np = gt_dist.squeeze(0).cpu().numpy()

            class_text = ""
            if class_logits is not None and class_target is not None:
                pred_class = int(class_logits.argmax(dim=1).item())
                gt_class = int(class_target.item())
                class_total += 1
                class_correct += int(pred_class == gt_class)
                class_counts[gt_class] += 1
                class_correct_counts[gt_class] += int(pred_class == gt_class)
                class_text = (
                    f" cls_loss={loss_cls.item():.6f} "
                    f"gt_class={CLASS_NAMES[gt_class]} pred_class={CLASS_NAMES[pred_class]}"
                )
            doa_text = (
                f"gt_doa_bin={int(gt_doa_np.argmax())} pred_doa_bin={int(pred_doa_np.argmax())} "
                f"gt_dist_bin={int(gt_dist_np.argmax())} pred_dist_bin={int(pred_dist_np.argmax())}"
                if bool(sample.get("has_doa", True))
                else "no_doa_label"
            )
            if args.print_samples:
                print(
                    f"idx={idx} loss={loss.item():.6f} doa_loss={loss_doa.item():.6f} dist_loss={loss_dist.item():.6f}"
                    f"{class_text} {doa_text} path={sample['path']}"
                )
            if args.save_vis:
                if args.doa_vis == "polar":
                    visualize_doa_polar(pred_doa_np, gt_doa_np, idx, Path(args.vis_dir) / f"sample_{idx:06d}.png")
                else:
                    visualize_distribution(
                        pred_doa_np,
                        gt_doa_np,
                        idx,
                        Path(args.vis_dir) / f"sample_{idx:06d}.png",
                        f"DOA sample {idx}",
                        "angle bin",
                    )
                visualize_distribution(
                    pred_dist_np,
                    gt_dist_np,
                    idx,
                    Path(args.vis_dist_dir) / f"sample_{idx:06d}.png",
                    f"Distance sample {idx}",
                    "distance bin",
                )

    if n_samples:
        print(f"Average loss on {n_samples} samples: {total_loss / n_samples:.6f}")
    if class_total:
        print(f"Classification accuracy: {class_correct / class_total:.6f} ({class_correct}/{class_total})")
        for class_id, class_name in enumerate(CLASS_NAMES):
            total = int(class_counts[class_id].item())
            correct = int(class_correct_counts[class_id].item())
            acc = correct / total if total else 0.0
            print(f"  {class_name}: acc={acc:.6f} ({correct}/{total})")
    if args.save_vis:
        print(f"Visualizations saved to {args.vis_dir} and {args.vis_dist_dir}")
    else:
        print("Visualization saving disabled. Use --save-vis to write images.")


if __name__ == "__main__":
    main()

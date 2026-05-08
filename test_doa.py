import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import random_split

from dataloader.ssl_dataset import SingleStepDataset
from dataloader.utils import parse_channel_pairs
from model_training.train_doa import combined_loss
from network.audionet.ssl_net import SSLNet_DOA, SSLNet_depth_DOA
from main_doa import infer_audio_in_channels, parse_int_tuple


def parse_args():
    parser = argparse.ArgumentParser(description="Test and visualize DOA + distance model.")
    parser.add_argument("--data-root", default="synced_dataset")
    parser.add_argument("--train-root", default=None, help="Train root used during training.")
    parser.add_argument("--val-root", default=None, help="Val root used during training.")
    parser.add_argument("--eval-split", default="val", choices=["val", "train", "all"])
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Must match the value used by main_doa.py.")
    parser.add_argument("--seed", type=int, default=7, help="Must match the value used by main_doa.py.")
    parser.add_argument("--odom", default="lio_odom", choices=["lio_odom", "lio_robo_odom"])
    parser.add_argument("--checkpoint", default="weights/ssl_doa_distance_synced/best_model.pth")
    parser.add_argument("--model", default="audio_depth", choices=["audio", "audio_depth"])
    parser.add_argument("--indices", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--vis-dir", default="vis_result_doa")
    parser.add_argument("--vis-dist-dir", default="vis_result_dist")
    parser.add_argument("--doa-vis", default="curve", choices=["curve", "polar"], help="Save one DOA visualization type.")
    parser.add_argument("--audio-feat", default="ipd", choices=["ipd", "spec", "phase", "both", "gcc_phat_complex"])
    parser.add_argument("--audio-channels", default="0,1,2,3")
    parser.add_argument("--ipd-pairs", default="0-1,0-2,0-3,1-2,1-3,2-3")
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


def build_model(args, audio_in_channels):
    if args.model == "audio":
        return SSLNet_DOA(use_compress=args.use_compress, audio_in_channels=audio_in_channels)
    return SSLNet_depth_DOA(
        use_compress=args.use_compress,
        audio_in_channels=audio_in_channels,
        pretrained_depth_encoder=not args.no_pretrained_depth,
        freeze_depth_encoder=False,
        drop_depth_prob=0.0,
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

    full_dataset = build_dataset(args.data_root, args)
    if args.eval_split == "all":
        return full_dataset, "all"

    val_size = max(1, int(len(full_dataset) * args.val_ratio))
    train_size = len(full_dataset) - val_size
    if train_size <= 0:
        raise RuntimeError("Dataset is too small for the requested val split")

    generator = torch.Generator().manual_seed(args.seed)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)
    return (train_dataset, "train") if args.eval_split == "train" else (val_dataset, "val")


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

            logits_doa, logits_dist = model(spectrogram, depth)
            pred_doa = torch.softmax(logits_doa, dim=1)
            pred_dist = torch.softmax(logits_dist, dim=1)
            loss, loss_doa, loss_dist = combined_loss(logits_doa, logits_dist, gt_doa, gt_dist)
            total_loss += loss.item()
            n_samples += 1

            pred_doa_np = pred_doa.squeeze(0).cpu().numpy()
            gt_doa_np = gt_doa.squeeze(0).cpu().numpy()
            pred_dist_np = pred_dist.squeeze(0).cpu().numpy()
            gt_dist_np = gt_dist.squeeze(0).cpu().numpy()

            print(
                f"idx={idx} loss={loss.item():.6f} doa_loss={loss_doa.item():.6f} dist_loss={loss_dist.item():.6f} "
                f"gt_doa_bin={int(gt_doa_np.argmax())} pred_doa_bin={int(pred_doa_np.argmax())} "
                f"gt_dist_bin={int(gt_dist_np.argmax())} pred_dist_bin={int(pred_dist_np.argmax())} "
                f"path={sample['path']}"
            )
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
    print(f"Visualizations saved to {args.vis_dir} and {args.vis_dist_dir}")


if __name__ == "__main__":
    main()

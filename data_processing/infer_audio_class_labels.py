#!/usr/bin/env python3
import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataloader.ssl_dataset import CLASS_NAMES, make_empty_depth
from dataloader.utils import (
    compute_spectrogram,
    compute_stft_phase_features,
    load_audio_wav,
    parse_channel_pairs,
)
from main_doa import infer_audio_in_channels, parse_int_tuple
from network.audionet.ssl_net import SSLNet_DOA, SSLNet_depth_DOA


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")

    audio_channels = parse_int_tuple(args.audio_channels)
    ipd_pairs = parse_channel_pairs(args.ipd_pairs)
    audio_in_channels = infer_audio_in_channels(args.audio_feat, audio_channels, ipd_pairs)

    model = build_model(args, audio_in_channels).to(device)
    load_checkpoint(model, args.checkpoint, device)
    model.eval()

    dataset_dirs = find_audio_dataset_dirs(
        Path(args.data_root),
        object_name=args.object_name,
        recursive=args.recursive,
        include_noise=args.include_noise,
    )
    print(f"Found {len(dataset_dirs)} audio dataset dir(s)")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Audio feature: {args.audio_feat}, input channels: {audio_in_channels}")

    total = 0
    for dataset_dir in dataset_dirs:
        count = infer_one_dataset(
            dataset_dir=dataset_dir,
            model=model,
            device=device,
            args=args,
            audio_channels=audio_channels,
            ipd_pairs=ipd_pairs,
        )
        total += count

    print(f"Finished. Wrote class labels for {total} audio segment(s).")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Infer 3-class audio labels for every wav segment and save class/*.npy files."
    )
    parser.add_argument("--data-root", default="synced_dataset")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--object-name", default=None, help="Optional prefix filter, e.g. clock/person/dryer. Noise dirs are still included with --include-noise.")
    parser.add_argument("--recursive", action="store_true", help="Search audio dirs recursively.")
    parser.add_argument("--include-noise", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--model", default="audio", choices=["audio", "audio_depth"])
    parser.add_argument("--audio-feat", default="spec", choices=["ipd", "spec", "phase", "both", "gcc_phat_complex"])
    parser.add_argument("--audio-channels", default="1,2,3,4")
    parser.add_argument("--ipd-pairs", default="0-1,0-2,0-3,1-2,1-3,2-3")
    parser.add_argument("--use-compress", action="store_true")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir-name", default="class")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--save-probs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If true, npy format is [pred_class, prob0, prob1, prob2]. If false, save only [pred_class].",
    )
    return parser.parse_args()


def build_model(args, audio_in_channels):
    if args.model == "audio":
        return SSLNet_DOA(
            use_compress=args.use_compress,
            audio_in_channels=audio_in_channels,
            num_classes=len(CLASS_NAMES),
        )
    return SSLNet_depth_DOA(
        use_compress=args.use_compress,
        audio_in_channels=audio_in_channels,
        pretrained_depth_encoder=False,
        freeze_depth_encoder=False,
        drop_depth_prob=0.0,
        num_classes=len(CLASS_NAMES),
    )


def load_checkpoint(model, checkpoint, device):
    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.exists():
        raise RuntimeError(f"Checkpoint does not exist: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt.get("model", ckpt.get("state_dict", ckpt)) if isinstance(ckpt, dict) else ckpt
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"Loaded checkpoint: {checkpoint_path}")
    if missing:
        print(f"[WARN] missing keys: {len(missing)}")
    if unexpected:
        print(f"[WARN] unexpected keys: {len(unexpected)}")


def find_audio_dataset_dirs(data_root, object_name=None, recursive=False, include_noise=True):
    data_root = Path(data_root)
    pattern = "**/audio" if recursive else "*/audio"
    if (data_root / "audio").is_dir():
        candidates = [data_root]
    else:
        candidates = sorted({path.parent for path in data_root.glob(pattern)})

    names = parse_name_filter(object_name)
    if names:
        filtered = []
        for path in candidates:
            if matches_object_filter(path.name, names) or (include_noise and path.name in ("robot_noise", "moving_sound")):
                filtered.append(path)
        candidates = filtered
    elif not include_noise:
        candidates = [path for path in candidates if path.name not in ("robot_noise", "moving_sound")]
    return candidates


def infer_one_dataset(dataset_dir, model, device, args, audio_channels, ipd_pairs):
    audio_dir = dataset_dir / "audio"
    output_dir = dataset_dir / args.output_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    wav_paths = sorted(audio_dir.glob("*.wav"), key=lambda path: numeric_stem(path.stem))
    for wav_path in wav_paths:
        output_path = output_dir / f"{wav_path.stem}.npy"
        if output_path.exists() and not args.overwrite:
            continue

        sample = build_audio_sample(
            wav_path=wav_path,
            args=args,
            audio_channels=audio_channels,
            ipd_pairs=ipd_pairs,
        )
        spectrogram = sample["spectrogram"].unsqueeze(0).to(device)
        depth = sample["depth"].unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(spectrogram, depth)
            if len(outputs) < 3:
                raise RuntimeError("Model does not have a classification head. Use a classification checkpoint/model.")
            probs = torch.softmax(outputs[2], dim=1).squeeze(0).cpu().numpy()

        pred_class = int(np.argmax(probs))
        if args.save_probs:
            saved = np.concatenate([[pred_class], probs.astype(np.float32)]).astype(np.float32)
        else:
            saved = np.array([pred_class], dtype=np.int64)
        np.save(output_path, saved)

        rows.append({
            "sample_id": wav_path.stem,
            "audio": str(wav_path.relative_to(dataset_dir)),
            "class_file": str(output_path.relative_to(dataset_dir)),
            "pred_class": pred_class,
            "pred_class_name": CLASS_NAMES[pred_class],
            **{f"prob_{CLASS_NAMES[i]}": float(probs[i]) for i in range(len(CLASS_NAMES))},
        })

    if rows:
        write_index_csv(output_dir / "index.csv", rows)
    print(f"{dataset_dir}: wrote {len(rows)} class label(s) -> {output_dir}")
    return len(rows)


def build_audio_sample(wav_path, args, audio_channels, ipd_pairs):
    audio, _ = load_audio_wav(wav_path, audio_channels)
    if args.audio_feat == "spec":
        spectrogram = compute_spectrogram(audio, args.use_compress)
    else:
        spectrogram = compute_stft_phase_features(audio, mode=args.audio_feat, pairs=ipd_pairs)
    spectrogram = torch.as_tensor(spectrogram, dtype=torch.float32).permute(2, 0, 1)
    depth = torch.as_tensor(make_empty_depth((args.image_size, args.image_size)), dtype=torch.float32).unsqueeze(0)
    return {
        "spectrogram": spectrogram,
        "depth": depth,
    }


def parse_name_filter(value):
    if value is None:
        return set()
    return {item.strip() for item in value.split(",") if item.strip()}


def object_prefix(sequence_name):
    return re.sub(r"\d+$", "", sequence_name)


def matches_object_filter(sequence_name, object_names):
    return sequence_name in object_names or object_prefix(sequence_name) in object_names


def numeric_stem(stem):
    try:
        return int(stem)
    except ValueError:
        return stem


def write_index_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import soundfile as sf


def main():
    parser = argparse.ArgumentParser(
        description="Split noise-only WAV files into fixed-length audio segments."
    )
    parser.add_argument(
        "--input-dir",
        default="audio_data/noise",
        help="Directory containing noise wav files.",
    )
    parser.add_argument(
        "--files",
        default="robot_noise.wav,moving_sound.wav",
        help="Comma-separated wav file names under --input-dir. Use 'all' for every wav in the directory.",
    )
    parser.add_argument(
        "--output-dir",
        default="synced_dataset",
        help=(
            "Output parent directory. Default is synced_dataset. "
            "With --layout synced, each source file is written to <output-dir>/<stem>/audio."
        ),
    )
    parser.add_argument(
        "--layout",
        default="synced",
        choices=["synced", "segments"],
        help=(
            "Output layout. 'synced' writes synced_dataset-like folders with audio/metadata. "
            "'segments' writes <stem>_segments with only wav/index.csv."
        ),
    )
    parser.add_argument("--segment-sec", type=float, default=1.0, help="Segment duration in seconds.")
    parser.add_argument("--hop-sec", type=float, default=None, help="Hop duration in seconds. Default equals --segment-sec.")
    parser.add_argument(
        "--channels",
        default=None,
        help=(
            "Optional 1-based channels to keep, e.g. 1,2,3,4. "
            "Default keeps all channels in the wav."
        ),
    )
    parser.add_argument(
        "--pad-last",
        action="store_true",
        help="Pad the last incomplete segment with zeros instead of dropping it.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into an existing segment directory.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_parent = Path(args.output_dir) if args.output_dir else input_dir
    wav_paths = resolve_wav_paths(input_dir, args.files)
    channels = parse_channels(args.channels)

    if args.segment_sec <= 0:
        raise RuntimeError("--segment-sec must be positive.")
    hop_sec = args.hop_sec if args.hop_sec is not None else args.segment_sec
    if hop_sec <= 0:
        raise RuntimeError("--hop-sec must be positive.")

    output_parent.mkdir(parents=True, exist_ok=True)
    total_segments = 0
    for wav_path in wav_paths:
        count = split_one_wav(
            wav_path=wav_path,
            output_parent=output_parent,
            layout=args.layout,
            segment_sec=args.segment_sec,
            hop_sec=hop_sec,
            channels=channels,
            pad_last=args.pad_last,
            overwrite=args.overwrite,
        )
        total_segments += count

    print(f"Finished. Split {len(wav_paths)} wav file(s), generated {total_segments} segment(s).")


def resolve_wav_paths(input_dir, files_arg):
    input_dir = Path(input_dir)
    if not input_dir.exists():
        raise RuntimeError(f"Input directory does not exist: {input_dir}")

    if files_arg.strip().lower() == "all":
        wav_paths = sorted(input_dir.glob("*.wav"))
    else:
        wav_paths = [
            input_dir / item.strip()
            for item in files_arg.split(",")
            if item.strip()
        ]

    missing = [path for path in wav_paths if not path.exists()]
    if missing:
        raise RuntimeError("Missing wav file(s): " + ", ".join(str(path) for path in missing))
    if not wav_paths:
        raise RuntimeError(f"No wav files selected under: {input_dir}")
    return wav_paths


def parse_channels(value):
    if value is None or str(value).strip() == "":
        return None
    channels = [int(item.strip()) for item in str(value).split(",") if item.strip()]
    if any(channel <= 0 for channel in channels):
        raise RuntimeError("--channels uses 1-based channel ids, so every value must be >= 1.")
    return tuple(channel - 1 for channel in channels)


def split_one_wav(
    wav_path,
    output_parent,
    layout,
    segment_sec,
    hop_sec,
    channels=None,
    pad_last=False,
    overwrite=False,
):
    audio, sample_rate = sf.read(str(wav_path), always_2d=True)
    if channels is not None:
        if max(channels) >= audio.shape[1]:
            requested = ",".join(str(channel + 1) for channel in channels)
            raise RuntimeError(
                f"{wav_path} has {audio.shape[1]} channel(s), cannot select 1-based channels: {requested}"
            )
        audio = audio[:, channels]

    segment_frames = int(round(segment_sec * sample_rate))
    hop_frames = int(round(hop_sec * sample_rate))
    if segment_frames <= 0 or hop_frames <= 0:
        raise RuntimeError("segment-sec and hop-sec are too small for this sample rate.")

    output_dir = build_output_dir(output_parent, wav_path, layout)
    wav_dir = output_dir / "audio" if layout == "synced" else output_dir
    metadata_dir = output_dir / "metadata" if layout == "synced" else None

    if wav_dir.exists() and not overwrite and any(wav_dir.glob("*.wav")):
        raise RuntimeError(
            f"Output directory already contains wav segments: {wav_dir}. "
            "Use --overwrite or choose another --output-dir."
        )
    wav_dir.mkdir(parents=True, exist_ok=True)
    if metadata_dir is not None:
        metadata_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    sample_id = 0
    start = 0
    while start < audio.shape[0]:
        end = start + segment_frames
        if end > audio.shape[0]:
            if not pad_last:
                break
            segment = np.zeros((segment_frames, audio.shape[1]), dtype=audio.dtype)
            valid = audio.shape[0] - start
            if valid <= 0:
                break
            segment[:valid] = audio[start:]
        else:
            segment = audio[start:end]

        sample_stem = f"{sample_id:06d}"
        sample_name = f"{sample_stem}.wav"
        segment_path = wav_dir / sample_name
        sf.write(str(segment_path), segment, sample_rate)

        row = {
            "sample_id": sample_stem,
            "source_file": str(wav_path),
            "segment_file": str(segment_path),
            "sample_rate": sample_rate,
            "channels": segment.shape[1],
            "start_frame": start,
            "end_frame": min(end, audio.shape[0]),
            "start_sec": start / sample_rate,
            "end_sec": min(end, audio.shape[0]) / sample_rate,
            "padded": int(end > audio.shape[0]),
        }
        rows.append(row)
        if metadata_dir is not None:
            write_json(metadata_dir / f"{sample_stem}.json", {
                "id": sample_stem,
                "source_file": str(wav_path),
                "audio": str(segment_path.relative_to(output_dir)),
                "start_frame": row["start_frame"],
                "end_frame": row["end_frame"],
                "start_sec": row["start_sec"],
                "end_sec": row["end_sec"],
                "sample_rate": sample_rate,
                "channels": segment.shape[1],
                "segment_sec": segment_sec,
                "hop_sec": hop_sec,
                "padded": bool(row["padded"]),
                "noise_only": True,
            })

        sample_id += 1
        start += hop_frames

    write_index_csv(output_dir / "index.csv", rows)
    if layout == "synced":
        write_json(output_dir / "dataset_manifest.json", {
            "dataset_type": "noise_only",
            "source_file": str(wav_path),
            "sample_rate": sample_rate,
            "channels": audio.shape[1],
            "segment_sec": segment_sec,
            "hop_sec": hop_sec,
            "pad_last": pad_last,
            "samples": rows,
        })
    print(f"{wav_path}: {sample_id} segment(s) -> {output_dir}")
    return sample_id


def build_output_dir(output_parent, wav_path, layout):
    if layout == "synced":
        return Path(output_parent) / wav_path.stem
    return Path(output_parent) / f"{wav_path.stem}_segments"


def write_index_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()

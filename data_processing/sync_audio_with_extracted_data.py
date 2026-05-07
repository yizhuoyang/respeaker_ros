#!/usr/bin/env python3
import argparse
import csv
import json
import re
import shutil
from pathlib import Path

import numpy as np
import soundfile as sf


NS_PER_SEC = 1_000_000_000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--extracted",
        required=True,
        help="One extracted bag directory, e.g. extracted_data/bag_001.",
    )
    parser.add_argument("--audio", required=True, help="Audio WAV recorded at the same time.")
    parser.add_argument(
        "--audio-start-ns",
        type=int,
        default=None,
        help="Audio start timestamp in ns. If omitted, parse from WAV filename.",
    )
    parser.add_argument("--output", required=True, help="Output synchronized sample directory.")
    parser.add_argument("--start-ns", type=int, default=None, help="Start timestamp in ns.")
    parser.add_argument("--end-ns", type=int, default=None, help="End timestamp in ns.")
    parser.add_argument("--segment-sec", type=float, default=1.0, help="Audio segment duration.")
    parser.add_argument("--hop-sec", type=float, default=None, help="Step between segments.")
    parser.add_argument(
        "--max-diff-sec",
        type=float,
        default=0.2,
        help="Maximum allowed nearest sensor timestamp difference.",
    )
    args = parser.parse_args()

    sync_dataset(
        extracted_dir=Path(args.extracted),
        audio_path=Path(args.audio),
        audio_start_ns=args.audio_start_ns,
        output_dir=Path(args.output),
        start_ns=args.start_ns,
        end_ns=args.end_ns,
        segment_sec=args.segment_sec,
        hop_sec=args.hop_sec if args.hop_sec is not None else args.segment_sec,
        max_diff_ns=int(args.max_diff_sec * NS_PER_SEC),
    )


def sync_dataset(
    extracted_dir,
    audio_path,
    audio_start_ns,
    output_dir,
    start_ns=None,
    end_ns=None,
    segment_sec=1.0,
    hop_sec=1.0,
    max_diff_ns=200_000_000,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_start_ns = resolve_audio_start_ns(audio_path, audio_start_ns)

    audio, sample_rate = sf.read(str(audio_path), always_2d=True)
    audio_duration_ns = int(round(audio.shape[0] / sample_rate * NS_PER_SEC))
    audio_end_ns = audio_start_ns + audio_duration_ns

    start_ns = audio_start_ns if start_ns is None else max(start_ns, audio_start_ns)
    end_ns = audio_end_ns if end_ns is None else min(end_ns, audio_end_ns)

    segment_ns = int(round(segment_sec * NS_PER_SEC))
    hop_ns = int(round(hop_sec * NS_PER_SEC))
    if segment_ns <= 0 or hop_ns <= 0:
        raise RuntimeError("segment-sec and hop-sec must be positive.")

    color_index = load_image_index(extracted_dir / "color" / "index.csv")
    depth_index = load_image_index(extracted_dir / "depth" / "index.csv")
    lio_odom = load_odom_csv(extracted_dir / "lio_odom.csv")
    lio_robo_odom = load_odom_csv(extracted_dir / "lio_robo_odom.csv")

    metadata_rows = []
    lio_odom_rows = []
    lio_robo_odom_rows = []
    prepare_modality_dirs(output_dir)

    sample_id = 0
    current_start_ns = start_ns
    while current_start_ns + segment_ns <= end_ns:
        current_end_ns = current_start_ns + segment_ns
        center_ns = current_start_ns + segment_ns // 2
        sync_ns = current_end_ns

        sample = {
            "id": f"{sample_id:06d}",
            "start_ns": current_start_ns,
            "end_ns": current_end_ns,
            "center_ns": center_ns,
            "sync_ns": sync_ns,
        }

        color = latest_record_at_or_before(color_index, sync_ns, max_diff_ns)
        depth = latest_record_at_or_before(depth_index, sync_ns, max_diff_ns)
        lio = latest_record_at_or_before(lio_odom, sync_ns, max_diff_ns)
        robo = latest_record_at_or_before(lio_robo_odom, sync_ns, max_diff_ns)

        if color is None and depth is None and lio is None and robo is None:
            current_start_ns += hop_ns
            continue

        sample_name = f"{sample_id:06d}"
        audio_file = output_dir / "audio" / f"{sample_name}.wav"
        write_audio_segment(
            audio=audio,
            sample_rate=sample_rate,
            audio_start_ns=audio_start_ns,
            segment_start_ns=current_start_ns,
            segment_end_ns=current_end_ns,
            output_file=audio_file,
        )
        sample["audio"] = str(audio_file.relative_to(output_dir))

        if color is not None:
            sample["color"] = copy_sensor_file(color, output_dir / "color", sample_name)
        if depth is not None:
            sample["depth"] = copy_sensor_file(depth, output_dir / "depth", sample_name)
        if lio is not None:
            sample["lio_odom"] = write_odom_npy(output_dir / "lio_odom", sample_name, lio)
            lio_odom_rows.append(odom_npz_row(sample_name, lio))
        if robo is not None:
            sample["lio_robo_odom"] = write_odom_npy(output_dir / "lio_robo_odom", sample_name, robo)
            lio_robo_odom_rows.append(odom_npz_row(sample_name, robo))

        write_json(output_dir / "metadata" / f"{sample_name}.json", sample)
        metadata_rows.append(sample)

        sample_id += 1
        current_start_ns += hop_ns

    write_odom_npz(output_dir / "lio_odom.npz", lio_odom_rows)
    write_odom_npz(output_dir / "lio_robo_odom.npz", lio_robo_odom_rows)

    write_json(output_dir / "dataset_manifest.json", {
        "extracted_dir": str(extracted_dir),
        "audio": str(audio_path),
        "audio_start_ns": audio_start_ns,
        "audio_end_ns": audio_end_ns,
        "sample_rate": sample_rate,
        "segment_sec": segment_sec,
        "hop_sec": hop_sec,
        "max_diff_ns": max_diff_ns,
        "samples": metadata_rows,
    })

    print(f"Generated {len(metadata_rows)} synchronized sample(s) under {output_dir}")


def prepare_modality_dirs(output_dir):
    for name in ("audio", "color", "depth", "lio_odom", "lio_robo_odom", "metadata"):
        (output_dir / name).mkdir(parents=True, exist_ok=True)


def resolve_audio_start_ns(audio_path, audio_start_ns=None):
    if audio_start_ns is not None:
        return audio_start_ns

    match = re.search(r"_(\d{12,})\.wav$", Path(audio_path).name)
    if not match:
        raise RuntimeError(
            "audio-start-ns was not provided and could not be parsed from "
            f"WAV filename: {audio_path}"
        )

    return int(match.group(1))


def load_image_index(index_csv):
    if not index_csv.exists():
        return []

    records = []
    base_dir = index_csv.parent
    with index_csv.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            timestamp_ns = row_timestamp_ns(row)
            records.append({
                **row,
                "timestamp_ns": timestamp_ns,
                "path": base_dir / row["file"],
            })
    return records


def load_odom_csv(csv_path):
    if not csv_path.exists():
        return []

    records = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            timestamp_ns = row_timestamp_ns(row)
            numeric_row = dict(row)
            for key, value in row.items():
                if key in ("frame_id", "child_frame_id"):
                    continue
                numeric_row[key] = parse_number(value)
            numeric_row["timestamp_ns"] = timestamp_ns
            records.append(numeric_row)
    return records


def row_timestamp_ns(row):
    if "stamp_sec" in row:
        sec_key = "stamp_sec"
    else:
        sec_key = "msg_stamp_sec"

    if "stamp_nanosec" in row:
        nsec_key = "stamp_nanosec"
    else:
        nsec_key = "msg_stamp_nanosec"

    return int(row[sec_key]) * NS_PER_SEC + int(row[nsec_key])


def nearest_record(records, timestamp_ns, max_diff_ns):
    if not records:
        return None

    best = min(records, key=lambda row: abs(row["timestamp_ns"] - timestamp_ns))
    diff_ns = abs(best["timestamp_ns"] - timestamp_ns)
    if diff_ns > max_diff_ns:
        return None

    return {**best, "diff_ns": diff_ns}


def latest_record_at_or_before(records, timestamp_ns, max_diff_ns):
    if not records:
        return None

    candidates = [row for row in records if row["timestamp_ns"] <= timestamp_ns]
    if candidates:
        best = max(candidates, key=lambda row: row["timestamp_ns"])
    else:
        best = min(records, key=lambda row: row["timestamp_ns"])

    diff_ns = abs(best["timestamp_ns"] - timestamp_ns)
    if diff_ns > max_diff_ns:
        return None

    return {**best, "diff_ns": diff_ns}


def write_audio_segment(audio, sample_rate, audio_start_ns, segment_start_ns, segment_end_ns, output_file):
    start_sample = int(round((segment_start_ns - audio_start_ns) / NS_PER_SEC * sample_rate))
    end_sample = int(round((segment_end_ns - audio_start_ns) / NS_PER_SEC * sample_rate))
    start_sample = max(0, start_sample)
    end_sample = min(audio.shape[0], end_sample)
    sf.write(str(output_file), audio[start_sample:end_sample], sample_rate)


def copy_sensor_file(record, modality_dir, sample_name):
    source = record["path"]
    target = modality_dir / f"{sample_name}{source.suffix}"
    shutil.copy2(source, target)
    return {
        "file": str(target.relative_to(modality_dir.parent)),
        "source": str(source),
        "timestamp_ns": record["timestamp_ns"],
        "diff_ns": record["diff_ns"],
    }


def write_odom_npy(modality_dir, sample_name, record):
    target = modality_dir / f"{sample_name}.npy"
    np.save(target, odom_vector(record))
    return {
        "file": str(target.relative_to(modality_dir.parent)),
        "timestamp_ns": record["timestamp_ns"],
        "diff_ns": record["diff_ns"],
        "fields": odom_fields(),
    }


def write_odom_npz(npz_path, rows):
    if not rows:
        return

    sample_ids = np.array([row["sample_id"] for row in rows])
    timestamps_ns = np.array([row["timestamp_ns"] for row in rows], dtype=np.int64)
    diff_ns = np.array([row["diff_ns"] for row in rows], dtype=np.int64)
    data = np.stack([row["data"] for row in rows], axis=0)
    fields = np.array(odom_fields())

    np.savez(
        npz_path,
        sample_ids=sample_ids,
        timestamps_ns=timestamps_ns,
        diff_ns=diff_ns,
        data=data,
        fields=fields,
    )


def odom_npz_row(sample_name, record):
    return {
        "sample_id": sample_name,
        "timestamp_ns": int(record["timestamp_ns"]),
        "diff_ns": int(record["diff_ns"]),
        "data": odom_vector(record),
    }


def odom_vector(record):
    return np.array([float(record[field]) for field in odom_fields()], dtype=np.float64)


def odom_fields():
    return [
        "px",
        "py",
        "pz",
        "qx",
        "qy",
        "qz",
        "qw",
        "linear_x",
        "linear_y",
        "linear_z",
        "angular_x",
        "angular_y",
        "angular_z",
    ]


def write_json(path, data):
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return path.name


def parse_number(value):
    if value == "":
        return value
    try:
        if "." in value or "e" in value.lower():
            return float(value)
        return int(value)
    except ValueError:
        return value


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

from export_audio_from_bag import export_audio
from export_audio_from_bag import parse_channel_list


def discover_bags(input_dir, recursive=False):
    root = Path(input_dir)
    if not root.exists():
        raise RuntimeError(f"Input path does not exist: {input_dir}")

    if root.is_file():
        if root.suffix == ".db3":
            return [root]
        raise RuntimeError(f"Input file is not a .db3 bag database: {input_dir}")

    pattern = "**/*.db3" if recursive else "*.db3"
    db3_files = sorted(root.glob(pattern))
    bag_paths = []
    seen = set()

    for db3_file in db3_files:
        bag_path = db3_file.parent
        key = bag_path.resolve()
        if key not in seen:
            seen.add(key)
            bag_paths.append(bag_path)

    if not recursive:
        for child in sorted(root.iterdir()):
            if child.is_dir() and any(child.glob("*.db3")):
                key = child.resolve()
                if key not in seen:
                    seen.add(key)
                    bag_paths.append(child)

    return bag_paths


def output_wav_path(bag_path, input_dir, output_dir, suffix, timestamp_ns=None):
    bag_path = Path(bag_path)
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    if bag_path.is_file():
        name = bag_path.stem
    elif bag_path == input_dir:
        db3_files = sorted(bag_path.glob("*.db3"))
        name = db3_files[0].stem if db3_files else bag_path.name
    else:
        name = bag_path.name

    timestamp_part = f"_{timestamp_ns}" if timestamp_ns is not None else ""
    return output_dir / f"{name}{suffix}{timestamp_part}.wav"


def batch_export(input_dir, output_dir, topic, selected_channels, recursive, suffix, timestamp_in_name):
    bags = discover_bags(input_dir=input_dir, recursive=recursive)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not bags:
        raise RuntimeError(f"No ROS 2 .db3 bags found under: {input_dir}")

    print(f"Found {len(bags)} bag(s).")

    failures = []
    metadata_rows = []
    for index, bag_path in enumerate(bags, start=1):
        temp_wav_path = output_wav_path(
            bag_path=bag_path,
            input_dir=input_dir,
            output_dir=output_dir,
            suffix=suffix,
        )

        print(f"[{index}/{len(bags)}] Exporting {bag_path} -> {temp_wav_path}")
        try:
            metadata = export_audio(
                bag_path=str(bag_path),
                topic=topic,
                output_wav=str(temp_wav_path),
                selected_channels=selected_channels,
            )
            final_wav_path = temp_wav_path
            if timestamp_in_name:
                final_wav_path = output_wav_path(
                    bag_path=bag_path,
                    input_dir=input_dir,
                    output_dir=output_dir,
                    suffix=suffix,
                    timestamp_ns=metadata["first_stamp_ns"],
                )
                if final_wav_path != temp_wav_path:
                    temp_wav_path.replace(final_wav_path)

            metadata["output_wav"] = str(final_wav_path)
            metadata_rows.append(metadata)
        except Exception as exc:
            failures.append((bag_path, exc))
            print(f"Failed: {bag_path}: {exc}")

    write_metadata_csv(output_dir / "audio_timestamps.csv", metadata_rows)

    print(f"Batch export finished. Success: {len(bags) - len(failures)}, Failed: {len(failures)}")

    if failures:
        print("Failures:")
        for bag_path, exc in failures:
            print(f"- {bag_path}: {exc}")


def write_metadata_csv(csv_path, rows):
    file_exists = csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "bag",
                "sqlite_database",
                "topic",
                "output_wav",
                "sample_rate",
                "channels",
                "frames",
                "saved_shape",
                "first_stamp_ns",
                "last_stamp_ns",
            ],
            extrasaction="ignore",
        )
        if not file_exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"Appended timestamp index: {csv_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing ROS 2 bag directories or .db3 files.",
    )
    parser.add_argument(
        "--output-dir",
        default="wav_exports",
        help="Directory for exported WAV files.",
    )
    parser.add_argument("--topic", default="/respeaker/audio_raw")
    parser.add_argument(
        "--channels",
        default="",
        help="Selected channel indices, e.g. '0,1'. Empty means all channels.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search for .db3 files under input-dir.",
    )
    parser.add_argument(
        "--suffix",
        default="",
        help="Suffix appended to each exported WAV file name.",
    )
    parser.add_argument(
        "--timestamp-in-name",
        dest="timestamp_in_name",
        action="store_true",
        default=True,
        help="Append the first audio message timestamp ns to each WAV filename. Enabled by default.",
    )
    parser.add_argument(
        "--no-timestamp-in-name",
        dest="timestamp_in_name",
        action="store_false",
        help="Do not append the first audio message timestamp ns to WAV filenames.",
    )

    args = parser.parse_args()
    selected_channels = parse_channel_list(args.channels)

    batch_export(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        topic=args.topic,
        selected_channels=selected_channels,
        recursive=args.recursive,
        suffix=args.suffix,
        timestamp_in_name=args.timestamp_in_name,
    )


if __name__ == "__main__":
    main()

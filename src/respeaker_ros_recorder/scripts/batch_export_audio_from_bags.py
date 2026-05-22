#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import rosbag

from export_audio_from_bag import export_audio, parse_channel_list


def iter_bag_files(bag_dir, recursive=False):
    pattern = "**/*.bag" if recursive else "*.bag"
    return sorted(path for path in bag_dir.glob(pattern) if path.is_file())


def get_first_audio_stamp(bag_path, topic):
    with rosbag.Bag(str(bag_path), "r") as bag:
        for _, msg, _ in bag.read_messages(topics=[topic]):
            return msg.header.stamp.to_sec()

    raise RuntimeError(f"No audio messages found on topic: {topic}")


def format_stamp_for_filename(stamp):
    return f"{stamp:.9f}".rstrip("0").rstrip(".")


def output_path_for_bag(bag_path, bag_dir, out_dir, first_stamp, recursive=False):
    output_name = f"{bag_path.stem}_{format_stamp_for_filename(first_stamp)}.wav"

    if recursive:
        relative_parent = bag_path.relative_to(bag_dir).parent
        return out_dir / relative_parent / output_name

    return out_dir / output_name


def batch_export(args):
    bag_dir = Path(args.bag_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else bag_dir / "wav_exports"

    if not bag_dir.exists():
        raise FileNotFoundError(f"Bag directory does not exist: {bag_dir}")
    if not bag_dir.is_dir():
        raise NotADirectoryError(f"--bag-dir must be a directory: {bag_dir}")

    bag_files = iter_bag_files(bag_dir, recursive=args.recursive)
    if not bag_files:
        raise RuntimeError(f"No .bag files found in: {bag_dir}")

    selected_channels = parse_channel_list(args.channels)

    ok_count = 0
    fail_count = 0
    skip_count = 0

    print(f"Found {len(bag_files)} bag file(s).")
    print(f"Topic: {args.topic}")
    print(f"Output directory: {out_dir}")

    for index, bag_path in enumerate(bag_files, start=1):
        print("")
        print(f"[{index}/{len(bag_files)}] {bag_path}")

        try:
            first_stamp = get_first_audio_stamp(bag_path, args.topic)
            wav_path = output_path_for_bag(
                bag_path=bag_path,
                bag_dir=bag_dir,
                out_dir=out_dir,
                first_stamp=first_stamp,
                recursive=args.recursive,
            )

            if wav_path.exists() and not args.overwrite:
                print(f"Skip existing wav: {wav_path}")
                skip_count += 1
                continue

            wav_path.parent.mkdir(parents=True, exist_ok=True)

            export_audio(
                bag_path=str(bag_path),
                topic=args.topic,
                output_wav=str(wav_path),
                selected_channels=selected_channels,
            )
            ok_count += 1
        except Exception as exc:
            fail_count += 1
            print(f"Failed: {exc}", file=sys.stderr)
            if not args.keep_going:
                raise

    print("")
    print("Batch export finished.")
    print(f"Exported: {ok_count}")
    print(f"Skipped: {skip_count}")
    print(f"Failed: {fail_count}")

    return 1 if fail_count else 0


def main():
    parser = argparse.ArgumentParser(
        description="Batch export ReSpeaker audio WAV files from ROS1 bag files."
    )
    parser.add_argument("--bag-dir", required=True, help="Directory containing .bag files")
    parser.add_argument(
        "--out-dir",
        default="",
        help="Output directory. Default: <bag-dir>/wav_exports",
    )
    parser.add_argument("--topic", default="/respeaker/audio_raw", help="Audio topic name")
    parser.add_argument(
        "--channels",
        default="",
        help="Selected channel indices, e.g. '0,1'. Empty means all channels.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search .bag files recursively and mirror subdirectories in the output.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing wav files.",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Continue exporting other bags if one bag fails.",
    )

    args = parser.parse_args()
    return batch_export(args)


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
import argparse
import shutil
import tempfile
from pathlib import Path

from extract_ros2_bag_data import DEFAULT_TOPICS
from extract_ros2_bag_data import discover_bags
from extract_ros2_bag_data import extract_bag
from extract_ros2_bag_data import parse_topic_labels
from sync_audio_with_extracted_data import sync_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag", required=True, help="ROS2 bag directory, .db3 file, or directory containing bags.")
    parser.add_argument("--audio", required=True, help="Audio WAV file or directory containing WAV files.")
    parser.add_argument(
        "--audio-start-ns",
        type=int,
        default=None,
        help="Audio start timestamp in ns. If omitted, parse from WAV filename.",
    )
    parser.add_argument("--output", required=True, help="Output synchronized sample directory.")
    parser.add_argument("--recursive", action="store_true", help="Recursively search for bags and WAV files.")
    parser.add_argument(
        "--topics",
        default="color,depth,lio_odom,lio_robo_odom",
        help=(
            "Comma-separated labels to extract before syncing. Available labels: "
            "color,depth,lio_odom,lio_robo_odom,livox"
        ),
    )
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
    parser.add_argument(
        "--keep-extracted",
        action="store_true",
        help="Keep intermediate extracted files under output/_extracted.",
    )
    parser.add_argument("--color-topic", default=DEFAULT_TOPICS["color"])
    parser.add_argument("--depth-topic", default=DEFAULT_TOPICS["depth"])
    parser.add_argument("--lio-odom-topic", default=DEFAULT_TOPICS["lio_odom"])
    parser.add_argument("--lio-robo-odom-topic", default=DEFAULT_TOPICS["lio_robo_odom"])
    parser.add_argument("--livox-topic", default=DEFAULT_TOPICS["livox"])
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_topics = {
        "color": args.color_topic,
        "depth": args.depth_topic,
        "lio_odom": args.lio_odom_topic,
        "lio_robo_odom": args.lio_robo_odom_topic,
        "livox": args.livox_topic,
    }
    selected_labels = parse_topic_labels(args.topics)
    topics = {label: all_topics[label] for label in selected_labels}

    pairs = pair_bags_and_audio(args.bag, args.audio, recursive=args.recursive)
    print(f"Found {len(pairs)} bag/audio pair(s).")

    for bag_db, audio_path in pairs:
        bag_name = bag_name_from_db(bag_db)
        pair_output_dir = output_dir / bag_name if len(pairs) > 1 else output_dir
        extracted_parent = pair_output_dir / "_extracted" if args.keep_extracted else None

        if extracted_parent is not None:
            extracted_parent.mkdir(parents=True, exist_ok=True)
            extracted_dir = extracted_parent / bag_name
            extract_and_sync(args, bag_db, audio_path, extracted_dir, pair_output_dir, topics)
        else:
            with tempfile.TemporaryDirectory(prefix="bag_extract_") as tmp:
                extracted_dir = Path(tmp) / bag_name
                extract_and_sync(args, bag_db, audio_path, extracted_dir, pair_output_dir, topics)


def pair_bags_and_audio(bag_input, audio_input, recursive=False):
    bags = discover_bags(bag_input, recursive=recursive)
    audio_files = discover_audio_files(audio_input, recursive=recursive)

    if not bags:
        raise RuntimeError(f"No bags found from --bag: {bag_input}")
    if not audio_files:
        raise RuntimeError(f"No WAV files found from --audio: {audio_input}")
    if len(bags) != len(audio_files):
        raise RuntimeError(
            f"Bag/audio count mismatch: {len(bags)} bag(s), {len(audio_files)} wav(s). "
            "For batch mode, files are paired by sorted order."
        )

    return list(zip(sorted(bags), sorted(audio_files)))


def discover_audio_files(audio_input, recursive=False):
    path = Path(audio_input)
    if path.is_file():
        return [path] if path.suffix.lower() == ".wav" else []
    if not path.is_dir():
        return []

    pattern = "**/*.wav" if recursive else "*.wav"
    return sorted(path.glob(pattern))


def extract_and_sync(args, bag_db, audio_path, extracted_dir, output_dir, topics):
    if extracted_dir.exists():
        shutil.rmtree(extracted_dir)
    extracted_dir.mkdir(parents=True, exist_ok=True)

    print(f"Extracting bag data: {bag_db} -> {extracted_dir}")
    extract_bag(
        bag_db=bag_db,
        output_dir=extracted_dir,
        topics=topics,
        start_ns=args.start_ns,
        end_ns=args.end_ns,
    )

    print(f"Generating synchronized samples -> {output_dir}")
    sync_dataset(
        extracted_dir=extracted_dir,
        audio_path=Path(audio_path),
        audio_start_ns=args.audio_start_ns,
        output_dir=output_dir,
        start_ns=args.start_ns,
        end_ns=args.end_ns,
        segment_sec=args.segment_sec,
        hop_sec=args.hop_sec if args.hop_sec is not None else args.segment_sec,
        max_diff_ns=int(args.max_diff_sec * 1_000_000_000),
    )


def bag_name_from_db(db_path):
    db_path = Path(db_path)
    return db_path.parent.name if db_path.parent.name else db_path.stem


if __name__ == "__main__":
    main()

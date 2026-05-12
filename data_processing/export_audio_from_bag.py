#!/usr/bin/env python3
import argparse
import csv
import sqlite3
from pathlib import Path

import numpy as np
import soundfile as sf
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


def export_audio(bag_path, topic, output_wav, selected_channels=None):
    try:
        import rosbag2_py
    except ImportError:
        return export_audio_from_sqlite_bag(
            bag_path=bag_path,
            topic=topic,
            output_wav=output_wav,
            selected_channels=selected_channels,
        )

    reader = rosbag2_py.SequentialReader()
    storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id="sqlite3")
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr",
    )
    reader.open(storage_options, converter_options)

    topic_types = {
        metadata.name: metadata.type
        for metadata in reader.get_all_topics_and_types()
    }
    if topic not in topic_types:
        raise RuntimeError(f"No topic named {topic} found in bag: {bag_path}")

    msg_type = get_message(topic_types[topic])
    all_audio = []
    sample_rate = None
    channels = None
    first_stamp_ns = None
    last_stamp_ns = None

    while reader.has_next():
        topic_name, serialized_data, _ = reader.read_next()
        if topic_name != topic:
            continue

        msg = deserialize_message(serialized_data, msg_type)
        if sample_rate is None:
            sample_rate = int(msg.sample_rate)
            channels = int(msg.channels)
            first_stamp_ns = stamp_to_ns(msg.header.stamp)

        last_stamp_ns = stamp_to_ns(msg.header.stamp)

        audio = np.array(msg.data, dtype=np.int16)
        audio = audio.reshape(-1, channels)

        if selected_channels is not None:
            audio = audio[:, selected_channels]

        all_audio.append(audio)

    if not all_audio:
        raise RuntimeError(f"No audio messages found on topic: {topic}")

    output_wav = resolve_output_wav(output_wav, first_stamp_ns)
    audio_all = np.concatenate(all_audio, axis=0)
    sf.write(output_wav, audio_all, sample_rate)

    print("Export finished.")
    print(f"Bag: {bag_path}")
    print(f"Topic: {topic}")
    print(f"Output wav: {output_wav}")
    print(f"Sample rate: {sample_rate}")
    print(f"Original channels: {channels}")
    print(f"Saved shape: {audio_all.shape}")
    print(f"First stamp ns: {first_stamp_ns}")
    print(f"Last stamp ns: {last_stamp_ns}")

    return {
        "bag": str(bag_path),
        "topic": topic,
        "output_wav": str(output_wav),
        "sample_rate": sample_rate,
        "channels": channels,
        "frames": int(audio_all.shape[0]),
        "saved_shape": tuple(audio_all.shape),
        "first_stamp_ns": first_stamp_ns,
        "last_stamp_ns": last_stamp_ns,
    }


def export_audio_from_sqlite_bag(bag_path, topic, output_wav, selected_channels=None):
    db3_path = find_sqlite_bag_file(bag_path)

    with sqlite3.connect(str(db3_path)) as conn:
        topic_row = conn.execute(
            "SELECT id, type FROM topics WHERE name = ?",
            (topic,),
        ).fetchone()
        if topic_row is None:
            raise RuntimeError(f"No topic named {topic} found in bag: {bag_path}")

        topic_id, topic_type = topic_row
        msg_type = get_message(topic_type)

        rows = conn.execute(
            "SELECT timestamp, data FROM messages WHERE topic_id = ? ORDER BY timestamp",
            (topic_id,),
        )

        all_audio = []
        sample_rate = None
        channels = None
        first_stamp_ns = None
        last_stamp_ns = None

        for _, serialized_data in rows:
            msg = deserialize_message(serialized_data, msg_type)
            if sample_rate is None:
                sample_rate = int(msg.sample_rate)
                channels = int(msg.channels)
                first_stamp_ns = stamp_to_ns(msg.header.stamp)

            last_stamp_ns = stamp_to_ns(msg.header.stamp)

            audio = np.array(msg.data, dtype=np.int16)
            audio = audio.reshape(-1, channels)

            if selected_channels is not None:
                audio = audio[:, selected_channels]

            all_audio.append(audio)

    if not all_audio:
        raise RuntimeError(f"No audio messages found on topic: {topic}")

    output_wav = resolve_output_wav(output_wav, first_stamp_ns)
    audio_all = np.concatenate(all_audio, axis=0)
    sf.write(output_wav, audio_all, sample_rate)

    print("Export finished.")
    print(f"Bag: {bag_path}")
    print(f"SQLite database: {db3_path}")
    print(f"Topic: {topic}")
    print(f"Output wav: {output_wav}")
    print(f"Sample rate: {sample_rate}")
    print(f"Original channels: {channels}")
    print(f"Saved shape: {audio_all.shape}")
    print(f"First stamp ns: {first_stamp_ns}")
    print(f"Last stamp ns: {last_stamp_ns}")

    return {
        "bag": str(bag_path),
        "sqlite_database": str(db3_path),
        "topic": topic,
        "output_wav": str(output_wav),
        "sample_rate": sample_rate,
        "channels": channels,
        "frames": int(audio_all.shape[0]),
        "saved_shape": tuple(audio_all.shape),
        "first_stamp_ns": first_stamp_ns,
        "last_stamp_ns": last_stamp_ns,
    }


def find_sqlite_bag_file(bag_path):
    path = Path(bag_path)
    if path.is_file() and path.suffix == ".db3":
        return path

    if not path.is_dir():
        raise RuntimeError(f"Bag path is not a directory or .db3 file: {bag_path}")

    db3_files = sorted(path.glob("*.db3"))
    if not db3_files:
        raise RuntimeError(f"No .db3 file found in ROS 2 bag directory: {bag_path}")

    return db3_files[0]


def discover_bags(bag_input, recursive=False):
    path = Path(bag_input)
    if path.is_file() and path.suffix == ".db3":
        return [path]
    if not path.is_dir():
        raise RuntimeError(f"Bag input is not a directory or .db3 file: {bag_input}")

    if list(path.glob("*.db3")):
        return [path]

    pattern = "**/*.db3" if recursive else "*/*.db3"
    bag_dirs = sorted({db3.parent for db3 in path.glob(pattern)})
    if not bag_dirs:
        raise RuntimeError(f"No ROS 2 bag .db3 files found under: {bag_input}")
    return bag_dirs


def resolve_output_wav(output_wav, first_stamp_ns):
    if first_stamp_ns is None:
        raise RuntimeError("Cannot build timestamped wav name because first audio stamp is missing.")

    output_path = Path(output_wav)
    filename = f"bag_{first_stamp_ns}.wav"

    if output_path.suffix.lower() == ".wav":
        output_path = output_path.parent / filename
    else:
        output_path = output_path / filename

    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def resolve_report_path(output_path):
    output_path = Path(output_path)
    if output_path.suffix.lower() == ".wav":
        return output_path.parent / "audio_export_report.csv"
    return output_path / "audio_export_report.csv"


def parse_channel_list(channel_str):
    if channel_str is None or channel_str == "":
        return None

    return [int(x) for x in channel_str.split(",")]


def stamp_to_ns(stamp):
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def write_batch_report(report_path, rows):
    if not rows:
        return
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "bag",
        "topic",
        "output_wav",
        "sample_rate",
        "channels",
        "frames",
        "first_stamp_ns",
        "last_stamp_ns",
        "status",
        "error",
    ]
    with report_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bag",
        required=True,
        help="Input ROS 2 bag directory, .db3 file, or a parent directory containing multiple bags.",
    )
    parser.add_argument("--topic", default="/respeaker/audio_raw")
    parser.add_argument(
        "--out",
        default="wav_exports",
        help="Output directory or wav path. Final file name is always bag_<first_stamp_ns>.wav.",
    )
    parser.add_argument(
        "--channels",
        default="",
        help="Selected channel indices, e.g. '1,2,3,4'. Empty means all channels.",
    )
    parser.add_argument("--recursive", action="store_true", help="Recursively search for bags under --bag.")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue exporting remaining bags if one bag fails.",
    )
    parser.add_argument(
        "--report",
        default=None,
        help="Optional CSV report path. Default: <out>/audio_export_report.csv when exporting multiple bags.",
    )

    args = parser.parse_args()
    selected_channels = parse_channel_list(args.channels)

    bag_paths = discover_bags(args.bag, recursive=args.recursive)
    print(f"Found {len(bag_paths)} bag(s).")

    rows = []
    for index, bag_path in enumerate(bag_paths, start=1):
        print(f"\n[{index}/{len(bag_paths)}] Exporting: {bag_path}")
        try:
            result = export_audio(
                bag_path=bag_path,
                topic=args.topic,
                output_wav=args.out,
                selected_channels=selected_channels,
            )
            result["status"] = "ok"
            result["error"] = ""
            rows.append(result)
        except Exception as exc:
            if not args.continue_on_error:
                raise
            print(f"[WARN] Failed to export {bag_path}: {exc}")
            rows.append({
                "bag": str(bag_path),
                "topic": args.topic,
                "status": "failed",
                "error": str(exc),
            })

    report_path = args.report
    if report_path is None and len(bag_paths) > 1:
        report_path = resolve_report_path(args.out)
    if report_path:
        write_batch_report(report_path, rows)
        print(f"\nWrote report: {report_path}")


if __name__ == "__main__":
    main()

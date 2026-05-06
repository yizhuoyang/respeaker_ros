#!/usr/bin/env python3
import argparse
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
    first_stamp = None
    last_stamp = None

    while reader.has_next():
        topic_name, serialized_data, _ = reader.read_next()
        if topic_name != topic:
            continue

        msg = deserialize_message(serialized_data, msg_type)
        if sample_rate is None:
            sample_rate = int(msg.sample_rate)
            channels = int(msg.channels)
            first_stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        last_stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        audio = np.array(msg.data, dtype=np.int16)
        audio = audio.reshape(-1, channels)

        if selected_channels is not None:
            audio = audio[:, selected_channels]

        all_audio.append(audio)

    if not all_audio:
        raise RuntimeError(f"No audio messages found on topic: {topic}")

    audio_all = np.concatenate(all_audio, axis=0)
    sf.write(output_wav, audio_all, sample_rate)

    print("Export finished.")
    print(f"Bag: {bag_path}")
    print(f"Topic: {topic}")
    print(f"Output wav: {output_wav}")
    print(f"Sample rate: {sample_rate}")
    print(f"Original channels: {channels}")
    print(f"Saved shape: {audio_all.shape}")
    print(f"First stamp: {first_stamp}")
    print(f"Last stamp: {last_stamp}")


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
        first_stamp = None
        last_stamp = None

        for _, serialized_data in rows:
            msg = deserialize_message(serialized_data, msg_type)
            if sample_rate is None:
                sample_rate = int(msg.sample_rate)
                channels = int(msg.channels)
                first_stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

            last_stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

            audio = np.array(msg.data, dtype=np.int16)
            audio = audio.reshape(-1, channels)

            if selected_channels is not None:
                audio = audio[:, selected_channels]

            all_audio.append(audio)

    if not all_audio:
        raise RuntimeError(f"No audio messages found on topic: {topic}")

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
    print(f"First stamp: {first_stamp}")
    print(f"Last stamp: {last_stamp}")


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


def parse_channel_list(channel_str):
    if channel_str is None or channel_str == "":
        return None

    return [int(x) for x in channel_str.split(",")]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag", required=True, help="Input ROS 2 bag directory")
    parser.add_argument("--topic", default="/respeaker/audio_raw")
    parser.add_argument("--out", default="respeaker_audio.wav")
    parser.add_argument(
        "--channels",
        default="",
        help="Selected channel indices, e.g. '1,2,3,4'. Empty means all channels.",
    )

    args = parser.parse_args()
    selected_channels = parse_channel_list(args.channels)

    export_audio(
        bag_path=args.bag,
        topic=args.topic,
        output_wav=args.out,
        selected_channels=selected_channels,
    )


if __name__ == "__main__":
    main()

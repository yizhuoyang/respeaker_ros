#!/usr/bin/env python3
import argparse
import rosbag
import numpy as np
import soundfile as sf


def export_audio(bag_path, topic, output_wav, selected_channels=None):
    all_audio = []
    sample_rate = None
    channels = None
    first_stamp = None
    last_stamp = None

    with rosbag.Bag(bag_path, "r") as bag:
        for _, msg, _ in bag.read_messages(topics=[topic]):
            if sample_rate is None:
                sample_rate = int(msg.sample_rate)
                channels = int(msg.channels)
                first_stamp = msg.header.stamp.to_sec()

            last_stamp = msg.header.stamp.to_sec()

            audio = np.array(msg.data, dtype=np.int16)
            audio = audio.reshape(-1, channels)

            if selected_channels is not None:
                audio = audio[:, selected_channels]

            all_audio.append(audio)

    if len(all_audio) == 0:
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


def parse_channel_list(channel_str):
    if channel_str is None or channel_str == "":
        return None

    return [int(x) for x in channel_str.split(",")]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag", required=True, help="Input rosbag path")
    parser.add_argument("--topic", default="/respeaker/audio_raw")
    parser.add_argument("--out", default="respeaker_audio.wav")
    parser.add_argument(
        "--channels",
        default="",
        help="Selected channel indices, e.g. '1,2,3,4'. Empty means all channels."
    )

    args = parser.parse_args()

    selected_channels = parse_channel_list(args.channels)

    export_audio(
        bag_path=args.bag,
        topic=args.topic,
        output_wav=args.out,
        selected_channels=selected_channels
    )

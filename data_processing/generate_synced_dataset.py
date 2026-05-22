#!/usr/bin/env python3
import argparse
import csv
import json
import math
import re
import sys
from bisect import bisect_left
from pathlib import Path

import numpy as np
import rosbag
import soundfile as sf


STAMP_RE = re.compile(r"_(\d+(?:\.\d+)?)$")


def stamp_to_sec(stamp):
    return float(stamp.secs) + float(stamp.nsecs) * 1e-9


def parse_wav_start_time(wav_path):
    match = STAMP_RE.search(wav_path.stem)
    if match is None:
        raise ValueError(
            f"Cannot parse audio start timestamp from wav name: {wav_path.name}. "
            "Expected something like bag_name_1716361234.56789.wav"
        )
    return float(match.group(1))


def find_bag_for_wav(wav_path, bag_by_stem):
    candidates = []
    for stem, bag_path in bag_by_stem.items():
        if wav_path.stem == stem or wav_path.stem.startswith(f"{stem}_"):
            candidates.append((len(stem), bag_path))

    if not candidates:
        return None

    candidates.sort(reverse=True)
    return candidates[0][1]


def image_msg_to_numpy(msg):
    height = int(msg.height)
    width = int(msg.width)
    encoding = msg.encoding.lower()

    dtype_by_encoding = {
        "mono8": np.uint8,
        "8uc1": np.uint8,
        "bgr8": np.uint8,
        "rgb8": np.uint8,
        "rgba8": np.uint8,
        "bgra8": np.uint8,
        "mono16": np.uint16,
        "16uc1": np.uint16,
        "32fc1": np.float32,
    }
    channels_by_encoding = {
        "mono8": 1,
        "8uc1": 1,
        "bgr8": 3,
        "rgb8": 3,
        "rgba8": 4,
        "bgra8": 4,
        "mono16": 1,
        "16uc1": 1,
        "32fc1": 1,
    }

    if encoding not in dtype_by_encoding:
        raise ValueError(f"Unsupported image encoding: {msg.encoding}")

    dtype = dtype_by_encoding[encoding]
    channels = channels_by_encoding[encoding]
    data = np.frombuffer(msg.data, dtype=dtype)

    if msg.is_bigendian and data.dtype.byteorder != ">":
        data = data.byteswap().newbyteorder()

    row_values = int(msg.step) // np.dtype(dtype).itemsize
    data = data.reshape(height, row_values)
    data = data[:, : width * channels]

    if channels == 1:
        return data.reshape(height, width).copy()

    return data.reshape(height, width, channels).copy()


def depth_array_for_png(depth, depth_scale):
    if depth.dtype == np.float32 or depth.dtype == np.float64:
        depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        depth = np.clip(depth * depth_scale, 0, np.iinfo(np.uint16).max)
        return depth.astype(np.uint16)

    if depth.dtype == np.uint16 or depth.dtype == np.uint8:
        return depth

    if np.issubdtype(depth.dtype, np.integer):
        depth = np.clip(depth, 0, np.iinfo(np.uint16).max)
        return depth.astype(np.uint16)

    raise ValueError(f"Unsupported depth dtype for png: {depth.dtype}")


def save_png(path, array, encoding=None):
    encoding = (encoding or "").lower()

    try:
        import cv2

        image = array
        if array.ndim == 3 and encoding in ("rgb8", "rgba8"):
            code = cv2.COLOR_RGB2BGR if encoding == "rgb8" else cv2.COLOR_RGBA2BGRA
            image = cv2.cvtColor(array, code)

        if not cv2.imwrite(str(path), image):
            raise RuntimeError(f"cv2.imwrite failed for: {path}")
        return
    except ImportError:
        pass

    try:
        from PIL import Image

        image = array
        if array.ndim == 3 and encoding == "bgr8":
            image = array[:, :, ::-1]
        elif array.ndim == 3 and encoding == "bgra8":
            image = array[:, :, [2, 1, 0, 3]]

        Image.fromarray(image).save(str(path))
    except ImportError as exc:
        raise RuntimeError("Saving png requires either opencv-python or Pillow") from exc


def odom_msg_to_numpy(msg):
    pose = msg.pose.pose
    twist = msg.twist.twist
    return np.array(
        [
            stamp_to_sec(msg.header.stamp),
            pose.position.x,
            pose.position.y,
            pose.position.z,
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
            twist.linear.x,
            twist.linear.y,
            twist.linear.z,
            twist.angular.x,
            twist.angular.y,
            twist.angular.z,
        ],
        dtype=np.float64,
    )


def collect_topic_messages(bag_path, topics):
    topic_data = {name: [] for name in topics}

    with rosbag.Bag(str(bag_path), "r") as bag:
        for topic, msg, _ in bag.read_messages(topics=list(topics.values())):
            for name, topic_name in topics.items():
                if topic == topic_name:
                    topic_data[name].append((stamp_to_sec(msg.header.stamp), msg))
                    break

    return topic_data


def nearest_message(topic_data, target_time, tolerance):
    if not topic_data:
        return None, None, None

    stamps = [item[0] for item in topic_data]
    index = bisect_left(stamps, target_time)
    candidates = []

    if index < len(topic_data):
        candidates.append(topic_data[index])
    if index > 0:
        candidates.append(topic_data[index - 1])

    best_stamp, best_msg = min(candidates, key=lambda item: abs(item[0] - target_time))
    delta = abs(best_stamp - target_time)
    if tolerance is not None and delta > tolerance:
        return None, None, delta

    return best_stamp, best_msg, delta


def write_metadata(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "index",
                "bag",
                "wav",
                "audio_start",
                "audio_end",
                "sync_time",
                "image_time",
                "depth_time",
                "odom_time",
                "image_delta",
                "depth_delta",
                "odom_delta",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def save_config(path, args):
    config = vars(args).copy()
    path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")


def make_output_dirs(root):
    dirs = {
        "audio": root / "audio",
        "image": root / "image",
        "depth": root / "depth",
        "odom": root / "odom",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def process_pair(wav_path, bag_path, out_root, args):
    audio, sample_rate = sf.read(str(wav_path), always_2d=True, dtype="float32")
    audio_start_time = parse_wav_start_time(wav_path)

    segment_samples = int(round(args.segment_sec * sample_rate))
    hop_samples = int(round(args.hop_sec * sample_rate))
    if segment_samples <= 0:
        raise ValueError("--segment-sec is too small")
    if hop_samples <= 0:
        raise ValueError("--hop-sec is too small")
    if audio.shape[0] < segment_samples:
        raise ValueError(f"Audio is shorter than one segment: {wav_path}")

    topics = {
        "image": args.image_topic,
        "depth": args.depth_topic,
        "odom": args.odom_topic,
    }
    topic_data = collect_topic_messages(bag_path, topics)

    bag_out = out_root / bag_path.stem
    dirs = make_output_dirs(bag_out)

    metadata_rows = []
    total_segments = 1 + (audio.shape[0] - segment_samples) // hop_samples
    saved_count = 0

    for seg_index in range(total_segments):
        start_sample = seg_index * hop_samples
        end_sample = start_sample + segment_samples
        audio_start = audio_start_time + float(start_sample) / float(sample_rate)
        audio_end = audio_start_time + float(end_sample) / float(sample_rate)
        sync_time = (audio_start + audio_end) * 0.5

        image_time, image_msg, image_delta = nearest_message(
            topic_data["image"], sync_time, args.max_delta_sec
        )
        depth_time, depth_msg, depth_delta = nearest_message(
            topic_data["depth"], sync_time, args.max_delta_sec
        )
        odom_time, odom_msg, odom_delta = nearest_message(
            topic_data["odom"], sync_time, args.max_delta_sec
        )

        if image_msg is None or depth_msg is None or odom_msg is None:
            if args.keep_incomplete:
                pass
            else:
                continue

        saved_count += 1
        stem = f"{saved_count:0{args.index_width}d}"

        sf.write(dirs["audio"] / f"{stem}.wav", audio[start_sample:end_sample], sample_rate)
        if image_msg is not None:
            save_png(
                dirs["image"] / f"{stem}.png",
                image_msg_to_numpy(image_msg),
                encoding=image_msg.encoding,
            )
        if depth_msg is not None:
            depth = image_msg_to_numpy(depth_msg)
            save_png(
                dirs["depth"] / f"{stem}.png",
                depth_array_for_png(depth, args.depth_scale),
                encoding=depth_msg.encoding,
            )
        if odom_msg is not None:
            np.save(dirs["odom"] / f"{stem}.npy", odom_msg_to_numpy(odom_msg))

        metadata_rows.append(
            {
                "index": saved_count,
                "bag": str(bag_path),
                "wav": str(wav_path),
                "audio_start": f"{audio_start:.9f}",
                "audio_end": f"{audio_end:.9f}",
                "sync_time": f"{sync_time:.9f}",
                "image_time": "" if image_time is None else f"{image_time:.9f}",
                "depth_time": "" if depth_time is None else f"{depth_time:.9f}",
                "odom_time": "" if odom_time is None else f"{odom_time:.9f}",
                "image_delta": "" if image_delta is None or math.isnan(image_delta) else f"{image_delta:.9f}",
                "depth_delta": "" if depth_delta is None or math.isnan(depth_delta) else f"{depth_delta:.9f}",
                "odom_delta": "" if odom_delta is None or math.isnan(odom_delta) else f"{odom_delta:.9f}",
            }
        )

    write_metadata(bag_out / "metadata.csv", metadata_rows)
    return total_segments, saved_count


def main():
    parser = argparse.ArgumentParser(
        description="Generate synchronized audio/image/depth/odom samples from WAV files and ROS1 bags."
    )
    parser.add_argument("--wav-dir", required=True, help="Directory containing exported wav files")
    parser.add_argument("--bag-dir", required=True, help="Directory containing matching .bag files")
    parser.add_argument("--out-dir", default="data", help="Output root directory")
    parser.add_argument("--segment-sec", type=float, default=1.0, help="Audio segment length in seconds")
    parser.add_argument("--hop-sec", type=float, default=0.1, help="Sliding window hop in seconds")
    parser.add_argument("--max-delta-sec", type=float, default=0.05, help="Max allowed sensor time delta")
    parser.add_argument("--image-topic", default="/camera/color/image_raw")
    parser.add_argument("--depth-topic", default="/camera/depth/image_raw")
    parser.add_argument("--odom-topic", default="/odom")
    parser.add_argument(
        "--depth-scale",
        type=float,
        default=1000.0,
        help="Scale for float depth images before uint16 png saving, e.g. meters to millimeters.",
    )
    parser.add_argument("--recursive", action="store_true", help="Search wav and bag directories recursively")
    parser.add_argument("--keep-incomplete", action="store_true", help="Save samples even if one modality is missing")
    parser.add_argument("--keep-going", action="store_true", help="Continue if one wav/bag pair fails")
    parser.add_argument("--index-width", type=int, default=4, help="Filename zero-padding width")

    args = parser.parse_args()

    wav_dir = Path(args.wav_dir).expanduser().resolve()
    bag_dir = Path(args.bag_dir).expanduser().resolve()
    out_root = Path(args.out_dir).expanduser().resolve()

    wav_pattern = "**/*.wav" if args.recursive else "*.wav"
    bag_pattern = "**/*.bag" if args.recursive else "*.bag"
    wav_files = sorted(wav_dir.glob(wav_pattern))
    bag_files = sorted(bag_dir.glob(bag_pattern))
    bag_by_stem = {path.stem: path for path in bag_files}

    if not wav_files:
        raise RuntimeError(f"No wav files found in: {wav_dir}")
    if not bag_files:
        raise RuntimeError(f"No bag files found in: {bag_dir}")

    out_root.mkdir(parents=True, exist_ok=True)
    save_config(out_root / "sync_config.json", args)

    ok_count = 0
    fail_count = 0
    missing_count = 0

    print(f"Found {len(wav_files)} wav file(s).")
    print(f"Found {len(bag_files)} bag file(s).")
    print(f"Output root: {out_root}")

    for wav_path in wav_files:
        bag_path = find_bag_for_wav(wav_path, bag_by_stem)
        if bag_path is None:
            missing_count += 1
            print(f"Missing matching bag for wav: {wav_path}", file=sys.stderr)
            continue

        print("")
        print(f"Processing wav: {wav_path}")
        print(f"Matching bag: {bag_path}")

        try:
            total, saved = process_pair(wav_path, bag_path, out_root, args)
            ok_count += 1
            print(f"Saved {saved}/{total} synchronized sample(s).")
        except Exception as exc:
            fail_count += 1
            print(f"Failed: {exc}", file=sys.stderr)
            if not args.keep_going:
                raise

    print("")
    print("Done.")
    print(f"Processed pairs: {ok_count}")
    print(f"Missing pairs: {missing_count}")
    print(f"Failed pairs: {fail_count}")

    return 1 if fail_count or missing_count else 0


if __name__ == "__main__":
    sys.exit(main())

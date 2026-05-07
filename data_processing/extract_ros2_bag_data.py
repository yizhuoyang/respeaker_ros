#!/usr/bin/env python3
import argparse
import csv
import json
import sqlite3
from pathlib import Path

import numpy as np
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


DEFAULT_TOPICS = {
    "color": "/camera/color/image_raw",
    "depth": "/camera/depth/image_raw",
    "lio_odom": "/lio/odom",
    "lio_robo_odom": "/lio/robo/odom",
    "livox": "/livox/lidar",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        required=True,
        help="ROS2 bag directory, .db3 file, or directory containing many bags.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory for extracted data.",
    )
    parser.add_argument("--recursive", action="store_true", help="Recursively search for bags.")
    parser.add_argument(
        "--topics",
        default="color,depth,lio_odom,lio_robo_odom,livox",
        help=(
            "Comma-separated labels to extract. Available labels: "
            "color,depth,lio_odom,lio_robo_odom,livox"
        ),
    )
    parser.add_argument("--start-ns", type=int, default=None, help="Start bag timestamp in ns.")
    parser.add_argument("--end-ns", type=int, default=None, help="End bag timestamp in ns.")
    parser.add_argument("--color-topic", default=DEFAULT_TOPICS["color"])
    parser.add_argument("--depth-topic", default=DEFAULT_TOPICS["depth"])
    parser.add_argument("--lio-odom-topic", default=DEFAULT_TOPICS["lio_odom"])
    parser.add_argument("--lio-robo-odom-topic", default=DEFAULT_TOPICS["lio_robo_odom"])
    parser.add_argument("--livox-topic", default=DEFAULT_TOPICS["livox"])
    args = parser.parse_args()

    all_topics = {
        "color": args.color_topic,
        "depth": args.depth_topic,
        "lio_odom": args.lio_odom_topic,
        "lio_robo_odom": args.lio_robo_odom_topic,
        "livox": args.livox_topic,
    }
    selected_labels = parse_topic_labels(args.topics)
    topics = {label: all_topics[label] for label in selected_labels}

    bags = discover_bags(args.input, recursive=args.recursive)
    if not bags:
        raise RuntimeError(f"No ROS2 .db3 bags found under: {args.input}")

    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(bags)} bag(s).")
    for index, bag_db in enumerate(bags, start=1):
        bag_name = bag_name_from_db(bag_db)
        bag_output = output_root / bag_name
        print(f"[{index}/{len(bags)}] Extracting {bag_db} -> {bag_output}")
        extract_bag(
            bag_db=bag_db,
            output_dir=bag_output,
            topics=topics,
            start_ns=args.start_ns,
            end_ns=args.end_ns,
        )


def parse_topic_labels(topic_labels):
    labels = [label.strip() for label in topic_labels.split(",") if label.strip()]
    unknown = sorted(set(labels) - set(DEFAULT_TOPICS))
    if unknown:
        raise RuntimeError(f"Unknown topic label(s): {','.join(unknown)}")
    return labels


def discover_bags(input_path, recursive=False):
    path = Path(input_path)
    if path.is_file():
        return [path] if path.suffix == ".db3" else []

    if not path.is_dir():
        return []

    patterns = ["**/*.db3"] if recursive else ["*.db3", "*/*.db3"]
    bags = []
    seen = set()
    for pattern in patterns:
        for db in sorted(path.glob(pattern)):
            key = db.resolve()
            if key not in seen:
                seen.add(key)
                bags.append(db)
    return bags


def bag_name_from_db(db_path):
    db_path = Path(db_path)
    if db_path.parent.name:
        return db_path.parent.name
    return db_path.stem


def extract_bag(bag_db, output_dir, topics, start_ns=None, end_ns=None):
    output_dir.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(str(bag_db)) as conn:
        topic_map = read_topic_map(conn)
        write_topic_manifest(output_dir, topic_map, topics, start_ns, end_ns)

        for label, topic_name in topics.items():
            resolved_topic_name = resolve_topic_name(topic_name, topic_map, label)
            if resolved_topic_name not in topic_map:
                print(f"  Skip missing topic: {topic_name}")
                continue

            topic_id, type_name = topic_map[resolved_topic_name]
            try:
                msg_type = get_message(type_name)
            except (AttributeError, ModuleNotFoundError, ValueError) as exc:
                print(f"  Skip {resolved_topic_name}: cannot import message type {type_name}: {exc}")
                continue

            if label in ("color", "depth"):
                extract_images(
                    conn,
                    topic_id,
                    msg_type,
                    type_name,
                    output_dir,
                    label,
                    resolved_topic_name,
                    start_ns,
                    end_ns,
                )
            elif label in ("lio_odom", "lio_robo_odom"):
                extract_odom(conn, topic_id, msg_type, output_dir, label, resolved_topic_name, start_ns, end_ns)
            elif label == "livox":
                extract_livox(conn, topic_id, msg_type, output_dir, label, resolved_topic_name, start_ns, end_ns)


def resolve_topic_name(topic_name, topic_map, label):
    if topic_name in topic_map:
        return topic_name

    if label in ("color", "depth"):
        compressed_topic = f"{topic_name}/compressed"
        if compressed_topic in topic_map:
            return compressed_topic

        compressed_depth_topic = f"{topic_name}/compressedDepth"
        if compressed_depth_topic in topic_map:
            return compressed_depth_topic

    return topic_name


def read_topic_map(conn):
    rows = conn.execute("SELECT id, name, type FROM topics ORDER BY id").fetchall()
    return {name: (topic_id, type_name) for topic_id, name, type_name in rows}


def write_topic_manifest(output_dir, topic_map, selected_topics, start_ns=None, end_ns=None):
    manifest = {
        "available_topics": {
            name: {"id": topic_id, "type": type_name}
            for name, (topic_id, type_name) in topic_map.items()
        },
        "selected_topics": selected_topics,
        "bag_timestamp_filter": {
            "start_ns": start_ns,
            "end_ns": end_ns,
        },
    }
    with (output_dir / "topics_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def read_messages(conn, topic_id, start_ns=None, end_ns=None):
    query = "SELECT timestamp, data FROM messages WHERE topic_id = ?"
    params = [topic_id]
    if start_ns is not None:
        query += " AND timestamp >= ?"
        params.append(start_ns)
    if end_ns is not None:
        query += " AND timestamp <= ?"
        params.append(end_ns)
    query += " ORDER BY timestamp"
    return conn.execute(query, params)


def extract_images(conn, topic_id, msg_type, type_name, output_dir, label, topic_name, start_ns=None, end_ns=None):
    image_dir = output_dir / label
    image_dir.mkdir(parents=True, exist_ok=True)
    index_csv = image_dir / "index.csv"
    is_compressed = type_name.endswith("/CompressedImage")

    with index_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "index",
            "bag_timestamp_ns",
            "msg_stamp_sec",
            "msg_stamp_nanosec",
            "encoding",
            "height",
            "width",
            "message_type",
            "format",
            "file",
        ])

        count = 0
        for count, (bag_stamp, serialized) in enumerate(
            read_messages(conn, topic_id, start_ns, end_ns),
            start=1,
        ):
            msg = deserialize_message(serialized, msg_type)
            stem = f"{count:06d}_{stamp_to_ns(msg.header.stamp)}"
            if is_compressed:
                saved_file, height, width = save_compressed_image(image_dir, stem, msg, label)
                encoding = "compressed"
                image_format = msg.format
            else:
                image = image_to_numpy(msg)
                saved_file = save_image_array(image_dir, stem, image, msg.encoding)
                height = msg.height
                width = msg.width
                encoding = msg.encoding
                image_format = ""
            writer.writerow([
                count,
                bag_stamp,
                msg.header.stamp.sec,
                msg.header.stamp.nanosec,
                encoding,
                height,
                width,
                type_name,
                image_format,
                saved_file.name,
            ])

    print(f"  {topic_name}: exported {count if 'count' in locals() else 0} image(s)")


def save_compressed_image(image_dir, stem, msg, label):
    decoded = decode_compressed_image(msg, label)
    if decoded is not None:
        path = image_dir / f"{stem}.png"
        try:
            import cv2
            cv2.imwrite(str(path), decoded)
            return path, decoded.shape[0], decoded.shape[1]
        except ImportError:
            path = image_dir / f"{stem}.npy"
            np.save(path, decoded)
            return path, decoded.shape[0], decoded.shape[1]

    ext = compressed_extension(msg.format)
    path = image_dir / f"{stem}{ext}"
    path.write_bytes(bytes(msg.data))
    return path, "", ""


def decode_compressed_image(msg, label):
    try:
        import cv2
    except ImportError:
        return None

    data = bytes(msg.data)
    image_bytes = extract_embedded_image_bytes(data)
    if image_bytes is None:
        image_bytes = data

    array = np.frombuffer(image_bytes, dtype=np.uint8)
    flag = cv2.IMREAD_UNCHANGED if label == "depth" else cv2.IMREAD_COLOR
    return cv2.imdecode(array, flag)


def extract_embedded_image_bytes(data):
    png_magic = b"\x89PNG\r\n\x1a\n"
    png_pos = data.find(png_magic)
    if png_pos >= 0:
        return data[png_pos:]

    jpeg_magic = b"\xff\xd8"
    jpeg_pos = data.find(jpeg_magic)
    if jpeg_pos >= 0:
        return data[jpeg_pos:]

    return None


def compressed_extension(image_format):
    fmt = image_format.lower()
    if "jpeg" in fmt or "jpg" in fmt:
        return ".jpg"
    if "png" in fmt or "compresseddepth" in fmt:
        return ".png"
    return ".bin"


def image_to_numpy(msg):
    encoding = msg.encoding.lower()
    dtype, channels = image_encoding_info(encoding)
    itemsize = np.dtype(dtype).itemsize
    row_bytes = msg.width * channels * itemsize

    raw = np.frombuffer(msg.data, dtype=np.uint8)
    rows = raw.reshape(msg.height, msg.step)
    rows = rows[:, :row_bytes]
    image = rows.reshape(msg.height, msg.width, channels, itemsize).copy()
    image = image.view(dtype).reshape(msg.height, msg.width, channels)

    if msg.is_bigendian:
        image = image.byteswap()

    if channels == 1:
        image = image[:, :, 0]

    return image


def image_encoding_info(encoding):
    if encoding in ("rgb8", "bgr8"):
        return np.uint8, 3
    if encoding in ("rgba8", "bgra8"):
        return np.uint8, 4
    if encoding in ("mono8", "8uc1"):
        return np.uint8, 1
    if encoding in ("mono16", "16uc1"):
        return np.uint16, 1
    if encoding == "32fc1":
        return np.float32, 1
    raise RuntimeError(f"Unsupported image encoding: {encoding}")


def save_image_array(image_dir, stem, image, encoding):
    encoding = encoding.lower()

    if encoding == "32fc1":
        path = image_dir / f"{stem}.npy"
        np.save(path, image)
        return path

    try:
        import cv2
    except ImportError:
        path = image_dir / f"{stem}.npy"
        np.save(path, image)
        return path

    output = image
    if encoding == "rgb8":
        output = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    elif encoding == "rgba8":
        output = cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)

    path = image_dir / f"{stem}.png"
    cv2.imwrite(str(path), output)
    return path


def extract_odom(conn, topic_id, msg_type, output_dir, label, topic_name, start_ns=None, end_ns=None):
    csv_path = output_dir / f"{label}.csv"
    count = 0

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "index",
            "bag_timestamp_ns",
            "stamp_sec",
            "stamp_nanosec",
            "frame_id",
            "child_frame_id",
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
        ])

        for count, (bag_stamp, serialized) in enumerate(
            read_messages(conn, topic_id, start_ns, end_ns),
            start=1,
        ):
            msg = deserialize_message(serialized, msg_type)
            p = msg.pose.pose.position
            q = msg.pose.pose.orientation
            linear = msg.twist.twist.linear
            angular = msg.twist.twist.angular
            writer.writerow([
                count,
                bag_stamp,
                msg.header.stamp.sec,
                msg.header.stamp.nanosec,
                msg.header.frame_id,
                msg.child_frame_id,
                p.x,
                p.y,
                p.z,
                q.x,
                q.y,
                q.z,
                q.w,
                linear.x,
                linear.y,
                linear.z,
                angular.x,
                angular.y,
                angular.z,
            ])

    print(f"  {topic_name}: exported {count} odom row(s) -> {csv_path.name}")


def extract_livox(conn, topic_id, msg_type, output_dir, label, topic_name, start_ns=None, end_ns=None):
    livox_dir = output_dir / label
    livox_dir.mkdir(parents=True, exist_ok=True)
    index_csv = livox_dir / "index.csv"

    with index_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "index",
            "bag_timestamp_ns",
            "stamp_sec",
            "stamp_nanosec",
            "timebase",
            "point_num",
            "lidar_id",
            "file",
        ])

        count = 0
        for count, (bag_stamp, serialized) in enumerate(
            read_messages(conn, topic_id, start_ns, end_ns),
            start=1,
        ):
            msg = deserialize_message(serialized, msg_type)
            stem = f"{count:06d}_{stamp_to_ns(msg.header.stamp)}"
            csv_file = livox_dir / f"{stem}.csv"
            write_livox_points(csv_file, msg)
            writer.writerow([
                count,
                bag_stamp,
                msg.header.stamp.sec,
                msg.header.stamp.nanosec,
                getattr(msg, "timebase", ""),
                getattr(msg, "point_num", len(getattr(msg, "points", []))),
                getattr(msg, "lidar_id", ""),
                csv_file.name,
            ])

    print(f"  {topic_name}: exported {count if 'count' in locals() else 0} lidar frame(s)")


def write_livox_points(csv_file, msg):
    with csv_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["offset_time", "x", "y", "z", "reflectivity", "tag", "line"])
        for point in msg.points:
            writer.writerow([
                getattr(point, "offset_time", ""),
                getattr(point, "x", ""),
                getattr(point, "y", ""),
                getattr(point, "z", ""),
                getattr(point, "reflectivity", ""),
                getattr(point, "tag", ""),
                getattr(point, "line", ""),
            ])


def stamp_to_ns(stamp):
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


if __name__ == "__main__":
    main()

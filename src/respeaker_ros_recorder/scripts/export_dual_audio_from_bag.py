#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

from export_audio_from_bag import export_audio, parse_channel_list


def export_dual_audio(
    bag_path,
    mic1_topic,
    mic2_topic,
    mic1_out,
    mic2_out,
    mic1_channels=None,
    mic2_channels=None,
):
    export_audio(
        bag_path=bag_path,
        topic=mic1_topic,
        output_wav=mic1_out,
        selected_channels=mic1_channels,
    )

    print("")

    export_audio(
        bag_path=bag_path,
        topic=mic2_topic,
        output_wav=mic2_out,
        selected_channels=mic2_channels,
    )


def iter_bag_files(bag_dir, recursive=False):
    pattern = "**/*.bag" if recursive else "*.bag"
    return sorted(path for path in bag_dir.glob(pattern) if path.is_file())


def default_output_paths(bag_path, out_dir, prefix, relative_parent=None):
    bag_stem = Path(bag_path).expanduser().stem
    output_prefix = prefix or bag_stem
    output_dir = Path(out_dir).expanduser() if out_dir else Path(".")

    if relative_parent is not None:
        output_dir = output_dir / relative_parent

    return (
        output_dir / f"{output_prefix}_mic1.wav",
        output_dir / f"{output_prefix}_mic2.wav",
    )


def export_one_bag(args, bag_path, mic1_out=None, mic2_out=None, relative_parent=None):
    default_mic1_out, default_mic2_out = default_output_paths(
        bag_path=bag_path,
        out_dir=args.out_dir,
        prefix=args.prefix if args.bag else "",
        relative_parent=relative_parent,
    )

    mic1_out = mic1_out or default_mic1_out
    mic2_out = mic2_out or default_mic2_out

    mic1_out.parent.mkdir(parents=True, exist_ok=True)
    mic2_out.parent.mkdir(parents=True, exist_ok=True)

    export_dual_audio(
        bag_path=str(bag_path),
        mic1_topic=args.mic1_topic,
        mic2_topic=args.mic2_topic,
        mic1_out=str(mic1_out),
        mic2_out=str(mic2_out),
        mic1_channels=parse_channel_list(args.mic1_channels),
        mic2_channels=parse_channel_list(args.mic2_channels),
    )

    print("")
    print("Dual export finished.")
    print(f"Mic 1 wav: {mic1_out}")
    print(f"Mic 2 wav: {mic2_out}")


def batch_export(args):
    bag_dir = Path(args.bag_dir).expanduser().resolve()
    if not bag_dir.exists():
        raise FileNotFoundError(f"Bag directory does not exist: {bag_dir}")
    if not bag_dir.is_dir():
        raise NotADirectoryError(f"--bag-dir must be a directory: {bag_dir}")

    args.out_dir = args.out_dir or str(bag_dir / "dual_wav_exports")
    bag_files = iter_bag_files(bag_dir, recursive=args.recursive)
    if not bag_files:
        raise RuntimeError(f"No .bag files found in: {bag_dir}")

    ok_count = 0
    fail_count = 0
    skip_count = 0

    print(f"Found {len(bag_files)} bag file(s).")
    print(f"Mic 1 topic: {args.mic1_topic}")
    print(f"Mic 2 topic: {args.mic2_topic}")
    print(f"Output directory: {args.out_dir}")

    for index, bag_path in enumerate(bag_files, start=1):
        print("")
        print(f"[{index}/{len(bag_files)}] {bag_path}")

        relative_parent = bag_path.relative_to(bag_dir).parent if args.recursive else None
        mic1_out, mic2_out = default_output_paths(
            bag_path=bag_path,
            out_dir=args.out_dir,
            prefix="",
            relative_parent=relative_parent,
        )

        if (mic1_out.exists() or mic2_out.exists()) and not args.overwrite:
            print(f"Skip existing wav: {mic1_out}")
            print(f"Skip existing wav: {mic2_out}")
            skip_count += 1
            continue

        try:
            export_one_bag(
                args=args,
                bag_path=bag_path,
                mic1_out=mic1_out,
                mic2_out=mic2_out,
                relative_parent=relative_parent,
            )
            ok_count += 1
        except Exception as exc:
            fail_count += 1
            print(f"Failed: {exc}", file=sys.stderr)
            if not args.keep_going:
                raise

    print("")
    print("Batch dual export finished.")
    print(f"Exported: {ok_count}")
    print(f"Skipped: {skip_count}")
    print(f"Failed: {fail_count}")

    return 1 if fail_count else 0


def main():
    parser = argparse.ArgumentParser(
        description="Export two WAV files from a dual-audio ROS1 bag."
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--bag", help="Input rosbag path")
    input_group.add_argument("--bag-dir", help="Directory containing .bag files")
    parser.add_argument("--mic1-topic", default="/mic1/audio_raw", help="Mic 1 audio topic")
    parser.add_argument("--mic2-topic", default="/mic2/audio_raw", help="Mic 2 audio topic")
    parser.add_argument(
        "--mic1-out",
        default="",
        help="Mic 1 output wav for single-bag export. Default: <out-dir>/<bag-name>_mic1.wav",
    )
    parser.add_argument(
        "--mic2-out",
        default="",
        help="Mic 2 output wav for single-bag export. Default: <out-dir>/<bag-name>_mic2.wav",
    )
    parser.add_argument(
        "--out-dir",
        default="",
        help="Output directory. For --bag-dir default: <bag-dir>/dual_wav_exports",
    )
    parser.add_argument(
        "--prefix",
        default="",
        help="Output filename prefix used when --mic1-out/--mic2-out are omitted.",
    )
    parser.add_argument(
        "--mic1-channels",
        default="",
        help="Mic 1 selected channel indices, e.g. '0,1'. Empty means all channels.",
    )
    parser.add_argument(
        "--mic2-channels",
        default="",
        help="Mic 2 selected channel indices, e.g. '0,1,2,3,4,5'. Empty means all channels.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search .bag files recursively with --bag-dir and mirror subdirectories.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing wav files when using --bag-dir.",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Continue exporting other bags if one bag fails when using --bag-dir.",
    )

    args = parser.parse_args()

    if args.bag_dir:
        if args.mic1_out or args.mic2_out or args.prefix:
            raise ValueError("--mic1-out, --mic2-out and --prefix are only for --bag")
        return batch_export(args)

    mic1_out = Path(args.mic1_out).expanduser() if args.mic1_out else None
    mic2_out = Path(args.mic2_out).expanduser() if args.mic2_out else None
    export_one_bag(args=args, bag_path=Path(args.bag), mic1_out=mic1_out, mic2_out=mic2_out)

    return 0


if __name__ == "__main__":
    sys.exit(main())

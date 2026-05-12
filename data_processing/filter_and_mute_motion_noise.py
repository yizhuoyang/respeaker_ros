#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, iirnotch, sosfiltfilt, tf2sos


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Simple robot-noise filtering plus hard muting around motion impacts. "
            "All selected channels share the same mute mask to keep inter-channel timing consistent."
        )
    )
    parser.add_argument("--input", required=True, help="Input multichannel wav.")
    parser.add_argument("--output", required=True, help="Output wav.")
    parser.add_argument(
        "--channels",
        default="1,2,3,4",
        help="Comma-separated wav channel indices to process. Default: 1,2,3,4.",
    )
    parser.add_argument(
        "--highpass-hz",
        type=float,
        default=120.0,
        help="High-pass cutoff for robot/body low-frequency noise. Use 0 to disable.",
    )
    parser.add_argument(
        "--notches-hz",
        default="48.8,66.4,179.7,341.8,867.2,1271.5,1845.7,2533.2,3783.2",
        help="Comma-separated notch frequencies for stationary robot noise. Empty string disables notches.",
    )
    parser.add_argument("--notch-q", type=float, default=35.0)
    parser.add_argument(
        "--motion-threshold",
        type=float,
        default=0.06,
        help="Mute samples whose selected-channel absolute amplitude exceeds this value.",
    )
    parser.add_argument(
        "--mute-window-sec",
        type=float,
        default=0.05,
        help="Mute this many seconds before and after every threshold crossing.",
    )
    parser.add_argument(
        "--mute-all-channels",
        action="store_true",
        help="Mute all wav channels. Default only mutes selected channels.",
    )
    parser.add_argument(
        "--no-filter",
        action="store_true",
        help="Disable robot-noise filtering and only apply hard muting.",
    )
    args = parser.parse_args()

    sample_rate, audio, input_dtype = read_wav_float(args.input)
    channels = parse_channels(args.channels, audio)
    output = audio.copy()

    selected = output[:, channels]
    if not args.no_filter:
        selected = apply_robot_filter(
            selected,
            sample_rate=sample_rate,
            highpass_hz=args.highpass_hz,
            notches_hz=parse_float_list(args.notches_hz),
            notch_q=args.notch_q,
        )

    mute_mask = make_motion_mute_mask(
        selected,
        sample_rate=sample_rate,
        threshold=args.motion_threshold,
        window_sec=args.mute_window_sec,
    )
    muted_samples = int(np.count_nonzero(mute_mask))
    if args.mute_all_channels:
        output[mute_mask, :] = 0.0
        output[:, channels] = selected[: output.shape[0], :]
        output[mute_mask, :] = 0.0
    else:
        selected[mute_mask, :] = 0.0
        output[:, channels] = selected[: output.shape[0], :]

    write_wav_float(args.output, sample_rate, output, input_dtype)
    duration = output.shape[0] / sample_rate
    muted_duration = muted_samples / sample_rate
    print(f"Wrote: {args.output}")
    print(f"Sample rate: {sample_rate} Hz, duration: {duration:.3f} s")
    print(f"Channels processed: {channels}")
    print(f"Muted samples: {muted_samples} ({muted_duration:.3f} s, {muted_duration / duration * 100:.2f}%)")


def read_wav_float(path):
    sample_rate, audio = wavfile.read(path)
    audio = np.asarray(audio)
    input_dtype = audio.dtype
    if audio.ndim == 1:
        audio = audio[:, None]

    if np.issubdtype(audio.dtype, np.integer):
        audio = audio.astype(np.float32) / float(np.iinfo(audio.dtype).max)
    else:
        audio = audio.astype(np.float32)
    return sample_rate, audio, input_dtype


def write_wav_float(path, sample_rate, audio, input_dtype):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    audio = np.asarray(audio, dtype=np.float32)
    if np.issubdtype(input_dtype, np.integer):
        info = np.iinfo(input_dtype)
        wavfile.write(str(path), sample_rate, (np.clip(audio, -1.0, 1.0) * info.max).astype(input_dtype))
    else:
        wavfile.write(str(path), sample_rate, audio)


def parse_channels(value, audio):
    channels = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not channels:
        raise RuntimeError("--channels cannot be empty")
    if min(channels) < 0 or max(channels) >= audio.shape[1]:
        raise RuntimeError(f"Input has {audio.shape[1]} channels, requested {channels}")
    return channels


def parse_float_list(value):
    if value is None or str(value).strip() == "":
        return []
    return [float(item.strip()) for item in str(value).split(",") if item.strip()]


def apply_robot_filter(audio, sample_rate, highpass_hz, notches_hz, notch_q):
    filtered = np.asarray(audio, dtype=np.float32)
    if highpass_hz and highpass_hz > 0:
        sos = butter(4, highpass_hz, btype="highpass", fs=sample_rate, output="sos")
        filtered = sosfiltfilt(sos, filtered, axis=0).astype(np.float32)

    for freq in notches_hz:
        if freq <= 0 or freq >= sample_rate / 2:
            continue
        b, a = iirnotch(freq, notch_q, fs=sample_rate)
        sos = tf2sos(b, a)
        filtered = sosfiltfilt(sos, filtered, axis=0).astype(np.float32)
    return filtered


def make_motion_mute_mask(audio, sample_rate, threshold, window_sec):
    if threshold <= 0:
        raise RuntimeError("--motion-threshold must be positive")
    if window_sec < 0:
        raise RuntimeError("--mute-window-sec must be non-negative")

    amplitude = np.max(np.abs(audio), axis=1)
    trigger = amplitude > threshold
    if not np.any(trigger):
        return np.zeros(audio.shape[0], dtype=bool)

    radius = int(round(window_sec * sample_rate))
    kernel = np.ones(radius * 2 + 1, dtype=np.int16)
    expanded = np.convolve(trigger.astype(np.int16), kernel, mode="same") > 0
    return expanded[: audio.shape[0]]


if __name__ == "__main__":
    main()

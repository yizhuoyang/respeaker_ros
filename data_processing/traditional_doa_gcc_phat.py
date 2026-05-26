#!/usr/bin/env python3
import argparse
import csv
import math
from itertools import combinations
from pathlib import Path

import numpy as np


def parse_audio_channels(value):
    text = value.strip()
    if "," in text:
        return tuple(int(item.strip()) for item in text.split(",") if item.strip())
    return tuple(int(item) for item in text)


def parse_pairs(value, num_channels):
    if value == "all":
        return tuple(combinations(range(num_channels), 2))

    pairs = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        first, second = item.replace(":", "-").split("-")
        pairs.append((int(first), int(second)))
    return tuple(pairs)


def load_audio_wav(path, channels):
    try:
        import soundfile as sf

        audio, sample_rate = sf.read(str(path), always_2d=True, dtype="float32")
    except ImportError:
        from scipy.io import wavfile

        sample_rate, audio = wavfile.read(path)
        audio = np.asarray(audio)
        if audio.ndim == 1:
            audio = audio[:, None]
        if np.issubdtype(audio.dtype, np.integer):
            audio = audio.astype(np.float32) / float(np.iinfo(audio.dtype).max)
        else:
            audio = audio.astype(np.float32)

    if max(channels) >= audio.shape[1]:
        raise RuntimeError(f"{path} has {audio.shape[1]} channels, requested {channels}")

    return audio[:, channels].T.astype(np.float32), sample_rate


def make_mic_positions(args, num_channels):
    if args.mic_positions:
        values = [float(item) for item in args.mic_positions.replace(";", ",").split(",") if item.strip()]
        if len(values) != num_channels * 2:
            raise ValueError(f"--mic-positions needs {num_channels * 2} values for {num_channels} channels")
        return np.asarray(values, dtype=np.float64).reshape(num_channels, 2)

    if args.geometry == "circle":
        angles = np.linspace(0.0, 2.0 * np.pi, num_channels, endpoint=False) + np.deg2rad(args.geometry_rotation_deg)
        return np.stack([args.radius * np.cos(angles), args.radius * np.sin(angles)], axis=1)

    if args.geometry == "square":
        if num_channels != 4:
            raise ValueError("square geometry requires 4 channels")
        half = args.spacing * 0.5
        return np.array(
            [
                [half, half],
                [-half, half],
                [-half, -half],
                [half, -half],
            ],
            dtype=np.float64,
        )

    if args.geometry == "linear":
        x = (np.arange(num_channels, dtype=np.float64) - (num_channels - 1) * 0.5) * args.spacing
        return np.stack([x, np.zeros_like(x)], axis=1)

    raise ValueError(f"Unknown geometry: {args.geometry}")


def gcc_phat_tdoa(sig_a, sig_b, sample_rate, max_tau, freq_min=None, freq_max=None, interp=8):
    n = sig_a.shape[0] + sig_b.shape[0]
    n_fft = 1
    while n_fft < n:
        n_fft *= 2

    spec_a = np.fft.rfft(sig_a, n=n_fft)
    spec_b = np.fft.rfft(sig_b, n=n_fft)
    cross = spec_a * np.conj(spec_b)

    if freq_min is not None or freq_max is not None:
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)
        mask = np.ones_like(freqs, dtype=bool)
        if freq_min is not None:
            mask &= freqs >= float(freq_min)
        if freq_max is not None:
            mask &= freqs <= float(freq_max)
        cross = np.where(mask, cross, 0.0)

    cross = cross / (np.abs(cross) + 1e-12)

    cc = np.fft.irfft(cross, n=n_fft * interp)
    max_shift = min(int(round(max_tau * sample_rate * interp)), cc.shape[0] // 2)
    cc = np.concatenate((cc[-max_shift:], cc[: max_shift + 1]))
    shift = np.argmax(np.abs(cc)) - max_shift
    return shift / float(sample_rate * interp)


def estimate_tdoas(audio, sample_rate, pairs, mic_positions, sound_speed, freq_min, freq_max):
    max_dist = 0.0
    for first, second in pairs:
        max_dist = max(max_dist, np.linalg.norm(mic_positions[second] - mic_positions[first]))
    max_tau = max_dist / sound_speed

    tdoas = {}
    for first, second in pairs:
        tdoas[(first, second)] = gcc_phat_tdoa(
            audio[first],
            audio[second],
            sample_rate=sample_rate,
            max_tau=max_tau,
            freq_min=freq_min,
            freq_max=freq_max,
        )
    return tdoas


def grid_search_doa(tdoas, pairs, mic_positions, sound_speed, num_azimuth_bins, tdoa_sigma, sign):
    azimuths = np.linspace(-np.pi, np.pi, num_azimuth_bins, endpoint=False)
    scores = np.zeros_like(azimuths)

    for index, azimuth in enumerate(azimuths):
        unit = np.array([np.cos(azimuth), np.sin(azimuth)], dtype=np.float64)
        error = 0.0
        for first, second in pairs:
            delta_pos = mic_positions[second] - mic_positions[first]
            pred_tdoa = sign * np.dot(delta_pos, unit) / sound_speed
            error += ((tdoas[(first, second)] - pred_tdoa) / tdoa_sigma) ** 2
        scores[index] = -error

    best_index = int(np.argmax(scores))
    return azimuths[best_index], scores


def circular_error_deg(pred_rad, gt_rad):
    diff = (pred_rad - gt_rad + np.pi) % (2.0 * np.pi) - np.pi
    return abs(np.rad2deg(diff))


def wrap_angle_rad(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def fit_global_offset(pred_azimuths, gt_azimuths, num_steps=720):
    offsets = np.linspace(-np.pi, np.pi, int(num_steps), endpoint=False)
    best_offset = 0.0
    best_error = float("inf")

    for offset in offsets:
        corrected = wrap_angle_rad(pred_azimuths + offset)
        errors = np.abs(np.rad2deg(wrap_angle_rad(corrected - gt_azimuths)))
        mean_error = float(np.mean(errors))
        if mean_error < best_error:
            best_error = mean_error
            best_offset = float(offset)

    return best_offset, best_error


def find_sequence_dirs(root):
    root = Path(root)
    if (root / "audio").is_dir() and (root / "doa").is_dir():
        return [root]
    return sorted(path for path in root.rglob("*") if path.is_dir() and (path / "audio").is_dir() and (path / "doa").is_dir())


def collect_samples(root, min_distance):
    samples = []
    for seq_dir in find_sequence_dirs(root):
        for audio_path in sorted((seq_dir / "audio").glob("*.wav")):
            doa_path = seq_dir / "doa" / f"{audio_path.stem}.npy"
            if not doa_path.exists():
                continue
            doa = np.load(doa_path).astype(np.float32)
            distance = float(doa[5])
            if distance < min_distance:
                continue
            samples.append((seq_dir, audio_path, doa_path))
    return samples


def main():
    parser = argparse.ArgumentParser(description="Traditional GCC-PHAT DOA baseline for synchronized dataset audio.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--audio-channels", default="1234")
    parser.add_argument("--pairs", default="all", help="'all' or comma list like 0-1,0-2")
    parser.add_argument("--geometry", default="square", choices=["circle", "square", "linear"])
    parser.add_argument("--radius", type=float, default=0.032, help="Circle geometry radius in meters.")
    parser.add_argument("--spacing", type=float, default=45.7 / 1000.0, help="Square/linear spacing in meters.")
    parser.add_argument("--geometry-rotation-deg", type=float, default=0.0)
    parser.add_argument("--mic-positions", default="", help="Custom x,y list in meters after channel selection.")
    parser.add_argument("--freq-min", type=float, default=300.0)
    parser.add_argument("--freq-max", type=float, default=3400.0)
    parser.add_argument("--sound-speed", type=float, default=343.0)
    parser.add_argument("--num-azimuth-bins", type=int, default=720)
    parser.add_argument("--tdoa-sigma", type=float, default=3e-5)
    parser.add_argument("--tdoa-sign", type=float, default=-1.0, choices=[-1.0, 1.0])
    parser.add_argument("--fit-offset", action="store_true", help="Fit one global azimuth offset on this split before reporting final errors.")
    parser.add_argument("--offset-search-steps", type=int, default=720)
    parser.add_argument("--min-distance", type=float, default=0.3)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--out-csv", default="traditional_doa_results.csv")
    args = parser.parse_args()

    channels = parse_audio_channels(args.audio_channels)
    pairs = parse_pairs(args.pairs, num_channels=len(channels))
    mic_positions = make_mic_positions(args, num_channels=len(channels))
    samples = collect_samples(args.data_root, min_distance=args.min_distance)
    if args.max_samples > 0:
        samples = samples[: args.max_samples]
    if not samples:
        raise RuntimeError(f"No samples found under: {args.data_root}")

    print(f"Samples: {len(samples)}")
    print(f"Channels: {channels}")
    print(f"Pairs: {pairs}")
    print(f"Mic positions after channel selection:\n{mic_positions}")

    rows = []
    errors = []
    pred_azimuths = []
    gt_azimuths = []
    for index, (seq_dir, audio_path, doa_path) in enumerate(samples, start=1):
        audio, sample_rate = load_audio_wav(audio_path, channels=channels)
        gt = np.load(doa_path).astype(np.float32)
        gt_azimuth = float(gt[0])
        distance = float(gt[5])

        tdoas = estimate_tdoas(
            audio=audio,
            sample_rate=sample_rate,
            pairs=pairs,
            mic_positions=mic_positions,
            sound_speed=args.sound_speed,
            freq_min=args.freq_min,
            freq_max=args.freq_max,
        )
        pred_azimuth, _ = grid_search_doa(
            tdoas=tdoas,
            pairs=pairs,
            mic_positions=mic_positions,
            sound_speed=args.sound_speed,
            num_azimuth_bins=args.num_azimuth_bins,
            tdoa_sigma=args.tdoa_sigma,
            sign=args.tdoa_sign,
        )
        error_deg = circular_error_deg(pred_azimuth, gt_azimuth)
        errors.append(error_deg)
        pred_azimuths.append(pred_azimuth)
        gt_azimuths.append(gt_azimuth)

        rows.append(
            {
                "seq": str(seq_dir),
                "sample_id": audio_path.stem,
                "audio": str(audio_path),
                "gt_azimuth_deg": f"{np.rad2deg(gt_azimuth):.6f}",
                "pred_azimuth_deg": f"{np.rad2deg(pred_azimuth):.6f}",
                "error_deg": f"{error_deg:.6f}",
                "pred_offset_deg": "",
                "pred_corrected_azimuth_deg": "",
                "corrected_error_deg": "",
                "distance_m": f"{distance:.6f}",
            }
        )

        if index % 100 == 0 or index == len(samples):
            print(f"[{index}/{len(samples)}] mean_err={np.mean(errors):.2f} deg median_err={np.median(errors):.2f} deg")

    pred_azimuths = np.asarray(pred_azimuths, dtype=np.float64)
    gt_azimuths = np.asarray(gt_azimuths, dtype=np.float64)
    corrected_errors = None
    best_offset = 0.0

    if args.fit_offset:
        best_offset, _ = fit_global_offset(
            pred_azimuths=pred_azimuths,
            gt_azimuths=gt_azimuths,
            num_steps=args.offset_search_steps,
        )
        corrected = wrap_angle_rad(pred_azimuths + best_offset)
        corrected_errors = np.abs(np.rad2deg(wrap_angle_rad(corrected - gt_azimuths)))
        for row, corrected_azimuth, corrected_error in zip(rows, corrected, corrected_errors):
            row["pred_offset_deg"] = f"{np.rad2deg(best_offset):.6f}"
            row["pred_corrected_azimuth_deg"] = f"{np.rad2deg(corrected_azimuth):.6f}"
            row["corrected_error_deg"] = f"{corrected_error:.6f}"

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("")
    print(f"Saved: {out_csv}")
    print(f"Mean error: {np.mean(errors):.3f} deg")
    print(f"Median error: {np.median(errors):.3f} deg")
    print(f"90th percentile error: {np.percentile(errors, 90):.3f} deg")
    if corrected_errors is not None:
        print(f"Best offset: {np.rad2deg(best_offset):.3f} deg")
        print(f"Corrected mean error: {np.mean(corrected_errors):.3f} deg")
        print(f"Corrected median error: {np.median(corrected_errors):.3f} deg")
        print(f"Corrected 90th percentile error: {np.percentile(corrected_errors, 90):.3f} deg")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, iirnotch, medfilt, sosfiltfilt, stft, istft, tf2sos


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Phase-friendly multichannel denoising for robot-mounted microphone audio. "
            "The same filters and the same TF gain are applied to all selected channels "
            "to preserve inter-channel phase differences for DOA."
        )
    )
    parser.add_argument("--input", required=True, help="Input multichannel wav.")
    parser.add_argument("--output", required=True, help="Output denoised wav.")
    parser.add_argument(
        "--noise",
        action="append",
        default=[],
        help="Backward-compatible always-on noise wav. Can be repeated.",
    )
    parser.add_argument(
        "--stationary-noise",
        action="append",
        default=[],
        help="Always-present noise wav, e.g. robot/lidar noise. Can be repeated.",
    )
    parser.add_argument(
        "--motion-noise",
        action="append",
        default=[],
        help="Motion-only noise wav. It is subtracted only on frames detected as motion-like. Can be repeated.",
    )
    parser.add_argument(
        "--channels",
        default=None,
        help="Comma-separated channels to denoise. Default: all non-silent channels.",
    )
    parser.add_argument("--highpass-hz", type=float, default=90.0, help="Shared high-pass cutoff. Use 0 to disable.")
    parser.add_argument(
        "--notches-hz",
        default="48,180,342,1845",
        help="Comma-separated shared notch frequencies. Empty string disables notches.",
    )
    parser.add_argument("--notch-q", type=float, default=35.0)
    parser.add_argument(
        "--spectral-strength",
        type=float,
        default=0.8,
        help="Noise spectral subtraction strength. 0 disables spectral gating.",
    )
    parser.add_argument(
        "--gain-floor",
        type=float,
        default=0.25,
        help="Minimum TF gain. Higher keeps more signal, lower removes more noise.",
    )
    parser.add_argument("--motion-strength", type=float, default=0.3)
    parser.add_argument("--motion-gain-floor", type=float, default=0.6)
    parser.add_argument("--motion-gate-threshold", type=float, default=1.6)
    parser.add_argument("--motion-gate-smooth-frames", type=int, default=5)
    parser.add_argument("--transient-attenuation", type=float, default=0.35)
    parser.add_argument("--transient-threshold", type=float, default=3.0)
    parser.add_argument("--transient-frame-ms", type=float, default=20.0)
    parser.add_argument("--transient-hop-ms", type=float, default=5.0)
    parser.add_argument("--transient-smooth-frames", type=int, default=5)
    parser.add_argument("--n-fft", type=int, default=1024)
    parser.add_argument("--hop", type=int, default=256)
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize output peak to input peak after denoising.",
    )
    args = parser.parse_args()

    sample_rate, audio, input_dtype = read_wav_float(args.input)
    channels = parse_channels(args.channels, audio)
    output = audio.copy()

    selected = output[:, channels]
    selected = apply_shared_time_filters(
        selected,
        sample_rate=sample_rate,
        highpass_hz=args.highpass_hz,
        notches_hz=parse_float_list(args.notches_hz),
        notch_q=args.notch_q,
    )

    stationary_noise_paths = args.stationary_noise or args.noise
    if stationary_noise_paths and args.spectral_strength > 0:
        noise_profile = estimate_noise_profile(
            noise_paths=stationary_noise_paths,
            sample_rate=sample_rate,
            channels=channels,
            n_fft=args.n_fft,
            hop=args.hop,
            highpass_hz=args.highpass_hz,
            notches_hz=parse_float_list(args.notches_hz),
            notch_q=args.notch_q,
        )
        selected = apply_shared_spectral_gate(
            selected,
            sample_rate=sample_rate,
            noise_power=noise_profile,
            n_fft=args.n_fft,
            hop=args.hop,
            strength=args.spectral_strength,
            gain_floor=args.gain_floor,
        )

    if args.motion_noise and args.motion_strength > 0:
        motion_noise_profile = estimate_noise_profile(
            noise_paths=args.motion_noise,
            sample_rate=sample_rate,
            channels=channels,
            n_fft=args.n_fft,
            hop=args.hop,
            highpass_hz=args.highpass_hz,
            notches_hz=parse_float_list(args.notches_hz),
            notch_q=args.notch_q,
        )
        selected = apply_motion_gated_spectral_gate(
            selected,
            sample_rate=sample_rate,
            noise_power=motion_noise_profile,
            n_fft=args.n_fft,
            hop=args.hop,
            strength=args.motion_strength,
            gain_floor=args.motion_gain_floor,
            gate_threshold=args.motion_gate_threshold,
            gate_smooth_frames=args.motion_gate_smooth_frames,
        )

    if args.transient_attenuation > 0:
        selected = suppress_shared_transients(
            selected,
            sample_rate=sample_rate,
            frame_ms=args.transient_frame_ms,
            hop_ms=args.transient_hop_ms,
            threshold=args.transient_threshold,
            attenuation=args.transient_attenuation,
            smooth_frames=args.transient_smooth_frames,
        )

    output[:, channels] = selected[: output.shape[0], :]
    if args.normalize:
        output = normalize_to_input_peak(output, audio)

    write_wav_float(args.output, sample_rate, output, input_dtype)
    print(f"Wrote denoised wav: {args.output}")


def read_wav_float(path):
    sample_rate, audio = wavfile.read(path)
    audio = np.asarray(audio)
    input_dtype = audio.dtype
    if audio.ndim == 1:
        audio = audio[:, None]

    if np.issubdtype(audio.dtype, np.integer):
        scale = float(np.iinfo(audio.dtype).max)
        audio = audio.astype(np.float32) / scale
    else:
        audio = audio.astype(np.float32)
    return sample_rate, audio, input_dtype


def write_wav_float(path, sample_rate, audio, input_dtype):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    audio = np.asarray(audio, dtype=np.float32)

    if np.issubdtype(input_dtype, np.integer):
        info = np.iinfo(input_dtype)
        audio_int = np.clip(audio, -1.0, 1.0) * info.max
        wavfile.write(str(path), sample_rate, audio_int.astype(input_dtype))
    else:
        wavfile.write(str(path), sample_rate, audio)


def parse_channels(value, audio):
    if value:
        return [int(item.strip()) for item in value.split(",") if item.strip()]

    rms = np.sqrt(np.mean(audio * audio, axis=0))
    active = [idx for idx, value in enumerate(rms) if value > 1e-8]
    return active if active else list(range(audio.shape[1]))


def parse_float_list(value):
    if value is None or value.strip() == "":
        return []
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def apply_shared_time_filters(audio, sample_rate, highpass_hz=90.0, notches_hz=None, notch_q=35.0):
    filtered = audio.astype(np.float32)
    if highpass_hz and highpass_hz > 0:
        sos = butter(4, highpass_hz, btype="highpass", fs=sample_rate, output="sos")
        filtered = sosfiltfilt(sos, filtered, axis=0).astype(np.float32)

    for freq in notches_hz or []:
        if freq <= 0 or freq >= sample_rate / 2:
            continue
        b, a = iirnotch(freq, notch_q, fs=sample_rate)
        sos = tf2sos(b, a)
        filtered = sosfiltfilt(sos, filtered, axis=0).astype(np.float32)
    return filtered


def estimate_noise_profile(
    noise_paths,
    sample_rate,
    channels,
    n_fft,
    hop,
    highpass_hz,
    notches_hz,
    notch_q,
):
    powers = []
    for path in noise_paths:
        noise_sr, noise, _ = read_wav_float(path)
        if noise_sr != sample_rate:
            raise RuntimeError(f"Noise sample rate mismatch: {path} has {noise_sr}, expected {sample_rate}")
        if max(channels) >= noise.shape[1]:
            raise RuntimeError(f"Noise wav {path} has {noise.shape[1]} channels, requested {channels}")
        noise = noise[:, channels]
        noise = apply_shared_time_filters(noise, sample_rate, highpass_hz, notches_hz, notch_q)
        _, _, noise_stft = stft(
            noise.T,
            fs=sample_rate,
            nperseg=n_fft,
            noverlap=n_fft - hop,
            axis=-1,
            boundary=None,
        )
        powers.append(np.mean(np.abs(noise_stft) ** 2, axis=(0, 2)))

    return np.mean(np.stack(powers, axis=0), axis=0)


def apply_shared_spectral_gate(audio, sample_rate, noise_power, n_fft, hop, strength, gain_floor):
    _, _, spectrum = stft(
        audio.T,
        fs=sample_rate,
        nperseg=n_fft,
        noverlap=n_fft - hop,
        axis=-1,
        boundary="zeros",
    )
    signal_power = np.mean(np.abs(spectrum) ** 2, axis=0)
    gain = 1.0 - strength * noise_power[:, None] / (signal_power + 1e-10)
    gain = np.clip(gain, gain_floor, 1.0).astype(np.float32)

    gated = spectrum * gain[None, :, :]
    restored_channels = []
    for channel in range(gated.shape[0]):
        _, restored_channel = istft(
            gated[channel],
            fs=sample_rate,
            nperseg=n_fft,
            noverlap=n_fft - hop,
            input_onesided=True,
        )
        restored_channels.append(restored_channel)
    restored = np.stack(restored_channels, axis=1).astype(np.float32)
    return restored[: audio.shape[0], :]


def apply_motion_gated_spectral_gate(
    audio,
    sample_rate,
    noise_power,
    n_fft,
    hop,
    strength,
    gain_floor,
    gate_threshold,
    gate_smooth_frames,
):
    _, _, spectrum = stft(
        audio.T,
        fs=sample_rate,
        nperseg=n_fft,
        noverlap=n_fft - hop,
        axis=-1,
        boundary="zeros",
    )
    signal_power = np.mean(np.abs(spectrum) ** 2, axis=0)
    weights = noise_power / (np.mean(noise_power) + 1e-10)
    weights = weights / (np.sum(weights) + 1e-10)
    expected_noise = float(np.sum(noise_power * weights)) + 1e-10
    motion_score = np.sum(signal_power * weights[:, None], axis=0) / expected_noise
    gate = (motion_score >= gate_threshold).astype(np.float32)
    if gate_smooth_frames and gate_smooth_frames > 1 and gate.size > 1:
        kernel = np.ones(int(gate_smooth_frames), dtype=np.float32)
        kernel = kernel / kernel.sum()
        gate = np.convolve(gate, kernel, mode="same").astype(np.float32)
        gate = np.clip(gate, 0.0, 1.0)

    if float(np.max(gate)) <= 0.0:
        return audio.astype(np.float32)

    gain = 1.0 - strength * gate[None, :] * noise_power[:, None] / (signal_power + 1e-10)
    gain = np.clip(gain, gain_floor, 1.0).astype(np.float32)

    gated = spectrum * gain[None, :, :]
    restored_channels = []
    for channel in range(gated.shape[0]):
        _, restored_channel = istft(
            gated[channel],
            fs=sample_rate,
            nperseg=n_fft,
            noverlap=n_fft - hop,
            input_onesided=True,
        )
        restored_channels.append(restored_channel)
    restored = np.stack(restored_channels, axis=1).astype(np.float32)
    return restored[: audio.shape[0], :]


def suppress_shared_transients(
    audio,
    sample_rate,
    frame_ms=20.0,
    hop_ms=5.0,
    threshold=3.0,
    attenuation=0.5,
    smooth_frames=5,
):
    audio = np.asarray(audio, dtype=np.float32)
    if attenuation <= 0 or threshold <= 0 or audio.shape[0] == 0:
        return audio

    frame = max(8, int(round(sample_rate * frame_ms / 1000.0)))
    hop = max(1, int(round(sample_rate * hop_ms / 1000.0)))
    num_samples = audio.shape[0]
    if num_samples < frame:
        return audio

    starts = list(range(0, num_samples - frame + 1, hop))
    energies = []
    for start in starts:
        segment = audio[start:start + frame, :]
        energies.append(float(np.mean(segment * segment)))
    energies = np.asarray(energies, dtype=np.float32)
    if energies.size < 3:
        return audio

    kernel_size = max(3, int(round(0.5 * sample_rate / hop)))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel_size = min(kernel_size, energies.size if energies.size % 2 == 1 else energies.size - 1)
    if kernel_size < 3:
        return audio

    baseline = medfilt(energies, kernel_size=kernel_size).astype(np.float32)
    baseline = np.maximum(baseline, np.percentile(energies, 20) + 1e-10)
    ratio = energies / baseline
    transient = np.clip((ratio - threshold) / max(threshold, 1e-6), 0.0, 1.0)

    if smooth_frames and smooth_frames > 1:
        kernel = np.ones(int(smooth_frames), dtype=np.float32)
        kernel = kernel / kernel.sum()
        transient = np.convolve(transient, kernel, mode="same").astype(np.float32)
        transient = np.clip(transient, 0.0, 1.0)

    frame_gain = 1.0 - attenuation * transient
    envelope = np.ones(num_samples, dtype=np.float32)
    weight = np.zeros(num_samples, dtype=np.float32)
    for start, gain in zip(starts, frame_gain):
        envelope[start:start + frame] += gain
        weight[start:start + frame] += 1.0
    valid = weight > 0
    envelope[valid] = envelope[valid] / (weight[valid] + 1.0)
    return (audio * envelope[:, None]).astype(np.float32)


def normalize_to_input_peak(output, original):
    original_peak = float(np.max(np.abs(original)))
    output_peak = float(np.max(np.abs(output)))
    if original_peak > 0 and output_peak > 0:
        output = output * min(1.0, original_peak / output_peak)
    return output


if __name__ == "__main__":
    main()

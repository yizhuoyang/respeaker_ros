#!/usr/bin/env python3
"""Streaming audio inference core for SSLNet DOA and distance distributions."""

from collections import deque
from pathlib import Path
import sys
import time

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataloader.utils import (  # noqa: E402
    apply_audio_bandpass,
    compute_spectrogram,
    compute_stft_phase_features,
    filter_and_mute_motion_impacts,
    parse_channel_pairs,
    parse_float_list,
)
from network.audionet.ssl_net import SSLNet_DOA  # noqa: E402


def parse_int_tuple(value):
    if isinstance(value, str):
        return tuple(int(item.strip()) for item in value.split(",") if item.strip())
    return tuple(int(item) for item in value)


def infer_audio_in_channels(audio_feat, audio_channels, ipd_pairs):
    if audio_feat == "spec":
        return len(audio_channels)
    if audio_feat == "phase":
        return len(audio_channels) * 2
    if audio_feat in ("ipd", "gcc_phat_complex"):
        return len(ipd_pairs) * 2
    if audio_feat == "both":
        return len(audio_channels) * 2 + len(ipd_pairs) * 2
    raise ValueError(f"Unsupported audio feature: {audio_feat}")


def checkpoint_arguments(checkpoint):
    if isinstance(checkpoint, dict) and isinstance(checkpoint.get("args"), dict):
        return checkpoint["args"]
    return {}


def class_count_from_state_dict(state_dict):
    weight = state_dict.get("class_head.3.weight")
    return int(weight.shape[0]) if weight is not None else 0


def signal_count_from_state_dict(state_dict):
    weight = state_dict.get("signal_head.3.weight")
    return int(weight.shape[0]) if weight is not None else 0


def load_trusted_checkpoint(path, device):
    """Load a local training checkpoint while supporting older PyTorch releases."""
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


class SSLNetStreamingPredictor:
    """Load an SSLNet audio checkpoint and infer one fixed-length audio window."""

    def __init__(
        self,
        checkpoint_path,
        device="cuda:0",
        audio_feat=None,
        audio_channels=None,
        ipd_pairs=None,
        use_compress=None,
        expected_sample_rate=16000,
        distance_min_m=0.0,
        distance_max_m=6.0,
        audio_bandpass_low_hz=None,
        audio_bandpass_high_hz=None,
        use_filter_mute_denoise=None,
        filter_mute_highpass_hz=None,
        filter_mute_notches_hz=None,
        filter_mute_threshold=None,
        filter_mute_window_sec=None,
        filter_mute_floor=None,
        filter_mute_edge_smooth_ms=None,
    ):
        self.checkpoint_path = str(checkpoint_path)
        self.device = torch.device(
            device if str(device).startswith("cuda") and torch.cuda.is_available() else "cpu"
        )
        checkpoint = load_trusted_checkpoint(self.checkpoint_path, self.device)
        state_dict = (
            checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
            if isinstance(checkpoint, dict)
            else checkpoint
        )
        saved_args = checkpoint_arguments(checkpoint)
        if saved_args.get("model", "audio") != "audio":
            raise ValueError(
                "ROS audio-only deployment requires a checkpoint trained with --model audio"
            )
        if saved_args.get("use_denoise", False):
            raise ValueError(
                "This checkpoint used --use-denoise with offline noise profiles; "
                "real-time deployment currently supports bandpass and --use-filter-mute-denoise only"
            )

        self.audio_feat = audio_feat or saved_args.get("audio_feat", "ipd")
        saved_audio_channels = saved_args.get("audio_channels", "1,2,3,4")
        self.audio_channels = parse_int_tuple(
            audio_channels if audio_channels is not None else saved_audio_channels
        )
        saved_ipd_pairs = saved_args.get("ipd_pairs", "0-1,0-2,0-3,1-2,1-3,2-3")
        self.ipd_pairs = parse_channel_pairs(
            ipd_pairs if ipd_pairs is not None else saved_ipd_pairs
        )
        self.use_compress = bool(
            saved_args.get("use_compress", False) if use_compress is None else use_compress
        )
        self.expected_sample_rate = int(expected_sample_rate)
        self.distance_min_m = float(distance_min_m)
        self.distance_max_m = float(distance_max_m)

        self.audio_bandpass_low_hz = float(
            saved_args.get("audio_bandpass_low_hz", 0.0)
            if audio_bandpass_low_hz is None else audio_bandpass_low_hz
        )
        self.audio_bandpass_high_hz = float(
            saved_args.get("audio_bandpass_high_hz", 0.0)
            if audio_bandpass_high_hz is None else audio_bandpass_high_hz
        )
        self.use_filter_mute_denoise = bool(
            saved_args.get("use_filter_mute_denoise", False)
            if use_filter_mute_denoise is None else use_filter_mute_denoise
        )
        self.filter_mute_highpass_hz = float(
            saved_args.get("filter_mute_highpass_hz", 120.0)
            if filter_mute_highpass_hz is None else filter_mute_highpass_hz
        )
        saved_notches = saved_args.get(
            "filter_mute_notches_hz",
            "48.8,66.4,179.7,341.8,867.2,1271.5,1845.7,2533.2,3783.2",
        )
        self.filter_mute_notches_hz = tuple(
            parse_float_list(filter_mute_notches_hz if filter_mute_notches_hz is not None else saved_notches)
        )
        self.filter_mute_threshold = float(
            saved_args.get("filter_mute_threshold", 0.06)
            if filter_mute_threshold is None else filter_mute_threshold
        )
        self.filter_mute_window_sec = float(
            saved_args.get("filter_mute_window_sec", 0.05)
            if filter_mute_window_sec is None else filter_mute_window_sec
        )
        self.filter_mute_floor = float(
            saved_args.get("filter_mute_floor", 0.02)
            if filter_mute_floor is None else filter_mute_floor
        )
        self.filter_mute_edge_smooth_ms = float(
            saved_args.get("filter_mute_edge_smooth_ms", 5.0)
            if filter_mute_edge_smooth_ms is None else filter_mute_edge_smooth_ms
        )

        audio_in_channels = infer_audio_in_channels(
            self.audio_feat, self.audio_channels, self.ipd_pairs
        )
        model = SSLNet_DOA(
            use_compress=self.use_compress,
            audio_in_channels=audio_in_channels,
            num_classes=class_count_from_state_dict(state_dict),
            signal_classes=signal_count_from_state_dict(state_dict),
        )
        model.load_state_dict(state_dict, strict=True)
        self.model = model.to(self.device).eval()

    def describe(self):
        return {
            "checkpoint": self.checkpoint_path,
            "device": str(self.device),
            "audio_feat": self.audio_feat,
            "audio_channels": self.audio_channels,
            "ipd_pairs": self.ipd_pairs,
            "use_compress": self.use_compress,
            "expected_sample_rate": self.expected_sample_rate,
            "use_filter_mute_denoise": self.use_filter_mute_denoise,
        }

    def _prepare_features(self, audio, sample_rate):
        if int(sample_rate) != self.expected_sample_rate:
            raise ValueError(
                f"Expected {self.expected_sample_rate} Hz audio, received {sample_rate} Hz"
            )
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim != 2:
            raise ValueError(f"Expected audio with shape (frames, channels), got {audio.shape}")
        if max(self.audio_channels) >= audio.shape[1]:
            raise ValueError(
                f"Requested channels {self.audio_channels}, received audio with {audio.shape[1]} channels"
            )
        selected = audio[:, self.audio_channels].T
        if self.use_filter_mute_denoise:
            selected = filter_and_mute_motion_impacts(
                selected,
                sample_rate=sample_rate,
                highpass_hz=self.filter_mute_highpass_hz,
                notches_hz=self.filter_mute_notches_hz,
                motion_threshold=self.filter_mute_threshold,
                mute_window_sec=self.filter_mute_window_sec,
                mute_floor=self.filter_mute_floor,
                edge_smooth_ms=self.filter_mute_edge_smooth_ms,
            )
        selected = apply_audio_bandpass(
            selected,
            sample_rate=sample_rate,
            low_hz=self.audio_bandpass_low_hz,
            high_hz=self.audio_bandpass_high_hz,
        )
        if self.audio_feat == "spec":
            feature = compute_spectrogram(selected, self.use_compress)
        else:
            feature = compute_stft_phase_features(
                selected, mode=self.audio_feat, pairs=self.ipd_pairs
            )
        return torch.as_tensor(feature, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)

    @torch.no_grad()
    def predict(self, audio, sample_rate):
        started_at = time.perf_counter()
        spectrogram = self._prepare_features(audio, sample_rate).to(self.device)
        outputs = self.model(spectrogram)
        doa_probability = torch.softmax(outputs[0], dim=1)[0].cpu().numpy()
        distance_probability = torch.softmax(outputs[1], dim=1)[0].cpu().numpy()
        output_index = 2
        doa_bin = int(np.argmax(doa_probability))
        distance_bin = int(np.argmax(distance_probability))
        distance_axis = np.linspace(
            self.distance_min_m, self.distance_max_m, len(distance_probability), dtype=np.float32
        )
        result = {
            "doa_deg": float(doa_bin * 360.0 / len(doa_probability)),
            "distance_m": float(distance_axis[distance_bin]),
            "doa_confidence": float(doa_probability[doa_bin]),
            "distance_confidence": float(distance_probability[distance_bin]),
            "doa_probability": doa_probability,
            "distance_probability": distance_probability,
            "inference_ms": float((time.perf_counter() - started_at) * 1000.0),
        }
        if self.model.class_head is not None:
            class_probability = torch.softmax(outputs[output_index], dim=1)[0].cpu().numpy()
            result["class_probability"] = class_probability
            result["class_id"] = int(np.argmax(class_probability))
            output_index += 1
        if self.model.signal_head is not None:
            signal_probability = torch.softmax(outputs[output_index], dim=1)[0].cpu().numpy()
            result["signal_probability"] = signal_probability
            result["signal_id"] = int(np.argmax(signal_probability))
            result["signal_prob"] = float(signal_probability[1]) if len(signal_probability) > 1 else 0.0
        return result


class SlidingAudioWindow:
    """Emit exact rolling windows as interleaved audio messages arrive."""

    def __init__(self, sample_rate, window_seconds=1.0, hop_seconds=None):
        self.sample_rate = int(sample_rate)
        self.window_frames = int(round(float(window_seconds) * self.sample_rate))
        hop_seconds = window_seconds if hop_seconds is None else hop_seconds
        self.hop_frames = int(round(float(hop_seconds) * self.sample_rate))
        if self.window_frames <= 0 or self.hop_frames <= 0:
            raise ValueError("window_seconds and hop_seconds must be positive")
        self._chunks = deque()
        self._buffer_start = 0
        self._total_frames = 0
        self._next_end = self.window_frames

    def reset(self):
        self._chunks.clear()
        self._buffer_start = 0
        self._total_frames = 0
        self._next_end = self.window_frames

    def append(self, audio):
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim != 2:
            raise ValueError(f"Expected audio with shape (frames, channels), got {audio.shape}")
        if audio.shape[0] == 0:
            return []
        self._chunks.append(audio)
        self._total_frames += audio.shape[0]
        joined = np.concatenate(tuple(self._chunks), axis=0)
        outputs = []
        while self._next_end <= self._total_frames:
            begin = self._next_end - self.window_frames - self._buffer_start
            end = self._next_end - self._buffer_start
            outputs.append(joined[begin:end].copy())
            self._next_end += self.hop_frames

        keep_from = max(0, self._next_end - self.window_frames - self._buffer_start)
        if keep_from:
            joined = joined[keep_from:]
            self._buffer_start += keep_from
        self._chunks.clear()
        self._chunks.append(joined)
        return outputs

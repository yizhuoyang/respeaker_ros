#!/usr/bin/env python3
"""TensorRT inference core for SSLNet audio DOA and distance distributions."""

import json
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
from deploy.sslnet_realtime import parse_int_tuple  # noqa: E402


def normalize_cuda_device(device):
    text = str(device).strip()
    if text.isdigit():
        text = f"cuda:{text}"
    if not text:
        text = "cuda:0"
    if not text.startswith("cuda"):
        raise ValueError("TensorRT inference requires a CUDA device such as cuda:0 or 0.")
    if not torch.cuda.is_available():
        raise RuntimeError("TensorRT inference requires CUDA, but torch.cuda.is_available() is false.")
    return torch.device(text)


def torch_dtype_from_numpy(dtype):
    mapping = {
        np.dtype(np.float32): torch.float32,
        np.dtype(np.float16): torch.float16,
        np.dtype(np.int32): torch.int32,
        np.dtype(np.int64): torch.int64,
        np.dtype(np.int8): torch.int8,
        np.dtype(np.bool_): torch.bool,
    }
    return mapping[np.dtype(dtype)]


class TensorRTEngineSession:
    """Execute a static SSLNet engine using torch CUDA tensors as bindings."""

    def __init__(self, engine_path, device):
        try:
            import tensorrt as trt
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Python TensorRT is required to run the SSLNet engine. "
                "Activate the TensorRT environment that contains `import tensorrt`."
            ) from exc

        self.trt = trt
        self.device = normalize_cuda_device(device)
        torch.cuda.set_device(self.device)
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as handle:
            serialized = handle.read()
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(serialized)
        if self.engine is None:
            raise RuntimeError(f"Cannot deserialize TensorRT engine: {engine_path}")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError(f"Cannot create TensorRT execution context: {engine_path}")
        self.tensor_api = hasattr(self.engine, "num_io_tensors")
        self.input_name, self.output_names = self._io_names()

    def _io_names(self):
        trt = self.trt
        if self.tensor_api:
            names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
            inputs = [
                name
                for name in names
                if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
            ]
            outputs = [
                name
                for name in names
                if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT
            ]
        else:
            names = [self.engine.get_binding_name(i) for i in range(self.engine.num_bindings)]
            inputs = [name for i, name in enumerate(names) if self.engine.binding_is_input(i)]
            outputs = [name for i, name in enumerate(names) if not self.engine.binding_is_input(i)]
        if len(inputs) != 1 or len(outputs) < 2:
            raise RuntimeError(
                f"Expected one SSLNet input and at least two outputs, got {inputs} and {outputs}."
            )
        return inputs[0], outputs

    def _allocate_output(self, name):
        if self.tensor_api:
            shape = tuple(self.context.get_tensor_shape(name))
            dtype = torch_dtype_from_numpy(self.trt.nptype(self.engine.get_tensor_dtype(name)))
        else:
            index = self.engine.get_binding_index(name)
            shape = tuple(self.context.get_binding_shape(index))
            dtype = torch_dtype_from_numpy(self.trt.nptype(self.engine.get_binding_dtype(index)))
        if any(dim < 0 for dim in shape):
            raise RuntimeError(f"TensorRT output shape for {name} remains dynamic: {shape}")
        return torch.empty(shape, dtype=dtype, device=self.device)

    @torch.no_grad()
    def infer(self, feature):
        feature = feature.to(self.device, non_blocking=True).contiguous()
        stream = torch.cuda.current_stream(self.device)
        if self.tensor_api:
            self.context.set_input_shape(self.input_name, tuple(feature.shape))
            outputs = {name: self._allocate_output(name) for name in self.output_names}
            self.context.set_tensor_address(self.input_name, int(feature.data_ptr()))
            for name, output in outputs.items():
                self.context.set_tensor_address(name, int(output.data_ptr()))
            if not self.context.execute_async_v3(stream_handle=stream.cuda_stream):
                raise RuntimeError("TensorRT execute_async_v3 failed.")
        else:
            input_index = self.engine.get_binding_index(self.input_name)
            self.context.set_binding_shape(input_index, tuple(feature.shape))
            outputs = {name: self._allocate_output(name) for name in self.output_names}
            bindings = [0] * self.engine.num_bindings
            bindings[input_index] = int(feature.data_ptr())
            for name, output in outputs.items():
                bindings[self.engine.get_binding_index(name)] = int(output.data_ptr())
            if not self.context.execute_async_v2(
                bindings=bindings, stream_handle=stream.cuda_stream
            ):
                raise RuntimeError("TensorRT execute_async_v2 failed.")
        stream.synchronize()
        return outputs


class SSLNetTensorRTPredictor:
    """Prepare live audio features and run an exported SSLNet TensorRT engine."""

    def __init__(
        self,
        engine_path,
        metadata_path=None,
        device="cuda:0",
        audio_feat=None,
        audio_channels=None,
        ipd_pairs=None,
        use_compress=None,
        expected_sample_rate=None,
        distance_min_m=None,
        distance_max_m=None,
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
        self.engine_path = str(engine_path)
        self.metadata_path = str(metadata_path or f"{self.engine_path}.json")
        with open(self.metadata_path, "r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        self.metadata = metadata
        self.audio_feat = audio_feat or metadata.get("audio_feat", "ipd")
        self.audio_channels = parse_int_tuple(
            audio_channels if audio_channels is not None else metadata.get("audio_channels", [1, 2, 3, 4])
        )
        saved_pairs = metadata.get("ipd_pairs", [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]])
        self.ipd_pairs = parse_channel_pairs(
            ipd_pairs if ipd_pairs is not None else saved_pairs
        )
        self.use_compress = bool(
            metadata.get("use_compress", False) if use_compress is None else use_compress
        )
        self.expected_sample_rate = int(
            metadata.get("sample_rate", 16000)
            if expected_sample_rate is None else expected_sample_rate
        )
        self.distance_min_m = float(
            metadata.get("distance_min_m", 0.0) if distance_min_m is None else distance_min_m
        )
        self.distance_max_m = float(
            metadata.get("distance_max_m", 6.0) if distance_max_m is None else distance_max_m
        )
        self.audio_bandpass_low_hz = float(
            metadata.get("audio_bandpass_low_hz", 0.0)
            if audio_bandpass_low_hz is None else audio_bandpass_low_hz
        )
        self.audio_bandpass_high_hz = float(
            metadata.get("audio_bandpass_high_hz", 0.0)
            if audio_bandpass_high_hz is None else audio_bandpass_high_hz
        )
        self.use_filter_mute_denoise = bool(
            metadata.get("use_filter_mute_denoise", False)
            if use_filter_mute_denoise is None else use_filter_mute_denoise
        )
        self.filter_mute_highpass_hz = float(
            metadata.get("filter_mute_highpass_hz", 120.0)
            if filter_mute_highpass_hz is None else filter_mute_highpass_hz
        )
        saved_notches = metadata.get(
            "filter_mute_notches_hz",
            [48.8, 66.4, 179.7, 341.8, 867.2, 1271.5, 1845.7, 2533.2, 3783.2],
        )
        notch_value = filter_mute_notches_hz if filter_mute_notches_hz is not None else saved_notches
        self.filter_mute_notches_hz = (
            tuple(parse_float_list(notch_value))
            if isinstance(notch_value, str)
            else tuple(float(value) for value in notch_value)
        )
        self.filter_mute_threshold = float(
            metadata.get("filter_mute_threshold", 0.06)
            if filter_mute_threshold is None else filter_mute_threshold
        )
        self.filter_mute_window_sec = float(
            metadata.get("filter_mute_window_sec", 0.05)
            if filter_mute_window_sec is None else filter_mute_window_sec
        )
        self.filter_mute_floor = float(
            metadata.get("filter_mute_floor", 0.02)
            if filter_mute_floor is None else filter_mute_floor
        )
        self.filter_mute_edge_smooth_ms = float(
            metadata.get("filter_mute_edge_smooth_ms", 5.0)
            if filter_mute_edge_smooth_ms is None else filter_mute_edge_smooth_ms
        )
        self.output_names = metadata.get("output_names", ["doa_logits", "distance_logits"])
        self.session = TensorRTEngineSession(self.engine_path, device)
        expected_shape = tuple(metadata.get("input_shape", []))
        if expected_shape and expected_shape[0] != 1:
            raise ValueError(f"Only batch-size 1 TensorRT engines are supported, got {expected_shape}.")

    def describe(self):
        return {
            "engine": self.engine_path,
            "metadata": self.metadata_path,
            "precision": self.metadata.get("precision", "unknown"),
            "device": str(self.session.device),
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
        tensor = torch.as_tensor(feature, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        expected_shape = tuple(self.metadata.get("input_shape", []))
        if expected_shape and tuple(tensor.shape) != expected_shape:
            raise ValueError(
                f"Feature shape {tuple(tensor.shape)} does not match exported engine input {expected_shape}. "
                "Use the same window_seconds and preprocessing as export."
            )
        return tensor

    @torch.no_grad()
    def predict(self, audio, sample_rate):
        started_at = time.perf_counter()
        outputs = self.session.infer(self._prepare_features(audio, sample_rate))
        if "doa_logits" in outputs and "distance_logits" in outputs:
            doa_logits = outputs["doa_logits"]
            distance_logits = outputs["distance_logits"]
        else:
            ordered = [outputs[name] for name in self.output_names if name in outputs]
            if len(ordered) < 2:
                ordered = list(outputs.values())
            doa_logits = ordered[0]
            distance_logits = ordered[1]
        doa_probability = torch.softmax(doa_logits, dim=1)[0].cpu().numpy()
        distance_probability = torch.softmax(distance_logits, dim=1)[0].cpu().numpy()
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
        if "class_logits" in outputs:
            class_probability = torch.softmax(outputs["class_logits"], dim=1)[0].cpu().numpy()
            result["class_probability"] = class_probability
            result["class_id"] = int(np.argmax(class_probability))
        if "signal_logits" in outputs:
            signal_probability = torch.softmax(outputs["signal_logits"], dim=1)[0].cpu().numpy()
            result["signal_probability"] = signal_probability
            result["signal_id"] = int(np.argmax(signal_probability))
            result["signal_prob"] = float(signal_probability[1]) if len(signal_probability) > 1 else 0.0
        return result

#!/usr/bin/env python3
"""Export an audio-only SSLNet checkpoint to ONNX and a TensorRT engine."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deploy.sslnet_realtime import SSLNetStreamingPredictor  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export SSLNet audio DOA/distance inference to TensorRT."
    )
    parser.add_argument(
        "--checkpoint",
        default="weights/pairs_ros1_sslnet_audio/best_model.pth",
        help="Audio-only SSLNet .pth checkpoint.",
    )
    parser.add_argument(
        "--onnx",
        default="weights/pairs_ros1_sslnet_audio/best_model.onnx",
        help="Intermediate ONNX output path.",
    )
    parser.add_argument(
        "--engine",
        default="weights/pairs_ros1_sslnet_audio/best_model.engine",
        help="TensorRT engine output path.",
    )
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--window-seconds", type=float, default=1.0)
    parser.add_argument("--device", default="cuda:0", help="PyTorch export device.")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--workspace-mib", type=int, default=2048)
    parser.add_argument("--fp16", action="store_true", help="Build an FP16 TensorRT engine.")
    parser.add_argument(
        "--trtexec",
        default="trtexec",
        help="Path or executable name for the TensorRT trtexec binary.",
    )
    parser.add_argument(
        "--builder",
        choices=["auto", "python", "trtexec"],
        default="auto",
        help="Engine builder. auto uses trtexec when found, otherwise Python TensorRT.",
    )
    parser.add_argument(
        "--onnx-only",
        action="store_true",
        help="Export ONNX and metadata without invoking trtexec.",
    )
    return parser.parse_args()


def resolved_device(requested):
    if str(requested).startswith("cuda") and torch.cuda.is_available():
        return str(requested)
    if str(requested).startswith("cuda"):
        print("[WARN] CUDA is unavailable to PyTorch; exporting ONNX on CPU.")
    return "cpu"


def metadata_path(engine_path):
    return Path(str(engine_path) + ".json")


def write_metadata(path, predictor, feature_shape, output_names, args):
    config = predictor.describe()
    payload = {
        "format": "sslnet_audio_tensorrt_v1",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "engine": str(Path(args.engine).resolve()),
        "onnx": str(Path(args.onnx).resolve()),
        "precision": "fp16" if args.fp16 else "fp32",
        "input_name": "spectrogram",
        "input_shape": list(feature_shape),
        "output_names": output_names,
        "has_signal_head": "signal_logits" in output_names,
        "num_doa_bins": 360,
        "num_distance_bins": 120,
        "distance_min_m": 0.0,
        "distance_max_m": 6.0,
        "sample_rate": int(args.sample_rate),
        "window_seconds": float(args.window_seconds),
        "audio_feat": config["audio_feat"],
        "audio_channels": list(config["audio_channels"]),
        "ipd_pairs": [list(pair) for pair in config["ipd_pairs"]],
        "use_compress": bool(config["use_compress"]),
        "use_filter_mute_denoise": bool(config["use_filter_mute_denoise"]),
        "audio_bandpass_low_hz": predictor.audio_bandpass_low_hz,
        "audio_bandpass_high_hz": predictor.audio_bandpass_high_hz,
        "filter_mute_highpass_hz": predictor.filter_mute_highpass_hz,
        "filter_mute_notches_hz": list(predictor.filter_mute_notches_hz),
        "filter_mute_threshold": predictor.filter_mute_threshold,
        "filter_mute_window_sec": predictor.filter_mute_window_sec,
        "filter_mute_floor": predictor.filter_mute_floor,
        "filter_mute_edge_smooth_ms": predictor.filter_mute_edge_smooth_ms,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def export_onnx(predictor, args):
    onnx_path = Path(args.onnx)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    required_channels = max(predictor.audio_channels) + 1
    frame_count = int(round(args.sample_rate * args.window_seconds))
    dummy_audio = np.zeros((frame_count, required_channels), dtype=np.float32)
    feature = predictor._prepare_features(dummy_audio, args.sample_rate).to(predictor.device)
    model = predictor.model.eval()
    with torch.no_grad():
        outputs = model(feature)
    output_names = ["doa_logits", "distance_logits"]
    if getattr(model, "class_head", None) is not None:
        output_names.append("class_logits")
    if getattr(model, "signal_head", None) is not None:
        output_names.append("signal_logits")
    try:
        torch.onnx.export(
            model,
            feature,
            str(onnx_path),
            input_names=["spectrogram"],
            output_names=output_names,
            opset_version=args.opset,
            do_constant_folding=True,
        )
    except Exception as exc:
        raise RuntimeError(
            "ONNX export failed. If the error says `Module onnx is not installed`, "
            "install an ONNX package compatible with the PyTorch environment, for example "
            f"`python -m pip install onnx`. Original error: {exc}"
        ) from exc
    return tuple(feature.shape), output_names


def build_engine_with_trtexec(args):
    executable = shutil.which(args.trtexec)
    if executable is None and Path(args.trtexec).is_file():
        executable = args.trtexec
    if executable is None:
        raise RuntimeError(
            "Cannot find trtexec. Run this script in the TensorRT environment or pass "
            "--trtexec /path/to/TensorRT/bin/trtexec."
        )
    command = [
        executable,
        f"--onnx={Path(args.onnx).resolve()}",
        f"--saveEngine={Path(args.engine).resolve()}",
        f"--workspace={int(args.workspace_mib)}",
    ]
    if args.fp16:
        command.append("--fp16")
    print("[INFO] Building TensorRT engine:")
    print(" ".join(command))
    subprocess.run(command, check=True)


def build_engine_with_python(args):
    try:
        import tensorrt as trt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Python TensorRT is not installed. Activate the TensorRT environment, "
            "or use --builder trtexec with a TensorRT binary."
        ) from exc
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(Path(args.onnx).resolve())):
        errors = [str(parser.get_error(index)) for index in range(parser.num_errors)]
        raise RuntimeError("TensorRT ONNX parser failed:\n" + "\n".join(errors))
    config = builder.create_builder_config()
    workspace_bytes = int(args.workspace_mib) * (1 << 20)
    if hasattr(config, "set_memory_pool_limit"):
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
    else:
        config.max_workspace_size = workspace_bytes
    if args.fp16:
        if not builder.platform_has_fast_fp16:
            print("[WARN] TensorRT reports no fast FP16 support; building with FP16 enabled anyway.")
        config.set_flag(trt.BuilderFlag.FP16)
    print("[INFO] Building TensorRT engine through Python TensorRT Builder.")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT Python Builder failed to produce a serialized engine.")
    Path(args.engine).write_bytes(bytes(serialized))


def build_engine(args):
    executable_available = shutil.which(args.trtexec) is not None or Path(args.trtexec).is_file()
    if args.builder == "trtexec" or (args.builder == "auto" and executable_available):
        build_engine_with_trtexec(args)
    else:
        build_engine_with_python(args)


def main():
    args = parse_args()
    device = resolved_device(args.device)
    predictor = SSLNetStreamingPredictor(
        checkpoint_path=args.checkpoint,
        device=device,
        expected_sample_rate=args.sample_rate,
    )
    feature_shape, output_names = export_onnx(predictor, args)
    metadata = metadata_path(args.engine)
    write_metadata(metadata, predictor, feature_shape, output_names, args)
    print(f"[INFO] Exported ONNX model: {Path(args.onnx).resolve()}")
    print(f"[INFO] Input shape: {feature_shape}; outputs: {output_names}")
    print(f"[INFO] Wrote runtime metadata: {metadata.resolve()}")
    if args.onnx_only:
        print("[INFO] Skipped TensorRT build because --onnx-only was set.")
        return
    Path(args.engine).parent.mkdir(parents=True, exist_ok=True)
    build_engine(args)
    print(f"[INFO] Exported TensorRT engine: {Path(args.engine).resolve()}")


if __name__ == "__main__":
    main()

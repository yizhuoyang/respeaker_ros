from pathlib import Path
import re

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from dataloader.utils import (
    apply_audio_bandpass,
    apply_motion_gated_spectral_gate,
    compute_spectrogram,
    compute_stft_phase_features,
    denoise_multichannel_audio,
    filter_and_mute_motion_impacts,
    load_audio_wav,
    load_image,
    load_noise_profile,
    make_doa_gaussian_from_azimuth_deg,
    make_doa_gaussian_from_yaw_deg,
    make_r_gaussian_1d,
    parse_float_list,
    parse_channel_pairs,
    suppress_shared_transients,
)


DOA_FIELDS = [
    "robot_x",
    "robot_y",
    "source_odom_x",
    "source_odom_y",
    "target_x",
    "target_y",
    "target_vector_world_x",
    "target_vector_world_y",
    "heading_world_x",
    "heading_world_y",
    "distance_xy",
    "distance_3d",
    "robot_heading_world_deg",
    "target_azimuth_world_deg",
    "heading_target_yaw_signed_deg",
    "heading_target_yaw_abs_deg",
]

DISTANCE_FIELDS = [
    "distance_xy",
    "distance_3d",
]

CLASS_NAMES = [
    "robot_noise",
    "moving_sound",
    "signal_static",
]

ROBOT_NOISE_CLASS = 0
MOVING_SOUND_CLASS = 1
SIGNAL_STATIC_CLASS = 2


class SingleStepDataset(Dataset):
    """Dataset for synced odom labels or pairs_ros1 labelCloud bbox labels."""

    def __init__(
        self,
        root_dir: str,
        transform=None,
        use_compress=True,
        mode="doa_distance",
        audio_feat="ipd",
        odom_name="lio_odom",
        audio_channels=(1, 2, 3, 4),
        ipd_pairs=((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)),
        image_size=(224, 224),
        require_depth=True,
        require_rgb=False,
        denoise_noise_paths=None,
        denoise_stationary_noise_paths=None,
        denoise_motion_noise_paths=None,
        denoise_highpass_hz=0.0,
        denoise_notches_hz="",
        denoise_spectral_strength=0.0,
        denoise_gain_floor=0.35,
        denoise_motion_strength=0.35,
        denoise_motion_gain_floor=0.55,
        denoise_motion_gate_threshold=1.6,
        denoise_motion_gate_smooth_frames=5,
        denoise_transient_attenuation=0.35,
        denoise_transient_threshold=3.0,
        denoise_transient_frame_ms=20.0,
        denoise_transient_hop_ms=5.0,
        denoise_transient_smooth_frames=5,
        denoise_n_fft=1024,
        denoise_hop=256,
        filter_mute_enabled=False,
        filter_mute_highpass_hz=120.0,
        filter_mute_notches_hz="48.8,66.4,179.7,341.8,867.2,1271.5,1845.7,2533.2,3783.2",
        filter_mute_notch_q=35.0,
        filter_mute_threshold=0.06,
        filter_mute_window_sec=0.05,
        filter_mute_floor=0.02,
        filter_mute_edge_smooth_ms=5.0,
        time_mask_enabled=False,
        time_mask_prob=0.0,
        time_mask_num=1,
        time_mask_max_width=12,
        time_mask_fill="zero",
        split=None,
        object_names=None,
        min_distance=None,
        max_distance=None,
        use_classification=False,
        max_signal_abs=None,
        audio_bandpass_low_hz=0.0,
        audio_bandpass_high_hz=0.0,
    ):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.use_compress = use_compress
        self.mode = mode
        self.audio_feat = audio_feat
        self.odom_name = odom_name
        self.audio_channels = tuple(audio_channels)
        self.ipd_pairs = parse_channel_pairs(ipd_pairs)
        self.image_size = image_size
        self.require_depth = require_depth
        self.require_rgb = require_rgb
        stationary_noise_paths = denoise_stationary_noise_paths
        if stationary_noise_paths is None:
            stationary_noise_paths = denoise_noise_paths
        self.denoise_stationary_noise_paths = tuple(stationary_noise_paths or [])
        self.denoise_motion_noise_paths = tuple(denoise_motion_noise_paths or [])
        self.denoise_highpass_hz = denoise_highpass_hz
        self.denoise_notches_hz = tuple(parse_float_list(denoise_notches_hz))
        self.denoise_spectral_strength = denoise_spectral_strength
        self.denoise_gain_floor = denoise_gain_floor
        self.denoise_motion_strength = denoise_motion_strength
        self.denoise_motion_gain_floor = denoise_motion_gain_floor
        self.denoise_motion_gate_threshold = denoise_motion_gate_threshold
        self.denoise_motion_gate_smooth_frames = denoise_motion_gate_smooth_frames
        self.denoise_transient_attenuation = denoise_transient_attenuation
        self.denoise_transient_threshold = denoise_transient_threshold
        self.denoise_transient_frame_ms = denoise_transient_frame_ms
        self.denoise_transient_hop_ms = denoise_transient_hop_ms
        self.denoise_transient_smooth_frames = denoise_transient_smooth_frames
        self.denoise_n_fft = denoise_n_fft
        self.denoise_hop = denoise_hop
        self.filter_mute_enabled = filter_mute_enabled
        self.filter_mute_highpass_hz = filter_mute_highpass_hz
        self.filter_mute_notches_hz = tuple(parse_float_list(filter_mute_notches_hz))
        self.filter_mute_notch_q = filter_mute_notch_q
        self.filter_mute_threshold = filter_mute_threshold
        self.filter_mute_window_sec = filter_mute_window_sec
        self.filter_mute_floor = filter_mute_floor
        self.filter_mute_edge_smooth_ms = filter_mute_edge_smooth_ms
        self.time_mask_enabled = time_mask_enabled
        self.time_mask_prob = time_mask_prob
        self.time_mask_num = time_mask_num
        self.time_mask_max_width = time_mask_max_width
        self.time_mask_fill = time_mask_fill
        self.split = split
        self.object_names = parse_name_filter(object_names)
        self.min_distance = min_distance
        self.max_distance = max_distance
        self.use_classification = use_classification
        self.max_signal_abs = max_signal_abs
        self.audio_bandpass_low_hz = float(audio_bandpass_low_hz or 0.0)
        self.audio_bandpass_high_hz = float(audio_bandpass_high_hz or 0.0)
        self.skipped_by_distance = 0
        self.skipped_by_signal_amplitude = 0
        self._stationary_noise_profile_by_sample_rate = {}
        self._motion_noise_profile_by_sample_rate = {}

        self.file_list = self._collect_samples()
        if not self.file_list:
            raise RuntimeError(f"No synchronized ROS2 samples found under: {root_dir}")
        if (self.min_distance is not None or self.max_distance is not None) and self.skipped_by_distance:
            print(
                f"[SingleStepDataset] split={self.split}: skipped "
                f"{self.skipped_by_distance} samples outside distance range "
                f"[{self.min_distance}, {self.max_distance}]"
            )
        if self.max_signal_abs and self.skipped_by_signal_amplitude:
            print(
                f"[SingleStepDataset] split={self.split}: skipped "
                f"{self.skipped_by_signal_amplitude} signal samples with max_abs > {self.max_signal_abs}"
            )

    def _collect_samples(self):
        samples = []
        for dataset_dir in self._find_dataset_dirs():
            if is_pairs_dataset_dir(dataset_dir):
                samples.extend(self._collect_pairs_samples(dataset_dir))
                continue

            audio_dir = dataset_dir / "audio"
            depth_dir = dataset_dir / "depth"
            color_dir = dataset_dir / "color"
            doa_dir = dataset_dir / f"doa_{self.odom_name}"
            distance_dir = dataset_dir / f"distance_{self.odom_name}"
            odom_dir = dataset_dir / self.odom_name
            class_label = class_label_from_sequence(dataset_dir.name)
            is_noise = class_label in (ROBOT_NOISE_CLASS, MOVING_SOUND_CLASS)

            if not audio_dir.exists():
                continue
            if not self.use_classification and is_noise:
                continue
            if not is_noise and any(not path.exists() for path in [doa_dir, distance_dir]):
                continue

            for audio_path in sorted(audio_dir.glob("*.wav"), key=lambda p: numeric_stem(p.stem)):
                sample_id = audio_path.stem
                depth_path = find_modality_file(depth_dir, sample_id, [".png", ".npy", ".jpg", ".jpeg"])
                color_path = find_modality_file(color_dir, sample_id, [".png", ".npy", ".jpg", ".jpeg"])
                doa_path = doa_dir / f"{sample_id}.npy"
                distance_path = distance_dir / f"{sample_id}.npy"
                odom_path = odom_dir / f"{sample_id}.npy"

                has_doa = doa_path.exists() and distance_path.exists()
                if not is_noise and not has_doa:
                    continue
                if is_noise:
                    doa_path = None
                    distance_path = None
                if self.require_depth and depth_path is None and not is_noise:
                    continue
                if self.require_rgb and color_path is None and not is_noise:
                    continue
                distance_xy = np.nan
                if has_doa:
                    distance_xy = self._load_distance_xy(doa_path, distance_path)
                    if self._should_skip_by_distance(distance_xy):
                        self.skipped_by_distance += 1
                        continue
                    if self._should_skip_signal_by_amplitude(audio_path):
                        self.skipped_by_signal_amplitude += 1
                        continue

                samples.append({
                    "dataset_dir": dataset_dir,
                    "sample_id": sample_id,
                    "audio": audio_path,
                    "depth": depth_path,
                    "rgb": color_path,
                    "doa": doa_path,
                    "distance": distance_path,
                    "distance_xy": distance_xy,
                    "odom": odom_path if odom_path.exists() else None,
                    "class_label": class_label,
                    "has_doa": has_doa,
                })
        return samples

    def _collect_pairs_samples(self, dataset_dir):
        samples = []
        audio_dir = dataset_dir / "audio"
        label_dir = dataset_dir / "labels"
        depth_dir = dataset_dir / "depth"
        image_dir = dataset_dir / "image"

        for audio_path in sorted(audio_dir.glob("*.wav"), key=lambda p: numeric_stem(p.stem)):
            sample_id = audio_path.stem
            label_path = label_dir / f"{sample_id}.txt"
            target = load_labelcloud_target(label_path)
            if target is None:
                continue
            if self._should_skip_by_distance(target["distance_xy"]):
                self.skipped_by_distance += 1
                continue
            if self._should_skip_signal_by_amplitude(audio_path):
                self.skipped_by_signal_amplitude += 1
                continue

            depth_path = find_modality_file(depth_dir, sample_id, [".png", ".npy", ".jpg", ".jpeg"])
            rgb_path = find_modality_file(image_dir, sample_id, [".png", ".npy", ".jpg", ".jpeg"])
            if self.require_depth and depth_path is None:
                continue
            if self.require_rgb and rgb_path is None:
                continue
            samples.append({
                "dataset_dir": dataset_dir,
                "sample_id": sample_id,
                "audio": audio_path,
                "depth": depth_path,
                "rgb": rgb_path,
                "label": label_path,
                "label_type": "labelcloud_xy",
                "azimuth_deg": target["azimuth_deg"],
                "distance_xy": target["distance_xy"],
                "distance_3d": target["distance_3d"],
                "target_x": target["x"],
                "target_y": target["y"],
                "class_label": SIGNAL_STATIC_CLASS,
                "has_doa": True,
            })
        return samples

    def _load_distance_xy(self, doa_path, distance_path):
        distance_values = np.load(distance_path).astype(np.float32).reshape(-1)
        if len(distance_values) > 0 and np.isfinite(distance_values[0]):
            return float(distance_values[0])
        doa_values = np.load(doa_path).astype(np.float32).reshape(-1)
        doa = array_to_named_values(doa_values, DOA_FIELDS)
        return float(doa.get("distance_xy", np.nan))

    def _should_skip_by_distance(self, distance_xy):
        if not np.isfinite(distance_xy):
            return self.min_distance is not None or self.max_distance is not None
        if self.min_distance is not None and distance_xy < self.min_distance:
            return True
        if self.max_distance is not None and distance_xy > self.max_distance:
            return True
        return False

    def _should_skip_signal_by_amplitude(self, audio_path):
        if self.max_signal_abs is None or self.max_signal_abs <= 0:
            return False
        audio, _ = load_audio_wav(audio_path, self.audio_channels)
        return float(np.max(np.abs(audio))) > float(self.max_signal_abs)

    def _find_dataset_dirs(self):
        search_root = self.root_dir
        if self.split:
            search_root = self.root_dir / self.split
            if not search_root.exists():
                raise RuntimeError(f"Dataset split does not exist: {search_root}")

        if is_dataset_dir(search_root):
            candidates = [search_root]
        else:
            candidates = sorted(path.parent for path in search_root.glob("**/audio") if is_dataset_dir(path.parent))

        if self.object_names:
            candidates = [
                path for path in candidates
                if matches_object_filter(path.name, self.object_names)
                or (self.use_classification and is_noise_sequence(path.name))
            ]

        return candidates

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx: int):
        item = self.file_list[idx]

        audio, sample_rate = load_audio_wav(item["audio"], self.audio_channels)
        audio = self._maybe_filter_mute_audio(audio, sample_rate)
        audio = self._maybe_denoise_audio(audio, sample_rate)
        audio = self._maybe_apply_audio_bandpass(audio, sample_rate)
        depth = (
            load_image(item["depth"], normalize_rgb=False, image_size=self.image_size)
            if item["depth"] is not None
            else make_empty_depth(self.image_size)
        )
        rgb = (
            load_image(item["rgb"], normalize_rgb=True, image_size=self.image_size)
            if item["rgb"] is not None
            else make_empty_rgb(depth)
        )

        has_doa = bool(item.get("has_doa", True))
        if item.get("label_type") == "labelcloud_xy":
            doa = {
                "robot_x": 0.0,
                "robot_y": 0.0,
                "target_x": item["target_x"],
                "target_y": item["target_y"],
                "robot_heading_world_deg": 0.0,
            }
            yaw_signed_deg = float(item["azimuth_deg"])
            distance_xy = float(item["distance_xy"])
            distance_3d = float(item["distance_3d"])
            doa_reference = "azimuth_xy"
        elif has_doa:
            doa_values = np.load(item["doa"]).astype(np.float32)
            distance_values = np.load(item["distance"]).astype(np.float32)
            doa = array_to_named_values(doa_values, DOA_FIELDS)
            distance = array_to_named_values(distance_values, DISTANCE_FIELDS)

            yaw_signed_deg = float(doa["heading_target_yaw_signed_deg"])
            distance_xy = float(distance.get("distance_xy", doa.get("distance_xy", 0.0)))
            distance_3d = float(distance.get("distance_3d", doa.get("distance_3d", distance_xy)))
            doa_reference = "heading_yaw"
        else:
            doa = {
                "robot_x": 0.0,
                "robot_y": 0.0,
                "target_x": 0.0,
                "target_y": 0.0,
                "robot_heading_world_deg": 0.0,
            }
            yaw_signed_deg = 0.0
            distance_xy = 0.0
            distance_3d = 0.0
            doa_reference = "heading_yaw"

        if self.audio_feat == "spec":
            spectrogram = compute_spectrogram(audio, self.use_compress)
        else:
            spectrogram = compute_stft_phase_features(audio, mode=self.audio_feat, pairs=self.ipd_pairs)
        spectrogram = self._maybe_apply_time_mask(spectrogram)

        if doa_reference == "azimuth_xy":
            doa_map = make_doa_gaussian_from_azimuth_deg(
                yaw_signed_deg,
                distance_m=distance_xy,
                num_bins=360,
                base_sigma_deg=3.0,
                sigma_scale_deg=1.0,
            )
        else:
            doa_map = make_doa_gaussian_from_yaw_deg(
                yaw_signed_deg,
                distance_m=distance_xy,
                num_bins=360,
                base_sigma_deg=3.0,
                sigma_scale_deg=1.0,
            )
        distant_map = make_r_gaussian_1d(
            distance_xy,
            num_bins=120,
            r_min=0.0,
            r_max=6.0,
            base_sigma=0.08,
            sigma_scale=0.05,
        )

        pose = np.array([doa["robot_x"], doa["robot_y"], 0.0], dtype=np.float32)
        source_position = np.array([doa["target_x"], doa["target_y"], 0.0], dtype=np.float32)
        heading_rad = np.deg2rad(float(doa["robot_heading_world_deg"]))

        sample = {
            "depth": image_to_chw_tensor(depth),
            "rgb": image_to_chw_tensor(rgb),
            "audio_wave": torch.as_tensor(audio, dtype=torch.float32),
            "spectrogram": torch.as_tensor(spectrogram, dtype=torch.float32).permute(2, 0, 1),
            "doa_map": torch.as_tensor(doa_map, dtype=torch.float32),
            "distant_map": torch.as_tensor(distant_map, dtype=torch.float32),
            "distance": torch.as_tensor([distance_xy, distance_3d], dtype=torch.float32),
            "doa_deg": torch.as_tensor(yaw_signed_deg, dtype=torch.float32),
            "doa_reference": doa_reference,
            "pose": torch.as_tensor(pose, dtype=torch.float32),
            "heading": heading_rad,
            "sound_source": torch.as_tensor(source_position, dtype=torch.float32),
            "path": str(item["audio"]),
            "sample_id": item["sample_id"],
            "dataset_dir": str(item["dataset_dir"]),
            "sample_rate": sample_rate,
            "class_label": torch.as_tensor(int(item.get("class_label", SIGNAL_STATIC_CLASS)), dtype=torch.long),
            "has_doa": torch.as_tensor(has_doa, dtype=torch.bool),
        }

        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def _maybe_denoise_audio(self, audio, sample_rate):
        use_time_filters = self.denoise_highpass_hz > 0 or len(self.denoise_notches_hz) > 0
        use_stationary = self.denoise_stationary_noise_paths and self.denoise_spectral_strength > 0
        use_motion = self.denoise_motion_noise_paths and self.denoise_motion_strength > 0
        use_transient = self.denoise_transient_attenuation > 0
        if not use_time_filters and not use_stationary and not use_motion and not use_transient:
            return audio

        stationary_noise_power = None
        if use_stationary:
            if sample_rate not in self._stationary_noise_profile_by_sample_rate:
                self._stationary_noise_profile_by_sample_rate[sample_rate] = load_noise_profile(
                    noise_paths=self.denoise_stationary_noise_paths,
                    channels=self.audio_channels,
                    sample_rate=sample_rate,
                    n_fft=self.denoise_n_fft,
                    hop=self.denoise_hop,
                    highpass_hz=self.denoise_highpass_hz,
                    notches_hz=self.denoise_notches_hz,
                    notch_q=35.0,
                )
            stationary_noise_power = self._stationary_noise_profile_by_sample_rate[sample_rate]

        denoised = denoise_multichannel_audio(
            audio=audio,
            sample_rate=sample_rate,
            noise_power=stationary_noise_power,
            highpass_hz=self.denoise_highpass_hz,
            notches_hz=self.denoise_notches_hz,
            spectral_strength=self.denoise_spectral_strength,
            gain_floor=self.denoise_gain_floor,
            n_fft=self.denoise_n_fft,
            hop=self.denoise_hop,
        )
        if not use_motion:
            return self._maybe_suppress_transients(denoised, sample_rate)

        if sample_rate not in self._motion_noise_profile_by_sample_rate:
            self._motion_noise_profile_by_sample_rate[sample_rate] = load_noise_profile(
                noise_paths=self.denoise_motion_noise_paths,
                channels=self.audio_channels,
                sample_rate=sample_rate,
                n_fft=self.denoise_n_fft,
                hop=self.denoise_hop,
                highpass_hz=self.denoise_highpass_hz,
                notches_hz=self.denoise_notches_hz,
                notch_q=35.0,
            )
        denoised = apply_motion_gated_spectral_gate(
            audio=denoised,
            sample_rate=sample_rate,
            noise_power=self._motion_noise_profile_by_sample_rate[sample_rate],
            n_fft=self.denoise_n_fft,
            hop=self.denoise_hop,
            strength=self.denoise_motion_strength,
            gain_floor=self.denoise_motion_gain_floor,
            gate_threshold=self.denoise_motion_gate_threshold,
            gate_smooth_frames=self.denoise_motion_gate_smooth_frames,
        )
        return self._maybe_suppress_transients(denoised, sample_rate)

    def _maybe_filter_mute_audio(self, audio, sample_rate):
        if not self.filter_mute_enabled:
            return audio
        return filter_and_mute_motion_impacts(
            audio,
            sample_rate=sample_rate,
            highpass_hz=self.filter_mute_highpass_hz,
            notches_hz=self.filter_mute_notches_hz,
            notch_q=self.filter_mute_notch_q,
            motion_threshold=self.filter_mute_threshold,
            mute_window_sec=self.filter_mute_window_sec,
            mute_floor=self.filter_mute_floor,
            edge_smooth_ms=self.filter_mute_edge_smooth_ms,
        )

    def _maybe_apply_audio_bandpass(self, audio, sample_rate):
        if self.audio_bandpass_low_hz <= 0 and self.audio_bandpass_high_hz <= 0:
            return audio
        return apply_audio_bandpass(
            audio,
            sample_rate=sample_rate,
            low_hz=self.audio_bandpass_low_hz,
            high_hz=self.audio_bandpass_high_hz,
        )

    def _maybe_apply_time_mask(self, spectrogram):
        if (
            not self.time_mask_enabled
            or self.time_mask_prob <= 0
            or self.time_mask_num <= 0
            or self.time_mask_max_width <= 0
        ):
            return spectrogram
        if np.random.random() > self.time_mask_prob:
            return spectrogram

        augmented = np.array(spectrogram, copy=True)
        num_frames = augmented.shape[1]
        if num_frames <= 1:
            return augmented

        max_width = min(int(self.time_mask_max_width), num_frames)
        for _ in range(int(self.time_mask_num)):
            width = np.random.randint(1, max_width + 1)
            start = np.random.randint(0, num_frames - width + 1)
            if self.time_mask_fill == "mean":
                fill_value = np.mean(augmented, axis=1, keepdims=True)
                augmented[:, start:start + width, :] = fill_value
            else:
                augmented[:, start:start + width, :] = 0.0
        return augmented.astype(np.float32)

    def _maybe_suppress_transients(self, audio, sample_rate):
        if self.denoise_transient_attenuation <= 0:
            return audio
        return suppress_shared_transients(
            audio,
            sample_rate=sample_rate,
            frame_ms=self.denoise_transient_frame_ms,
            hop_ms=self.denoise_transient_hop_ms,
            threshold=self.denoise_transient_threshold,
            attenuation=self.denoise_transient_attenuation,
            smooth_frames=self.denoise_transient_smooth_frames,
        )

    def get_yaw_deg(self, idx: int):
        item = self.file_list[idx]
        if item.get("label_type") == "labelcloud_xy":
            return float(item["azimuth_deg"])
        if not item.get("has_doa", True):
            return 0.0
        doa_values = np.load(item["doa"]).astype(np.float32)
        doa = array_to_named_values(doa_values, DOA_FIELDS)
        return float(doa["heading_target_yaw_signed_deg"])

    def get_class_label(self, idx: int):
        item = self.file_list[idx]
        return int(item.get("class_label", SIGNAL_STATIC_CLASS))


def array_to_named_values(values, fields):
    return {name: values[i] for i, name in enumerate(fields[:len(values)])}


def numeric_stem(stem):
    try:
        return int(stem)
    except ValueError:
        return stem


def is_dataset_dir(path):
    return (path / "audio").exists()


def is_pairs_dataset_dir(path):
    return (path / "audio").exists() and (path / "labels").exists()


def load_labelcloud_target(path):
    if not path.exists():
        return None
    lines = [line for line in path.read_text().splitlines() if line.strip()]
    targets = []
    for line in lines:
        fields = line.split()
        if len(fields) < 15:
            raise ValueError(f"Expected at least 15 labelCloud fields in {path}, got {len(fields)}")
        if fields[0].lower() not in {"person", "pedestrian"}:
            continue
        x, y, z = map(float, fields[11:14])
        distance_xy = float(np.hypot(x, y))
        targets.append({
            "x": x,
            "y": y,
            "z": z,
            "distance_xy": distance_xy,
            "distance_3d": float(np.sqrt(x * x + y * y + z * z)),
            "azimuth_deg": float(np.degrees(np.arctan2(y, x)) % 360.0),
        })
    if not targets:
        return None
    return min(targets, key=lambda target: target["distance_xy"])


def parse_name_filter(value):
    if value is None:
        return set()
    if isinstance(value, str):
        return {item.strip() for item in value.split(",") if item.strip()}
    return {str(item).strip() for item in value if str(item).strip()}


def object_prefix(sequence_name):
    return re.sub(r"\d+$", "", sequence_name)


def matches_object_filter(sequence_name, object_names):
    return sequence_name in object_names or object_prefix(sequence_name) in object_names


def class_label_from_sequence(sequence_name):
    if sequence_name == "robot_noise":
        return ROBOT_NOISE_CLASS
    if sequence_name == "moving_sound":
        return MOVING_SOUND_CLASS
    return SIGNAL_STATIC_CLASS


def is_noise_sequence(sequence_name):
    return class_label_from_sequence(sequence_name) in (ROBOT_NOISE_CLASS, MOVING_SOUND_CLASS)


def find_modality_file(directory, sample_id, suffixes):
    if not directory.exists():
        return None
    for suffix in suffixes:
        path = directory / f"{sample_id}{suffix}"
        if path.exists():
            return path
    return None


def image_to_chw_tensor(image):
    tensor = torch.as_tensor(np.array(image, copy=True), dtype=torch.float32)
    if tensor.ndim == 2:
        return tensor.unsqueeze(0)
    return tensor.permute(2, 0, 1)


def make_empty_rgb(reference):
    height, width = reference.shape[:2]
    return np.zeros((height, width, 3), dtype=np.float32)


def make_empty_depth(image_size):
    height, width = image_size
    return np.zeros((height, width), dtype=np.float32)


if __name__ == "__main__":
    root_dir = "/home/kemove/yyz/audio-nav/ws_col/synced_dataset"
    dataset = SingleStepDataset(root_dir, use_compress=False, audio_feat="ipd")
    print(len(dataset), "samples found in the dataset.")
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
    batch = next(iter(loader))
    for key in ["depth", "rgb", "audio_wave", "spectrogram", "doa_map", "distant_map", "distance"]:
        print(key, batch[key].shape)

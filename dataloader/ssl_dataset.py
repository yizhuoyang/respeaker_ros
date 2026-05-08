from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from dataloader.utils import (
    compute_spectrogram,
    compute_stft_phase_features,
    load_audio_wav,
    load_image,
    make_doa_gaussian_from_yaw_deg,
    make_r_gaussian_1d,
    parse_channel_pairs,
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


class SingleStepDataset(Dataset):
    """Dataset for synchronized ROS2 audio/depth/DOA/distance samples."""

    def __init__(
        self,
        root_dir: str,
        transform=None,
        use_compress=True,
        mode="doa_distance",
        audio_feat="ipd",
        odom_name="lio_odom",
        audio_channels=(0, 1, 2, 3),
        ipd_pairs=((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)),
        image_size=(224, 224),
        require_depth=True,
        require_rgb=False,
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

        self.file_list = self._collect_samples()
        if not self.file_list:
            raise RuntimeError(f"No synchronized ROS2 samples found under: {root_dir}")

    def _collect_samples(self):
        samples = []
        for dataset_dir in self._find_dataset_dirs():
            audio_dir = dataset_dir / "audio"
            depth_dir = dataset_dir / "depth"
            color_dir = dataset_dir / "color"
            doa_dir = dataset_dir / f"doa_{self.odom_name}"
            distance_dir = dataset_dir / f"distance_{self.odom_name}"
            odom_dir = dataset_dir / self.odom_name

            if any(not path.exists() for path in [audio_dir, doa_dir, distance_dir]):
                continue

            for audio_path in sorted(audio_dir.glob("*.wav"), key=lambda p: numeric_stem(p.stem)):
                sample_id = audio_path.stem
                depth_path = find_modality_file(depth_dir, sample_id, [".png", ".npy", ".jpg", ".jpeg"])
                color_path = find_modality_file(color_dir, sample_id, [".png", ".npy", ".jpg", ".jpeg"])
                doa_path = doa_dir / f"{sample_id}.npy"
                distance_path = distance_dir / f"{sample_id}.npy"
                odom_path = odom_dir / f"{sample_id}.npy"

                if not doa_path.exists() or not distance_path.exists():
                    continue
                if self.require_depth and depth_path is None:
                    continue
                if self.require_rgb and color_path is None:
                    continue

                samples.append({
                    "dataset_dir": dataset_dir,
                    "sample_id": sample_id,
                    "audio": audio_path,
                    "depth": depth_path,
                    "rgb": color_path,
                    "doa": doa_path,
                    "distance": distance_path,
                    "odom": odom_path if odom_path.exists() else None,
                })
        return samples

    def _find_dataset_dirs(self):
        if (self.root_dir / "audio").exists():
            return [self.root_dir]
        return sorted(path for path in self.root_dir.iterdir() if path.is_dir())

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx: int):
        item = self.file_list[idx]

        audio, sample_rate = load_audio_wav(item["audio"], self.audio_channels)
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

        doa_values = np.load(item["doa"]).astype(np.float32)
        distance_values = np.load(item["distance"]).astype(np.float32)
        doa = array_to_named_values(doa_values, DOA_FIELDS)
        distance = array_to_named_values(distance_values, DISTANCE_FIELDS)

        yaw_signed_deg = float(doa["heading_target_yaw_signed_deg"])
        distance_xy = float(distance.get("distance_xy", doa.get("distance_xy", 0.0)))
        distance_3d = float(distance.get("distance_3d", doa.get("distance_3d", distance_xy)))

        if self.audio_feat == "spec":
            spectrogram = compute_spectrogram(audio, self.use_compress)
        else:
            spectrogram = compute_stft_phase_features(audio, mode=self.audio_feat, pairs=self.ipd_pairs)

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
            "pose": torch.as_tensor(pose, dtype=torch.float32),
            "heading": heading_rad,
            "sound_source": torch.as_tensor(source_position, dtype=torch.float32),
            "path": str(item["audio"]),
            "sample_id": item["sample_id"],
            "dataset_dir": str(item["dataset_dir"]),
            "sample_rate": sample_rate,
        }

        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def get_yaw_deg(self, idx: int):
        item = self.file_list[idx]
        doa_values = np.load(item["doa"]).astype(np.float32)
        doa = array_to_named_values(doa_values, DOA_FIELDS)
        return float(doa["heading_target_yaw_signed_deg"])


def array_to_named_values(values, fields):
    return {name: values[i] for i, name in enumerate(fields[:len(values)])}


def numeric_stem(stem):
    try:
        return int(stem)
    except ValueError:
        return stem


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

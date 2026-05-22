from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


DOA_FIELDS = [
    "azimuth_rad",
    "elevation_rad",
    "unit_x",
    "unit_y",
    "unit_z",
    "distance_m",
]


class AudioVisualDoaDataset(Dataset):
    """Dataset for synchronized audio/image/depth/DOA samples.

    Expected directory layout:

        root_dir/
          bag_name/
            audio/0001.wav
            image/0001.png
            depth/0001.png
            doa/0001.npy

    DOA npy format:
        [azimuth_rad, elevation_rad, unit_x, unit_y, unit_z, distance_m]
    """

    def __init__(
        self,
        root_dir,
        transform=None,
        audio_channels=(1, 2, 3, 4),
        image_size=(224, 224),
        require_image=True,
        require_depth=True,
        audio_feat=None,
        use_compress=True,
        ipd_pairs=((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)),
        depth_scale=1000.0,
        doa_num_bins=360,
        doa_sigma_deg=3.0,
        distance_num_bins=120,
        distance_min=0.0,
        distance_max=6.0,
        distance_sigma=0.08,
    ):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.audio_channels = parse_audio_channels(audio_channels)
        self.image_size = image_size
        self.require_image = require_image
        self.require_depth = require_depth
        self.audio_feat = audio_feat
        self.use_compress = use_compress
        self.ipd_pairs = parse_channel_pairs(ipd_pairs)
        self.depth_scale = float(depth_scale)
        self.doa_num_bins = int(doa_num_bins)
        self.doa_sigma_deg = float(doa_sigma_deg)
        self.distance_num_bins = int(distance_num_bins)
        self.distance_min = float(distance_min)
        self.distance_max = float(distance_max)
        self.distance_sigma = float(distance_sigma)

        self.file_list = self._collect_samples()
        if not self.file_list:
            raise RuntimeError(f"No synchronized audio/image/depth/doa samples found under: {root_dir}")

    def _collect_samples(self):
        samples = []
        for dataset_dir in self._find_dataset_dirs():
            audio_dir = dataset_dir / "audio"
            image_dir = dataset_dir / "image"
            depth_dir = dataset_dir / "depth"
            doa_dir = dataset_dir / "doa"

            if not audio_dir.exists() or not doa_dir.exists():
                continue

            for audio_path in sorted(audio_dir.glob("*.wav"), key=lambda p: numeric_stem(p.stem)):
                sample_id = audio_path.stem
                image_path = find_modality_file(image_dir, sample_id, [".png", ".jpg", ".jpeg", ".npy"])
                depth_path = find_modality_file(depth_dir, sample_id, [".png", ".npy"])
                doa_path = doa_dir / f"{sample_id}.npy"

                if not doa_path.exists():
                    continue
                if self.require_image and image_path is None:
                    continue
                if self.require_depth and depth_path is None:
                    continue

                samples.append(
                    {
                        "dataset_dir": dataset_dir,
                        "sample_id": sample_id,
                        "audio": audio_path,
                        "image": image_path,
                        "depth": depth_path,
                        "doa": doa_path,
                    }
                )

        return samples

    def _find_dataset_dirs(self):
        if (self.root_dir / "audio").exists():
            return [self.root_dir]
        return sorted(path for path in self.root_dir.iterdir() if path.is_dir())

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        item = self.file_list[idx]

        audio, sample_rate = load_audio_wav(item["audio"], channels=self.audio_channels)
        image = (
            load_image(item["image"], normalize_rgb=True, image_size=self.image_size)
            if item["image"] is not None
            else make_empty_image(self.image_size)
        )
        depth = (
            load_depth(item["depth"], image_size=self.image_size, depth_scale=self.depth_scale)
            if item["depth"] is not None
            else make_empty_depth(self.image_size)
        )

        doa_values = np.load(item["doa"]).astype(np.float32)
        doa = array_to_named_values(doa_values, DOA_FIELDS)
        azimuth_rad = float(doa["azimuth_rad"])
        distance_m = float(doa["distance_m"])
        doa_map = make_doa_gaussian_from_azimuth_rad(
            azimuth_rad,
            num_bins=self.doa_num_bins,
            sigma_deg=self.doa_sigma_deg,
        )
        distance_map = make_distance_gaussian_1d(
            distance_m,
            num_bins=self.distance_num_bins,
            r_min=self.distance_min,
            r_max=self.distance_max,
            sigma=self.distance_sigma,
        )

        sample = {
            "audio_wave": torch.as_tensor(audio, dtype=torch.float32),
            "image": image_to_chw_tensor(image),
            "depth": image_to_chw_tensor(depth),
            "doa_map": torch.as_tensor(doa_map, dtype=torch.float32),
            "distance_map": torch.as_tensor(distance_map, dtype=torch.float32),
            "azimuth_rad": torch.as_tensor(azimuth_rad, dtype=torch.float32),
            "distance_m": torch.as_tensor(distance_m, dtype=torch.float32),
            "target": torch.as_tensor([azimuth_rad, distance_m], dtype=torch.float32),
            "path": str(item["audio"]),
            "sample_id": item["sample_id"],
            "dataset_dir": str(item["dataset_dir"]),
            "sample_rate": sample_rate,
        }

        if self.audio_feat is not None:
            sample["spectrogram"] = torch.as_tensor(
                compute_audio_feature(
                    audio,
                    mode=self.audio_feat,
                    use_compress=self.use_compress,
                    pairs=self.ipd_pairs,
                ),
                dtype=torch.float32,
            ).permute(2, 0, 1)

        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def get_azimuth_rad(self, idx):
        doa_values = np.load(self.file_list[idx]["doa"]).astype(np.float32)
        return float(doa_values[0])

    def get_distance_m(self, idx):
        doa_values = np.load(self.file_list[idx]["doa"]).astype(np.float32)
        return float(doa_values[5])


def array_to_named_values(values, fields):
    return {name: values[i] for i, name in enumerate(fields[: len(values)])}


def parse_audio_channels(value):
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if "," in text:
            return tuple(int(item.strip()) for item in text.split(",") if item.strip())
        return tuple(int(item) for item in text)
    return tuple(value)


def make_doa_gaussian_from_azimuth_rad(azimuth_rad, num_bins=360, sigma_deg=3.0):
    azimuth_deg = np.rad2deg(float(azimuth_rad))
    angle_deg = azimuth_deg % 360.0
    angles_deg = np.linspace(0.0, 360.0, int(num_bins), endpoint=False)
    diff_deg = (angles_deg - angle_deg + 180.0) % 360.0 - 180.0
    sigma_deg = max(float(sigma_deg), 1e-6)
    probs = np.exp(-0.5 * (diff_deg / sigma_deg) ** 2)
    if probs.sum() > 0.0:
        probs = probs / probs.sum()
    return probs.astype(np.float32)


def make_distance_gaussian_1d(distance_m, num_bins=120, r_min=0.0, r_max=6.0, sigma=0.08):
    r_axis = np.linspace(float(r_min), float(r_max), int(num_bins), dtype=np.float32)
    sigma = max(float(sigma), 1e-6)
    probs = np.exp(-0.5 * ((r_axis - float(distance_m)) / sigma) ** 2)
    if probs.sum() > 0.0:
        probs = probs / probs.sum()
    return probs.astype(np.float32)


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


def load_audio_wav(path, channels=None):
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

    if channels is not None:
        if max(channels) >= audio.shape[1]:
            raise RuntimeError(f"Audio file {path} has {audio.shape[1]} channels, requested {channels}")
        audio = audio[:, channels]

    return audio.T.astype(np.float32), sample_rate


def load_image(path, normalize_rgb=True, image_size=None):
    path = Path(path)
    if path.suffix == ".npy":
        image = np.load(path)
    else:
        image = np.asarray(Image.open(path))

    image = image.astype(np.float32)
    if image.ndim == 3 and normalize_rgb and image.max() > 1.0:
        image = image / 255.0
    if image_size is not None:
        image = resize_image(image, image_size, normalize_rgb=normalize_rgb)
    return image


def load_depth(path, image_size=None, depth_scale=1000.0):
    depth = load_image(path, normalize_rgb=False, image_size=None)
    if depth.ndim == 3:
        depth = depth[:, :, 0]
    if depth.max() > 100.0:
        depth = depth / float(depth_scale)
    if image_size is not None:
        depth = resize_image(depth, image_size, normalize_rgb=False)
    return depth.astype(np.float32)


def resize_image(image, image_size, normalize_rgb=True):
    height, width = image_size
    if image.ndim == 2:
        pil_image = Image.fromarray(image.astype(np.float32), mode="F")
        return np.asarray(pil_image.resize((width, height), resample=Image.BILINEAR), dtype=np.float32)

    image = np.asarray(image)
    if normalize_rgb:
        pil_image = Image.fromarray((np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8))
        resized = np.asarray(pil_image.resize((width, height), resample=Image.BILINEAR), dtype=np.float32)
        return resized / 255.0

    pil_image = Image.fromarray(image.astype(np.uint8))
    return np.asarray(pil_image.resize((width, height), resample=Image.BILINEAR), dtype=np.float32)


def image_to_chw_tensor(image):
    tensor = torch.as_tensor(np.array(image, copy=True), dtype=torch.float32)
    if tensor.ndim == 2:
        return tensor.unsqueeze(0)
    return tensor.permute(2, 0, 1)


def make_empty_image(image_size):
    height, width = image_size
    return np.zeros((height, width, 3), dtype=np.float32)


def make_empty_depth(image_size):
    height, width = image_size
    return np.zeros((height, width), dtype=np.float32)


def parse_channel_pairs(value):
    if isinstance(value, str):
        pairs = []
        for item in value.split(","):
            item = item.strip()
            if not item:
                continue
            first, second = item.replace(":", "-").split("-")
            pairs.append((int(first), int(second)))
        return tuple(pairs)
    return tuple(tuple(pair) for pair in value)


def compute_audio_feature(audio, mode="ipd", use_compress=True, pairs=((0, 1),)):
    if mode == "spec":
        return compute_spectrogram(audio, use_compress=use_compress)
    return compute_stft_phase_features(audio, mode=mode, pairs=pairs)


def compute_spectrogram(audio, use_compress=True):
    channels = [np.log1p(compute_stft(channel, use_compress)) for channel in audio]
    return np.stack(channels, axis=-1).astype(np.float32)


def compute_stft(signal, use_compress=True):
    import librosa

    stft = np.abs(librosa.stft(signal, n_fft=512, hop_length=160, win_length=400))
    if use_compress:
        stft = block_reduce_mean(stft, block_size=(4, 4))
    return stft


def compute_stft_phase_features(
    audio,
    mode="ipd",
    pairs=((0, 1),),
    eps=1e-8,
):
    import librosa

    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim != 2:
        raise RuntimeError(f"Expected audio shape (C, N), got {audio.shape}")

    spectra = [librosa.stft(audio[channel], n_fft=512, hop_length=160, win_length=400) for channel in range(audio.shape[0])]
    phases = [np.angle(spec) for spec in spectra]

    phase_feats = []
    if mode in ("phase", "both"):
        for phase in phases:
            phase_feats.extend([np.cos(phase), np.sin(phase)])

    pair_feats = []
    if mode in ("ipd", "both"):
        for first, second in pairs:
            check_pair(first, second, audio.shape[0])
            ipd = np.angle(np.exp(1j * (phases[first] - phases[second])))
            pair_feats.extend([np.cos(ipd), np.sin(ipd)])

    if mode == "gcc_phat_complex":
        for first, second in pairs:
            check_pair(first, second, audio.shape[0])
            cross = spectra[first] * np.conj(spectra[second])
            cross = cross / (np.abs(cross) + eps)
            pair_feats.extend([cross.real, cross.imag])

    if mode == "phase":
        return np.stack(phase_feats, axis=-1).astype(np.float32)
    if mode == "ipd":
        return np.stack(pair_feats, axis=-1).astype(np.float32)
    if mode == "both":
        return np.stack(phase_feats + pair_feats, axis=-1).astype(np.float32)
    if mode == "gcc_phat_complex":
        return np.stack(pair_feats, axis=-1).astype(np.float32)

    raise ValueError(f"Unknown audio feature mode: {mode}")


def block_reduce_mean(array, block_size):
    height_block, width_block = block_size
    height = array.shape[0] // height_block * height_block
    width = array.shape[1] // width_block * width_block
    trimmed = array[:height, :width]
    reshaped = trimmed.reshape(height // height_block, height_block, width // width_block, width_block)
    return reshaped.mean(axis=(1, 3))


def check_pair(first, second, num_channels):
    if first >= num_channels or second >= num_channels:
        raise RuntimeError(
            f"IPD pair ({first}, {second}) requires {max(first, second) + 1} channels, "
            f"but audio has {num_channels} channels"
        )


if __name__ == "__main__":
    dataset = AudioVisualDoaDataset("data", audio_channels=(1, 2, 3, 4), audio_feat=None)
    print(len(dataset), "samples found in the dataset.")
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
    batch = next(iter(loader))
    for key in ["audio_wave", "image", "depth", "doa_map", "distance_map", "azimuth_rad", "distance_m"]:
        print(key, batch[key].shape)

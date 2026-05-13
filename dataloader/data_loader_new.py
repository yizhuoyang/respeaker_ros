import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as Trans
import torch
import matplotlib.pyplot as plt
import librosa
import torch.utils.data as data
import os
import re
import pandas as pd
import numpy as np
import random
import soundfile as sf
import scipy.io as sio

from dataloader.utils import (
    filter_and_mute_motion_impacts,
    load_audio_wav,
    parse_float_list,
)

try:
    sys.path.append('/home/kemove/yyz/SubspaceNet/DeepMucis_plus')
    from DeepMucis_plus.dataset.data_loader import load_dataframe, load_numpy
    from DeepMucis_plus.utlis.util import filter_folders, downsample_audio, normalize_magnitude, normalize_phase, \
        ModeVector_torch, doa_xy_deg_from_xyz, load_gt_dict, stem, CalibFromMat, parse_gt_xyz, doa_xz_deg_from_xyz_cav3d
    from DeepMucis_plus.dataset.data_augmentation import simulate_microphone_desync, add_gaussian_noise
    import pysensing.acoustic.preprocessing.transform as transform
except ModuleNotFoundError:
    load_dataframe = load_numpy = None
    simulate_microphone_desync = None

    def filter_folders(folder_list, n):
        if n == 0:
            return folder_list
        return [folder for folder in folder_list if f"_{n}" in folder]

    def normalize_magnitude(magnitude):
        magnitude = torch.log1p(magnitude)
        mean = magnitude.mean()
        std = magnitude.std().clamp_min(1e-6)
        return (magnitude - mean) / std

    def normalize_phase(phase):
        return phase / torch.pi

    def add_gaussian_noise(audio_data, target_snr_db):
        audio = np.asarray(audio_data, dtype=np.float32)
        signal_power = np.mean(audio * audio) + 1e-10
        noise_power = signal_power / (10 ** (target_snr_db / 10.0))
        noise = np.random.normal(0.0, np.sqrt(noise_power), size=audio.shape).astype(np.float32)
        return audio + noise

    class _LocalSTFT:
        def __init__(self, n_fft=512, hop_length=256):
            self.n_fft = n_fft
            self.hop_length = hop_length

        def __call__(self, audio_data):
            return torch.stft(
                audio_data,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.n_fft,
                window=torch.hann_window(self.n_fft, device=audio_data.device),
                return_complex=True,
            )

    class _LocalTransform:
        @staticmethod
        def stft(n_fft=512, hop_length=256):
            return _LocalSTFT(n_fft=n_fft, hop_length=hop_length)

    transform = _LocalTransform()
# random.seed(0)


def compute_correlation_matrices_torch(X: torch.Tensor) -> torch.Tensor:
    X = X.permute(2, 1, 0)  # (T, F, C)
    C_hat = torch.matmul(X.unsqueeze(-1), X.unsqueeze(-2).conj())
    C_hat = C_hat.mean(dim=0)
    return C_hat


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


class SyncedDeepMusicDataset(Dataset):
    """DeepMUSIC-style dataset for this repo's synced_dataset layout.

    Returns:
        spectrogram: float tensor (2 * num_mics, 257, 64)
        doa: float tensor (1,), angle bin in [0, 360). 90 deg is robot front.
        steering_vector: complex tensor (257, num_mics, 360)
        correlation: complex tensor (257, num_mics, num_mics)
    """

    def __init__(
        self,
        root="synced_dataset",
        split="train",
        odom_name="lio_odom",
        object_names=None,
        audio_channels=(1, 2, 3, 4),
        sample_rate=16000,
        n_fft=512,
        hop_length=256,
        output_time_frames=64,
        min_freq_hz=2000.0,
        max_audio_abs=0.06,
        min_distance=None,
        max_distance=None,
        speed_of_sound=343.0,
        mic_geometry="respeaker_v3",
        mic_radius=0.032,
        mic_rotation_deg=0.0,
        mic_channel_order=None,
        mic_positions=None,
        geometry_aug=False,
        geometry_aug_step_deg=1.0,
        noise_aug=False,
        snr_min_db=0.0,
        snr_max_db=25.0,
        time_mask=False,
        time_mask_prob=0.5,
        time_mask_max_width=8,
        use_filter_mute=False,
        filter_mute_highpass_hz=120.0,
        filter_mute_notches_hz="48.8,66.4,179.7,341.8,867.2,1271.5,1845.7,2533.2,3783.2",
        filter_mute_threshold=0.06,
        filter_mute_window_sec=0.05,
        filter_mute_floor=0.02,
    ):
        self.root = Path(root)
        self.split = split
        self.odom_name = odom_name
        self.object_names = parse_name_filter(object_names)
        self.audio_channels = tuple(audio_channels)
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.output_time_frames = output_time_frames
        self.min_freq_hz = min_freq_hz
        self.max_audio_abs = max_audio_abs
        self.min_distance = min_distance
        self.max_distance = max_distance
        self.speed_of_sound = speed_of_sound
        self.mic_geometry = mic_geometry
        self.mic_positions = resolve_mic_positions(
            num_mics=len(self.audio_channels),
            geometry=mic_geometry,
            radius=mic_radius,
            rotation_deg=mic_rotation_deg,
            channel_order=mic_channel_order,
            explicit_positions=mic_positions,
        )
        self.geometry_aug = geometry_aug and split == "train"
        self.geometry_aug_step_deg = geometry_aug_step_deg
        self.noise_aug = noise_aug and split == "train"
        self.snr_min_db = snr_min_db
        self.snr_max_db = snr_max_db
        self.time_mask = time_mask and split == "train"
        self.time_mask_prob = time_mask_prob
        self.time_mask_max_width = time_mask_max_width
        self.use_filter_mute = use_filter_mute
        self.filter_mute_highpass_hz = filter_mute_highpass_hz
        self.filter_mute_notches_hz = tuple(parse_float_list(filter_mute_notches_hz))
        self.filter_mute_threshold = filter_mute_threshold
        self.filter_mute_window_sec = filter_mute_window_sec
        self.filter_mute_floor = filter_mute_floor
        self.skipped_by_amplitude = 0
        self.skipped_by_distance = 0
        self.samples = self._collect_samples()
        if not self.samples:
            raise RuntimeError(f"No DeepMUSIC samples found under {root} split={split}")
        if self.max_audio_abs and self.skipped_by_amplitude:
            print(
                f"[SyncedDeepMusicDataset] split={split}: skipped "
                f"{self.skipped_by_amplitude} samples with max_abs > {self.max_audio_abs}"
            )
        if (self.min_distance is not None or self.max_distance is not None) and self.skipped_by_distance:
            print(
                f"[SyncedDeepMusicDataset] split={split}: skipped "
                f"{self.skipped_by_distance} samples outside distance range "
                f"[{self.min_distance}, {self.max_distance}]"
            )

    def _collect_samples(self):
        split_root = self.root / self.split
        if not split_root.exists():
            raise RuntimeError(f"Missing dataset split: {split_root}")

        samples = []
        for seq_dir in sorted(path for path in split_root.iterdir() if path.is_dir()):
            if self.object_names and not matches_object_filter(seq_dir.name, self.object_names):
                continue
            audio_dir = seq_dir / "audio"
            doa_dir = seq_dir / f"doa_{self.odom_name}"
            distance_dir = seq_dir / f"distance_{self.odom_name}"
            if not audio_dir.exists() or not doa_dir.exists():
                continue
            for audio_path in sorted(audio_dir.glob("*.wav"), key=lambda p: numeric_stem(p.stem)):
                doa_path = doa_dir / f"{audio_path.stem}.npy"
                if not doa_path.exists():
                    continue
                distance_path = distance_dir / f"{audio_path.stem}.npy"
                distance_xy = self._load_distance_xy(doa_path, distance_path)
                if self._should_skip_by_distance(distance_xy):
                    self.skipped_by_distance += 1
                    continue
                if self._should_skip_audio_by_amplitude(audio_path):
                    self.skipped_by_amplitude += 1
                    continue
                samples.append({
                    "sequence": seq_dir.name,
                    "audio": audio_path,
                    "doa": doa_path,
                    "distance": distance_path if distance_path.exists() else None,
                    "distance_xy": distance_xy,
                    "sample_id": audio_path.stem,
                })
        return samples

    def _load_distance_xy(self, doa_path, distance_path):
        if distance_path.exists():
            distance_values = np.load(distance_path).astype(np.float32).reshape(-1)
            if len(distance_values) > 0:
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

    def _should_skip_audio_by_amplitude(self, audio_path):
        if self.max_audio_abs is None or self.max_audio_abs <= 0:
            return False
        audio, sample_rate = load_audio_wav(audio_path, self.audio_channels)
        if sample_rate != self.sample_rate:
            raise RuntimeError(f"{audio_path} has sample_rate={sample_rate}, expected {self.sample_rate}")
        return float(np.max(np.abs(audio))) > float(self.max_audio_abs)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        item = self.samples[index]
        audio, sample_rate = load_audio_wav(item["audio"], self.audio_channels)
        if sample_rate != self.sample_rate:
            raise RuntimeError(f"{item['audio']} has sample_rate={sample_rate}, expected {self.sample_rate}")

        if self.use_filter_mute:
            audio = filter_and_mute_motion_impacts(
                audio,
                sample_rate=sample_rate,
                highpass_hz=self.filter_mute_highpass_hz,
                notches_hz=self.filter_mute_notches_hz,
                motion_threshold=self.filter_mute_threshold,
                mute_window_sec=self.filter_mute_window_sec,
                mute_floor=self.filter_mute_floor,
            )

        if self.noise_aug:
            audio = add_gaussian_noise(audio, random.uniform(self.snr_min_db, self.snr_max_db))

        audio_tensor = torch.as_tensor(audio, dtype=torch.float32)
        stft_tensor = torch.stft(
            audio_tensor,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=torch.hann_window(self.n_fft),
            return_complex=True,
        )
        stft_tensor = self.crop_high_frequency_stft(stft_tensor, sample_rate)
        correlation = compute_correlation_matrices_torch(stft_tensor)
        correlation = resize_complex_frequency_axis(correlation, self.n_fft // 2 + 1)
        spectrogram = self.spectrogram_process(stft_tensor)
        if self.time_mask:
            spectrogram = apply_time_mask(spectrogram, self.time_mask_prob, self.time_mask_max_width)

        doa_values = np.load(item["doa"]).astype(np.float32)
        doa = array_to_named_values(doa_values, DOA_FIELDS)
        angle_deg = yaw_to_deepmusic_angle(float(doa["heading_target_yaw_signed_deg"]))

        rotation_deg = self._sample_rotation_deg()
        target_deg = (angle_deg + rotation_deg) % 360.0
        mic_positions = rotate_mic_positions(self.mic_positions, rotation_deg)
        steering_vector = make_far_field_steering_vector(
            mic_positions=mic_positions,
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            min_freq_hz=self.min_freq_hz,
            num_freq_bins=self.n_fft // 2 + 1,
            speed_of_sound=self.speed_of_sound,
        )

        return (
            spectrogram,
            torch.tensor([target_deg], dtype=torch.float32),
            steering_vector,
            correlation,
        )

    def spectrogram_process(self, stft_tensor):
        magnitude = normalize_magnitude(torch.abs(stft_tensor))
        phase = normalize_phase(torch.angle(stft_tensor))
        spectrogram = torch.stack([magnitude, phase], dim=1).view(2 * stft_tensor.shape[0], stft_tensor.shape[1], stft_tensor.shape[2])
        spectrogram = torch.nn.functional.interpolate(
            spectrogram.unsqueeze(0),
            size=(self.n_fft // 2 + 1, self.output_time_frames),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        return spectrogram.float()

    def crop_high_frequency_stft(self, stft_tensor, sample_rate):
        if self.min_freq_hz is None or self.min_freq_hz <= 0:
            return stft_tensor
        freq_bins = torch.fft.rfftfreq(self.n_fft, d=1.0 / float(sample_rate))
        keep_mask = freq_bins >= float(self.min_freq_hz)
        if not bool(torch.any(keep_mask)):
            raise RuntimeError(
                f"min_freq_hz={self.min_freq_hz} is higher than Nyquist frequency "
                f"{sample_rate / 2.0}"
            )
        if bool(torch.all(keep_mask)):
            return stft_tensor
        return stft_tensor[:, keep_mask, :]

    def _sample_rotation_deg(self):
        if not self.geometry_aug:
            return 0.0
        if self.geometry_aug_step_deg and self.geometry_aug_step_deg > 0:
            steps = int(round(360.0 / self.geometry_aug_step_deg))
            return random.randint(0, steps - 1) * self.geometry_aug_step_deg
        return random.uniform(0.0, 360.0)


def resolve_mic_positions(
    num_mics,
    geometry="respeaker_v3",
    radius=0.032,
    rotation_deg=0.0,
    channel_order=None,
    explicit_positions=None,
):
    if explicit_positions is not None:
        positions = np.asarray(explicit_positions, dtype=np.float32)
    elif geometry in ("respeaker_v3", "circular"):
        positions = make_circular_mic_positions(num_mics, radius=radius, rotation_deg=rotation_deg)
    else:
        raise RuntimeError(f"Unknown mic geometry: {geometry}")

    if positions.shape != (num_mics, 3):
        raise RuntimeError(f"Mic positions must have shape ({num_mics}, 3), got {positions.shape}")

    if channel_order:
        order = [int(item.strip()) for item in str(channel_order).split(",") if item.strip()]
        if sorted(order) != list(range(num_mics)):
            raise RuntimeError(f"--mic-channel-order must be a permutation of 0..{num_mics - 1}, got {order}")
        positions = positions[order]
    return positions.astype(np.float32)


def make_circular_mic_positions(num_mics, radius, rotation_deg=0.0):
    angles = np.linspace(0.0, 2.0 * np.pi, num_mics, endpoint=False) + np.radians(rotation_deg)
    return np.stack([
        radius * np.cos(angles),
        radius * np.sin(angles),
        np.zeros_like(angles),
    ], axis=1).astype(np.float32)


def rotate_mic_positions(mic_positions, rotation_deg):
    theta = np.radians(rotation_deg)
    rotation = np.array([
        [np.cos(theta), -np.sin(theta), 0.0],
        [np.sin(theta), np.cos(theta), 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)
    return np.asarray(mic_positions, dtype=np.float32) @ rotation.T


def make_far_field_steering_vector(
    mic_positions,
    sample_rate=16000,
    n_fft=512,
    min_freq_hz=0.0,
    num_freq_bins=None,
    speed_of_sound=343.0,
):
    mic_positions = torch.as_tensor(mic_positions, dtype=torch.float32)
    angles = torch.arange(360, dtype=torch.float32) * torch.pi / 180.0
    directions = torch.stack([
        torch.cos(angles),
        torch.sin(angles),
        torch.zeros_like(angles),
    ], dim=1)
    tau = directions @ mic_positions.T / speed_of_sound
    if min_freq_hz is not None and min_freq_hz > 0:
        num_freq_bins = num_freq_bins or (n_fft // 2 + 1)
        nyquist_hz = sample_rate / 2.0
        if min_freq_hz > nyquist_hz:
            raise RuntimeError(f"min_freq_hz={min_freq_hz} is higher than Nyquist frequency {nyquist_hz}")
        freqs = torch.linspace(float(min_freq_hz), float(nyquist_hz), int(num_freq_bins), dtype=torch.float32)
    else:
        freqs = torch.arange(n_fft // 2 + 1, dtype=torch.float32) * sample_rate / n_fft
    phase = 2.0 * torch.pi * freqs[:, None, None] * tau.T[None, :, :]
    return torch.exp(1j * phase).to(torch.complex64)


def resize_complex_frequency_axis(tensor, num_freq_bins):
    if tensor.shape[0] == num_freq_bins:
        return tensor
    freq_bins, rows, cols = tensor.shape
    real = tensor.real.permute(1, 2, 0).reshape(1, rows * cols, freq_bins)
    imag = tensor.imag.permute(1, 2, 0).reshape(1, rows * cols, freq_bins)
    real = torch.nn.functional.interpolate(real, size=num_freq_bins, mode="linear", align_corners=False)
    imag = torch.nn.functional.interpolate(imag, size=num_freq_bins, mode="linear", align_corners=False)
    real = real.reshape(rows, cols, num_freq_bins).permute(2, 0, 1)
    imag = imag.reshape(rows, cols, num_freq_bins).permute(2, 0, 1)
    return torch.complex(real, imag).to(tensor.dtype)


def yaw_to_deepmusic_angle(yaw_signed_deg):
    return (90.0 + yaw_signed_deg) % 360.0


def circular_abs_diff_deg(first, second):
    return abs((first - second + 180.0) % 360.0 - 180.0)


def array_to_named_values(values, fields):
    return {name: values[i] for i, name in enumerate(fields[:len(values)])}


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


def numeric_stem(stem_value):
    try:
        return int(stem_value)
    except ValueError:
        return stem_value


def apply_time_mask(spectrogram, prob=0.5, max_width=8):
    if prob <= 0 or max_width <= 0 or random.random() > prob:
        return spectrogram
    output = spectrogram.clone()
    num_frames = output.shape[-1]
    width = random.randint(1, min(int(max_width), num_frames))
    start = random.randint(0, num_frames - width)
    output[:, :, start:start + width] = 0.0
    return output

class Grid:
    def __init__(self):
        self.x = np.load('/home/kemove/yyz/SubspaceNet/DeepMucis_plus/grid_x.npy')
        self.y = np.load('/home/kemove/yyz/SubspaceNet/DeepMucis_plus/grid_y.npy')
        self.z = np.load('/home/kemove/yyz/SubspaceNet/DeepMucis_plus/grid_z.npy')

def array_aug(mic_offsets,locs,transform,interval=None,mic_center=np.array([[3, 3, 1]])):
    if transform:
        if interval == None:
            random_integer = random.randint(0, 360)
        else:
            random_integer = random.randint(0,int(360/interval))*interval
    else:
        random_integer = 0
    grid = Grid()
    rotation_degree = random_integer
    theta = np.deg2rad(rotation_degree)
    R = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta),  0],
        [0,             0,              1]
    ])
    rotated_offsets = mic_offsets @ R.T
    mic_locs_rotated = mic_center + rotated_offsets
    mic_positions = mic_locs_rotated.T
    steer_vector_calc = ModeVector_torch(torch.tensor(mic_positions), 16000, 512, 343, grid, "far", precompute=True)
    sv = steer_vector_calc.mode_vec
    return sv,(locs+rotation_degree)%360

class DAMUSIC_Loader(Dataset):
    def __init__(self, root,mic_offsets, subset = "train",coherent=2,num_source=1,noise_aug=False,time_aug=False,geometry_aug=False,model='DAMUSIC',feature='spectrogram',num_percent=1,snr=None,mode='all')-> None:
        self.audio_data = []
        self.root = root
        self.noise_aug = noise_aug
        self.time_aug = time_aug
        self.geometry_aug = geometry_aug
        self.model = model
        self.coherent = coherent
        self.num_source = num_source
        self.feature = feature
        self.num_percent = num_percent
        self.mic_offsets = mic_offsets
        self.snr = snr
        self.mode = mode
        self.array_aug = array_aug
        # if subset!='train':
        if 'train' not in subset:
            self.transform = False
        else:
            self.transform = True
        if mode!='all':
            self.transform=False
        subset_data_dirs = []

        if coherent in [1, 2]:
            coherent_path = os.path.join(self.root, subset, 'coherent')
            if os.path.exists(coherent_path):
                subset_data_dirs += [os.path.join(coherent_path, d) for d in os.listdir(coherent_path)
                                     if os.path.isdir(os.path.join(coherent_path, d))]

        if coherent in [0, 2]:
            incoherent_path = os.path.join(self.root, subset, 'incoherent')
            if os.path.exists(incoherent_path):
                subset_data_dirs += [os.path.join(incoherent_path, d) for d in os.listdir(incoherent_path)
                                     if os.path.isdir(os.path.join(incoherent_path, d))]

        if not geometry_aug:
            subset_data_dirs = [d for d in subset_data_dirs if os.path.basename(d).startswith("N")]

        subset_data_dirs = filter_folders(subset_data_dirs, self.num_source)
        for seq_path in subset_data_dirs:
            audio_file_list = os.listdir(seq_path)
            if 'train' in subset:
                audio_file_list = audio_file_list[:int(len(audio_file_list) * self.num_percent)]
            for audio_file in audio_file_list:
                if self.mode == 'all':
                    self.audio_data.append(os.path.join(seq_path, audio_file))
                else:
                    match = re.search(r'degree_([-+]?\d*\.?\d+)', audio_file)
                    if match:
                        degree_val = float(match.group(1))
                        if self.mode == 'range':
                            if 0 <= degree_val <= 315 and abs(degree_val - round(degree_val / 5) * 5) < 1e-3:
                                self.audio_data.append(os.path.join(seq_path, audio_file))
                        elif self.mode == 'step':
                            if 0 <= degree_val <= 355 and abs(degree_val - round(degree_val / 5) * 5) < 1e-3:
                                self.audio_data.append(os.path.join(seq_path, audio_file))
                        elif self.mode == 'step_10':
                            if 0 <= degree_val <= 350 and abs(degree_val - round(degree_val / 10) * 10) < 1e-3:
                                self.audio_data.append(os.path.join(seq_path, audio_file))
                        elif self.mode == 'gap':
                            for base in range(0, 331, 40):  # 351 to include 340
                                if base <= degree_val < base + 20:
                                    self.audio_data.append(os.path.join(seq_path, audio_file))
        # for seq_path in subset_data_dirs:
        #     audio_file_list = os.listdir(seq_path)
        #     if 'train' in subset:
        #         audio_file_list = audio_file_list[:int(len(audio_file_list)*self.num_percent)]
        #     for audio_file in audio_file_list:
        #         self.audio_data.append(os.path.join(seq_path, audio_file))
        self.stft_trans = transform.stft(n_fft=512,hop_length=256)

    def extract_degree_numbers(self,file_path):
        file_name = os.path.basename(file_path)  # Extract file name
        match = re.search(r"degree_([\d.-]+)_+times", file_name)  # Handle both single and double underscores
        degree_part = match.group(1)  # Extract the matched part
        number_list = [float(num) if '.' in num else int(num) for num in degree_part.split('-')]  # Convert numbers
        return number_list

    def spectrogram_process(self, audio_data):
        stft = self.stft_trans(audio_data)
        correlation  = compute_correlation_matrices_torch(stft)
        magnitude = torch.abs(stft)
        C, F, T = magnitude.shape
        magnitude = normalize_magnitude(magnitude)
        phase = torch.angle(stft)
        phase = normalize_phase(phase)
        spectrogram = torch.stack([magnitude, phase], dim=1)  # Shape: (C, 2, F, T)
        spectrogram = spectrogram.view(2 * C, F, T)
        resize_transform = Trans.Resize((257,64),antialias=True)
        spectrogram = resize_transform(spectrogram)
        return spectrogram,correlation

    def noise_aug_algo(self,audio_data):
        if self.snr==None:
            rand_num_noise = random.random()
            if rand_num_noise>=0.4:
                target_snr_db = random.randint(-10,30)
                audio_data = add_gaussian_noise(audio_data,target_snr_db)
            # if rand_num_noise>=0.1:
            #     target_snr_db = random.randint(-30,10)
            #     audio_data = add_gaussian_noise(audio_data,target_snr_db)
        else:
            target_snr_db = self.snr
            audio_data = add_gaussian_noise(audio_data,target_snr_db)
        return audio_data

    def __len__(self):
        return len(self.audio_data)

    def __getitem__(self, index):
        audio_path = self.audio_data[index]
        doas       = torch.tensor(self.extract_degree_numbers(audio_path))
        audio_data  = np.load(audio_path)
        if self.noise_aug:
            audio_data = self.noise_aug_algo(audio_data)
        audio_data  = torch.from_numpy(audio_data).float()

        sv,doas = array_aug(self.mic_offsets,doas,self.transform)
        if self.feature!='raw':
            spectrogram,correlation = self.spectrogram_process(audio_data)
            return spectrogram,doas,sv,correlation
        else:
            return audio_data,doas,sv

class DAMUSIC_Loader_pretrain(Dataset):
    def __init__(self, root, subset = "train",coherent=2,num_source=1,noise_aug=False,time_aug=False,geometry_aug=False,model='DAMUSIC',feature='spectrogram',num_muc=4)-> None:
        self.audio_data = []
        self.root = root
        self.noise_aug = noise_aug
        self.time_aug = time_aug
        self.geometry_aug = geometry_aug
        self.model = model
        self.coherent = coherent
        self.num_source = num_source
        self.feature = feature
        self.num_mic = num_muc
        subset_data_dirs = []
        if coherent in [1, 2]:
            coherent_path = os.path.join(self.root, subset, 'coherent')
            if os.path.exists(coherent_path):
                subset_data_dirs += [os.path.join(coherent_path, d) for d in os.listdir(coherent_path)
                                     if os.path.isdir(os.path.join(coherent_path, d))]

        if coherent in [0, 2]:
            incoherent_path = os.path.join(self.root, subset, 'incoherent')
            if os.path.exists(incoherent_path):
                subset_data_dirs += [os.path.join(incoherent_path, d) for d in os.listdir(incoherent_path)
                                     if os.path.isdir(os.path.join(incoherent_path, d))]

        if not geometry_aug:
            subset_data_dirs = [d for d in subset_data_dirs if os.path.basename(d).startswith("N")]

        subset_data_dirs = filter_folders(subset_data_dirs, self.num_source)
        for seq_path in subset_data_dirs:
            for audio_file in os.listdir(seq_path):
                self.audio_data.append(os.path.join(seq_path, audio_file))

        self.stft_trans = transform.stft(n_fft=512,hop_length=256)

    def apply_mic_masking(self,spectrogram):
        C, F, T = spectrogram.shape  # C = 2 * num_mics (magnitude & phase interleaved)
        mic_idx = 0
        mag_channel = mic_idx * 2      # Magnitude channel
        phase_channel = mag_channel + 1  # Phase channel
        mask = torch.zeros_like(spectrogram[:2,:,:], dtype=torch.float32)
        masking_type = random.choice(["time", "frequency", "patch"])
        if masking_type == "time":
            num_mask = int(0.1 * T)
            mask_start = random.randint(0, T - num_mask)
            spectrogram[mag_channel, :, mask_start:mask_start + num_mask] = 0
            spectrogram[phase_channel, :, mask_start:mask_start + num_mask] = 0
            mask[mag_channel, :, mask_start:mask_start + num_mask] = 1
            mask[phase_channel, :, mask_start:mask_start + num_mask] = 1
        elif masking_type == "frequency":
            num_mask = int(0.1 * F)
            mask_start = random.randint(0, F - num_mask)
            spectrogram[mag_channel, mask_start:mask_start + num_mask, :] = 0
            spectrogram[phase_channel, mask_start:mask_start + num_mask, :] = 0
            mask[mag_channel, mask_start:mask_start + num_mask, :] = 1
            mask[phase_channel, mask_start:mask_start + num_mask, :] = 1
        elif masking_type == "patch":
            patch_size = 10
            num_patches = int(0.1 * (F * T) // (patch_size * patch_size))  # Number of patches
            for _ in range(num_patches):
                freq_start = random.randint(0, F - patch_size)
                time_start = random.randint(0, T - patch_size)
                spectrogram[mag_channel, freq_start:freq_start + patch_size, time_start:time_start + patch_size] = 0
                spectrogram[phase_channel, freq_start:freq_start + patch_size, time_start:time_start + patch_size] = 0
                mask[mag_channel, freq_start:freq_start + patch_size, time_start:time_start + patch_size] = 1
                mask[phase_channel, freq_start:freq_start + patch_size, time_start:time_start + patch_size] = 1
        return spectrogram, mask

    def spectrogram_process(self, audio_data):
        stft = self.stft_trans(audio_data)
        magnitude = torch.abs(stft)
        C, F, T = magnitude.shape
        magnitude = normalize_magnitude(magnitude)
        phase = torch.angle(stft)
        phase = normalize_phase(phase)
        spectrogram = torch.stack([magnitude, phase], dim=1)  # Shape: (C, 2, F, T)
        spectrogram = spectrogram.view(2 * C, F, T)
        resize_transform = Trans.Resize((257,64),antialias=True)
        spectrogram_original = resize_transform(spectrogram)
        spectrogram_original_copy = spectrogram_original.clone()
        spectrogram,mask = self.apply_mic_masking(spectrogram_original_copy)
        return spectrogram,spectrogram_original[:2,:,:],mask

    def noise_aug_algo(self,audio_data):
        rand_num_noise = random.random()
        if rand_num_noise>=0.4:
            target_snr_db = random.randint(-10,30)
            audio_data = add_gaussian_noise(audio_data,target_snr_db)
        return audio_data

    def __len__(self):
        return len(self.audio_data)

    def __getitem__(self, index):
        audio_path = self.audio_data[index]
        audio_data  = np.load(audio_path)
        if self.noise_aug:
            audio_data = self.noise_aug_algo(audio_data)
        audio_data  = torch.from_numpy(audio_data).float()
        if self.feature!='raw':
            spectrogram,spectrogram_ori,mask = self.spectrogram_process(audio_data)
            return spectrogram,spectrogram_ori,mask
        else:
            return audio_data

class SoClas_database_pretrain(Dataset):
    def __init__(self, root,subset='train',noise_aug=False,time_aug=False,model='DAMUSIC',feature='spectrogram')-> None:
        self.audio_data = []
        self.root = root
        self.noise_aug = noise_aug
        self.time_aug = time_aug
        self.model = model
        self.subset = subset
        self.feature = feature
        all_class_list = [os.path.join(root, audio_class) for audio_class in os.listdir(root)]
        for class_dir in all_class_list:
            subset_data_dir = os.listdir(class_dir)
            for seq in subset_data_dir :
                subseq_data_path  = os.path.join(class_dir,seq)
                audio_file_dir = os.listdir(subseq_data_path)
                if subset=='train':
                    for audio_file in audio_file_dir[:int(0.8*len(audio_file_dir))]:
                        self.audio_data.append(os.path.join(subseq_data_path,audio_file))
                else:
                    for audio_file in audio_file_dir[int(0.8*len(audio_file_dir)):]:
                        self.audio_data.append(os.path.join(subseq_data_path,audio_file))

        self.stft_trans = transform.stft(n_fft=512,hop_length=256)

    def __len__(self):
        return len(self.audio_data)

    def apply_mic_masking(self,spectrogram):
        C, F, T = spectrogram.shape  # C = 2 * num_mics (magnitude & phase interleaved)
        mic_idx = 0
        mag_channel = mic_idx * 2      # Magnitude channel
        phase_channel = mag_channel + 1  # Phase channel
        mask = torch.zeros_like(spectrogram[:2,:,:], dtype=torch.float32)
        masking_type = random.choice(["time", "frequency", "patch"])
        if masking_type == "time":
            num_mask = int(0.1 * T)
            mask_start = random.randint(0, T - num_mask)
            spectrogram[mag_channel, :, mask_start:mask_start + num_mask] = 0
            spectrogram[phase_channel, :, mask_start:mask_start + num_mask] = 0
            mask[mag_channel, :, mask_start:mask_start + num_mask] = 1
            mask[phase_channel, :, mask_start:mask_start + num_mask] = 1
        elif masking_type == "frequency":
            num_mask = int(0.1 * F)
            mask_start = random.randint(0, F - num_mask)
            spectrogram[mag_channel, mask_start:mask_start + num_mask, :] = 0
            spectrogram[phase_channel, mask_start:mask_start + num_mask, :] = 0
            mask[mag_channel, mask_start:mask_start + num_mask, :] = 1
            mask[phase_channel, mask_start:mask_start + num_mask, :] = 1

        elif masking_type == "patch":
            patch_size = 10
            num_patches = int(0.1 * (F * T) // (patch_size * patch_size))  # Number of patches
            for _ in range(num_patches):
                freq_start = random.randint(0, F - patch_size)
                time_start = random.randint(0, T - patch_size)
                spectrogram[mag_channel, freq_start:freq_start + patch_size, time_start:time_start + patch_size] = 0
                spectrogram[phase_channel, freq_start:freq_start + patch_size, time_start:time_start + patch_size] = 0
                mask[mag_channel, freq_start:freq_start + patch_size, time_start:time_start + patch_size] = 1
                mask[phase_channel, freq_start:freq_start + patch_size, time_start:time_start + patch_size] = 1

        return spectrogram, mask

    def spectrogram_process(self, audio_data):
        stft = self.stft_trans(audio_data)
        magnitude = torch.abs(stft)
        C, F, T = magnitude.shape
        magnitude = normalize_magnitude(magnitude)
        phase = torch.angle(stft)
        phase = normalize_phase(phase)
        spectrogram = torch.stack([magnitude, phase], dim=1)  # Shape: (C, 2, F, T)
        spectrogram = spectrogram.view(2 * C, F, T)
        resize_transform = Trans.Resize((257,64),antialias=True)
        spectrogram_original = resize_transform(spectrogram)
        spectrogram_original_copy = spectrogram_original.clone()
        spectrogram,mask = self.apply_mic_masking(spectrogram_original_copy)
        return spectrogram,spectrogram_original[:2,:,:],mask

    def noise_aug_algo(self,audio_data):
        rand_num_noise = random.random()
        if rand_num_noise>=0.4:
            target_snr_db = random.randint(-10,30)
            audio_data = add_gaussian_noise(audio_data,target_snr_db)
        return audio_data

    def __getitem__(self, index):
        audio_path = self.audio_data[index]
        audio_data, sr = librosa.load(audio_path,sr=16000,mono=False)
        if self.noise_aug:
            audio_data = self.noise_aug_algo(audio_data)
        audio_data  = torch.from_numpy(audio_data).float()
        if self.feature!='raw':
            spectrogram,spectrogram_ori,mask = self.spectrogram_process(audio_data)
            return spectrogram,spectrogram_ori,mask
        else:
            if self.model=='DAMUSIC':
                audio_data = downsample_audio(audio_data,1600)
            return audio_data

class SoClas_database(Dataset):
    def __init__(self, root,mic_offsets,subset='train',noise_aug=False,time_aug=False,model='DAMUSIC',feature='spectrogram',num_percent=1.0)-> None:
        self.audio_data = []
        self.root = root
        self.noise_aug = noise_aug
        self.time_aug = time_aug
        self.model = model
        self.subset = subset
        self.feature = feature
        self.num_percent = num_percent
        self.mic_offsets = mic_offsets
        if subset!='train':
            self.transform = False
        else:
            # self.transform = True
            self.transform = False
        all_class_list = [os.path.join(root, audio_class) for audio_class in os.listdir(root)]
        for class_dir in all_class_list:
            subset_data_dir = os.listdir(class_dir)
            for seq in subset_data_dir :
                subseq_data_path  = os.path.join(class_dir,seq)
                audio_file_dir = os.listdir(subseq_data_path)
                if subset=='train':
                    for audio_file in audio_file_dir[:int(0.8*len(audio_file_dir)*self.num_percent)]:
                        self.audio_data.append(os.path.join(subseq_data_path,audio_file))
                else:
                    for audio_file in audio_file_dir[int(0.8*len(audio_file_dir)):]:
                        self.audio_data.append(os.path.join(subseq_data_path,audio_file))

        self.stft_trans = transform.stft(n_fft=512,hop_length=256)

    def extract_degree_numbers(self,file_path):
        match = re.search(r'class\d+_(\d+)', file_path)
        extracted_number = int(match.group(1))
        return [extracted_number]

    def noise_aug_algo(self,audio_data):
        rand_num_noise = random.random()
        if rand_num_noise>=0.5:
            target_snr_db = random.randint(-10,30)
            audio_data = add_gaussian_noise(audio_data,target_snr_db)
        return audio_data

    def spectrogram_process(self, audio_data):
        stft = self.stft_trans(audio_data)
        correlation  = compute_correlation_matrices_torch(stft)
        magnitude = torch.abs(stft)
        C, F, T = magnitude.shape
        magnitude = normalize_magnitude(magnitude)
        phase = torch.angle(stft)
        phase = normalize_phase(phase)
        spectrogram = torch.stack([magnitude, phase], dim=1)  # Shape: (C, 2, F, T)
        spectrogram = spectrogram.view(2 * C, F, T)
        resize_transform = Trans.Resize((257,64),antialias=True)
        spectrogram = resize_transform(spectrogram)
        return spectrogram,correlation

    # def spectrogram_process(self, audio_data):
    #     # audio_data  = torch.permute(audio_data,(1,0))
    #     stft = self.stft_trans(audio_data)
    #     correlation  = compute_correlation_matrices_torch(stft)
    #     real = stft.real
    #     imag = stft.imag
    #     magnitude = torch.abs(stft)
    #     phase = torch.angle(stft)
    #     spectrogram = torch.cat([magnitude,phase], dim=0)
    #     resize_transform = Trans.Resize((257,64),antialias=True)
    #     spectrogram = resize_transform(spectrogram)
    #     # spectrogram = spectrogram[:,:64,:]
    #     return spectrogram,correlation

    def __len__(self):
        return len(self.audio_data)

    def __getitem__(self, index):
        audio_path = self.audio_data[index]
        doas       = torch.tensor(self.extract_degree_numbers(audio_path))
        audio_data, sr = librosa.load(audio_path,sr=16000,mono=False)
        if self.noise_aug:
            audio_data = self.noise_aug_algo(audio_data)
        audio_data  = torch.from_numpy(audio_data).float()
        sv,doas = array_aug(self.mic_offsets,doas,self.transform,interval=5)
        if self.feature!='raw':
            spectrogram,correlation = self.spectrogram_process(audio_data)
            return spectrogram,doas,sv,correlation
        else:
            if self.model=='DAMUSIC':
                audio_data = downsample_audio(audio_data,1600)
            return audio_data,doas,sv


class SoClas_database_mul(Dataset):
    def __init__(
        self,
        root,
        mic_offsets,
        subset='train',
        noise_aug=False,
        time_aug=False,
        model='DAMUSIC',
        feature='spectrogram',
        num_percent=1.0,

        # ===== multi-source options =====
        multi_source=False,         # True -> 两声源混合（直接相加）
        doa_min_sep=10,             # 两个 DOA 至少相差多少度（圆周距离）
        val_seed=0,                 # 验证集固定配对的随机种子

        # ===== split txt options =====
        split_txt_path=None,        # e.g. "splits/soclas_train.txt" / "splits/soclas_val.txt"
        save_split_txt=False        # True: txt 不存在则保存；存在则直接加载
    ) -> None:
        super().__init__()
        self.root = root
        self.noise_aug = noise_aug
        self.time_aug = time_aug
        self.model = model
        self.subset = subset
        self.feature = feature
        self.num_percent = num_percent
        self.mic_offsets = mic_offsets

        # multi-source
        self.multi_source = multi_source
        self.doa_min_sep = doa_min_sep
        self.val_seed = val_seed

        # split txt
        self.split_txt_path = split_txt_path
        self.save_split_txt = save_split_txt

        # 你原逻辑：train transform=False（保持一致）
        if subset != 'train':
            self.transform = False
        else:
            self.transform = False

        # ===== file list =====
        self.audio_data = self._load_or_build_split_list()
        if len(self.audio_data) == 0:
            raise RuntimeError(f"Empty split list. root={root}, subset={subset}, split_txt_path={split_txt_path}")

        # cache doas
        self.doas_all = np.array([self.extract_degree_numbers(p)[0] for p in self.audio_data], dtype=np.int32)

        # stft
        self.stft_trans = transform.stft(n_fft=512, hop_length=256)

        # fixed pairs for val/test
        self.pairs = None
        if self.multi_source and self.subset != 'train':
            self.build_fixed_pairs()

    # =========================
    # split list helpers
    # =========================
    def _load_or_build_split_list(self):
        if self.split_txt_path is not None and os.path.exists(self.split_txt_path):
            return self._load_split_txt(self.split_txt_path)

        audio_list = self._build_split_by_scanning()

        if self.split_txt_path is not None and self.save_split_txt:
            os.makedirs(os.path.dirname(self.split_txt_path), exist_ok=True)
            self._save_split_txt(self.split_txt_path, audio_list)

        return audio_list

    def _load_split_txt(self, txt_path):
        audio_list = []
        with open(txt_path, "r") as f:
            for line in f:
                p = line.strip()
                if p:
                    audio_list.append(p)
        return audio_list

    def _save_split_txt(self, txt_path, audio_list):
        with open(txt_path, "w") as f:
            for p in audio_list:
                f.write(p + "\n")

    def _build_split_by_scanning(self):
        audio_list = []
        all_class_list = sorted([os.path.join(self.root, d) for d in os.listdir(self.root)])
        for class_dir in all_class_list:
            if not os.path.isdir(class_dir):
                continue
            subset_data_dir = sorted(os.listdir(class_dir))
            for seq in subset_data_dir:
                subseq_data_path = os.path.join(class_dir, seq)
                if not os.path.isdir(subseq_data_path):
                    continue
                audio_file_dir = sorted(os.listdir(subseq_data_path))

                if self.subset == 'train':
                    end = int(0.8 * len(audio_file_dir) * self.num_percent)
                    pick = audio_file_dir[:end]
                else:
                    start = int(0.8 * len(audio_file_dir))
                    pick = audio_file_dir[start:]

                for audio_file in pick:
                    audio_list.append(os.path.join(subseq_data_path, audio_file))
        return audio_list

    # =========================
    # parsing / augmentation
    # =========================
    def extract_degree_numbers(self, file_path):
        match = re.search(r'class\d+_(\d+)', file_path)
        if match is None:
            raise ValueError(f"Cannot parse doa from path: {file_path}")
        return [int(match.group(1))]

    def noise_aug_algo(self, audio_data):
        rand_num_noise = random.random()
        if rand_num_noise >= 0.5:
            target_snr_db = random.randint(-10, 30)
            audio_data = add_gaussian_noise(audio_data, target_snr_db)
        return audio_data

    # =========================
    # audio utils
    # =========================
    def load_audio(self, path, target_sr=16000):
        audio, sr = librosa.load(path, sr=target_sr, mono=False)
        if audio.ndim == 1:
            audio = audio[None, :]
        return audio  # numpy (C,T)

    def pad_or_crop(self, x, T):
        C, t = x.shape
        if t == T:
            return x
        if t > T:
            start = np.random.randint(0, t - T + 1)
            return x[:, start:start+T]
        pad = T - t
        return np.pad(x, ((0, 0), (0, pad)), mode="constant")

    def circular_sep(self, a, b):
        return abs(((a - b + 180) % 360) - 180)

    # =========================
    # fixed pairs for val/test
    # =========================
    def build_fixed_pairs(self):
        rng = np.random.RandomState(self.val_seed)
        N = len(self.audio_data)
        pairs = []

        for i in range(N):
            doa1 = int(self.doas_all[i])

            found = False
            j = i
            for _ in range(50):
                jj = int(rng.randint(0, N))
                if jj == i:
                    continue
                doa2 = int(self.doas_all[jj])
                if self.circular_sep(doa2, doa1) >= self.doa_min_sep:
                    j = jj
                    found = True
                    break

            if not found:
                seps = np.array([self.circular_sep(int(d), doa1) for d in self.doas_all], dtype=np.float32)
                seps[i] = -1
                j = int(np.argmax(seps))

            pairs.append(j)

        self.pairs = pairs  # pairs[i] = j

    # =========================
    # feature
    # =========================
    def spectrogram_process(self, audio_data):
        stft = self.stft_trans(audio_data)
        correlation = compute_correlation_matrices_torch(stft)

        magnitude = torch.abs(stft)
        C, F, T = magnitude.shape
        magnitude = normalize_magnitude(magnitude)

        phase = torch.angle(stft)
        phase = normalize_phase(phase)

        spectrogram = torch.stack([magnitude, phase], dim=1)  # (C,2,F,T)
        spectrogram = spectrogram.view(2 * C, F, T)

        resize_transform = Trans.Resize((257, 64), antialias=True)
        spectrogram = resize_transform(spectrogram)
        return spectrogram, correlation

    def __len__(self):
        return len(self.audio_data)

    def __getitem__(self, index):
        path1 = self.audio_data[index]
        doa1 = int(self.doas_all[index])

        audio1 = self.load_audio(path1, target_sr=16000)
        if self.noise_aug:
            audio1 = self.noise_aug_algo(audio1)

        if self.multi_source:
            if self.subset == 'train':
                # train：随机 second source（无强弱差、无时间偏移）
                N = len(self.audio_data)
                j = np.random.randint(0, N)
                for _ in range(20):
                    jj = np.random.randint(0, N)
                    if jj == index:
                        continue
                    if self.circular_sep(int(self.doas_all[jj]), doa1) >= self.doa_min_sep:
                        j = jj
                        break
            else:
                # val/test：固定 second source
                j = int(self.pairs[index])

            path2 = self.audio_data[j]
            doa2 = int(self.doas_all[j])

            audio2 = self.load_audio(path2, target_sr=16000)
            if self.noise_aug:
                audio2 = self.noise_aug_algo(audio2)

            # length align then DIRECT SUM
            T = audio1.shape[1]
            audio2 = self.pad_or_crop(audio2, T)
            mix = audio1 + audio2

            # 轻微防爆（可选）：只在峰值过大时缩放
            peak = np.max(np.abs(mix)) + 1e-8
            if peak > 1.0:
                mix = mix / peak

            audio_data = torch.from_numpy(mix).float()
            doas = torch.tensor(sorted([doa1, doa2]), dtype=torch.float32)
        else:
            audio_data = torch.from_numpy(audio1).float()
            doas = torch.tensor([doa1], dtype=torch.float32)

        sv, doas = array_aug(self.mic_offsets, doas, self.transform, interval=5)

        if self.feature != 'raw':
            spectrogram, correlation = self.spectrogram_process(audio_data)
            return spectrogram, doas, sv, correlation
        else:
            if self.model == 'DAMUSIC':
                audio_data = downsample_audio(audio_data, 1600)
            return audio_data, doas, sv




class AFPILD_raw_Dataset_pretrain(data.Dataset):
    def __init__(self, dataset_dir, data_type='train', covariant_type='cloth',data_transform='stft',noise_aug=False):
        meta_file = os.path.join(dataset_dir, f"AFPILD_FE1_{covariant_type+'_'+data_type}.csv")
        self.data_arr = load_dataframe(meta_file)
        self.data_dir = dataset_dir
        self.stft_trans = transform.stft(n_fft=512,hop_length=256)
        self.data_transorm   = data_transform
        self.noise_aug = noise_aug
    def __len__(self):
        return len(self.data_arr)

    def noise_aug_algo(self,audio_data):
        rand_num_noise = random.random()
        if rand_num_noise>=0.5:
            target_snr_db = random.randint(-10,30)
            audio_data = add_gaussian_noise(audio_data,target_snr_db)
        return audio_data

    def apply_mic_masking(self,spectrogram):
        C, F, T = spectrogram.shape  # C = 2 * num_mics (magnitude & phase interleaved)
        mic_idx = 0
        mag_channel = mic_idx * 2      # Magnitude channel
        phase_channel = mag_channel + 1  # Phase channel
        mask = torch.zeros_like(spectrogram[:2,:,:], dtype=torch.float32)
        masking_type = random.choice(["time", "frequency", "patch"])
        if masking_type == "time":
            num_mask = int(0.1 * T)
            mask_start = random.randint(0, T - num_mask)
            spectrogram[mag_channel, :, mask_start:mask_start + num_mask] = 0
            spectrogram[phase_channel, :, mask_start:mask_start + num_mask] = 0
            mask[mag_channel, :, mask_start:mask_start + num_mask] = 1
            mask[phase_channel, :, mask_start:mask_start + num_mask] = 1
        elif masking_type == "frequency":
            num_mask = int(0.1 * F)
            mask_start = random.randint(0, F - num_mask)
            spectrogram[mag_channel, mask_start:mask_start + num_mask, :] = 0
            spectrogram[phase_channel, mask_start:mask_start + num_mask, :] = 0
            mask[mag_channel, mask_start:mask_start + num_mask, :] = 1
            mask[phase_channel, mask_start:mask_start + num_mask, :] = 1

        elif masking_type == "patch":
            patch_size = 10
            num_patches = int(0.1 * (F * T) // (patch_size * patch_size))  # Number of patches
            for _ in range(num_patches):
                freq_start = random.randint(0, F - patch_size)
                time_start = random.randint(0, T - patch_size)
                spectrogram[mag_channel, freq_start:freq_start + patch_size, time_start:time_start + patch_size] = 0
                spectrogram[phase_channel, freq_start:freq_start + patch_size, time_start:time_start + patch_size] = 0
                mask[mag_channel, freq_start:freq_start + patch_size, time_start:time_start + patch_size] = 1
                mask[phase_channel, freq_start:freq_start + patch_size, time_start:time_start + patch_size] = 1
        return spectrogram, mask

    def spectrogram_process(self, audio_data):
        stft = self.stft_trans(audio_data)
        magnitude = torch.abs(stft)
        C, F, T = magnitude.shape
        magnitude = normalize_magnitude(magnitude)
        phase = torch.angle(stft)
        phase = normalize_phase(phase)
        spectrogram = torch.stack([magnitude, phase], dim=1)  # Shape: (C, 2, F, T)
        spectrogram = spectrogram.view(2 * C, F, T)
        resize_transform = Trans.Resize((257,64),antialias=True)
        spectrogram_original = resize_transform(spectrogram)
        spectrogram_original_copy = spectrogram_original.clone()
        spectrogram,mask = self.apply_mic_masking(spectrogram_original_copy)
        return spectrogram,spectrogram_original[:2,:,:],mask

    def __getitem__(self, idx):
        item = self.data_arr[idx]
        loc_theta = item['loc_azimuth']
        audio = load_numpy(os.path.join(self.data_dir, item['raw']))

        if self.noise_aug:
            audio = self.noise_aug_algo(audio)

        if self.data_transorm!='raw':
            spectrogram,spectrogram_ori,mask = self.spectrogram_process(torch.tensor(audio))

        return spectrogram,spectrogram_ori,mask

class AFPILD_raw_Dataset(data.Dataset):
    def __init__(self, dataset_dir,mic_offsets, data_type='train', covariant_type='cloth',data_transform='stft',noise_aug=False,num_percent=1):
        meta_file = os.path.join(dataset_dir, f"AFPILD_FE1_{covariant_type+'_'+data_type}.csv")
        self.data_arr = load_dataframe(meta_file)
        self.data_arr = self.data_arr[:int(len(self.data_arr)*num_percent)]
        self.data_dir = dataset_dir
        self.stft_trans = transform.stft(n_fft=512,hop_length=256)
        self.data_transorm   = data_transform
        self.noise_aug = noise_aug
        self.num_percent = num_percent
        self.mic_offsets = mic_offsets
        if data_type!='train':
            self.transform = False
        else:
            self.transform = True
    def __len__(self):
        return len(self.data_arr)

    def noise_aug_algo(self,audio_data):
        rand_num_noise = random.random()
        if rand_num_noise>=0.5:
            target_snr_db = random.randint(-10,30)
            audio_data = add_gaussian_noise(audio_data,target_snr_db)
        return audio_data

    # def spectrogram_process(self, audio_data):
    #     stft = self.stft_trans(audio_data)
    #     correlation  = compute_correlation_matrices_torch(stft)
    #     magnitude = torch.abs(stft)
    #     C, F, T = magnitude.shape
    #     magnitude = normalize_magnitude(magnitude)
    #     phase = torch.angle(stft)
    #     phase = normalize_phase(phase)
    #     spectrogram = torch.stack([magnitude, phase], dim=1)  # Shape: (C, 2, F, T)
    #     spectrogram = spectrogram.view(2 * C, F, T)
    #     resize_transform = Trans.Resize((257,64),antialias=True)
    #     spectrogram = resize_transform(spectrogram)
    #     return spectrogram,correlation

    def spectrogram_process(self, audio_data):
        # audio_data  = torch.permute(audio_data,(1,0))
        stft = self.stft_trans(audio_data)
        correlation  = compute_correlation_matrices_torch(stft)
        real = stft.real
        imag = stft.imag
        magnitude = torch.abs(stft)
        phase = torch.angle(stft)
        spectrogram = torch.cat([magnitude,phase], dim=0)
        resize_transform = Trans.Resize((257,64),antialias=True)
        spectrogram = resize_transform(spectrogram)
        # spectrogram = spectrogram[:,:64,:]
        return spectrogram,correlation

    def __getitem__(self, idx):
        item = self.data_arr[idx]
        loc_theta = item['loc_azimuth']
        audio = load_numpy(os.path.join(self.data_dir, item['raw']))

        if self.noise_aug:
            audio = self.noise_aug_algo(audio)

        if self.data_transorm!='raw':
            spectrogram,correlation = self.spectrogram_process(torch.tensor(audio))
        sv,doas = array_aug(self.mic_offsets,loc_theta,self.transform)
        return spectrogram,torch.tensor(np.array([doas])),sv,correlation

class RSL_database(Dataset):
    def __init__(self, root,mic_offsets,subset='train',noise_aug=False,time_aug=False,model='DAMUSIC',feature='spectrogram',num_percent=1.0)-> None:
        self.audio_data = []
        self.root = root
        self.noise_aug = noise_aug
        self.time_aug = time_aug
        self.model = model
        self.subset = subset
        self.feature = feature
        self.num_percent = num_percent
        self.mic_offsets = mic_offsets
        if subset!='train':
            self.transform = False
        else:
            self.transform = True
        all_class_list = [os.path.join(root, audio_class) for audio_class in os.listdir(root)]
        for class_dir in all_class_list:
            subset_data_dir = os.listdir(class_dir)
            for seq in subset_data_dir :
                subseq_data_path  = os.path.join(class_dir,seq)
                audio_file_dir = os.listdir(subseq_data_path)
                valid_files = [f for f in audio_file_dir if not (f.endswith("NOISE.wav") or f.endswith("TSP.wav"))]
                id_file_dict = defaultdict(list)
                for f in valid_files:
                    match = re.search(r'RSL_\d+_\d+_(F\d{3}|M\d{3})_', f)
                    if match:
                        speaker_id = match.group(1)
                        id_file_dict[speaker_id].append(f)

                selected_files = []
                for speaker_id, files in id_file_dict.items():
                    files = sorted(files)
                    split_idx = int(0.8 * len(files))
                    if subset == 'train':
                        selected_files.extend(files[:int(split_idx* self.num_percent)])
                    else:
                        selected_files.extend(files[split_idx:])
                self.audio_data+=([os.path.join(subseq_data_path, f) for f in selected_files])
        self.stft_trans = transform.stft(n_fft=512,hop_length=256)

    def extract_degree_numbers(self, file_path):
        match = re.search(r'RSL_\d+_(\d+)', file_path)
        return [int(match.group(1))]

    def noise_aug_algo(self,audio_data):
        rand_num_noise = random.random()
        if rand_num_noise>=0.5:
            target_snr_db = random.randint(-10,30)
            audio_data = add_gaussian_noise(audio_data,target_snr_db)
        return audio_data

    def spectrogram_process(self, audio_data):
        stft = self.stft_trans(audio_data)
        correlation  = compute_correlation_matrices_torch(stft)
        magnitude = torch.abs(stft)
        C, F, T = magnitude.shape
        magnitude = normalize_magnitude(magnitude)
        phase = torch.angle(stft)
        phase = normalize_phase(phase)
        spectrogram = torch.stack([magnitude, phase], dim=1)  # Shape: (C, 2, F, T)
        spectrogram = spectrogram.view(2 * C, F, T)
        resize_transform = Trans.Resize((257,64),antialias=True)
        spectrogram = resize_transform(spectrogram)
        return spectrogram,correlation

    def __len__(self):
        return len(self.audio_data)

    def __getitem__(self, index):
        audio_path = self.audio_data[index]
        # print(audio_path)
        doas       = torch.tensor(self.extract_degree_numbers(audio_path))
        audio_data, sr = librosa.load(audio_path,sr=48000,mono=False)
        if self.noise_aug:
            audio_data = self.noise_aug_algo(audio_data)
        audio_data  = torch.from_numpy(audio_data).float()
        sv,doas = array_aug(self.mic_offsets,doas,self.transform)
        if self.feature!='raw':
            spectrogram,correlation = self.spectrogram_process(audio_data)
            return spectrogram,doas,sv,correlation
        else:
            if self.model=='DAMUSIC':
                audio_data = downsample_audio(audio_data,1600)
            return audio_data,doas,sv

class RSL_database_pretrain(Dataset):
    def __init__(self, root,subset='train',noise_aug=False,time_aug=False,model='DAMUSIC',feature='spectrogram')-> None:
        self.audio_data = []
        self.root = root
        self.noise_aug = noise_aug
        self.time_aug = time_aug
        self.model = model
        self.subset = subset
        self.feature = feature
        all_class_list = [os.path.join(root, audio_class) for audio_class in os.listdir(root)]
        for class_dir in all_class_list:
            subset_data_dir = os.listdir(class_dir)
            for seq in subset_data_dir :
                subseq_data_path  = os.path.join(class_dir,seq)
                audio_file_dir = os.listdir(subseq_data_path)
                valid_files = [f for f in audio_file_dir if not (f.endswith("NOISE.wav") or f.endswith("TSP.wav"))]
                f_files = sorted([f for f in valid_files if 'F' in f])
                m_files = sorted([f for f in valid_files if 'M' in f])
                split_f = int(0.8 * len(f_files))
                split_m = int(0.8 * len(m_files))
                random.shuffle(f_files)
                random.shuffle(m_files)
                if subset == 'train':
                    selected_files = f_files[:int(split_f)] + m_files[:int(split_m)]
                else:
                    selected_files = f_files[split_f:] + m_files[split_m:]

                for audio_file in selected_files:
                    self.audio_data.append(os.path.join(subseq_data_path, audio_file))

        self.stft_trans = transform.stft(n_fft=512,hop_length=256)

    def __len__(self):
        return len(self.audio_data)

    def apply_mic_masking(self,spectrogram):
        C, F, T = spectrogram.shape  # C = 2 * num_mics (magnitude & phase interleaved)
        mic_idx = 0
        mag_channel = mic_idx * 2      # Magnitude channel
        phase_channel = mag_channel + 1  # Phase channel
        mask = torch.zeros_like(spectrogram[:2,:,:], dtype=torch.float32)
        masking_type = random.choice(["time", "frequency", "patch"])
        if masking_type == "time":
            num_mask = int(0.1 * T)
            mask_start = random.randint(0, T - num_mask)
            spectrogram[mag_channel, :, mask_start:mask_start + num_mask] = 0
            spectrogram[phase_channel, :, mask_start:mask_start + num_mask] = 0
            mask[mag_channel, :, mask_start:mask_start + num_mask] = 1
            mask[phase_channel, :, mask_start:mask_start + num_mask] = 1
        elif masking_type == "frequency":
            num_mask = int(0.1 * F)
            mask_start = random.randint(0, F - num_mask)
            spectrogram[mag_channel, mask_start:mask_start + num_mask, :] = 0
            spectrogram[phase_channel, mask_start:mask_start + num_mask, :] = 0
            mask[mag_channel, mask_start:mask_start + num_mask, :] = 1
            mask[phase_channel, mask_start:mask_start + num_mask, :] = 1

        elif masking_type == "patch":
            patch_size = 10
            num_patches = int(0.1 * (F * T) // (patch_size * patch_size))  # Number of patches
            for _ in range(num_patches):
                freq_start = random.randint(0, F - patch_size)
                time_start = random.randint(0, T - patch_size)
                spectrogram[mag_channel, freq_start:freq_start + patch_size, time_start:time_start + patch_size] = 0
                spectrogram[phase_channel, freq_start:freq_start + patch_size, time_start:time_start + patch_size] = 0
                mask[mag_channel, freq_start:freq_start + patch_size, time_start:time_start + patch_size] = 1
                mask[phase_channel, freq_start:freq_start + patch_size, time_start:time_start + patch_size] = 1

        return spectrogram, mask

    def spectrogram_process(self, audio_data):
        stft = self.stft_trans(audio_data)
        magnitude = torch.abs(stft)
        C, F, T = magnitude.shape
        magnitude = normalize_magnitude(magnitude)
        phase = torch.angle(stft)
        phase = normalize_phase(phase)
        spectrogram = torch.stack([magnitude, phase], dim=1)  # Shape: (C, 2, F, T)
        spectrogram = spectrogram.view(2 * C, F, T)
        resize_transform = Trans.Resize((257,64),antialias=True)
        spectrogram_original = resize_transform(spectrogram)
        spectrogram_original_copy = spectrogram_original.clone()
        spectrogram,mask = self.apply_mic_masking(spectrogram_original_copy)
        return spectrogram,spectrogram_original[:2,:,:],mask

    def noise_aug_algo(self,audio_data):
        rand_num_noise = random.random()
        if rand_num_noise>=0.4:
            target_snr_db = random.randint(-10,30)
            audio_data = add_gaussian_noise(audio_data,target_snr_db)
        return audio_data

    def __getitem__(self, index):
        audio_path = self.audio_data[index]
        audio_data, sr = librosa.load(audio_path,sr=16000,mono=False)
        if self.noise_aug:
            audio_data = self.noise_aug_algo(audio_data)
        audio_data  = torch.from_numpy(audio_data).float()
        if self.feature!='raw':
            spectrogram,spectrogram_ori,mask = self.spectrogram_process(audio_data)
            return spectrogram,spectrogram_ori,mask
        else:
            if self.model=='DAMUSIC':
                audio_data = downsample_audio(audio_data,1600)
            return audio_data



class AV16_Dataset_video(Dataset):
    def __init__(
        self,
        processed_root,
        mic_offsets,
        subset='train',
        data_transform="stft",   # "raw" or "stft"
        noise_aug=False,
        resize_hw=(257, 64),
        cam=1,
        model='DAMUSIC',feature='spectrogram',num_percent=1.0
    ):
        self.root = Path(processed_root)
        if not self.root.exists():
            raise FileNotFoundError(self.root)

        self.model = model
        self.subset = subset
        self.feature = feature
        self.mic_offsets = mic_offsets
        self.data_transform = data_transform
        self.noise_aug = noise_aug
        self.feature = feature
        self.num_percent = num_percent
        self.mic_offsets = mic_offsets
        self.stft_trans = transform.stft(n_fft=512,hop_length=256)
        self.resize_transform = Trans.Resize(resize_hw, antialias=True)
        self.items = []  # (audio_path, gt_path, seq_name)
        if subset!='train':
            self.transform = False
        else:
            self.transform = True

        for seq_dir in sorted([p for p in self.root.iterdir() if p.is_dir()]):
            print(seq_dir)
            audio_dir = seq_dir / "audio"
            gt_dir = seq_dir / "gt"
            img_dir = seq_dir / "image" / f"cam{cam}"  # not required for return

            if not (audio_dir.exists() and gt_dir.exists()):
                continue

            audio_files = sorted(audio_dir.glob("*.npy"))
            for ap in audio_files[:int(len(audio_files)*self.num_percent)]:
                s = stem(ap)
                gp = gt_dir / f"{s}.npy"
                if gp.exists():
                    self.items.append((ap, gp, seq_dir.name))

        if not self.items:
            raise RuntimeError(f"No matched (audio, gt) found under: {self.root}")

        print(f"[OK] indexed {len(self.items)} samples from {self.root}")

    def __len__(self):
        return len(self.items)

    def noise_aug_algo(self, audio_data):
        # audio_data: (C,T)
        rand_num_noise = random.random()
        if rand_num_noise >= 0.5:
            target_snr_db = random.randint(-10, 30)
            audio_data = add_gaussian_noise(audio_data, target_snr_db)
        return audio_data

    def spectrogram_process(self, audio_data):
        stft = self.stft_trans(audio_data)
        correlation  = compute_correlation_matrices_torch(stft)
        magnitude = torch.abs(stft)
        C, F, T = magnitude.shape
        magnitude = normalize_magnitude(magnitude)
        phase = torch.angle(stft)
        phase = normalize_phase(phase)
        spectrogram = torch.stack([magnitude, phase], dim=1)  # Shape: (C, 2, F, T)
        spectrogram = spectrogram.view(2 * C, F, T)
        resize_transform = Trans.Resize((257,64),antialias=True)
        spectrogram = resize_transform(spectrogram)
        return spectrogram,correlation

    def __getitem__(self, idx):
        ap, gp, seq_name = self.items[idx]
        audio = np.load(str(ap)).astype(np.float32)
        if self.noise_aug:
            audio = self.noise_aug_algo(audio)
        audio = torch.from_numpy(audio).float()  

        gt = load_gt_dict(gp)
        xyz = np.asarray(gt["gt3d_xyz"], dtype=np.float32)
        indicator = gt["indicator"]  
        loc_theta_deg = np.array(doa_xy_deg_from_xyz(xyz))  
        doas = torch.tensor(loc_theta_deg, dtype=torch.float32)          

        sv,doas = array_aug(self.mic_offsets,doas,self.transform,mic_center=np.array([0.0,0.0,0.0]))
        if self.feature!='raw':
            spectrogram,correlation = self.spectrogram_process(audio)
            return spectrogram,doas,sv,correlation,indicator
        else:
            if self.model=='DAMUSIC':
                audio_data = downsample_audio(audio,1600)
            return audio_data,doas,sv,indicator
        



class AV16_Dataset(Dataset):
    def __init__(
        self,
        processed_root,
        mic_offsets,
        subset='train',
        data_transform="stft",   # "raw" or "stft"
        noise_aug=False,
        resize_hw=(257, 64),
        cam=1,
        model='DAMUSIC',feature='spectrogram',num_percent=1.0
    ):
        self.root = Path(processed_root)
        if not self.root.exists():
            raise FileNotFoundError(self.root)

        self.model = model
        self.subset = subset
        self.feature = feature
        self.mic_offsets = mic_offsets
        self.data_transform = data_transform
        self.noise_aug = noise_aug
        self.feature = feature
        self.num_percent = num_percent
        self.mic_offsets = mic_offsets
        self.stft_trans = transform.stft(n_fft=512,hop_length=256)
        self.resize_transform = Trans.Resize(resize_hw, antialias=True)
        self.items = []  # (audio_path, gt_path, seq_name)
        if subset!='train':
            self.transform = False
        else:
            self.transform = True

        for seq_dir in sorted([p for p in self.root.iterdir() if p.is_dir()]):
            print(seq_dir)
            audio_dir = seq_dir / "audio"
            gt_dir = seq_dir / "gt"
            img_dir = seq_dir / "image" / f"cam{cam}"  # not required for return

            if not (audio_dir.exists() and gt_dir.exists()):
                continue

            audio_files = sorted(audio_dir.glob("*.npy"))
            for ap in audio_files[:int(len(audio_files)*self.num_percent)]:
                s = stem(ap)
                gp = gt_dir / f"{s}.npy"
                if gp.exists():
                    self.items.append((ap, gp, seq_dir.name))

        if not self.items:
            raise RuntimeError(f"No matched (audio, gt) found under: {self.root}")

        print(f"[OK] indexed {len(self.items)} samples from {self.root}")

    def __len__(self):
        return len(self.items)

    def noise_aug_algo(self, audio_data):
        # audio_data: (C,T)
        rand_num_noise = random.random()
        if rand_num_noise >= 0.5:
            target_snr_db = random.randint(-10, 30)
            audio_data = add_gaussian_noise(audio_data, target_snr_db)
        return audio_data

    def spectrogram_process(self, audio_data):
        stft = self.stft_trans(audio_data)
        correlation  = compute_correlation_matrices_torch(stft)
        magnitude = torch.abs(stft)
        C, F, T = magnitude.shape
        magnitude = normalize_magnitude(magnitude)
        phase = torch.angle(stft)
        phase = normalize_phase(phase)
        spectrogram = torch.stack([magnitude, phase], dim=1)  # Shape: (C, 2, F, T)
        spectrogram = spectrogram.view(2 * C, F, T)
        resize_transform = Trans.Resize((257,64),antialias=True)
        spectrogram = resize_transform(spectrogram)
        return spectrogram,correlation

    def __getitem__(self, idx):
        ap, gp, seq_name = self.items[idx]
        audio = np.load(str(ap)).astype(np.float32)
        if self.noise_aug:
            audio = self.noise_aug_algo(audio)
        audio = torch.from_numpy(audio).float()  

        gt = load_gt_dict(gp)
        xyz = np.asarray(gt["gt3d_xyz"], dtype=np.float32)  
        loc_theta_deg = np.array(doa_xy_deg_from_xyz(xyz))  
        doas = torch.tensor(loc_theta_deg, dtype=torch.float32)          

        sv,doas = array_aug(self.mic_offsets,doas,self.transform,mic_center=np.array([0.0,0.0,0.0]))
        if self.feature!='raw':
            spectrogram,correlation = self.spectrogram_process(audio)
            return spectrogram,doas,sv,correlation
        else:
            if self.model=='DAMUSIC':
                audio_data = downsample_audio(audio,1600)
            return audio_data,doas,sv

class AV16_Dataset_pretrain(Dataset):
    def __init__(
        self,
        processed_root,
        subset='train',
        data_transform="stft",   # "raw" or "stft"
        noise_aug=False,
        resize_hw=(257, 64),
        cam=1,
        model='DAMUSIC',feature='spectrogram',num_percent=1.0
    ):
        self.root = Path(processed_root)
        if not self.root.exists():
            raise FileNotFoundError(self.root)

        self.model = model
        self.subset = subset
        self.feature = feature
        self.data_transform = data_transform
        self.noise_aug = noise_aug
        self.feature = feature
        self.num_percent = num_percent
        self.stft_trans = transform.stft(n_fft=512,hop_length=256)
        self.resize_transform = Trans.Resize(resize_hw, antialias=True)
        self.items = []  # (audio_path, gt_path, seq_name)
        if subset!='train':
            self.transform = False
        else:
            self.transform = True

        for seq_dir in sorted([p for p in self.root.iterdir() if p.is_dir()]):
            audio_dir = seq_dir / "audio"
            gt_dir = seq_dir / "gt"
            img_dir = seq_dir / "image" / f"cam{cam}"  # not required for return

            if not (audio_dir.exists() and gt_dir.exists()):
                continue

            audio_files = sorted(audio_dir.glob("*.npy"))
            for ap in audio_files[:int(0.8*len(audio_files)*self.num_percent)]:
                s = stem(ap)
                gp = gt_dir / f"{s}.npy"
                if gp.exists():
                    self.items.append((ap, gp, seq_dir.name))

        if not self.items:
            raise RuntimeError(f"No matched (audio, gt) found under: {self.root}")

        print(f"[OK] indexed {len(self.items)} samples from {self.root}")

    def apply_mic_masking(self,spectrogram):
        C, F, T = spectrogram.shape  # C = 2 * num_mics (magnitude & phase interleaved)
        mic_idx = 0
        mag_channel = mic_idx * 2      # Magnitude channel
        phase_channel = mag_channel + 1  # Phase channel
        mask = torch.zeros_like(spectrogram[:2,:,:], dtype=torch.float32)
        masking_type = random.choice(["time", "frequency", "patch"])
        if masking_type == "time":
            num_mask = int(0.1 * T)
            mask_start = random.randint(0, T - num_mask)
            spectrogram[mag_channel, :, mask_start:mask_start + num_mask] = 0
            spectrogram[phase_channel, :, mask_start:mask_start + num_mask] = 0
            mask[mag_channel, :, mask_start:mask_start + num_mask] = 1
            mask[phase_channel, :, mask_start:mask_start + num_mask] = 1
        elif masking_type == "frequency":
            num_mask = int(0.1 * F)
            mask_start = random.randint(0, F - num_mask)
            spectrogram[mag_channel, mask_start:mask_start + num_mask, :] = 0
            spectrogram[phase_channel, mask_start:mask_start + num_mask, :] = 0
            mask[mag_channel, mask_start:mask_start + num_mask, :] = 1
            mask[phase_channel, mask_start:mask_start + num_mask, :] = 1

        elif masking_type == "patch":
            patch_size = 10
            num_patches = int(0.1 * (F * T) // (patch_size * patch_size))  # Number of patches
            for _ in range(num_patches):
                freq_start = random.randint(0, F - patch_size)
                time_start = random.randint(0, T - patch_size)
                spectrogram[mag_channel, freq_start:freq_start + patch_size, time_start:time_start + patch_size] = 0
                spectrogram[phase_channel, freq_start:freq_start + patch_size, time_start:time_start + patch_size] = 0
                mask[mag_channel, freq_start:freq_start + patch_size, time_start:time_start + patch_size] = 1
                mask[phase_channel, freq_start:freq_start + patch_size, time_start:time_start + patch_size] = 1

        return spectrogram, mask

    def __len__(self):
        return len(self.items)

    def noise_aug_algo(self, audio_data):
        # audio_data: (C,T)
        rand_num_noise = random.random()
        if rand_num_noise >= 0.4:
            target_snr_db = random.randint(-10, 30)
            audio_data = add_gaussian_noise(audio_data, target_snr_db)
        return audio_data

    def spectrogram_process(self, audio_data):
        stft = self.stft_trans(audio_data)
        magnitude = torch.abs(stft)
        C, F, T = magnitude.shape
        magnitude = normalize_magnitude(magnitude)
        phase = torch.angle(stft)
        phase = normalize_phase(phase)
        spectrogram = torch.stack([magnitude, phase], dim=1)  # Shape: (C, 2, F, T)
        spectrogram = spectrogram.view(2 * C, F, T)
        resize_transform = Trans.Resize((257,64),antialias=True)
        spectrogram_original = resize_transform(spectrogram)
        spectrogram_original_copy = spectrogram_original.clone()
        spectrogram,mask = self.apply_mic_masking(spectrogram_original_copy)
        return spectrogram,spectrogram_original[:2,:,:],mask

    def __getitem__(self, idx):
        ap, gp, seq_name = self.items[idx]
        audio = np.load(str(ap)).astype(np.float32)
        if self.noise_aug:
            audio = self.noise_aug_algo(audio)
        audio_data = torch.from_numpy(audio).float()  

        if self.feature!='raw':
            spectrogram,spectrogram_ori,mask = self.spectrogram_process(audio_data)
            return spectrogram,spectrogram_ori,mask
        else:
            if self.model=='DAMUSIC':
                audio_data = downsample_audio(audio_data,1600)
            return audio_data

class CAV3D_Dataset(Dataset):
    def __init__(
        self,
        processed_root,
        mic_offsets,
        calib_mat_path,               
        subset='train',
        data_transform="stft",         # "raw" or "stft"
        noise_aug=False,
        resize_hw=(257, 64),
        cam=5,                         
        model='DAMUSIC',
        feature='spectrogram',
        num_percent=1.0,
        assume_gt_in_camera_frame=False,
        mirror_x=False,               
    ):
        self.root = Path(processed_root)
        if not self.root.exists():
            raise FileNotFoundError(self.root)

        self.model = model
        self.subset = subset
        self.feature = feature
        self.mic_offsets = mic_offsets
        self.data_transform = data_transform
        self.noise_aug = noise_aug
        self.num_percent = float(num_percent)
        self.assume_gt_in_camera_frame = bool(assume_gt_in_camera_frame)
        self.mirror_x = bool(mirror_x)

        # ---- calib ----
        self.calib = CalibFromMat.from_mat(calib_mat_path)

        self.stft_trans = transform.stft(n_fft=512, hop_length=256)
        self.resize_transform = Trans.Resize(resize_hw, antialias=True)

        self.items = []  # (audio_path, gt_path, seq_name)

        if subset != 'train':
            self.transform = False
        else:
            self.transform = True

        split_ratio = 0.8
        seed = 0  # 想每次都一样就固定；想每次都不一样就用 None 或不设

        for seq_dir in sorted([p for p in self.root.iterdir() if p.is_dir()]):
            audio_dir = seq_dir / "audio"
            gt_dir = seq_dir / "gt"
            img_dir = seq_dir / "image" / f"cam{cam}"  # not required for return

            if not (audio_dir.exists() and gt_dir.exists()):
                continue

            audio_files = sorted(list(audio_dir.glob("*.wav")) + list(audio_dir.glob("*.npy")))
            if len(audio_files) == 0:
                continue

            # ===== 0) 只保留有对应 gt 的样本（建议先过滤再 split，避免 split 后大量缺 gt）
            valid_pairs = []
            for ap in audio_files:
                s = stem(ap)
                gp = gt_dir / f"{s}.npy"
                if gp.exists():
                    valid_pairs.append((ap, gp))
            if len(valid_pairs) == 0:
                continue

            # ===== 1) per-seq 随机划分：shuffle 后 80/20
            rng = np.random.default_rng(seed + (abs(hash(seq_dir.name)) % 1000000) if seed is not None else None)
            idx = np.arange(len(valid_pairs))
            rng.shuffle(idx)
            valid_pairs = [valid_pairs[i] for i in idx]

            n_total = len(valid_pairs)
            n_split = int(n_total * split_ratio)
            n_split = max(1, min(n_total - 1, n_split))  # 保证两边都至少有1个

            if subset == "train":
                split_pairs = valid_pairs[:n_split]
            else:
                split_pairs = valid_pairs[n_split:]
                
            n_take = int(len(split_pairs) * self.num_percent)
            n_take = max(1, min(len(split_pairs), n_take))
            split_pairs = split_pairs[:n_take]
            for ap, gp in split_pairs:
                self.items.append((ap, gp, seq_dir.name))

        # for seq_dir in sorted([p for p in self.root.iterdir() if p.is_dir()]):
        #     audio_dir = seq_dir / "audio"
        #     gt_dir = seq_dir / "gt"
        #     img_dir = seq_dir / "image" / f"cam{cam}"  # not required for return

        #     if not (audio_dir.exists() and gt_dir.exists()):
        #         continue

        #     audio_files = sorted(list(audio_dir.glob("*.wav")) + list(audio_dir.glob("*.npy")))
        #     if len(audio_files) == 0:
        #         continue

        #     n_take = int(len(audio_files) * self.num_percent) if subset == "train" else int(len(audio_files) * self.num_percent)
        #     n_take = max(1, min(len(audio_files), n_take))

        #     for ap in audio_files:
        #         s = stem(ap)  
        #         gp = gt_dir / f"{s}.npy"
        #         if gp.exists():
        #             self.items.append((ap, gp, seq_dir.name))

        # if not self.items:
        #     raise RuntimeError(f"No matched (audio, gt) found under: {self.root}")

        # print(f"[OK] indexed {len(self.items)} samples from {self.root}")

    def __len__(self):
        return len(self.items)

    def noise_aug_algo(self, audio_data: np.ndarray):
        # audio_data: (C,T) numpy
        rand_num_noise = random.random()
        if rand_num_noise >= 0.5:
            target_snr_db = random.randint(-10, 30)
            audio_data = add_gaussian_noise(audio_data, target_snr_db)
        return audio_data

    def spectrogram_process(self, audio_data: torch.Tensor):
        # audio_data: (C,T)
        stft = self.stft_trans(audio_data)
        correlation = compute_correlation_matrices_torch(stft)

        magnitude = torch.abs(stft)
        C, F, T = magnitude.shape
        magnitude = normalize_magnitude(magnitude)

        phase = torch.angle(stft)
        phase = normalize_phase(phase)


        spectrogram = torch.stack([phase, phase], dim=1)  # (C,2,F,T)
        spectrogram = spectrogram.view(2 * C, F, T)
        spectrogram = self.resize_transform(spectrogram)
        return spectrogram, correlation

    def _load_audio_CT(self, ap: Path) -> np.ndarray:
        if ap.suffix.lower() == ".wav":
            wav, sr = sf.read(str(ap), always_2d=True)  # (T,C)
            audio = wav.T.astype(np.float32)            # (C,T)
            return audio
        elif ap.suffix.lower() == ".npy":
            audio = np.load(str(ap)).astype(np.float32)
            if audio.ndim == 1:
                audio = audio[None, :]  # (1,T)
            elif audio.ndim == 2:
                if audio.shape[0] > audio.shape[1]:
                    pass
                if audio.shape[0] > 16 and audio.shape[1] <= 16:
                    audio = audio.T
            return audio.astype(np.float32)
        else:
            raise ValueError(f"Unsupported audio format: {ap}")

    def __getitem__(self, idx):
        ap, gp, seq_name = self.items[idx]
        audio_np = self._load_audio_CT(Path(ap))
        if self.noise_aug:
            audio_np = self.noise_aug_algo(audio_np)
        audio = torch.from_numpy(audio_np).float()  # (C,T)

        gt_obj = np.load(str(gp), allow_pickle=True)
        if isinstance(gt_obj, np.ndarray) and gt_obj.dtype == object:
            try:
                gt_dict = gt_obj.item()
                xyz = np.asarray(gt_dict.get("gt3d_xyz", gt_dict.get("xyz", None)), dtype=np.float32)
                if xyz is None:
                    raise KeyError("gt dict has no gt3d_xyz/xyz")
            except Exception:
                gt_dict = load_gt_dict(gp) 
                xyz = np.asarray(gt_dict["gt3d_xyz"], dtype=np.float32)
        else:
            xyz = np.asarray(gt_obj, dtype=np.float32)

        xyz = parse_gt_xyz(xyz)  # (K,3)

        loc_theta_deg = doa_xz_deg_from_xyz_cav3d(
            xyz,
            calib=self.calib,
            assume_gt_in_camera_frame=self.assume_gt_in_camera_frame,
            mirror_x=self.mirror_x,
        ) 
        doas = torch.tensor(loc_theta_deg, dtype=torch.float32)
        sv, doas = array_aug(self.mic_offsets, doas, self.transform,mic_center=np.array([[0,0,0]]))
        if self.feature != 'raw':
            spectrogram, correlation = self.spectrogram_process(audio)
            return spectrogram, doas, sv, correlation
        else:
            audio_data = audio
            if self.model == 'DAMUSIC':
                audio_data = downsample_audio(audio_data, 1600)
            return audio_data, doas, sv

class CAV3D_Dataset_pretrain(Dataset):
    def __init__(
        self,
        processed_root,              
        subset='train',
        data_transform="stft",         # "raw" or "stft"
        noise_aug=False,
        resize_hw=(257, 64),
        cam=5,                         
        model='DAMUSIC',
        feature='spectrogram',
        num_percent=1.0,           
    ):
        self.root = Path(processed_root)
        if not self.root.exists():
            raise FileNotFoundError(self.root)

        self.model = model
        self.subset = subset
        self.feature = feature
        self.data_transform = data_transform
        self.noise_aug = noise_aug
        self.num_percent = float(num_percent)
        self.stft_trans = transform.stft(n_fft=512, hop_length=256)
        self.resize_transform = Trans.Resize(resize_hw, antialias=True)
        self.items = []  # (audio_path, gt_path, seq_name)

        if subset != 'train':
            self.transform = False
        else:
            self.transform = True

        for seq_dir in sorted([p for p in self.root.iterdir() if p.is_dir()]):
            audio_dir = seq_dir / "audio"
            gt_dir = seq_dir / "gt"
            img_dir = seq_dir / "image" / f"cam{cam}"  # not required for return

            if not (audio_dir.exists() and gt_dir.exists()):
                continue

            audio_files = sorted(list(audio_dir.glob("*.wav")) + list(audio_dir.glob("*.npy")))
            if len(audio_files) == 0:
                continue

            n_take = int(len(audio_files) * self.num_percent) if subset == "train" else int(len(audio_files) * self.num_percent)
            n_take = max(1, min(len(audio_files), n_take))

            for ap in audio_files[:n_take]:
                s = stem(ap)  
                gp = gt_dir / f"{s}.npy"
                if gp.exists():
                    self.items.append((ap, gp, seq_dir.name))

        if not self.items:
            raise RuntimeError(f"No matched (audio, gt) found under: {self.root}")

        print(f"[OK] indexed {len(self.items)} samples from {self.root}")

    def __len__(self):
        return len(self.items)

    def noise_aug_algo(self, audio_data: np.ndarray):
        # audio_data: (C,T) numpy
        rand_num_noise = random.random()
        if rand_num_noise >= 0.5:
            target_snr_db = random.randint(-10, 30)
            audio_data = add_gaussian_noise(audio_data, target_snr_db)
        return audio_data

    def apply_mic_masking(self,spectrogram):
        C, F, T = spectrogram.shape  # C = 2 * num_mics (magnitude & phase interleaved)
        mic_idx = 0
        mag_channel = mic_idx * 2      # Magnitude channel
        phase_channel = mag_channel + 1  # Phase channel
        mask = torch.zeros_like(spectrogram[:2,:,:], dtype=torch.float32)
        masking_type = random.choice(["time", "frequency", "patch"])
        if masking_type == "time":
            num_mask = int(0.1 * T)
            mask_start = random.randint(0, T - num_mask)
            spectrogram[mag_channel, :, mask_start:mask_start + num_mask] = 0
            spectrogram[phase_channel, :, mask_start:mask_start + num_mask] = 0
            mask[mag_channel, :, mask_start:mask_start + num_mask] = 1
            mask[phase_channel, :, mask_start:mask_start + num_mask] = 1
        elif masking_type == "frequency":
            num_mask = int(0.1 * F)
            mask_start = random.randint(0, F - num_mask)
            spectrogram[mag_channel, mask_start:mask_start + num_mask, :] = 0
            spectrogram[phase_channel, mask_start:mask_start + num_mask, :] = 0
            mask[mag_channel, mask_start:mask_start + num_mask, :] = 1
            mask[phase_channel, mask_start:mask_start + num_mask, :] = 1

        elif masking_type == "patch":
            patch_size = 10
            num_patches = int(0.1 * (F * T) // (patch_size * patch_size))  # Number of patches
            for _ in range(num_patches):
                freq_start = random.randint(0, F - patch_size)
                time_start = random.randint(0, T - patch_size)
                spectrogram[mag_channel, freq_start:freq_start + patch_size, time_start:time_start + patch_size] = 0
                spectrogram[phase_channel, freq_start:freq_start + patch_size, time_start:time_start + patch_size] = 0
                mask[mag_channel, freq_start:freq_start + patch_size, time_start:time_start + patch_size] = 1
                mask[phase_channel, freq_start:freq_start + patch_size, time_start:time_start + patch_size] = 1

        return spectrogram, mask

    def spectrogram_process(self, audio_data):
        stft = self.stft_trans(audio_data)
        magnitude = torch.abs(stft)
        C, F, T = magnitude.shape
        magnitude = normalize_magnitude(magnitude)
        phase = torch.angle(stft)
        phase = normalize_phase(phase)
        spectrogram = torch.stack([magnitude, phase], dim=1)  # Shape: (C, 2, F, T)
        spectrogram = spectrogram.view(2 * C, F, T)
        resize_transform = Trans.Resize((257,64),antialias=True)
        spectrogram_original = resize_transform(spectrogram)
        spectrogram_original_copy = spectrogram_original.clone()
        spectrogram,mask = self.apply_mic_masking(spectrogram_original_copy)
        return spectrogram,spectrogram_original[:2,:,:],mask

    def _load_audio_CT(self, ap: Path) -> np.ndarray:
        if ap.suffix.lower() == ".wav":
            wav, sr = sf.read(str(ap), always_2d=True)  # (T,C)
            audio = wav.T.astype(np.float32)            # (C,T)
            return audio
        elif ap.suffix.lower() == ".npy":
            audio = np.load(str(ap)).astype(np.float32)
            if audio.ndim == 1:
                audio = audio[None, :]  # (1,T)
            elif audio.ndim == 2:
                if audio.shape[0] > audio.shape[1]:
                    pass
                if audio.shape[0] > 16 and audio.shape[1] <= 16:
                    audio = audio.T
            return audio.astype(np.float32)
        else:
            raise ValueError(f"Unsupported audio format: {ap}")

    def __getitem__(self, idx):
        ap, gp, seq_name = self.items[idx]
        audio_np = self._load_audio_CT(Path(ap))
        if self.noise_aug:
            audio_np = self.noise_aug_algo(audio_np)
        audio_data = torch.from_numpy(audio_np).float()  # (C,T)
        if self.feature!='raw':
            spectrogram,spectrogram_ori,mask = self.spectrogram_process(audio_data)
            return spectrogram,spectrogram_ori,mask
        else:
            if self.model=='DAMUSIC':
                audio_data = downsample_audio(audio_data,1600)
            return audio_data



def quick_inspect_batch(batch):
    """
    兼容 dataset 的两种返回：
      1) spectrogram, doas, sv, correlation
      2) audio_data, doas, sv
    """
    if len(batch) == 4:
        x, doas, sv, corr = batch
        print("[BATCH] spectrogram:", tuple(x.shape), x.dtype)
        print("[BATCH] correlation:", tuple(corr.shape), corr.dtype)
    elif len(batch) == 3:
        x, doas, sv = batch
        print("[BATCH] audio/raw:", tuple(x.shape), x.dtype)
    else:
        raise ValueError("Unexpected batch length:", len(batch))

    print("[BATCH] doas:", tuple(doas.shape), doas.dtype, "min/max:", float(doas.min()), float(doas.max()))
    print("[BATCH] sv:", type(sv), "shape:", (tuple(sv.shape) if torch.is_tensor(sv) else None))

if __name__ == "__main__":
    index = 300
    # mic_offset = np.array([[ 45.7/1000/2, 45.7/1000/2, 0.0],
    #         [ -45.7/1000/2,  45.7/1000/2, 0.0],
    #         [-45.7/1000/2,  -45.7/1000/2, 0.0],
    #         [45.7/1000/2,  -45.7/1000/2, 0.0]])
    
    # mic_offset = np.array([
    # [-0.10000,  0.40000,  0.0],
    # [-0.07071,  0.32929,  0.0],
    # [ 0.00000,  0.30000,  0.0],
    # [ 0.07071,  0.32929,  0.0],
    # [ 0.10000,  0.40000,  0.0],
    # [ 0.07071,  0.47071,  0.0],
    # [ 0.00000,  0.50000,  0.0],
    # [-0.07071,  0.47071,  0.0],

    # [-0.10000, -0.40000,  0.0],
    # [-0.07071, -0.47071,  0.0],
    # [ 0.00000, -0.50000,  0.0],
    # [ 0.07071, -0.47071,  0.0],
    # [ 0.10000, -0.40000,  0.0],
    # [ 0.07071, -0.32929,  0.0],
    # [ 0.00000, -0.30000,  0.0],
    # [-0.07071, -0.32929,  0.0],
    # ])

    mic_offsets = np.array([
        [ 10.1,   -0.075,  0.0],   # t1
        [  7.3,    6.925,  0.0],   # t2
        [  0.1,    9.925,  0.0],   # t3
        [ -7.0,    7.025,  0.0],   # t4
        [ -9.9,    0.125,  0.0],   # t5
        [ -7.2,   -6.875,  0.0],   # t6
        [ -0.2,   -9.975,  0.0],   # t7
        [  6.8,   -7.075,  0.0],   # t8
    ])/100


    PROCESSED_ROOT = "/home/Disk/yyz/deepmusic++/AV3T/data/processed_cav/SOT"
    CALIB_MAT = "/home/Disk/yyz/deepmusic++/AV3T/data/CAV3D-release1.0/calib/C5SOT.mat"

    # ====== 2) 构建 dataset ======
    ds = CAV3D_Dataset(
        processed_root=PROCESSED_ROOT,
        mic_offsets=mic_offsets,
        calib_mat_path=CALIB_MAT,
        subset="train",
        feature="spectrogram",     # "raw" or "spectrogram"
        model="DAMUSIC",
        noise_aug=False,
        num_percent=1.0,
        assume_gt_in_camera_frame=False,
        mirror_x=False,
    )

    print("Total samples:", len(ds))
    assert len(ds) > 0

    # ====== 3) 单样本测试 ======
    idx = min(10, len(ds) - 1)
    sample = ds[idx]
    print("\n=== Single sample check (idx=%d) ===" % idx)
    if not isinstance(sample, (tuple, list)):
        raise TypeError("Dataset __getitem__ should return tuple/list, got:", type(sample))
    quick_inspect_batch(sample)

    # ====== 4) 多次随机抽样看 DOA/形状是否稳定 ======
    print("\n=== Random samples summary ===")
    for k in range(5):
        ii = random.randint(0, len(ds) - 1)
        out = ds[ii]
        if len(out) == 4:
            x, doas, sv, corr = out
        else:
            x, doas, sv = out

        print(f"[{k}] idx={ii} | doas_num={int(doas.numel())} | doa(min,max)=({float(doas.min()):.1f},{float(doas.max()):.1f})")

    # ====== 5) DataLoader 测试（可选） ======
    from torch.utils.data import DataLoader

    def collate_keep_list(batch):
        # 由于 doas 的人数可能不同，默认 collate 会报错
        # 这里直接返回 list，方便你先验证 pipeline 没问题
        return batch

    dl = DataLoader(ds, batch_size=4, shuffle=True, num_workers=0, collate_fn=collate_keep_list)
    b = next(iter(dl))
    print("\n=== DataLoader batch check ===")
    print("Batch type:", type(b), "len:", len(b))
    for i, item in enumerate(b):
        print(f"  - item {i}:")
        quick_inspect_batch(item)
        print("")





    # data_dir = '/media/kemove/T9/sound_source_loc/locate_dataset/RSL2019'
    # root = '/media/kemove/T9/sound_source_loc/afpild_data'
    # data_dir = '/media/kemove/T9/sound_source_loc/SoClas_database/SoClas_database/Segmented_Sound'
    # dataset   = RSL_database(root=data_dir,subset='train',noise_aug=True,num_percent=0.1,mic_offsets=mic_offset)
    # train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
    #
    # for i, (spectrograms, doas, svs, correlations) in enumerate(train_loader):
    #     print(svs.shape)
    #     # print(f"Batch {i}")
    #     # print(f"  Spectrograms shape: {spectrograms.shape}")   # torch.Size([B, C, F, T])
    #     # print(f"  DOAs: {doas}")                              # list of tensors or lists (variable length)
    #     # print(f"  SVs: {svs[0].shape}")                           # list of steering vectors
    #     # print(f"  Correlation shape: {correlations.shape}")   # torch.Size([B, C, C, F])
    #
    #     if i == 1:  # Just show two batches
    #         break
    # root = '/media/kemove/T9/sound_source_loc/simulation_data'
    # datset = DAMUSIC_Loader(root,mic_offsets=mic_offset,subset='train',num_percent=0.2,mode='step_10')
    # print(datset.audio_data)
    # _,b,_,_ = datset.__getitem__(1)
    # print(b)
    # dataset = SoClas_database(root=data_dir,num_percent=0.1)
    # dataset = AFPILD_raw_Dataset(dataset_dir=root,data_type='test',num_percent=0.1)

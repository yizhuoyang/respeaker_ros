import numpy as np
import librosa
from skimage.measure import block_reduce
from PIL import Image
from scipy.io import wavfile


def compute_stft(signal, use_compress=True):
    n_fft = 512
    hop_length = 160
    win_length = 400
    stft = np.abs(librosa.stft(signal, n_fft=n_fft, hop_length=hop_length, win_length=win_length))
    if use_compress:
        stft = block_reduce(stft, block_size=(4, 4), func=np.mean)
    return stft


def compute_spectrogram(audio_data, use_compress=True):
    channels = [np.log1p(compute_stft(channel, use_compress)) for channel in audio_data]
    return np.stack(channels, axis=-1).astype(np.float32)


def compute_stft_phase_features(
    audiogoal,
    mode="ipd",
    pairs=((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)),
    eps=1e-8,
):
    """
    Build phase features from multi-channel audio.

    Args:
        audiogoal: np.ndarray, shape (C, N).
        mode:
            - "ipd": for every pair, output cos(ipd), sin(ipd).
              With 4 channels and all pairs, output shape is (F,T,12).
            - "phase": output cos/sin phase for every selected channel.
            - "both": concatenate phase and ipd features.
            - "gcc_phat_complex": for every pair, output PHAT real/imag.
        pairs: channel index pairs used by IPD/GCC. Indices refer to audiogoal after
            channel selection in the dataset.
    """
    audio = np.asarray(audiogoal, dtype=np.float32)
    if audio.ndim != 2:
        raise RuntimeError(f"Expected audio shape (C, N), got {audio.shape}")

    def stft_complex(signal):
        return librosa.stft(signal, n_fft=512, hop_length=160, win_length=400)

    spectra = [stft_complex(audio[channel]) for channel in range(audio.shape[0])]
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

    raise ValueError(f"Unknown mode: {mode}")


def check_pair(first, second, num_channels):
    if first >= num_channels or second >= num_channels:
        raise RuntimeError(
            f"IPD pair ({first}, {second}) requires {max(first, second) + 1} channels, "
            f"but audio has {num_channels} channels"
        )


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


def load_audio_wav(path, channels=(0, 1, 2, 3)):
    sample_rate, audio = wavfile.read(path)
    audio = np.asarray(audio)
    if audio.ndim == 1:
        audio = audio[:, None]

    if np.issubdtype(audio.dtype, np.integer):
        audio = audio.astype(np.float32) / float(np.iinfo(audio.dtype).max)
    else:
        audio = audio.astype(np.float32)

    channels = tuple(channels)
    if max(channels) >= audio.shape[1]:
        raise RuntimeError(
            f"Audio file {path} has {audio.shape[1]} channels, requested channels {channels}"
        )
    return audio[:, channels].T, sample_rate


def load_image(path, normalize_rgb=True, image_size=None):
    path = str(path)
    if path.endswith(".npy"):
        image = np.load(path)
    else:
        image = np.asarray(Image.open(path))

    image = image.astype(np.float32)
    if image.ndim == 3 and normalize_rgb and image.max() > 1.0:
        image = image / 255.0
    elif image.ndim == 2 and image.max() > 100.0:
        image = image / 1000.0
    if image_size is not None:
        image = resize_image(image, image_size)
    return image


def resize_image(image, image_size):
    height, width = image_size
    if image.ndim == 2:
        pil_mode = "F"
        pil_image = Image.fromarray(image.astype(np.float32), mode=pil_mode)
        return np.asarray(pil_image.resize((width, height), resample=Image.BILINEAR), dtype=np.float32)

    pil_image = Image.fromarray((np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8))
    resized = np.asarray(pil_image.resize((width, height), resample=Image.BILINEAR), dtype=np.float32)
    return resized / 255.0


def make_doa_gaussian_from_yaw_deg(
    yaw_signed_deg,
    distance_m=0.0,
    num_bins=360,
    base_sigma_deg=3.0,
    sigma_scale_deg=1.0,
):
    # 0 deg bin is right, 90 deg bin is front. Positive yaw means target is left.
    angle_deg = (90.0 + yaw_signed_deg) % 360.0
    sigma_deg = max(base_sigma_deg + sigma_scale_deg * distance_m, 1e-3)
    angles_deg = np.linspace(0.0, 360.0, num_bins, endpoint=False)
    diff_deg = (angles_deg - angle_deg + 180.0) % 360.0 - 180.0
    probs = np.exp(-0.5 * (diff_deg / sigma_deg) ** 2)
    if probs.sum() > 0:
        probs = probs / probs.sum()
    return probs.astype(np.float32)


def make_r_gaussian_1d(
    r,
    num_bins=120,
    r_min=0.0,
    r_max=6.0,
    base_sigma=0.08,
    sigma_scale=0.05,
):
    r_axis = np.linspace(r_min, r_max, num_bins).astype(np.float32)
    sigma = max(base_sigma + sigma_scale * r, 1e-3)
    probs = np.exp(-0.5 * ((r_axis - r) / sigma) ** 2)
    if probs.sum() > 0:
        probs = probs / probs.sum()
    return probs.astype(np.float32)

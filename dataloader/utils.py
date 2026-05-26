import os
import tempfile

# librosa lazily imports numba, which needs a writable cache outside site-packages.
os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join(tempfile.gettempdir(), "respeaker_numba_cache"))

import numpy as np
import librosa
from skimage.measure import block_reduce
from PIL import Image
from scipy.io import wavfile
from scipy.signal import butter, iirnotch, istft, medfilt, sosfiltfilt, stft, tf2sos


def apply_audio_bandpass(audio, sample_rate, low_hz=None, high_hz=None, order=6):
    """Apply the same zero-phase band filter to every audio channel.

    The input is shaped (C, N). Using the same filter for every channel keeps
    inter-channel timing/phase cues comparable for IPD-style features.
    """
    low_hz = float(low_hz or 0.0)
    high_hz = float(high_hz or 0.0)
    if low_hz <= 0 and high_hz <= 0:
        return np.asarray(audio, dtype=np.float32)

    nyquist = float(sample_rate) * 0.5
    if high_hz <= 0 or high_hz >= nyquist:
        high_hz = nyquist * 0.999
    if low_hz <= 0:
        sos = butter(order, high_hz / nyquist, btype="lowpass", output="sos")
    elif low_hz >= high_hz:
        raise ValueError(
            f"Invalid audio bandpass range: low_hz={low_hz}, high_hz={high_hz}, "
            f"nyquist={nyquist}"
        )
    else:
        sos = butter(order, [low_hz / nyquist, high_hz / nyquist], btype="bandpass", output="sos")
    return sosfiltfilt(sos, np.asarray(audio, dtype=np.float32), axis=-1).astype(np.float32)


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


def load_audio_wav(path, channels=(1, 2, 3, 4)):
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


def parse_float_list(value):
    if value is None or str(value).strip() == "":
        return []
    return [float(item.strip()) for item in str(value).split(",") if item.strip()]


def load_noise_profile(
    noise_paths,
    channels,
    sample_rate,
    n_fft=1024,
    hop=256,
    highpass_hz=90.0,
    notches_hz=(48.0, 180.0, 342.0, 1845.0),
    notch_q=35.0,
):
    powers = []
    for path in noise_paths:
        noise, noise_sr = load_audio_wav(path, channels)
        if noise_sr != sample_rate:
            raise RuntimeError(
                f"Noise sample rate mismatch: {path} has {noise_sr}, expected {sample_rate}"
            )
        noise = apply_shared_time_filters(
            noise,
            sample_rate=sample_rate,
            highpass_hz=highpass_hz,
            notches_hz=notches_hz,
            notch_q=notch_q,
        )
        _, _, noise_stft = stft(
            noise,
            fs=sample_rate,
            nperseg=n_fft,
            noverlap=n_fft - hop,
            axis=-1,
            boundary=None,
        )
        powers.append(np.mean(np.abs(noise_stft) ** 2, axis=(0, 2)))

    if not powers:
        return None
    return np.mean(np.stack(powers, axis=0), axis=0).astype(np.float32)


def denoise_multichannel_audio(
    audio,
    sample_rate,
    noise_power=None,
    highpass_hz=90.0,
    notches_hz=(48.0, 180.0, 342.0, 1845.0),
    notch_q=35.0,
    spectral_strength=0.6,
    gain_floor=0.35,
    n_fft=1024,
    hop=256,
):
    """Denoise audio shaped (C, N) while preserving inter-channel phase cues."""
    denoised = apply_shared_time_filters(
        audio,
        sample_rate=sample_rate,
        highpass_hz=highpass_hz,
        notches_hz=notches_hz,
        notch_q=notch_q,
    )
    if noise_power is None or spectral_strength <= 0:
        return denoised.astype(np.float32)

    return apply_shared_spectral_gate(
        denoised,
        sample_rate=sample_rate,
        noise_power=noise_power,
        n_fft=n_fft,
        hop=hop,
        strength=spectral_strength,
        gain_floor=gain_floor,
    )


def apply_motion_gated_spectral_gate(
    audio,
    sample_rate,
    noise_power,
    n_fft=1024,
    hop=256,
    strength=0.35,
    gain_floor=0.55,
    gate_threshold=1.6,
    gate_smooth_frames=5,
):
    """Apply spectral subtraction only on frames that look like motion noise.

    The gate is computed from a noise-profile-weighted energy ratio. The same
    time-frequency gain is then applied to every channel, keeping IPD cues intact.
    """
    if noise_power is None or strength <= 0:
        return np.asarray(audio, dtype=np.float32)

    _, _, spectrum = stft(
        audio,
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
        return np.asarray(audio, dtype=np.float32)

    gain = 1.0 - strength * gate[None, :] * noise_power[:, None] / (signal_power + 1e-10)
    gain = np.clip(gain, gain_floor, 1.0).astype(np.float32)
    gated = spectrum * gain[None, :, :]

    restored_channels = []
    for channel in range(gated.shape[0]):
        _, restored = istft(
            gated[channel],
            fs=sample_rate,
            nperseg=n_fft,
            noverlap=n_fft - hop,
            input_onesided=True,
        )
        restored_channels.append(restored)
    restored = np.stack(restored_channels, axis=0).astype(np.float32)
    return restored[:, : audio.shape[1]]


def suppress_shared_transients(
    audio,
    sample_rate,
    frame_ms=20.0,
    hop_ms=5.0,
    threshold=3.0,
    attenuation=0.5,
    smooth_frames=5,
):
    """Suppress short broadband impulses with one shared gain envelope.

    This targets footstep-like ticks/thuds. The same envelope is applied to all
    channels so inter-channel phase and relative timing are not independently
    distorted.
    """
    audio = np.asarray(audio, dtype=np.float32)
    if attenuation <= 0 or threshold <= 0 or audio.shape[-1] == 0:
        return audio

    frame = max(8, int(round(sample_rate * frame_ms / 1000.0)))
    hop = max(1, int(round(sample_rate * hop_ms / 1000.0)))
    num_samples = audio.shape[1]
    if num_samples < frame:
        return audio

    energies = []
    starts = list(range(0, num_samples - frame + 1, hop))
    for start in starts:
        segment = audio[:, start:start + frame]
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
    return (audio * envelope[None, :]).astype(np.float32)


def filter_and_mute_motion_impacts(
    audio,
    sample_rate,
    highpass_hz=120.0,
    notches_hz=(48.8, 66.4, 179.7, 341.8, 867.2, 1271.5, 1845.7, 2533.2, 3783.2),
    notch_q=35.0,
    motion_threshold=0.06,
    mute_window_sec=0.05,
    mute_floor=0.02,
    edge_smooth_ms=5.0,
):
    """Fixed robot-noise filtering plus shared motion-impact muting.

    audio is shaped (C, N). The mute envelope is shared across channels to avoid
    channel-dependent phase/timing distortion. mute_floor=0 gives hard zeros;
    a small positive floor avoids undefined/unstable phase in STFT features.
    """
    filtered = apply_shared_time_filters(
        audio,
        sample_rate=sample_rate,
        highpass_hz=highpass_hz,
        notches_hz=notches_hz,
        notch_q=notch_q,
    )
    mask = make_motion_mute_mask(
        filtered,
        sample_rate=sample_rate,
        threshold=motion_threshold,
        window_sec=mute_window_sec,
    )
    if not np.any(mask):
        return filtered.astype(np.float32)

    floor = float(np.clip(mute_floor, 0.0, 1.0))
    gain = np.ones(filtered.shape[1], dtype=np.float32)
    gain[mask] = floor

    edge_samples = int(round(edge_smooth_ms * sample_rate / 1000.0))
    if edge_samples > 1 and floor > 0.0:
        kernel = np.ones(edge_samples, dtype=np.float32)
        kernel = kernel / kernel.sum()
        gain = np.convolve(gain, kernel, mode="same").astype(np.float32)
        gain = np.clip(gain, floor, 1.0)

    return (filtered * gain[None, :]).astype(np.float32)


def make_motion_mute_mask(audio, sample_rate, threshold=0.06, window_sec=0.05):
    audio = np.asarray(audio, dtype=np.float32)
    if threshold <= 0:
        raise RuntimeError("motion_threshold must be positive")
    if window_sec < 0:
        raise RuntimeError("mute_window_sec must be non-negative")
    if audio.ndim != 2:
        raise RuntimeError(f"Expected audio shape (C, N), got {audio.shape}")

    amplitude = np.max(np.abs(audio), axis=0)
    trigger = amplitude > threshold
    if not np.any(trigger):
        return np.zeros(audio.shape[1], dtype=bool)

    radius = int(round(window_sec * sample_rate))
    kernel = np.ones(radius * 2 + 1, dtype=np.int16)
    expanded = np.convolve(trigger.astype(np.int16), kernel, mode="same") > 0
    return expanded[: audio.shape[1]]


def apply_shared_time_filters(audio, sample_rate, highpass_hz=90.0, notches_hz=None, notch_q=35.0):
    filtered = np.asarray(audio, dtype=np.float32).T
    if highpass_hz and highpass_hz > 0:
        sos = butter(4, highpass_hz, btype="highpass", fs=sample_rate, output="sos")
        filtered = sosfiltfilt(sos, filtered, axis=0).astype(np.float32)

    for freq in notches_hz or []:
        if freq <= 0 or freq >= sample_rate / 2:
            continue
        b, a = iirnotch(freq, notch_q, fs=sample_rate)
        sos = tf2sos(b, a)
        filtered = sosfiltfilt(sos, filtered, axis=0).astype(np.float32)
    return filtered.T.astype(np.float32)


def apply_shared_spectral_gate(audio, sample_rate, noise_power, n_fft=1024, hop=256, strength=0.6, gain_floor=0.35):
    _, _, spectrum = stft(
        audio,
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
        _, restored = istft(
            gated[channel],
            fs=sample_rate,
            nperseg=n_fft,
            noverlap=n_fft - hop,
            input_onesided=True,
        )
        restored_channels.append(restored)
    restored = np.stack(restored_channels, axis=0).astype(np.float32)
    return restored[:, : audio.shape[1]]


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


def make_doa_gaussian_from_azimuth_deg(
    azimuth_deg,
    distance_m=0.0,
    num_bins=360,
    base_sigma_deg=3.0,
    sigma_scale_deg=1.0,
):
    """Render a circular target where bin 0 is +x and bin 90 is +y."""
    sigma_deg = max(base_sigma_deg + sigma_scale_deg * distance_m, 1e-3)
    angles_deg = np.linspace(0.0, 360.0, num_bins, endpoint=False)
    diff_deg = (angles_deg - float(azimuth_deg) + 180.0) % 360.0 - 180.0
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

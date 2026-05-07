import numpy as np
import librosa
from skimage.measure import block_reduce


def compute_stft(signal,use_compress=True):
    n_fft = 512
    hop_length = 160
    win_length = 400
    stft = np.abs(librosa.stft(signal, n_fft=n_fft, hop_length=hop_length, win_length=win_length))
    if use_compress:
        stft = block_reduce(stft, block_size=(4, 4), func=np.mean)
    return stft

def compute_spectrogram(audio_data,use_compress=True):
    channel1_magnitude = np.log1p(compute_stft(audio_data[0],use_compress))
    channel2_magnitude = np.log1p(compute_stft(audio_data[1],use_compress))
    spectrogram = np.stack([channel1_magnitude, channel2_magnitude], axis=-1)

    return spectrogram

def compute_stft_phase_features(audiogoal, mode="ipd", eps=1e-8):
    """
    Args:
        audiogoal: array-like, shape (2, N)  (stereo / 2-mic)
        mode:
            - "phase": return per-channel phase sin/cos  -> (F, T, 4) = [cos(phi1), sin(phi1), cos(phi2), sin(phi2)]
            - "ipd":   return inter-channel phase diff sin/cos -> (F, T, 2) = [cos(ipd), sin(ipd)]
            - "both":  return both of above -> (F, T, 6)
            - "gcc_phat_complex": return normalized cross-spectrum real/imag (PHAT) -> (F, T, 2)
    Returns:
        feat: np.ndarray, float32
    """
    def stft_complex(signal):
        n_fft = 512
        hop_length = 160
        win_length = 400
        # complex STFT: (F, T)
        return librosa.stft(signal, n_fft=n_fft, hop_length=hop_length, win_length=win_length)

    # complex STFT for each channel
    X1 = stft_complex(audiogoal[0])
    X2 = stft_complex(audiogoal[1])

    # phase in [-pi, pi]
    phi1 = np.angle(X1)
    phi2 = np.angle(X2)

    # per-channel sin/cos
    c1, s1 = np.cos(phi1), np.sin(phi1)
    c2, s2 = np.cos(phi2), np.sin(phi2)

    # IPD = phi1 - phi2, wrapped to [-pi, pi]
    ipd = np.angle(np.exp(1j * (phi1 - phi2)))
    cipd, sipd = np.cos(ipd), np.sin(ipd)

    if mode == "phase":
        feat = np.stack([c1, s1, c2, s2], axis=-1).astype(np.float32)  # (F,T,4)
        return feat

    if mode == "ipd":
        feat = np.stack([cipd, sipd], axis=-1).astype(np.float32)  # (F,T,2)
        return feat

    if mode == "both":
        feat = np.stack([c1, s1, c2, s2, cipd, sipd], axis=-1).astype(np.float32)  # (F,T,6)
        return feat

    if mode == "gcc_phat_complex":
        # PHAT normalized cross-spectrum: X1*conj(X2) / |X1*conj(X2)|
        C = X1 * np.conj(X2)
        C_phat = C / (np.abs(C) + eps)
        feat = np.stack([C_phat.real, C_phat.imag], axis=-1).astype(np.float32)  # (F,T,2)
        return feat

    raise ValueError(f"Unknown mode: {mode}")


def make_source_heatmap(
    front: float,
    right: float,
    map_size: int = 64,
    meters_per_pixel: float = 0.5,
    base_sigma: float = 0.2,
    sigma_scale: float = 0.5,
):

    H = W = map_size
    cx = W // 2
    cy = H // 2


    dx_pix = right / meters_per_pixel
    dy_pix = -front / meters_per_pixel 

    source_x = cx + dx_pix
    source_y = cy + dy_pix


    dist_m = np.sqrt(front**2 + right**2)  
    sigma = base_sigma + sigma_scale * dist_m
    sigma = max(sigma, 1e-3)  

    ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')

    gauss = np.exp(
        - ((xs - source_x)**2 + (ys - source_y)**2) / (2 * sigma**2)
    )

    s = gauss.sum()
    if s > 0:
        gauss = gauss / s

    gauss = (gauss-gauss.min())/(gauss.max()-gauss.min()+1e-8)
    return gauss

def make_doa_gaussian(
    front: float,
    right: float,
    num_bins: int = 360,
    base_sigma_deg: float = 5.0,
    sigma_scale_deg: float = 3.0,
):
    """
    根据 (front, right) 生成一个 DOA 高斯分布。

    坐标约定:
        - front > 0: 声源在前方
        - right > 0: 声源在右侧

    角度约定 (重要！！！):
        - 0 度: 正右 (right 正方向)
        - 90 度: 正前 (front 正方向)
        - 180 度: 正左
        - 270 度: 正后

    参数:
        num_bins: DOA 离散格数，360 表示 1 度一个 bin
        base_sigma_deg:  距离很近时的基础角度标准差（度）
        sigma_scale_deg: 距离每增加 1 米，角度标准差增加多少度

    返回:
        doa_gauss: (num_bins,) 角度上的高斯分布，范围 [0,1]
                   其第 i 个元素对应角度:
                       angle_i_deg = i * 360 / num_bins
    """
    # 1. 根据 front/right 计算 DOA（弧度）
    #    这里用 atan2(front, right)，保证：
    #      front=0, right>0 → 0 度 (右)
    #      front>0, right=0 → 90 度 (前)
    doa = np.arctan2(-front, right)  # 结果范围 [-pi, pi]

    if doa < 0:
        doa += 2 * np.pi

    dist_m = np.sqrt(front**2 + right**2)

    sigma_deg = base_sigma_deg + sigma_scale_deg * dist_m
    sigma_deg = max(sigma_deg, 1e-3)
    sigma_rad = np.deg2rad(sigma_deg)

    angles = np.linspace(0.0, 2 * np.pi, num_bins, endpoint=False)  # (num_bins,)

    diff = np.angle(np.exp(1j * (angles - doa)))  # [-pi, pi)
    doa_gauss = np.exp(- (diff ** 2) / (2 * sigma_rad ** 2))
    s = doa_gauss.sum()
    if s > 0:
        doa_gauss = doa_gauss / s

    doa_gauss = (doa_gauss - doa_gauss.min()) / (doa_gauss.max() - doa_gauss.min() + 1e-8)

    return doa_gauss

def make_r_gaussian_1d(r, num_bins=100, r_min=0.0, r_max=20.0,
                       base_sigma=0.2, sigma_scale=0.3):


    r_axis = np.linspace(r_min, r_max, num_bins).astype(np.float32)
    sigma = base_sigma + sigma_scale * r
    sigma = max(sigma, 1e-3)
    probs = np.exp(-0.5 * ((r_axis - r) / sigma)**2)
    if probs.sum() > 0:
        probs /= probs.sum()
    probs = (probs - probs.min()) / (probs.max() - probs.min() + 1e-8)
    return probs


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    hm_near = make_source_heatmap(front=0.5, right=1.0,
                                  map_size=64, meters_per_pixel=0.5,
                                  base_sigma=0.2, sigma_scale=0.5)

    hm_far = make_source_heatmap(front=10.0, right=1.0,
                                 map_size=64, meters_per_pixel=0.5,
                                 base_sigma=0.2, sigma_scale=0.5)
    print(hm_near.max(), hm_near.min())

    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.title("near source")
    plt.imshow(hm_near, cmap="hot")
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.title("far source")
    plt.imshow(hm_far, cmap="hot")
    plt.colorbar()

    plt.tight_layout()
    plt.show()

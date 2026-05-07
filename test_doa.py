import os
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from dataloader.ssl_dataset import SingleStepDataset
from network.audionet.ssl_net import SSLNet, SSLNet_DOA,SSLNet_depth_DOA
from utlis.loss import _neg_loss
import numpy as np

# ========================
# 手动配置区（自己改）
# ========================
VAL_META   = "/home/Disk/sound-space/ssl_data_semantic/val"
SAVE_DIR   = "/home/Disk/yyz/sound-spaces/weights/ssl_doa_r_audio_depth_freeze_tune"
CKPT_NAME  = "last_model.pth"
CKPT_PATH  = os.path.join(SAVE_DIR, CKPT_NAME)
CKPT_PATH = "/home/Disk/yyz/sound-spaces/data/models/savi_final_ipd_tune/laset_epoch.pth"

VIS_DIR    = "vis_result_doa_new"
VIS_DIR2   = "vis_result_dist"
INDICES    = list(range(1, 40 + 1))
use_compress = False


def visualize_sample_dist(pred_hm, gt_hm, idx, vis_dir,heading):
    if vis_dir is not None:
        os.makedirs(vis_dir, exist_ok=True)

    plt.figure(figsize=(6, 3))

    # GT

    plt.plot(gt_hm)
    plt.legend("GT",loc='upper right')
    plt.plot(pred_hm)
    plt.legend("Pred",loc='upper right')
    # plt.axis("off")

    save_path = os.path.join(vis_dir, f"sample_{idx}.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def visualize_sample(pred_doa, gt_doa, idx, vis_dir,heading):
    """
    pred_doa, gt_doa: 1D numpy 数组 (K,), 对应 DOA bins
    以 0~360 度的圆环（极坐标）可视化，并标出两者的最大值位置。
    """
    if vis_dir is not None:
        os.makedirs(vis_dir, exist_ok=True)

    K = len(gt_doa)

    # 每个 bin 对应一个角度 [0, 2π)
    theta = np.linspace(0.0, 2 * np.pi, K, endpoint=False)

    r_gt   = gt_doa
    r_pred = pred_doa

    # 找到 GT 和 Pred 的最大值 bin 和对应角度
    gt_idx   = int(np.argmax(r_gt))
    pred_idx = int(np.argmax(r_pred))

    gt_theta   = theta[gt_idx]
    pred_theta = theta[pred_idx]

    gt_angle_deg   = gt_idx * 360.0 / K
    pred_angle_deg = pred_idx * 360.0 / K

    plt.figure(figsize=(6, 6))
    ax = plt.subplot(111, projection="polar")

    # 极坐标设置：0° 在右侧，逆时针增加
    ax.set_theta_zero_location("E")  # East = 右侧
    ax.set_theta_direction(1)        # 逆时针

    # 曲线
    ax.plot(theta, r_gt,   label="GT",   linewidth=2)
    ax.plot(theta, r_pred, label="Pred", linewidth=2, linestyle="--")

    # 标出最大值位置（用 marker）
    ax.scatter([gt_theta], [r_gt[gt_idx]],   s=50, c="C0", marker="o", label=f"GT peak ({gt_angle_deg:.1f}°)")
    ax.scatter([pred_theta], [r_pred[pred_idx]], s=50, c="C1", marker="x", label=f"Pred peak ({pred_angle_deg:.1f}°)")

    # 在点附近写上角度（可选）
    # GT 文本
    ax.text(
        gt_theta,
        r_gt[gt_idx] + 0.05,   # 往外挪一点
        f"{gt_angle_deg:.1f}°",
        color="C0",
        fontsize=8,
        ha="center",
        va="bottom",
    )
    # Pred 文本
    ax.text(
        pred_theta,
        r_pred[pred_idx] + 0.05,
        f"{pred_angle_deg:.1f}°",
        color="C1",
        fontsize=8,
        ha="center",
        va="bottom",
    )


    # 半径范围（你可以根据数据调）
    ax.set_rmin(0.0)
    ax.set_rmax(1.0)

    # 角度刻度
    ax.set_thetagrids(
        [0, 90, 180, 270],
        labels=["0° (Right)", "90° (Front)", "180° (Left)", "270° (Back)"],
    )

    ax.set_title(f"DOA Map idx={idx}", va="bottom")

    # legend 里会有 GT 曲线 / Pred 曲线 / 两个 peak
    ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.2), fontsize=8)

    save_path = os.path.join(vis_dir, f"sample_{idx}.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def main():
    if VIS_DIR is not None:
        os.makedirs(VIS_DIR, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    criterion = nn.MSELoss()

    # ========================
    # 1. Dataset
    # ========================
    val_dataset = SingleStepDataset(
        root_dir=VAL_META,
        use_compress=use_compress,
        mode='doa_distance',
        audio_feat='ipd' 
    )
    print(f"Val samples: {len(val_dataset)}")

    # ========================
    # 2. Model + checkpoint
    # ========================
    model = SSLNet_DOA(use_compress=use_compress).to(device)
    # model = SSLNet_depth_DOA(use_compress=use_compress).to(device)

    if CKPT_PATH is not None and os.path.exists(CKPT_PATH):
        ckpt = torch.load(CKPT_PATH, map_location="cpu")
        if "audiogoal_predictor" in ckpt:
            model.load_state_dict(ckpt["audiogoal_predictor"], strict=False)
            print(f"[INFO] loaded ckpt: {CKPT_PATH}")
        else:
            model.load_state_dict(ckpt, strict=False)
            print(f"[INFO] loaded ckpt (raw state_dict): {CKPT_PATH}")
    else:
        print("[WARN] ckpt not loaded (path is None or not exists).")
    
    # if not os.path.isfile(CKPT_PATH):
    #     raise FileNotFoundError(f"Checkpoint not found: {CKPT_PATH}")
    # ckpt = torch.load(CKPT_PATH, map_location=device)
    # model.load_state_dict(ckpt)
    # print(f"Loaded checkpoint from {CKPT_PATH}")

    model.eval()

    # ========================
    # 3. 对指定 index 做预测 & 可视化
    # ========================
    total_loss = 0.0
    n_samples = 0

    with torch.no_grad():
        for idx in INDICES:
            if idx < 0 or idx >= len(val_dataset):
                print(f"[WARN] index {idx} out of range, skip.")
                continue

            sample = val_dataset[idx]

            depth       = sample["depth"].unsqueeze(0).to(device)        # (1, C, H, W)
            spectrogram = sample["spectrogram"].unsqueeze(0).to(device)  # (1, 2, 65, 26)
            gt_doa      = sample["doa_map"].unsqueeze(0).to(device)      # 期望形状 (1, K) 或 (1,1,K)
            gt_dist     = sample["distant_map"].unsqueeze(0).to(device)
            heading     = sample["heading"]
            path        = sample["path"]
            print(f"Processing sample index: {idx}, heading: {heading}", path)


            # 预测
            pred_logits,pred_dist = model(spectrogram, depth)                      # 期望形状 (1, K) 或 (1,1,K)

            # 对 pred / gt 做形状压缩成 (K,)
            pred_vec = pred_logits.squeeze().detach().cpu()  # 去掉 batch/多余维度
            gt_vec   = gt_doa.squeeze().detach().cpu()

            # 确保是一维向量
            pred_vec = pred_vec.view(-1)   # (K,)
            gt_vec   = gt_vec.view(-1)     # (K,)

            pred_dist = pred_dist.squeeze().detach().cpu()
            gt_dist   = gt_dist.squeeze().detach().cpu()
            pred_dist = pred_dist.view(-1)
            gt_dist   = gt_dist.view(-1)

            # loss（可有可无，方便看数值）
            loss = criterion(pred_vec, gt_vec)
            total_loss += loss.item()
            n_samples += 1

            # 转 numpy 用于画图
            pred_np = pred_vec.numpy()
            gt_np   = gt_vec.numpy()
            pred_dist = pred_dist.numpy()
            gt_dist   = gt_dist.numpy()
            # 画 1D doa_map 曲线
            visualize_sample(pred_np, gt_np, idx, VIS_DIR,heading)
            visualize_sample_dist(pred_dist, gt_dist, idx, VIS_DIR2,heading)

    if n_samples > 0:
        avg_loss = total_loss / n_samples
        print(f"\nDone. Average loss on {n_samples} selected samples: {avg_loss:.6f}")
    else:
        print("\nNo valid indices were tested.")

    print(f"Visualizations saved in: {VIS_DIR}")


if __name__ == "__main__":
    main()

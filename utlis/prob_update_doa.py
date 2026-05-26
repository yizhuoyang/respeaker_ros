import os
import numpy as np

try:
    from scipy.ndimage import gaussian_filter as _gaussian_filter
except Exception:
    _gaussian_filter = None

def align_for_occ(P, mode="180"):
    """
    只用于显示/叠加，不改变 P 的世界语义。
    mode:
      - "cw90": 顺时针90° (上->右)
      - "ccw90": 逆时针90°
      - "180": 旋转180°
      - "none": 不变
    """
    if mode == "cw90":
        return np.rot90(P, k=-1)
    if mode == "ccw90":
        return np.rot90(P, k=+1)
    if mode == "180":
        return np.rot90(P, k=2)
    return P

def wrap_to_2pi(a):
    return np.mod(a, 2 * np.pi)


def gaussian_blur(P, sigma_cells):
    if sigma_cells is None or sigma_cells <= 0:
        return P
    if _gaussian_filter is not None:
        return _gaussian_filter(P, sigma=sigma_cells, mode="nearest")

    radius = int(np.ceil(3 * sigma_cells))
    if radius <= 0:
        return P
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    k = np.exp(-(x**2) / (2 * (sigma_cells**2) + 1e-12))
    k = (k / (k.sum() + 1e-12)).astype(np.float32)

    def conv1d_axis(A, axis):
        pad_width = [(0, 0)] * A.ndim
        pad_width[axis] = (radius, radius)
        Ap = np.pad(A, pad_width, mode="edge")
        Ap = np.moveaxis(Ap, axis, -1)
        out = np.empty((*Ap.shape[:-1], Ap.shape[-1] - 2 * radius), dtype=np.float32)
        for i in range(out.shape[-1]):
            out[..., i] = (Ap[..., i:i + 2 * radius + 1] * k).sum(axis=-1)
        out = np.moveaxis(out, -1, axis)
        return out

    P = P.astype(np.float32, copy=False)
    P = conv1d_axis(P, axis=0)
    P = conv1d_axis(P, axis=1)
    return P


def circular_interp_0_2pi(theta_bins, p_theta, theta_query):
    theta_bins = np.asarray(theta_bins, dtype=np.float32)
    p_theta = np.asarray(p_theta, dtype=np.float32)

    tq = wrap_to_2pi(np.asarray(theta_query, dtype=np.float32))

    tb_ext = np.concatenate([theta_bins, theta_bins[:1] + 2 * np.pi], axis=0)
    p_ext  = np.concatenate([p_theta,    p_theta[:1]], axis=0)

    t0 = tb_ext[0]
    tq_shift = (tq - t0) % (2 * np.pi) + t0

    flat = tq_shift.reshape(-1)
    vals = np.interp(flat, tb_ext, p_ext).astype(np.float32)
    return vals.reshape(tq.shape)


def linear_interp(x_bins, p_x, x_query):
    x_bins = np.asarray(x_bins, dtype=np.float32)
    p_x = np.asarray(p_x, dtype=np.float32)
    xq = np.asarray(x_query, dtype=np.float32)

    flat = xq.reshape(-1)
    vals = np.interp(flat, x_bins, p_x, left=0.0, right=0.0).astype(np.float32)
    return vals.reshape(xq.shape)

def _snap_idx(i, step, max_i):
    # snap to nearest multiple of step, clamp to [0, max_i]
    j = int(np.round(i / step) * step)
    return int(np.clip(j, 0, max_i))


def to_1d_prob(x, eps=1e-12, use_softmax=False):
    x = np.asarray(x)
    x = np.squeeze(x)
    if x.ndim != 1:
        raise ValueError(f"Expect 1D distribution after squeeze, got shape={x.shape}")

    if use_softmax:
        x = x.astype(np.float32)
        x = x - np.max(x)
        x = np.exp(x)
    else:
        x = np.maximum(x, 0.0)

    s = float(x.sum())
    if s < eps:
        return np.ones_like(x, dtype=np.float32) / len(x)
    return (x / s).astype(np.float32)


def entropy_confidence(p, w_min=0.2):
    p = np.asarray(p, dtype=np.float32)
    p = p / (p.sum() + 1e-12)
    H = -(p * np.log(p + 1e-12)).sum()
    c = 1.0 - float(H / (np.log(len(p) + 1e-12) + 1e-12))
    c = float(np.clip(c, 0.0, 1.0))
    return float(np.clip(c, w_min, 1.0))


def save_map_png(path, P, bounds, agent_x, agent_z, heading, est=None, title=""):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    x_min, x_max, z_min, z_max = bounds
    extent = [x_min, x_max, z_min, z_max]

    plt.figure(figsize=(6, 6))
    plt.imshow(P, extent=extent, origin="lower", aspect="equal")

    plt.scatter([agent_x], [agent_z], marker="o")

    dx = -np.sin(heading)
    dz = -np.cos(heading)
    plt.arrow(agent_x, agent_z, 0.8*dx, 0.8*dz, length_includes_head=True, head_width=0.12)

    if est is not None:
        plt.scatter([est[0]], [est[1]], marker="x", s=80)
    

    plt.title(title)
    plt.xlabel("world x")
    plt.ylabel("world z  (front=-z when heading=0)")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


class StreamingSourceMapFusion:
    """
    流式融合版本：每次喂入一帧 (pred_theta, pred_r, pose, heading, intensity)
    - 第一帧(或 reset 后第一帧)自动把 agent 放到地图中心
    - intensity <= eps 时跳过 update（只保留上一帧的 P，不做 blur，不做 update）
    - 提供 reset() 用于新任务清空 heatmap
    """

    def __init__(
        self,
        map_size_m=60.0,
        res=0.1,
        node_res=1.0,
        sigma_Q_cells=0.0,
        beta_r=0.2,
        use_entropy_weight=True,
        w_min=0.2,
        free_mask=None,
        eps=1e-12,
        r_max=30.0,
        use_softmax=False,
        intensity_zero_eps=0.0,
        out_dir=None,
        save_every=1,
    ):
        self.map_size_m = float(map_size_m)
        self.res = float(res)
        self.node_res = float(node_res)
        self.sigma_Q_cells = float(sigma_Q_cells)
        self.beta_r = float(beta_r)
        self.use_entropy_weight = bool(use_entropy_weight)
        self.w_min = float(w_min)
        self.free_mask = free_mask
        self.eps = float(eps)

        self.r_max = float(r_max)
        self.use_softmax = bool(use_softmax)
        self.intensity_zero_eps = float(intensity_zero_eps)

        self.W = int(np.round(self.map_size_m / self.res))
        self.H = int(np.round(self.map_size_m / self.res))
#         if self.W % 2 == 0:
#             self.W += 1
#         if self.H % 2 == 0:
#             self.H += 1
            
        self.map_size_m = self.W * self.res

        self.inited = False
        self.theta_bins = None
        self.r_bins = None

        self.x_min = None
        self.x_max = None
        self.z_min = None
        self.z_max = None
        self.Xg = None
        self.Zg = None

        self.P = None

        self.t = 0
        self.out_dir = out_dir
        self.save_every = int(save_every)
        if self.out_dir is not None:
            os.makedirs(self.out_dir, exist_ok=True)

    def _init_bins_if_needed(self, pred_theta, pred_r):
        if self.theta_bins is None:
            N_theta = int(np.squeeze(np.asarray(pred_theta)).shape[-1])
            self.theta_bins = np.linspace(0.0, 2*np.pi, N_theta, endpoint=False).astype(np.float32)

        if self.r_bins is None:
            N_r = int(np.squeeze(np.asarray(pred_r)).shape[-1])
            self.r_bins = np.linspace(0.0, self.r_max, N_r).astype(np.float32)

    def _init_grid(self, agent_x0, agent_z0):
        # 让中心格子坐标 == agent 初始位置
        cx = self.W // 2
        cz = self.H // 2

        self.x_min = float(agent_x0 - cx * self.res)
        self.z_min = float(agent_z0 - cz * self.res)

        # 这里的 x_max/z_max 是最后一个格点坐标（不是边界）
        self.x_max = float(self.x_min + (self.W - 1) * self.res)
        self.z_max = float(self.z_min + (self.H - 1) * self.res)

        xs = self.x_min + np.arange(self.W, dtype=np.float32) * self.res
        zs = self.z_min + np.arange(self.H, dtype=np.float32) * self.res
        self.Xg, self.Zg = np.meshgrid(xs, zs)

        P0 = np.ones((self.H, self.W), dtype=np.float32)
        if self.free_mask is not None:
            P0 = P0 * self.free_mask.astype(np.float32)
            s = float(P0.sum())
            if s < 1e-6:
                raise ValueError("free_mask has no free cells.")
            P0 /= s
        else:
            P0 /= float(P0.sum())

        self.P = P0
        self.inited = True

    def reset(self, new_center_pose=None, clear_bins=False):
        """
        新任务开始调用：
        - new_center_pose: (agent_x, agent_z) 或完整 pose(只用前后两个)；
          如果给了，就以它作为“地图中心”的参考点（下一次 update_frame 会直接用这个初始化 grid）
          如果不给，就等下一帧 update_frame 来初始化
        - clear_bins: True 则同时清空 theta_bins/r_bins（当新任务输出维度可能不同）
        """
        self.inited = False
        self.P = None

        self.x_min = self.x_max = None
        self.z_min = self.z_max = None
        self.Xg = self.Zg = None

        self.t = 0

        if clear_bins:
            self.theta_bins = None
            self.r_bins = None

        if new_center_pose is not None:
            p = np.asarray(new_center_pose).reshape(-1)
            agent_x0 = float(p[0])
            agent_z0 = float(p[-1])
            self._init_grid(agent_x0, agent_z0)

    def map_to_vu(self, x_map: float, z_map: float):

        x_axis = self.Xg[0, :]   # (W,)
        z_axis = self.Zg[:, 0]   # (H,)

        u = int(np.argmin(np.abs(x_axis - x_map)))
        v = int(np.argmin(np.abs(z_axis - z_map)))

        # snap 到 node_res 对齐
        step = int(round(self.node_res / self.res))
        v2 = _snap_idx(v, step, self.H - 1)
        u2 = _snap_idx(u, step, self.W - 1)
        return int(v2), int(u2)


    def _readout(self, agent_x, agent_z):
        v, u = np.unravel_index(np.argmax(self.P), self.P.shape)

        step = int(round(self.node_res / self.res))  # e.g. 1.0/0.1=10
        v2 = _snap_idx(v, step, self.H - 1)
        u2 = _snap_idx(u, step, self.W - 1)

        x_map = float(self.Xg[v2, u2])
        z_map = float(self.Zg[v2, u2])
        return (x_map, z_map), (int(v2), int(u2))
    
    def update_frame(self, pred_theta, pred_r, pose, heading, audio_intensity=None, save_vis=False ,id_name = None,):
        """
        输入一帧数据，返回当前融合后的 map 与 argmax。
        - audio_intensity 为 0(或 <= intensity_zero_eps) 时：跳过更新，P 不变
        """
        pose = np.asarray(pose).reshape(-1)
        agent_x = float(pose[0])
        agent_z = float(pose[-1])
        heading = float(heading)

        if pred_theta is not None and pred_r is not None:
            self._init_bins_if_needed(pred_theta, pred_r)

        if not self.inited:
            self._init_grid(agent_x, agent_z)

        # decide update
        if audio_intensity is None:
            do_update = True
            intensity_val = None
        else:
            intensity_val = float(audio_intensity)
            do_update = (intensity_val > self.intensity_zero_eps)

        if not do_update:
            est, ij = self._readout(agent_x, agent_z)
            out = {
                "P": self.P,
                "map_argmax_world": est,
                "argmax_ij": ij,
                "bounds": (self.x_min, self.x_max, self.z_min, self.z_max),
                "do_update": False,
                "intensity": intensity_val,
                "t": self.t,
            }
            if self.out_dir is not None and save_vis and (self.t % self.save_every == 0):
                png = os.path.join(self.out_dir, f"{id_name}_{self.t:03d}.png")
                title = f"t={self.t:03d}  SKIP  intensity={intensity_val:.3f}  est=({est[0]:+.2f},{est[1]:+.2f})"
                save_map_png(png, self.P, out["bounds"], agent_x, agent_z, heading, est=est, title=title)
                out["png"] = png
            self.t += 1
            return out

        # --- update ---
        p_theta = to_1d_prob(pred_theta, use_softmax=self.use_softmax)
        p_r     = to_1d_prob(pred_r,     use_softmax=self.use_softmax)

        # predict
        P_pred = gaussian_blur(self.P, self.sigma_Q_cells)
        P_pred = np.clip(P_pred, 0.0, None)
        P_pred /= (P_pred.sum() + self.eps)

        # likelihood
        dx = self.Xg - agent_x
        dz = self.Zg - agent_z

        forward_x = -np.sin(heading)
        forward_z = -np.cos(heading)
        right_x   =  np.cos(heading)
        right_z   = -np.sin(heading)

        front = dx * forward_x + dz * forward_z
        right = dx * right_x + dz * right_z

        theta_hat = wrap_to_2pi(np.arctan2(-front, right))
        r_hat = np.sqrt(dx * dx + dz * dz)

        L_theta = circular_interp_0_2pi(self.theta_bins, p_theta, theta_hat)

        if self.beta_r is None or self.beta_r <= 0:
            M = L_theta
        else:
            L_r = linear_interp(self.r_bins, p_r, r_hat)
            M = L_theta * np.power(np.maximum(L_r, 0.0), float(self.beta_r))

        if self.use_entropy_weight:
            w = entropy_confidence(p_theta, w_min=self.w_min)
            M = np.power(np.maximum(M, 0.0), w)

        # if self.free_mask is not None:
        #     M = M * self.free_mask.astype(np.float32)
        M = np.clip(M, 0.0, None)

        # rotate the sound map then multiply with free_mask
        # P_overlay = align_for_occ(out["P"], mode="cw90")
        # if self.free_mask is not None:
        #     M = M * self.free_mask.astype(np.float32)

        if M.max() <= 0:
            self.P = P_pred
        else:
            # TODO
            P_new = P_pred * (M + self.eps)
            # P_new = M + self.eps
            P_new = np.clip(P_new, 0.0, None)
            P_new /= (P_new.sum() + self.eps)
            P_new = (P_new-P_new.min())/(P_new.max()-P_new.min()+ self.eps)
            self.P = P_new

        est, ij = self._readout(agent_x, agent_z)



        out = {
            "P": self.P,
            "map_argmax_world": est,
            "argmax_ij": ij,
            "bounds": (self.x_min, self.x_max, self.z_min, self.z_max),
            "do_update": True,
            "intensity": intensity_val,
            "t": self.t,
        }
        # print(out["map_argmax_world"])
        if self.out_dir is not None and save_vis and (self.t % self.save_every == 0):
            png = os.path.join(self.out_dir, f"{id_name}_{self.t:03d}.png")
            if intensity_val is None:
                title = f"t={self.t:03d}  UPDATE  est=({est[0]:+.2f},{est[1]:+.2f})"
            else:
                title = f"t={self.t:03d}  UPDATE  intensity={intensity_val:.3f}  est=({est[0]:+.2f},{est[1]:+.2f})"
            save_map_png(png, self.P, out["bounds"], agent_x, agent_z, heading, est=est, title=title)
            out["png"] = png

        self.t += 1
        return out

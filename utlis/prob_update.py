import numpy as np

def source_in_agent_frame(source_x, source_z,
                          agent_x, agent_z,
                          heading):

    dx = source_x - agent_x
    dz = source_z - agent_z

    forward_x = -np.sin(heading)
    forward_z = -np.cos(heading)

    right_x = np.cos(heading)
    right_z =  -np.sin(heading)

    front = dx * forward_x + dz * forward_z
    right = dx * right_x   + dz * right_z

    return front, right

def source_from_agent_frame(front, right,
                            agent_x, agent_z,
                            heading):

    forward_x = -np.sin(heading)
    forward_z = -np.cos(heading)

    right_x =  np.cos(heading)
    right_z =  -np.sin(heading)

    dx = front * forward_x + right * right_x
    dz = front * forward_z + right * right_z

    source_x = agent_x + dx
    source_z = agent_z + dz

    return source_x, source_z


class SoundFieldFusionNumpy:
    """
    Global sound field fusion using numpy (log-odds + temporal decay).

    - Maintains self.log_odds: [H, W]
    - update(obs_prob, obs_mask, weight): fuse one observation that is already in the global frame
    - get_prob(): return current [H, W] probability map
    """

    def __init__(
        self,
        map_height: int,
        map_width: int,
        prior_prob: float = 0.5,
        decay: float = 0.9,
        eps: float = 1e-6,
    ):
        """
        Args:
            map_height, map_width: size of the global sound field map
            prior_prob: prior probability P(src) (0~1)
            decay: temporal decay factor (<1 means older information fades faster)
            eps: small constant for numerical stability
        """
        assert 0.0 < prior_prob < 1.0
        assert 0.0 < decay <= 1.0

        self.H = map_height
        self.W = map_width
        self.decay = decay
        self.eps = eps

        self.prior_log_odds = np.log(prior_prob / (1.0 - prior_prob))

        # Current global log-odds map
        self.log_odds = np.full(
            (self.H, self.W),
            self.prior_log_odds,
            dtype=np.float32,
        )

    def reset(self):
        """Reset log-odds to the prior (for a new scene/episode)."""
        self.log_odds.fill(self.prior_log_odds)

    def update(self, obs_prob: np.ndarray, obs_mask: np.ndarray = None, weight: float = 1.0):
        """
        Fuse one observation that is already in the global coordinate frame.

        Args:
            obs_prob: [H, W] or [1, H, W], probability map in the global frame (0~1)
            obs_mask: [H, W] bool, True where the cell has a valid observation.
                      If None, all cells are treated as observed.
            weight:   weight of the current observation (>1 means trusting it more)
        """
        if obs_prob.ndim == 3 and obs_prob.shape[0] == 1:
            obs_prob = obs_prob[0]
        assert obs_prob.shape == (self.H, self.W), \
            f"obs_prob shape must be {(self.H, self.W)}, got {obs_prob.shape}"

        obs_prob = np.clip(obs_prob, self.eps, 1.0 - self.eps)

        if obs_mask is None:
            obs_mask = np.ones((self.H, self.W), dtype=bool)
        else:
            assert obs_mask.shape == (self.H, self.W)
            obs_mask = obs_mask.astype(bool)

        # 1) Temporal decay: older information fades out
        self.log_odds *= self.decay

        # 2) Convert observation probability to log-odds increment
        obs_log_odds = np.log(obs_prob / (1.0 - obs_prob))
        delta_log_odds = obs_log_odds

        self.log_odds[obs_mask] += weight * delta_log_odds[obs_mask]

    def get_prob(self) -> np.ndarray:
        """Return current global sound field probability map P(src), shape [H, W]."""
        return 1.0 / (1.0 + np.exp(-self.log_odds))


# ================== Alignment + fusion with the 1st frame as global ==================

class GlobalSoundMapRefiner:
    """
    Use the first agent frame as the global reference coordinate frame:

    - The heatmap of the first frame is directly taken as the initial global map.
    - For each subsequent frame:
        local heatmap (agent frame) -> world coordinates -> reference agent frame
        -> align to the reference heatmap grid -> fuse using log-odds + decay.

    After each add_frame() call, it returns:
        global_prob: current fused global heatmap (in the reference frame grid)
        max_x_world, max_z_world: world coordinates of the maximum in the global heatmap
    """

    def __init__(self,
                 H_l: int,
                 W_l: int,
                 meters_per_pixel: float,
                 decay: float = 0.9,
                 prior_prob: float = 0.5,
                 eps: float = 1e-6):
        """
        Args:
            H_l, W_l: size of each per-frame heatmap
            meters_per_pixel: physical scale of each heatmap pixel in (front/right) space (meters)
            decay: temporal decay factor for log-odds fusion
            prior_prob: initial prior for log-odds
        """
        self.H = H_l
        self.W = W_l
        self.mpp = meters_per_pixel
        self.eps = eps

        self.fusion = SoundFieldFusionNumpy(
            map_height=H_l,
            map_width=W_l,
            prior_prob=prior_prob,
            decay=decay,
            eps=eps,
        )

        # World pose of the first frame (used as the global reference)
        self.ref_agent_x = None
        self.ref_agent_z = None
        self.ref_heading = None

        self.initialized = False

        # Precompute (front, right) coordinates of each heatmap pixel (in the reference agent frame)
        center_i = H_l // 2      # e.g. 64 // 2 = 32
        center_j = W_l // 2

        ii, jj = np.meshgrid(
            np.arange(H_l, dtype=np.float32),
            np.arange(W_l, dtype=np.float32),
            indexing="ij"
        )
        # Convention: image top = front (+front), image right = +right
        self.front_grid = (center_i - ii) * meters_per_pixel  # [H_l, W_l]
        self.right_grid = (jj - center_j) * meters_per_pixel  # [H_l, W_l]

        self.center_i = center_i
        self.center_j = center_j

    def reset(self):
        """
        Call this at the start of a new scene/episode:
        - Reset the global fusion (log-odds to prior)
        - Clear reference pose
        - Next add_frame() will treat its frame as the new global reference
        """
        self.fusion.reset()
        self.ref_agent_x = None
        self.ref_agent_z = None
        self.ref_heading = None
        self.initialized = False

    def _argmax_world(self, global_prob: np.ndarray):
        """
        Given the current global probability map (in the reference agent grid),
        find the cell with maximum probability, convert its (i,j) to (front0,right0)
        in the reference frame, then to world coordinates (x,z).
        """
        # 1) Index of the maximum
        idx_flat = np.argmax(global_prob)
        i_max, j_max = np.unravel_index(idx_flat, global_prob.shape)

        # 2) (front0, right0) in the reference agent frame
        front0 = self.front_grid[i_max, j_max]
        right0 = self.right_grid[i_max, j_max]

        # 3) Convert to world coordinates using the reference agent pose
        max_x_world, max_z_world = source_from_agent_frame(
            front0, right0,
            self.ref_agent_x,
            self.ref_agent_z,
            self.ref_heading,
        )
        return max_x_world, max_z_world

    def add_frame(self,
                  local_heatmap: np.ndarray,
                  agent_x: float,
                  agent_z: float,
                  heading: float,
                  weight: float = 1.0):
        """
        Fuse one frame and return the current global heatmap and the world position
        of its maximum.

        - First frame: used to initialize the global map and define the reference frame.
        - Subsequent frames: warped into the reference frame, then fused via log-odds + decay.

        Args:
            local_heatmap: [H, W], probability heatmap in the current agent's local frame (0~1)
            agent_x, agent_z, heading: current agent pose in world coordinates
            weight: weight of this observation

        Returns:
            global_prob: [H, W], current fused global sound field (in the reference agent grid)
            max_x_world, max_z_world: world coordinates of the global heatmap maximum
        """
        assert local_heatmap.shape == (self.H, self.W)

        if not self.initialized:
            # ========= First frame: define the global reference and initialize =========
            self.ref_agent_x = agent_x
            self.ref_agent_z = agent_z
            self.ref_heading = heading

            prob0 = np.clip(local_heatmap, self.eps, 1.0 - self.eps)
            self.fusion.log_odds = np.log(prob0 / (1.0 - prob0))
            self.initialized = True

            global_prob = self.fusion.get_prob()
            max_x_world, max_z_world = self._argmax_world(global_prob)
            return global_prob, max_x_world, max_z_world

        # ========= Subsequent frames: align to the reference frame and fuse =========

        # 1) Current frame: each pixel (front,right) -> world (x,z)
        source_x, source_z = source_from_agent_frame(
            self.front_grid,
            self.right_grid,
            agent_x,
            agent_z,
            heading
        )  # [H, W]

        # 2) Then world (x,z) -> (front0,right0) in the reference agent frame
        front0, right0 = source_in_agent_frame(
            source_x,
            source_z,
            self.ref_agent_x,
            self.ref_agent_z,
            self.ref_heading
        )  # [H, W]

        # 3) Map (front0,right0) back to pixel indices (i0,j0) in the reference heatmap grid
        i0 = self.center_i - front0 / self.mpp   # float
        j0 = self.center_j + right0 / self.mpp   # float

        i0 = np.round(i0).astype(np.int32)
        j0 = np.round(j0).astype(np.int32)

        # 4) Build observation maps obs_prob_global & obs_mask_global for this frame
        obs_prob_global = np.full((self.H, self.W), 0.5, dtype=np.float32)
        obs_mask_global = np.zeros((self.H, self.W), dtype=bool)

        valid = (
            (i0 >= 0) & (i0 < self.H) &
            (j0 >= 0) & (j0 < self.W)
        )

        i0_valid = i0[valid]
        j0_valid = j0[valid]
        local_val = local_heatmap[valid]

        # If multiple local pixels map to the same global cell in this frame,
        # we take the max (strongest evidence)
        for ii_idx, jj_idx, p in zip(i0_valid, j0_valid, local_val):
            if not obs_mask_global[ii_idx, jj_idx]:
                obs_prob_global[ii_idx, jj_idx] = p
                obs_mask_global[ii_idx, jj_idx] = True
            else:
                obs_prob_global[ii_idx, jj_idx] = max(
                    obs_prob_global[ii_idx, jj_idx], p
                )

        # 5) Fuse into the global map via log-odds + decay
        self.fusion.update(obs_prob_global, obs_mask=obs_mask_global, weight=weight)

        # 6) Return current global probability map and argmax in world coordinates
        global_prob = self.fusion.get_prob()
        max_x_world, max_z_world = self._argmax_world(global_prob)
        return global_prob, max_x_world, max_z_world


def quaternion_to_heading_y(qw, qx, qy, qz):
    """
    Convert quaternion (w, x, y, z) to rotation angle around the Y axis.

    Coordinate system:
        +Y = up
        +X = right
        -Z = forward
        heading = 0 rad -> +X
        heading = pi/2  -> +Z

    Returns:
        heading (float): rotation angle in radians
    """
    # yaw = rotation around Y axis
    yaw = np.arctan2(
        2 * (qw * qy + qx * qz),
        1 - 2 * (qy * qy + qz * qz)
    )
    return yaw

def localmap_argmax_world(local_heatmap, agent_x, agent_z, heading, meters_per_pixel):
    """
    Args:
      local_heatmap: [H,W] (float), local map in agent-centered grid
      agent_x, agent_z, heading: agent pose in world
      meters_per_pixel: scale of each pixel in meters
    Returns:
      (max_x_world, max_z_world), (i_max, j_max), max_value
    """
    H, W = local_heatmap.shape
    center_i = H // 2
    center_j = W // 2

    # 1) argmax in pixel coordinates
    idx = int(np.argmax(local_heatmap))
    i_max, j_max = np.unravel_index(idx, local_heatmap.shape)
    max_val = float(local_heatmap[i_max, j_max])

    # 2) pixel -> (front,right) in meters
    # image top is +front, image right is +right
    front = (center_i - i_max) * meters_per_pixel
    right = (j_max - center_j) * meters_per_pixel
    print("front,right:", front, right)
    # 3) (front,right) -> world (x,z)
    max_x_world, max_z_world = source_from_agent_frame(
        front, right, float(agent_x), float(agent_z), float(heading)
    )

    return float(max_x_world), float(max_z_world)
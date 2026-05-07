import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from dataloader.utils import compute_stft_phase_features,quaternion_to_heading_y, compute_spectrogram,source_in_agent_frame, make_source_heatmap,make_doa_gaussian,make_r_gaussian_1d
# from utils import quaternion_to_heading_y, compute_spectrogram,source_in_agent_frame, make_source_heatmap,make_doa_gaussian,make_r_gaussian_1d

class SingleStepDataset(Dataset):
    """
    A dataset that treats each NPZ file as a single sample.
    """

    def __init__(self, root_dir: str, transform=None,use_compress=True,mode='heatmap',audio_feat='spec'):
        """
        Args:
            root_dir (str): Root directory containing all scene/episode folders.
            transform (callable, optional): Optional transform to be applied
                on a sample (after converting to torch.Tensor).
        """
        self.root_dir = root_dir
        self.transform = transform

        # Collect all npz file paths once during initialization
        self.file_list = self._collect_npz_files()
        self.use_compress = use_compress
        self.mode = mode
        self.audio_feat = audio_feat    

        if len(self.file_list) == 0:
            raise RuntimeError(f"No 'step_*.npz' files found under: {root_dir}")

    def _collect_npz_files(self):
        """
        Walk through root_dir and collect all 'step_*.npz' files
        in immediate subdirectories.
        """
        file_list = []

        # Iterate over all entries in root_dir
        for entry in os.listdir(self.root_dir):
            scene_dir = os.path.join(self.root_dir, entry)
            if not os.path.isdir(scene_dir):
                continue

            # Collect npz files that match the pattern 'step_*.npz'
            npz_files = glob.glob(os.path.join(scene_dir, "step_*.npz"))
            # npz_files.sort()  # optional: sort for deterministic ordering
            npz_files = sorted(npz_files, key=lambda x: float(os.path.basename(x).split("_")[1].split(".")[0]))
            file_list.extend(npz_files)

        return file_list

    def __len__(self):
        """Return the total number of samples."""
        return len(self.file_list)

    def __getitem__(self, idx: int):
        """
        Load one NPZ file and return all fields as tensors in a dictionary.
        """
        npz_path = self.file_list[idx]
        data = np.load(npz_path, allow_pickle=True)

        pose_all = data["pose_all"]
        depth    = data["depth"]
        audio    = data["audio_wave"]
        ego_map  = data["ego_map"]
        rgb      = data['rgb']

        ego_map = torch.as_tensor(ego_map, dtype=torch.float32)
        ego_map = torch.permute(ego_map, (2,0,1)) 

        rgb     = torch.as_tensor(rgb, dtype=torch.float32)
        rgb     = torch.permute(rgb, (2,0,1))  # (H,W,C) -> (C,H,W)

        depth = torch.as_tensor(depth, dtype=torch.float32)
        if depth.ndim == 2:
            depth = depth.unsqueeze(0)
        else:
            depth = torch.permute(depth, (2,0,1))  # (H,W,C) -> (C,H,W)

        state_position = pose_all[4:7]
        source_position = data["source_loc"]

        heading = quaternion_to_heading_y(pose_all[-4],pose_all[-3],pose_all[-2],pose_all[-1])
        front,right = source_in_agent_frame(source_position[0],source_position[-1],state_position[0],state_position[-1],heading)  
        
        if self.audio_feat == 'spec':
            spectrogram = compute_spectrogram(audio,self.use_compress)
        else:
            spectrogram = compute_stft_phase_features(audio)

        spectrogram = torch.as_tensor(spectrogram, dtype=torch.float32)
        spectrogram = torch.permute(spectrogram, (2, 0,1)) 

        if self.mode=='heatmap':
            heatmap = make_source_heatmap(front, right, map_size=64, meters_per_pixel=1.0,base_sigma=0.1,sigma_scale=0.2)
            heatmap = heatmap[np.newaxis, :, :]
            sample = {
                "depth":      depth,
                "ego_map":    ego_map,
                "audio_wave": torch.as_tensor(audio, dtype=torch.float32),
                "spectrogram": spectrogram,
                "heatmap":    torch.as_tensor(heatmap, dtype=torch.float32),
                "path"   :    npz_path,
                "rgb":         rgb,
                "pose":        pose_all[4:7],
                "heading":    heading,
                "sound_source": source_position,
                "path"        :npz_path
            }

        elif self.mode=='doa_distance':
            doa_map = make_doa_gaussian(front,right,num_bins=360,base_sigma_deg=0.5,sigma_scale_deg=1.0)
            doa_map = torch.as_tensor(doa_map, dtype=torch.float32)

            distant_map = np.sqrt(front**2+right**2)
            distant_map = make_r_gaussian_1d(distant_map, num_bins=120, r_min=0.0, r_max=30.0,base_sigma=0.05, sigma_scale=0.2)
            distant_map = torch.as_tensor(distant_map, dtype=torch.float32)
            sample = {
                "depth":      depth,
                "ego_map":    ego_map,
                "audio_wave": torch.as_tensor(audio, dtype=torch.float32),
                "spectrogram": spectrogram,
                "doa_map":    doa_map,
                "distant_map":distant_map,
                "path"   :    npz_path,
                "rgb":         rgb,
                "pose":        pose_all[4:7],
                "heading":    heading,
                "sound_source": source_position,
                "path"        :npz_path
            }

        return sample
    


def wrap_to_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi



if __name__ == "__main__":
    import matplotlib.pyplot as plt

    root_dir = "/home/Disk/sound-space/ssl_data/train"
    print(f"Loading dataset from: {root_dir}")

    try:
        dataset = SingleStepDataset(root_dir,use_compress=False)
    except RuntimeError as e:
        print("Error:", e)
        exit(1)
    print(dataset.__len__(), "samples found in the dataset.")
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
    
    for batch in loader:
        print("Batch loaded:")
        print("  depth:", batch["depth"].shape)
        print("  audio_wave:", batch["audio_wave"].shape)
        print("  spectrogram:", batch["spectrogram"].shape)
        print("  heatmap:", batch["heatmap"].shape)
        print("doa_map:", batch["doa_map"].shape)
        print("distance_map",batch["distant_map"].shape)

        heatmap = batch["heatmap"][0]   # 可能是 (1,H,W) 或 (H,W)
        doa_map = batch["doa_map"][0]   # 可能是 (K,) 或 (1,K)
        distance_map = batch["distant_map"][0]

        if heatmap.ndim == 3:
            heatmap = heatmap[0]
        heatmap_np = heatmap.detach().cpu().numpy()

        if doa_map.ndim == 2:
            doa_map = doa_map[0]
            distance_map = distance_map[0]
        doa_np = doa_map.detach().cpu().numpy()
        distance_map_np = distance_map.detach().cpu().numpy()

        plt.figure(figsize=(8, 4))
        plt.subplot(1, 3, 1)
        plt.title("Heatmap")
        plt.imshow(heatmap_np, cmap="hot", origin="upper")
        plt.colorbar()
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.title("DOA Map")
        plt.plot(doa_np)
        plt.xlabel("Angle bin")
        plt.ylabel("Value")
        plt.grid(True)

        plt.subplot(1, 3, 3)
        plt.title("Distance Map")
        plt.plot(distance_map_np)
        plt.xlabel("Distance bin")
        plt.ylabel("Value")
        plt.grid(True)

        plt.tight_layout()
        plt.show()

        break  

    print("Dataset test completed.")


import os
from glob import glob

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


# =========================
# Toy synthetic dataset (moving square)
# =========================

def create_synthetic_sequence(num_frames=20, height=32, width=32):
    frames = []
    for t in range(num_frames):
        img = np.zeros((height, width), dtype=np.float32)
        top = 10
        left = 2 + 3 * t  # square moves right
        bottom = top + 6
        right = left + 6
        bottom = min(bottom, height)
        right = min(right, width)
        img[top:bottom, left:right] = 1.0
        frames.append(img)
    frames = np.stack(frames, axis=0)  # (num_frames, height, width)
    return frames


class FrameInterpolationDataset(Dataset):
    def __init__(self, frames, gap=2):
        """
        frames: numpy array of shape (num_frames, height, width)
        gap: difference between input frames (e.g., t and t+gap)
        For gap=2, we use t and t+2 as input, and t+1 as target.
        """
        self.frames = frames
        self.gap = gap
        self.num_frames = frames.shape[0]

        # t from 0 to num_frames - gap - 1
        self.num_samples = self.num_frames - self.gap - 1

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        t = idx
        t_mid = t + 1      # for gap=2, middle is t+1
        t_end = t + self.gap

        frame_t = self.frames[t]
        frame_t_mid_target = self.frames[t_mid]
        frame_t_end = self.frames[t_end]

        # Stack input frames: shape (2, H, W)
        input_frames = np.stack([frame_t, frame_t_end], axis=0)

        # Convert to tensors
        input_tensor = torch.from_numpy(input_frames).float()          # (2, H, W)
        target_tensor = torch.from_numpy(frame_t_mid_target).float()   # (H, W)

        return input_tensor, target_tensor


# =========================
# Satellite temporal dataset
# =========================

class SatTemporalDataset(Dataset):
    def __init__(self, root_dir, gap=2, resize_to=(256, 256)):
        """
        root_dir: path to 'data' folder containing sequence_* subfolders
        gap: temporal gap between input frames (e.g., t and t+gap)
        resize_to: (W, H) to resize all frames so shapes match
        """
        self.root_dir = root_dir
        self.gap = gap
        self.resize_to = resize_to
        self.samples = []  # list of (frame_paths, t_start)

        seq_dirs = sorted(glob(os.path.join(root_dir, "sequence_*")))
        for seq in seq_dirs:
            frame_paths = sorted(glob(os.path.join(seq, "t*.png")))
            num_frames = len(frame_paths)
            # for gap=2, we need t, t+1, t+2
            for t_start in range(0, num_frames - gap):
                t_mid = t_start + 1
                t_end = t_start + gap
                if t_mid < num_frames and t_end < num_frames:
                    self.samples.append((frame_paths, t_start))

        print(f"[SatTemporalDataset] Found {len(self.samples)} triplets")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frame_paths, t = self.samples[idx]
        t_mid = t + 1
        t_end = t + self.gap

        def load_frame(path):
            img = Image.open(path).convert("L")   # grayscale
            if self.resize_to is not None:
                img = img.resize(self.resize_to)  # force same size, e.g. 256x256
            arr = np.array(img).astype(np.float32) / 255.0
            return arr

        frame_t = load_frame(frame_paths[t])
        frame_mid = load_frame(frame_paths[t_mid])
        frame_end = load_frame(frame_paths[t_end])

        # Stack inputs (2, H, W)
        input_frames = np.stack([frame_t, frame_end], axis=0)
        input_tensor = torch.from_numpy(input_frames).float()
        target_tensor = torch.from_numpy(frame_mid).float()

        return input_tensor, target_tensor
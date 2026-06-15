import numpy as np
import torch
from torch.utils.data import Dataset

# ---- 1. Create synthetic image sequence (same as Day 1) ----
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

# ---- 2. Define a Dataset class for frame interpolation ----
class FrameInterpolationDataset(Dataset):
    def __init__(self, frames, gap=2):
        """
        frames: numpy array of shape (num_frames, height, width)
        gap: how many frames apart for input (e.g., t and t+gap)
        We will create triplets:
          input: [frame_t, frame_{t+gap}]
          target: frame_{t + gap/2} (if gap is even) or nearest middle
        For simplicity, we'll use gap=2 and target = t+1.
        """
        self.frames = frames
        self.gap = gap
        self.num_frames = frames.shape[0]

        # We will create triplets from t = 0 to num_frames - gap - 1
        # So we can have t and t+gap, and target t+gap/2 (if gap==2 => t+1)
        self.num_samples = self.num_frames - self.gap - 1

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        t = idx
        t_mid = t + 1  # for gap=2, target is t+1
        t_end = t + self.gap

        frame_t = self.frames[t]
        frame_t_mid_target = self.frames[t_mid]
        frame_t_end = self.frames[t_end]

        # Stack input frames: [frame_t, frame_t_end] -> shape (2, H, W)
        input_frames = np.stack([frame_t, frame_t_end], axis=0)

        # Convert to tensors
        input_tensor = torch.from_numpy(input_frames).float()      # (2, H, W)
        target_tensor = torch.from_numpy(frame_t_mid_target).float() # (H, W)

        return input_tensor, target_tensor

# ---- 3. Create dataset and test loop ----
if __name__ == "__main__":
    # Create a sequence of 20 frames
    frames = create_synthetic_sequence(num_frames=20, height=32, width=32)
    print("Frames shape:", frames.shape)  # (20, 32, 32)

    # Create dataset
    dataset = FrameInterpolationDataset(frames, gap=2)

    print("Number of samples:", len(dataset))  # 20 - 2 - 1 = 17

    # Get one sample
    input_tensor, target_tensor = dataset[0]

    print("Input tensor shape:", input_tensor.shape)  # should be (2, 32, 32)
    print("Target tensor shape:", target_tensor.shape)  # should be (32, 32)

    # Print some simple debug info
    print("Sample from input[0] (frame t):", input_tensor[0, 0, 0])
    print("Sample from target (frame t+1):", target_tensor[0, 0])

    # Show a few samples by converting back to numpy and printing min/max
    print("Input range:", input_tensor.min(), input_tensor.max())
    print("Target range:", target_tensor.min(), target_tensor.max())
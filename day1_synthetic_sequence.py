import numpy as np
import matplotlib.pyplot as plt

# 1) Create a simple synthetic "video" of a moving square
# We'll make 5 frames of size 32x32

num_frames = 5
height, width = 32, 32

frames = []

for t in range(num_frames):
    # Start with all zeros (black)
    img = np.zeros((height, width), dtype=np.float32)

    # Define a small square that moves to the right over time
    # Square size: 6x6, starting near left
    top = 10
    left = 2 + 3 * t  # moves right as t increases

    bottom = top + 6
    right = left + 6

    # Clip to image bounds just in case
    bottom = min(bottom, height)
    right = min(right, width)

    img[top:bottom, left:right] = 1.0  # white square

    frames.append(img)

frames = np.stack(frames, axis=0)  # shape: (5, 32, 32)

print("Frames shape:", frames.shape)

# 2) Plot three frames: t=0, t=2, t=4
fig, axes = plt.subplots(1, 3, figsize=(9, 3))

indices = [0, 2, 4]
titles = ["Frame t=0", "Frame t=2", "Frame t=4"]

for ax, idx, title in zip(axes, indices, titles):
    ax.imshow(frames[idx], cmap="gray", vmin=0, vmax=1)
    ax.set_title(title)
    ax.axis("off")

plt.tight_layout()
plt.show()
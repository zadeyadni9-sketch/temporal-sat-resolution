import os
import importlib.util

import torch
import numpy as np
from matplotlib import pyplot as plt

# Locate project root and src
project_root = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(project_root, "src")

# Dynamically load dataset.py
dataset_path = os.path.join(src_dir, "dataset.py")
spec_ds = importlib.util.spec_from_file_location("dataset", dataset_path)
dataset_module = importlib.util.module_from_spec(spec_ds)
spec_ds.loader.exec_module(dataset_module)

# Dynamically load models.py
models_path = os.path.join(src_dir, "models.py")
spec_m = importlib.util.spec_from_file_location("models", models_path)
models_module = importlib.util.module_from_spec(spec_m)
spec_m.loader.exec_module(models_module)

create_synthetic_sequence = dataset_module.create_synthetic_sequence
SimpleInterpCNN = models_module.SimpleInterpCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Generate synthetic sequence
frames = create_synthetic_sequence(num_frames=20, height=32, width=32)
# frames shape: (T, H, W)

# 2. Pick two frames and the middle one
t = 5
t_end = t + 2
t_mid = t + 1

frame_t = frames[t]
frame_t_end = frames[t_end]
frame_mid_gt = frames[t_mid]

# 3. Prepare input tensor
input_frames = np.stack([frame_t, frame_t_end], axis=0)  # (2, H, W)
input_tensor = torch.from_numpy(input_frames).unsqueeze(0).float().to(device)  # (1, 2, H, W)

# 4. Load model
model_path = os.path.join(project_root, "simple_interp_cnn_day3.pth")
model = SimpleInterpCNN().to(device)
state = torch.load(model_path, map_location=device)
model.load_state_dict(state)
model.eval()

# 5. Inference
with torch.no_grad():
    pred = model(input_tensor)  # (1, 1, H, W)
pred_frame = pred.squeeze().cpu().numpy()

# 6. Visualize and save
os.makedirs("outputs", exist_ok=True)

fig, axes = plt.subplots(1, 4, figsize=(10, 3))
axes[0].imshow(frame_t, cmap="gray")
axes[0].set_title("Frame t")
axes[0].axis("off")

axes[1].imshow(frame_t_end, cmap="gray")
axes[1].set_title("Frame t+2")
axes[1].axis("off")

axes[2].imshow(frame_mid_gt, cmap="gray")
axes[2].set_title("GT t+1")
axes[2].axis("off")

axes[3].imshow(pred_frame, cmap="gray")
axes[3].set_title("Pred t+1")
axes[3].axis("off")

plt.tight_layout()
out_path = os.path.join("outputs", "interp_example_day3.png")
plt.savefig(out_path)
print(f"Saved visualization to {out_path}")
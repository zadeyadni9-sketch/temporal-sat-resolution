import os
import torch
import numpy as np
from PIL import Image
import importlib.util

# Paths
project_root = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(project_root, "src", "dataset.py")
models_path = os.path.join(project_root, "src", "models.py")
model_path = os.path.join(project_root, "sat_interp_cnn.pth")
data_seq = os.path.join(project_root, "data", "sequence_001")

# Load modules
spec_dataset = importlib.util.spec_from_file_location("dataset", dataset_path)
dataset_module = importlib.util.module_from_spec(spec_dataset)
spec_dataset.loader.exec_module(dataset_module)

spec_models = importlib.util.spec_from_file_location("models", models_path)
models_module = importlib.util.module_from_spec(spec_models)
spec_models.loader.exec_module(models_module)

SatTemporalDataset = dataset_module.SatTemporalDataset
SimpleInterpCNN = models_module.SimpleInterpCNN

# Dataset just to get image size/processing
dataset = SatTemporalDataset(root_dir=os.path.join(project_root, "data"), gap=2, resize_to=(256, 256))

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleInterpCNN().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Use first triplet from dataset
inputs, target = dataset[0]  # inputs: (2,H,W), target: (H,W)
with torch.no_grad():
    inp = inputs.unsqueeze(0).to(device)  # (1,2,H,W)
    pred = model(inp)                     # (1,1,H,W)
    pred = pred.squeeze(0).squeeze(0).cpu().numpy()  # (H,W)

# Convert to images (0–255)
def to_img(arr):
    arr = np.clip(arr, 0.0, 1.0)
    arr = (arr * 255).astype(np.uint8)
    return Image.fromarray(arr)

frame_t0 = to_img(inputs[0].numpy())
frame_t2 = to_img(inputs[1].numpy())
frame_real_mid = to_img(target.numpy())
frame_pred_mid = to_img(pred)

# Make a side-by-side strip: [t0 | pred_mid | real_mid | t2]
w, h = frame_t0.size
combined = Image.new("L", (w * 4, h))
combined.paste(frame_t0, (0, 0))
combined.paste(frame_pred_mid, (w, 0))
combined.paste(frame_real_mid, (2 * w, 0))
combined.paste(frame_t2, (3 * w, 0))

out_path = os.path.join(project_root, "satellite_result.png")
combined.save(out_path)
print(f"Saved result to {out_path}")
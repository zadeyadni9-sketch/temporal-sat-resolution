import os
import importlib.util

print("=== train_day3.py starting ===")
print("Current working directory:", os.getcwd())
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = current_dir
print("Calculated project_root:", project_root)

# ---- Manually load src/dataset.py ----
dataset_path = os.path.join(project_root, "src", "dataset.py")
print("Dataset path:", dataset_path)

if not os.path.isfile(dataset_path):
    raise FileNotFoundError(f"dataset.py not found at {dataset_path}")

spec_dataset = importlib.util.spec_from_file_location("dataset", dataset_path)
dataset_module = importlib.util.module_from_spec(spec_dataset)
spec_dataset.loader.exec_module(dataset_module)

# ---- Manually load src/models.py ----
models_path = os.path.join(project_root, "src", "models.py")
print("Models path:", models_path)

if not os.path.isfile(models_path):
    raise FileNotFoundError(f"models.py not found at {models_path}")

spec_models = importlib.util.spec_from_file_location("models", models_path)
models_module = importlib.util.module_from_spec(spec_models)
spec_models.loader.exec_module(models_module)

# Import objects from loaded modules
create_synthetic_sequence = dataset_module.create_synthetic_sequence
FrameInterpolationDataset = dataset_module.FrameInterpolationDataset
SimpleInterpCNN = models_module.SimpleInterpCNN

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim


def main():
    # 1. Prepare data
    frames = create_synthetic_sequence(num_frames=50, height=32, width=32)
    dataset = FrameInterpolationDataset(frames, gap=2)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    # 2. Prepare model, loss, optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = SimpleInterpCNN().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 10

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for inputs, targets in dataloader:
            inputs = inputs.to(device)    # (B, 2, H, W)
            targets = targets.to(device)  # (B, H, W)
            targets = targets.unsqueeze(1)  # (B, 1, H, W)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * inputs.size(0)

        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}")

    torch.save(model.state_dict(), "simple_interp_cnn_day3.pth")
    print("Training complete. Model saved to simple_interp_cnn_day3.pth")


if __name__ == "__main__":
    main()
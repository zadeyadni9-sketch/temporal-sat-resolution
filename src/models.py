import torch.nn as nn

class SimpleInterpCNN(nn.Module):
    def __init__(self):
        super(SimpleInterpCNN, self).__init__()
        # Input: (batch, 2, H, W)
        # Output: (batch, 1, H, W)

        self.net = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=3, padding=1)
        )

    def forward(self, x):
        out = self.net(x)
        return out
import torch
import torch.nn as nn


class BinaryAbstractionCNN(nn.Module):
    def __init__(self, img_size=32, num_classes=2):
        super().__init__()
        # Input: 2 channels (lower/upper bounds for binary pixels)
        self.features = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),  # 2 input channels
            nn.ReLU(),
            nn.MaxPool2d(2),  # e.g., 28x28 → 14x14
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 14x14 → 7x7
        )
        # Calculate flattened size based on img_size
        self.flattened_size = 64 * (img_size // 4) * (img_size // 4)
        self.classifier = nn.Sequential(
            nn.Linear(self.flattened_size, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
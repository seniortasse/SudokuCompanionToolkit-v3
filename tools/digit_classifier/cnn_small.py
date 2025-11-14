# vision/models/cnn_small.py
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN28(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool  = nn.MaxPool2d(2, 2)      # 28->14->7
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.fc1   = nn.Linear(128*7*7, 128)
        self.fc2   = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 1x28x28 -> 32x14x14
        x = self.pool(F.relu(self.conv2(x)))  # 32x14x14 -> 64x7x7
        x = F.relu(self.conv3(x))             # 64x7x7 -> 128x7x7
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)                    # logits
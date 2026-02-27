"""Small CNN for Fashion-MNIST (ResNet-style blocks)."""
import torch
import torch.nn as nn


class Block(nn.Module):
    """Conv-BN-ReLU block with residual connection when dimensions match."""

    def __init__(self, in_c: int, out_c: int, stride: int = 1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_c),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.relu(self.conv(x) + self.shortcut(x))


class SmallResNet(nn.Module):
    """ResNet-style CNN for 28x28 grayscale (e.g. Fashion-MNIST)."""

    def __init__(self, num_classes: int = 10, dropout: float = 0.3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.layers = nn.Sequential(
            Block(32, 64, stride=2),
            Block(64, 128, stride=2),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layers(x)
        return self.fc(x)

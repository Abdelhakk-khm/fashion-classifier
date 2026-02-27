"""Unit tests for SmallResNet model."""
import torch
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.model import SmallResNet, Block


def test_block_forward_shape():
    """Block should preserve spatial size when stride=1 and in_c=out_c."""
    block = Block(32, 32, stride=1)
    x = torch.randn(2, 32, 14, 14)
    out = block(x)
    assert out.shape == x.shape


def test_small_resnet_forward_shape():
    """SmallResNet should map (B, 1, 28, 28) -> (B, num_classes)."""
    model = SmallResNet(num_classes=10, dropout=0.0)
    x = torch.randn(4, 1, 28, 28)
    out = model(x)
    assert out.shape == (4, 10)


def test_small_resnet_gradient_flow():
    """Backward pass should run without errors."""
    model = SmallResNet(num_classes=10, dropout=0.0)
    x = torch.randn(2, 1, 28, 28)
    out = model(x)
    loss = out.sum()
    loss.backward()

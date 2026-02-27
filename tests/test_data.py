"""Unit tests for data loading and preprocessing."""
import torch
import pytest
from torchvision import transforms

# Import after setting path if needed
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Config
from src.data import get_transforms, get_dataloaders


def test_transforms_output_shape():
    """Check that train and test transforms produce [1, 28, 28] tensors."""
    train_tf = get_transforms(is_train=True)
    test_tf = get_transforms(is_train=False)
    # Simulate PIL Image (Fashion-MNIST: 28x28 grayscale)
    fake_input = torch.randint(0, 255, (28, 28), dtype=torch.uint8).numpy()
    from PIL import Image
    pil_img = Image.fromarray(fake_input, mode="L")
    out_train = train_tf(pil_img)
    out_test = test_tf(pil_img)
    assert out_train.shape == (1, 28, 28), "Train transform should output (1, 28, 28)"
    assert out_test.shape == (1, 28, 28), "Test transform should output (1, 28, 28)"
    assert out_train.dtype == torch.float32
    assert out_test.dtype == torch.float32


def test_dataloaders_splits_and_batch_size():
    """Check train/val/test loaders exist and batch size is respected."""
    config = Config(batch_size=32, seed=42, num_workers=0)
    train_loader, val_loader, test_loader = get_dataloaders(config)
    batch = next(iter(train_loader))
    x, y = batch
    assert x.shape[0] <= 32 and x.shape[1:] == (1, 28, 28)
    assert y.shape[0] == x.shape[0]
    assert len(train_loader) > 0 and len(val_loader) > 0 and len(test_loader) > 0

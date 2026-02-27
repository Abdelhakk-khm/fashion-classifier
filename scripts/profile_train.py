"""
One-off profiling script: run a few steps and print PyTorch profiler summary.
Usage: python scripts/profile_train.py
"""
import torch
from torch.profiler import profile, record_function, ProfilerActivity

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Config
from src.data import get_dataloaders
from src.model import SmallResNet


def main():
    config = Config(batch_size=64, num_workers=0)
    train_loader, _, _ = get_dataloaders(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SmallResNet(num_classes=10, dropout=0.3).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA] if device.type == "cuda" else [ProfilerActivity.CPU],
        record_shapes=True,
        with_stack=False,
    ) as prof:
        for i, (x, y) in enumerate(train_loader):
            if i >= 5:
                break
            x, y = x.to(device), y.to(device)
            with record_function("forward"):
                logits = model(x)
                loss = logits.sum()
            with record_function("backward"):
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=15))


if __name__ == "__main__":
    main()

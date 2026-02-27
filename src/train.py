"""
Training script for Fashion-MNIST with reproducibility and logging.
Run: python -m src.train --lr 1e-3 --epochs 20
"""
import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from src.config import Config
from src.data import get_dataloaders
from src.model import SmallResNet


def set_seed(seed: int):
    """Set seeds for PyTorch, CUDA, and NumPy for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import random
        random.seed(seed)
    except ImportError:
        pass


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        n += x.size(0)
    return total_loss / n, correct / n


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        n += x.size(0)
    return total_loss / n, correct / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--patience", type=int, default=5)
    args = parser.parse_args()

    config = Config(
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        dropout=args.dropout,
        early_stopping_patience=args.patience,
    )
    set_seed(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader = get_dataloaders(config)
    model = SmallResNet(num_classes=config.num_classes, dropout=config.dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    run_dir = Path(config.log_dir) / f"run_{int(time.time())}"
    writer = SummaryWriter(run_dir)
    best_val_acc, best_epoch, patience_counter = 0.0, 0, 0

    for epoch in range(config.epochs):
        t0 = time.perf_counter()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        elapsed = time.perf_counter() - t0

        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val", val_loss, epoch)
        writer.add_scalar("acc/train", train_acc, epoch)
        writer.add_scalar("acc/val", val_acc, epoch)

        log_line = (
            f"Epoch {epoch+1}/{config.epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} time={elapsed:.1f}s"
        )
        print(log_line)

        if val_acc > best_val_acc:
            best_val_acc, best_epoch = val_acc, epoch + 1
            ckpt_path = Path(config.checkpoint_dir) / "best.pt"
            torch.save({"epoch": epoch, "model": model.state_dict(), "val_acc": val_acc}, ckpt_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.early_stopping_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    writer.close()
    final_log = (
        f"Final best: epoch={best_epoch} val_acc={best_val_acc:.4f} "
        f"checkpoint={config.checkpoint_dir}/best.pt"
    )
    print(final_log)
    (run_dir / "config.json").write_text(json.dumps(vars(config), indent=2))


if __name__ == "__main__":
    main()

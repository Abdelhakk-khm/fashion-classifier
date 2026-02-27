"""
Run the trained model on test images and print predictions.
Train first: python -m src.train --epochs 20
Then run:    python scripts/predict.py
"""
import sys
from pathlib import Path

import torch
from torchvision import datasets, transforms

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.model import SmallResNet
from src.config import Config

# Fashion-MNIST class names (official order)
CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


def main():
    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = Path(config.checkpoint_dir) / "best.pt"

    if not ckpt_path.exists():
        print(f"No checkpoint found at {ckpt_path}. Train first with:")
        print("  python -m src.train --lr 1e-3 --epochs 20")
        return

    # Load model
    model = SmallResNet(num_classes=config.num_classes, dropout=0.0)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    # Load test set (same transform as in data.py)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),
    ])
    test_set = datasets.FashionMNIST(root=config.data_dir, train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=False)

    # Predict on one batch
    images, labels = next(iter(test_loader))
    images, labels = images.to(device), labels
    with torch.no_grad():
        logits = model(images)
        preds = logits.argmax(dim=1)

    # Print results
    print(f"Checkpoint: {ckpt_path} (val_acc ≈ {ckpt.get('val_acc', 'N/A')})\n")
    print("Sample predictions (first 10):")
    print("-" * 50)
    for i in range(min(10, len(labels))):
        gt = CLASS_NAMES[labels[i].item()]
        pred = CLASS_NAMES[preds[i].item()]
        ok = "✓" if preds[i].item() == labels[i].item() else "✗"
        print(f"  {i+1}. True: {gt:12s}  →  Pred: {pred:12s}  {ok}")
    correct = (preds == labels).sum().item()
    total = labels.size(0)
    print("-" * 50)
    print(f"Batch accuracy: {correct}/{total} = {100*correct/total:.1f}%")


if __name__ == "__main__":
    main()

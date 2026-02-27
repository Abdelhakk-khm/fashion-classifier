"""
Donne une image au modèle et affiche la prédiction.
Usage: python scripts/predict_image.py chemin/vers/image.jpg

L'image sera redimensionnée en 28x28 et convertie en niveaux de gris (comme Fashion-MNIST).
"""
import argparse
import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.model import SmallResNet
from src.config import Config

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


def main():
    parser = argparse.ArgumentParser(description="Prédire la classe d'une image (vêtement)")
    parser.add_argument("image", type=str, help="Chemin vers l'image (ex: photo.jpg)")
    args = parser.parse_args()

    path = Path(args.image)
    if not path.exists():
        print(f"Fichier introuvable: {path}")
        return

    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = Path(config.checkpoint_dir) / "best.pt"

    if not ckpt_path.exists():
        print(f"Modèle non trouvé dans {ckpt_path}. Lance d'abord:")
        print("  python -m src.train --lr 1e-3 --epochs 20")
        return

    # Même préparation que pour Fashion-MNIST: 28x28, gris, normalisation
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),
    ])

    img = Image.open(path).convert("RGB")  # ouvrir en RGB, Grayscale() gère la conversion
    tensor = transform(img).unsqueeze(0).to(device)  # batch de 1

    model = SmallResNet(num_classes=config.num_classes, dropout=0.0)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
        pred_idx = logits.argmax(dim=1).item()

    print(f"Image: {path.name}")
    print(f"Prédiction: {CLASS_NAMES[pred_idx]}")
    print(f"Confiance: {100 * probs[pred_idx].item():.1f}%")


if __name__ == "__main__":
    main()

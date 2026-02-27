"""
Exporte des images du jeu de test Fashion-MNIST pour tester predict_image.py.
Lance: python scripts/export_sample_images.py
Puis:   python scripts/predict_image.py sample_images/0_tshirt.png
"""
import sys
from pathlib import Path

from torchvision import datasets

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import Config

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]

def main():
    config = Config()
    out_dir = Path(__file__).resolve().parent.parent / "sample_images"
    out_dir.mkdir(exist_ok=True)

    # Charger le test set sans transformation (images PIL 28x28 gris)
    test_set = datasets.FashionMNIST(root=config.data_dir, train=False, download=False)
    if len(test_set) == 0:
        print("Pas de données dans data/FashionMNIST/. Lance l'entraînement ou place les données.")
        return

    # Un exemple par classe (0 à 9)
    by_class = {}
    for idx in range(len(test_set)):
        img, label = test_set[idx]
        if label not in by_class:
            by_class[label] = img
    for label in range(10):
        img = by_class[label]
        name = f"{label}_{CLASS_NAMES[label].replace('/', '-').replace(' ', '_')}.png"
        path = out_dir / name
        img.save(path)
        print(f"  {path.name}  (vraie classe: {CLASS_NAMES[label]})")

    print(f"\n10 images dans {out_dir}")
    print("Teste avec: python scripts/predict_image.py sample_images/0_T-shirt-top.png")


if __name__ == "__main__":
    main()

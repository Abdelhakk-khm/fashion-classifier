"""Data loading and preprocessing for Fashion-MNIST."""
import gzip
import shutil
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from .config import Config

# Fallback: official GitHub if torchvision's default mirror fails
FASHION_MNIST_GITHUB = "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion"
FILES = [
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
]

# Sizes in bytes (for sanity check); train-images is ~26MB
EXPECTED_SIZES = {
    "train-images-idx3-ubyte.gz": 26_421_880,
    "train-labels-idx1-ubyte.gz": 28_881,
    "t10k-images-idx3-ubyte.gz": 4_422_936,
    "t10k-labels-idx1-ubyte.gz": 4_542,
}


def _download_fashion_mnist_from_github(root: str) -> None:
    """Download Fashion-MNIST from official GitHub and extract to torchvision layout."""
    import urllib.request

    raw_folder = Path(root) / "FashionMNIST" / "raw"
    raw_folder.mkdir(parents=True, exist_ok=True)
    for f in FILES:
        path = raw_folder / f
        out_path = path.with_suffix("")
        if out_path.exists():
            continue
        url = f"{FASHION_MNIST_GITHUB}/{f}"
        expected = EXPECTED_SIZES.get(f)
        for attempt in range(1, 5):
            if path.exists():
                path.unlink(missing_ok=True)
            print(f"Downloading {f} from GitHub... (attempt {attempt}/4)")
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=60) as resp:
                    with open(path, "wb") as out:
                        shutil.copyfileobj(resp, out, length=1024 * 1024)
                if expected and path.stat().st_size != expected:
                    raise RuntimeError(f"Size mismatch: got {path.stat().st_size}, expected {expected}")
                break
            except Exception as e:
                print(f"  Failed: {e}")
                if attempt == 4:
                    raise
                time.sleep(5)
        with gzip.open(path, "rb") as f_in:
            with open(out_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
        path.unlink()
        print(f"  Done: {out_path.name}")


def get_transforms(is_train: bool):
    """Train vs test transforms; normalisation from dataset statistics."""
    base = [
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),  # Fashion-MNIST mean, std
    ]
    if is_train:
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            *base,
        ])
    return transforms.Compose(base)


def get_dataloaders(config: Config):
    """Build train/val/test loaders with fixed seed for split reproducibility."""
    train_transform = get_transforms(is_train=True)
    test_transform = get_transforms(is_train=False)

    try:
        full_train = datasets.FashionMNIST(
            root=config.data_dir,
            train=True,
            download=True,
            transform=train_transform,
        )
        test_set = datasets.FashionMNIST(
            root=config.data_dir,
            train=False,
            download=True,
            transform=test_transform,
        )
    except RuntimeError as e:
        if "not found" in str(e).lower() or "corrupted" in str(e).lower():
            _download_fashion_mnist_from_github(config.data_dir)
            full_train = datasets.FashionMNIST(
                root=config.data_dir,
                train=True,
                download=False,
                transform=train_transform,
            )
            test_set = datasets.FashionMNIST(
                root=config.data_dir,
                train=False,
                download=False,
                transform=test_transform,
            )
        else:
            raise

    n = len(full_train)
    n_val = int(n * config.val_ratio)
    n_train = n - n_val
    train_set, val_set = random_split(
        full_train, [n_train, n_val],
        generator=torch.Generator().manual_seed(config.seed),
    )

    train_loader = DataLoader(
        train_set,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),  # only on CUDA; not supported on MPS (Apple)
    )
    val_loader = DataLoader(val_set, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=config.batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

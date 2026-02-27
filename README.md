# Fashion-MNIST classification with SmallResNet

Image classification on Fashion-MNIST using a small ResNet-style CNN. PyTorch, reproducible training, TensorBoard, early stopping.

---

## What the model does

The model **classifies an image into one of 10 clothing/accessory classes**:

| Class | Label |
|-------|--------|
| 0 | T-shirt/top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle boot |

It was trained on **Fashion-MNIST**: small 28×28 **grayscale** images, with the **item centered on a dark background** (product-photo style). The script resizes and converts any image you give it to match that format before predicting.

---

## Which images work best

**Works as expected:**
- Images that look like Fashion-MNIST: **one garment or accessory**, **centered**, **plain/dark background**.  
  Use the built-in samples: run `python scripts/export_sample_images.py`, then try `predict_image.py` on files in `sample_images/` — those are real test-set images and the model is accurate on them.

**Often wrong or unreliable:**
- **Normal photos** (e.g. t-shirt on a person, colorful background, high resolution): after resizing to 28×28 and grayscale, the shape can look very different from training data, so the model may confuse classes (e.g. predict “Bag” for a T-shirt).
- **Non-garment images** (cars, faces, etc.): the model will still output one of the 10 classes, but the result is meaningless.

**Summary:** For reliable predictions, use either the exported **sample images** from the test set, or photos that are **close to Fashion-MNIST style** (single item, centered, neutral/dark background). For random real-world photos, expect mixed or wrong results.

---

## Step 1: Clone the repo

```bash
git clone https://github.com/Abdelhakk-khm/fashion-classifier.git
cd fashion-classifier
```

---

## Step 2: Install dependencies

**Option A – pip (recommended)**

```bash
python3 -m venv .venv
source .venv/bin/activate          # Linux / macOS
# or:  .venv\Scripts\activate     # Windows

pip install -r requirements.txt
```

**Option B – conda**

```bash
conda env create -f environment.yml
conda activate fashion-classifier
```

- **Python**: 3.10+
- **CUDA**: Optional; runs on CPU or GPU (PyTorch 2.x)

---

## Step 3: Get the data

Fashion-MNIST is downloaded automatically on first run. If the default mirror fails, place the 4 extracted files in `data/FashionMNIST/raw/` (see [STRUCTURE.md](STRUCTURE.md)).

---

## Step 4: Train the model

From the project root:

```bash
python -m src.train --lr 1e-3 --epochs 20 --batch_size 64 --seed 42
```

- Best model is saved to **`checkpoints/best.pt`** when validation accuracy improves.
- Logs go to **`runs/run_<timestamp>/`**. View with: `tensorboard --logdir runs`
- Early stopping (default patience=5) stops if validation accuracy does not improve.

---

## Step 5: Predict on the test set

Run the trained model on a batch of test images and print accuracy:

```bash
python scripts/predict.py
```

Requires `checkpoints/best.pt` (from Step 4).

---

## Step 6: Predict on a single image

Classify one image (e.g. a garment photo). Image is resized to 28×28 and converted to grayscale.

```bash
python scripts/predict_image.py path/to/your/image.jpg
```

Example with a sample image (after exporting samples, see below):

```bash
python scripts/export_sample_images.py   # creates sample_images/
python scripts/predict_image.py sample_images/0_T-shirt-top.png
```

**Output example:** `Prédiction: T-shirt/top` and confidence %.

---

## Optional: Export sample images

Export one image per class from the test set to try `predict_image.py`:

```bash
python scripts/export_sample_images.py
```

Images are written to **`sample_images/`**.

---

## Optional: Run tests

```bash
python -m pytest tests/ -v
```

This runs unit tests for the data loaders and the `SmallResNet` model.

---

## Summary

| Step | Command |
|------|---------|
| 1. Clone | `git clone <repo_url> && cd fashion-classifier` |
| 2. Install | `pip install -r requirements.txt` (inside venv) |
| 3. Data | Auto-download on first train, or place files in `data/FashionMNIST/raw/` |
| 4. Train | `python -m src.train --lr 1e-3 --epochs 20` |
| 5. Predict (test set) | `python scripts/predict.py` |
| 6. Predict (one image) | `python scripts/predict_image.py path/to/image.jpg` |

---

## License

MIT. Fashion-MNIST: MIT. PyTorch/torchvision: BSD-style.

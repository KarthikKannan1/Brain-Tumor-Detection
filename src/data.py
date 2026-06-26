"""
data.py — Image loading, label encoding, and train/val/test splitting.

Dataset layout expected:
    data/
        Training/{glioma,meningioma,notumor,pituitary}/  — 1400 images each
        Testing/ {glioma,meningioma,notumor,pituitary}/  — 400  images each

Two label schemes:
    Binary     — tumor (1) vs. notumor (0)
    Multiclass — glioma=0, meningioma=1, notumor=2, pituitary=3
"""

from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
from sklearn.model_selection import train_test_split

# ── Label maps ───────────────────────────────────────────────────────────────

CLASS_NAMES: List[str] = ["glioma", "meningioma", "notumor", "pituitary"]

MULTICLASS_LABEL: Dict[str, int] = {name: i for i, name in enumerate(CLASS_NAMES)}
# glioma=0, meningioma=1, notumor=2, pituitary=3

BINARY_LABEL: Dict[str, int] = {
    name: (0 if name == "notumor" else 1) for name in CLASS_NAMES
}
# notumor=0, tumor=1


# ── Internal helpers ─────────────────────────────────────────────────────────

def _collect_paths(split_dir: Path):
    """Walk a split folder and return parallel lists of paths + labels."""
    paths, y_bin, y_multi = [], [], []
    for cls in CLASS_NAMES:
        cls_dir = split_dir / cls
        if not cls_dir.exists():
            raise FileNotFoundError(f"Expected class folder not found: {cls_dir}")
        for img_path in sorted(cls_dir.glob("*.jpg")):
            paths.append(img_path)
            y_bin.append(BINARY_LABEL[cls])
            y_multi.append(MULTICLASS_LABEL[cls])
    return paths, np.array(y_bin, dtype=np.int32), np.array(y_multi, dtype=np.int32)


# ── Public API ────────────────────────────────────────────────────────────────

def load_dataset(
    data_dir,
    val_fraction: float = 0.15,
    random_state: int = 42,
) -> Dict[str, Dict]:
    """
    Return train / val / test splits as path + label dicts.

    The existing Training/ folder is stratified-split into train + val.
    Testing/ is kept as the held-out test set.

    Args:
        data_dir:      Root folder that contains Training/ and Testing/.
        val_fraction:  Fraction of Training images reserved for validation.
        random_state:  Seed for reproducibility.

    Returns:
        {
            "train": {"paths": [...], "y_binary": ndarray, "y_multiclass": ndarray},
            "val":   {...},
            "test":  {...},
        }
    """
    data_dir = Path(data_dir)

    train_paths, train_ybin, train_ymulti = _collect_paths(data_dir / "Training")
    test_paths, test_ybin, test_ymulti = _collect_paths(data_dir / "Testing")

    idx = np.arange(len(train_paths))
    idx_train, idx_val = train_test_split(
        idx,
        test_size=val_fraction,
        stratify=train_ymulti,
        random_state=random_state,
    )

    return {
        "train": {
            "paths": [train_paths[i] for i in idx_train],
            "y_binary": train_ybin[idx_train],
            "y_multiclass": train_ymulti[idx_train],
        },
        "val": {
            "paths": [train_paths[i] for i in idx_val],
            "y_binary": train_ybin[idx_val],
            "y_multiclass": train_ymulti[idx_val],
        },
        "test": {
            "paths": test_paths,
            "y_binary": test_ybin,
            "y_multiclass": test_ymulti,
        },
    }


def load_images(paths, image_size: tuple = (256, 256)) -> np.ndarray:
    """
    Load images as grayscale uint8 arrays.

    Args:
        paths:      Iterable of Path or str image paths.
        image_size: (width, height) passed to cv2.resize.

    Returns:
        Array of shape (N, H, W), dtype uint8.
    """
    imgs = []
    for p in paths:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {p}")
        img = cv2.resize(img, image_size, interpolation=cv2.INTER_AREA)
        imgs.append(img)
    return np.stack(imgs, axis=0)


def dataset_summary(splits: Dict) -> None:
    """Print a quick class-balance table for each split."""
    for split_name, split in splits.items():
        y = split["y_multiclass"]
        y_bin = split["y_binary"]
        print(f"\n{split_name.upper()}  (n={len(y)})")
        print(f"  {'Class':<14} {'Count':>6}  {'Binary':>6}")
        print(f"  {'-'*30}")
        for cls in CLASS_NAMES:
            idx = MULTICLASS_LABEL[cls]
            count = int((y == idx).sum())
            blabel = BINARY_LABEL[cls]
            print(f"  {cls:<14} {count:>6}  {'tumor' if blabel else 'notumor':>7}")
        print(f"  {'total':<14} {len(y):>6}  tumor={int(y_bin.sum()):>4} / notumor={int((y_bin==0).sum())}")

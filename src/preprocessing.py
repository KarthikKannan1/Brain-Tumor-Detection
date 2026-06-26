"""
preprocessing.py — Wavelet decomposition + GLCM / statistical feature extraction.

Pipeline per image (defaults):
  1. 2D DWT ('db1', level 1) → approximation cA + detail subbands cH, cV, cD
  2. For each of the 4 subbands:
       GLCM (6 props, averaged over 4 angles): contrast, dissimilarity,
           homogeneity, energy, correlation, ASM
       Statistics (5):  mean, variance, skewness, kurtosis, entropy
  3. Concatenate into a flat float32 vector of length 4 × 11 = 44 dims
"""

from typing import List
import numpy as np
import pywt
from skimage.feature import graycomatrix, graycoprops
from scipy.stats import skew, kurtosis as _kurtosis
from tqdm.auto import tqdm

# ── Constants ─────────────────────────────────────────────────────────────────

GLCM_PROPS: List[str] = [
    "contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"
]
_GLCM_ANGLES = [0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]


# ── Private helpers ───────────────────────────────────────────────────────────

def _to_uint8(arr: np.ndarray) -> np.ndarray:
    """Min-max normalise a 2D array to [0, 255] uint8."""
    a = arr.astype(np.float64)
    lo, hi = a.min(), a.max()
    if hi - lo < 1e-8:
        return np.zeros_like(arr, dtype=np.uint8)
    return ((a - lo) / (hi - lo) * 255.0).astype(np.uint8)


def _glcm_features(band: np.ndarray, levels: int = 64) -> List[float]:
    """Six GLCM texture properties, averaged over 4 angles for rotation invariance."""
    # Scale [0-255] → [0, levels-1]
    band_q = np.round(band.astype(np.float32) * (levels - 1) / 255.0) \
               .clip(0, levels - 1).astype(np.uint8)
    glcm = graycomatrix(
        band_q,
        distances=[1],
        angles=_GLCM_ANGLES,
        levels=levels,
        symmetric=True,
        normed=True,
    )
    return [float(np.nan_to_num(graycoprops(glcm, p).mean(), nan=0.0))
            for p in GLCM_PROPS]


def _stat_features(band: np.ndarray) -> List[float]:
    """Five statistical descriptors: mean, variance, skewness, kurtosis, entropy."""
    flat = band.astype(np.float64).ravel()
    hist, _ = np.histogram(flat, bins=64)
    p = hist / (hist.sum() + 1e-12)
    p_nz = p[p > 0]
    entropy = float(-np.sum(p_nz * np.log2(p_nz)))
    return [
        float(flat.mean()),
        float(flat.var()),
        float(skew(flat)),
        float(_kurtosis(flat)),
        entropy,
    ]


# ── Public API ────────────────────────────────────────────────────────────────

def extract_features(
    image: np.ndarray,
    wavelet: str = "db1",
    level: int = 1,
    glcm_levels: int = 64,
) -> np.ndarray:
    """
    Extract a wavelet + GLCM feature vector from a single grayscale image.

    Args:
        image:       2D uint8 array (H, W).
        wavelet:     PyWavelets wavelet name ('db1', 'haar', 'sym2', …).
        level:       DWT decomposition level.
        glcm_levels: Gray-level quantisation for GLCM (default 64).

    Returns:
        1D float32 array of length (1 + 3 * level) * 11.
        With defaults: 4 subbands × 11 features = 44 dims.
    """
    coeffs = pywt.wavedec2(image.astype(np.float64), wavelet=wavelet, level=level)
    # coeffs[0]        → cA  (approximation at deepest level)
    # coeffs[1..level] → (cH, cV, cD) detail tuples

    features: List[float] = []

    ca = _to_uint8(coeffs[0])
    features.extend(_glcm_features(ca, glcm_levels))
    features.extend(_stat_features(ca))

    for detail_tuple in coeffs[1:]:
        for band in detail_tuple:        # cH, cV, cD
            b = _to_uint8(band)
            features.extend(_glcm_features(b, glcm_levels))
            features.extend(_stat_features(b))

    return np.array(features, dtype=np.float32)


def feature_names(wavelet: str = "db1", level: int = 1) -> List[str]:
    """Ordered feature names matching the output of extract_features."""
    subband_labels = ["cA"]
    for lvl in range(level, 0, -1):
        for d in ("cH", "cV", "cD"):
            subband_labels.append(f"{d}_L{lvl}")

    names: List[str] = []
    for label in subband_labels:
        names.extend([f"{label}_{p.lower()}" for p in GLCM_PROPS])
        names.extend([f"{label}_{s}" for s in
                      ("mean", "var", "skewness", "kurtosis", "entropy")])
    return names


def extract_features_batch(
    images: np.ndarray,
    wavelet: str = "db1",
    level: int = 1,
    glcm_levels: int = 64,
    desc: str = "Extracting features",
) -> np.ndarray:
    """
    Extract features from a batch of images.

    Args:
        images: uint8 array (N, H, W).

    Returns:
        float32 array (N, n_features).
    """
    feats = [
        extract_features(img, wavelet=wavelet, level=level, glcm_levels=glcm_levels)
        for img in tqdm(images, desc=desc, unit="img")
    ]
    return np.stack(feats, axis=0)


def save_features(path, **arrays: np.ndarray) -> None:
    """Save feature arrays to a compressed .npz file."""
    np.savez_compressed(path, **arrays)


def load_features(path) -> dict:
    """Load feature arrays from a .npz file. Returns a plain dict."""
    data = np.load(path)
    return {k: data[k] for k in data.files}

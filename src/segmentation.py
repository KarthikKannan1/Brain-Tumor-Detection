"""
segmentation.py — Fuzzy C-Means (FCM) pixel-level segmentation.

Operates on single grayscale images.  Pixels are clustered into n_clusters
regions; clusters are re-ordered so that 0 = darkest (background) and
n_clusters-1 = brightest (likely tumour core).
"""

from typing import Optional, Tuple

import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt


def fuzzy_cmeans_segment(
    image: np.ndarray,
    n_clusters: int = 2,
    m: float = 2.0,
    error: float = 0.005,
    maxiter: int = 1000,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Segment a grayscale image with Fuzzy C-Means clustering.

    Args:
        image:      2D uint8 array (H, W).
        n_clusters: Number of fuzzy clusters (2 = background + foreground).
        m:          Fuzziness exponent; must be > 1 (typically 2).
        error:      Convergence threshold for the objective function.
        maxiter:    Maximum number of FCM iterations.

    Returns:
        mask:       Hard-assignment label map (H, W), int32.
                    0 = darkest cluster, n_clusters-1 = brightest.
        membership: Soft membership matrix (n_clusters, H, W), float64,
                    with clusters sorted by ascending centre intensity.
    """
    H, W = image.shape
    # skfuzzy expects shape (n_features, n_samples); we have 1 feature (intensity)
    data = image.astype(np.float64).ravel() / 255.0
    data = data.reshape(1, -1)

    centers, membership, _, _, _, _, _ = fuzz.cmeans(
        data, c=n_clusters, m=m, error=error, maxiter=maxiter, init=None
    )

    # Hard assignment by maximum membership
    hard = np.argmax(membership, axis=0)

    # Re-order clusters: 0 = darkest centre → n_clusters-1 = brightest
    order  = np.argsort(centers.ravel())
    remap  = np.empty(n_clusters, dtype=np.int32)
    for new_lbl, old_lbl in enumerate(order):
        remap[old_lbl] = new_lbl

    mask       = remap[hard].reshape(H, W).astype(np.int32)
    membership = membership[order].reshape(n_clusters, H, W)

    return mask, membership


def visualize_segmentation(
    image: np.ndarray,
    mask: np.ndarray,
    title: str = "",
    ax: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Side-by-side plot of original image and FCM segmentation mask.

    Args:
        image: 2D uint8 grayscale image.
        mask:  2D int label map from fuzzy_cmeans_segment.
        title: Optional figure title.
        ax:    Array of 2 Axes; if None, a new figure is created.

    Returns:
        Array of the two Axes used.
    """
    if ax is None:
        _, ax = plt.subplots(1, 2, figsize=(8, 4))

    ax[0].imshow(image, cmap="gray")
    ax[0].set_title("Original")
    ax[0].axis("off")

    ax[1].imshow(mask, cmap="hot")
    ax[1].set_title("FCM Segmentation")
    ax[1].axis("off")

    if title:
        ax[0].figure.suptitle(title, fontsize=11)

    return ax

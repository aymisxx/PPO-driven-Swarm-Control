"""RGB satellite image -> scalar VARI-based utility field (Section 1).

VARI = (G - R) / (G + R - B + eps), then clipped to [-1, 1] and min-max
normalized to [0, 1]. Optional Gaussian smoothing.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from .config import CFG, DATA_DIR


def load_rgb_image(image_path: Path) -> np.ndarray:
    """Load an image from disk and return an HxWx3 RGB uint8 array."""
    if not Path(image_path).exists():
        raise FileNotFoundError(
            f"Satellite image not found at: {image_path}\n"
            "Place your RGB satellite image as data/field_satellite.jpg"
        )
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise RuntimeError(f"cv2.imread failed to load image: {image_path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def compute_vari_field(
    rgb: np.ndarray,
    eps: float = CFG.vari_epsilon,
    smooth: bool = CFG.use_gaussian_smoothing,
    kernel: int = CFG.gaussian_blur_kernel,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the raw and (optionally) smoothed VARI-based utility field.

    Returns
    -------
    phi_raw : (H, W) float32 in [0, 1]
    phi_final : (H, W) float32 in [0, 1], Gaussian-smoothed if requested
    """
    rgb_f = rgb.astype(np.float32) / 255.0
    R = rgb_f[:, :, 0]
    G = rgb_f[:, :, 1]
    B = rgb_f[:, :, 2]

    vari_raw = (G - R) / (G + R - B + eps)
    vari_clipped = np.clip(vari_raw, -1.0, 1.0)

    vmin = float(vari_clipped.min())
    vmax = float(vari_clipped.max())
    phi_raw = (vari_clipped - vmin) / (vmax - vmin + eps)
    phi_raw = np.clip(phi_raw, 0.0, 1.0).astype(np.float32)

    if smooth:
        k = int(kernel)
        if k < 3:
            k = 3
        if k % 2 == 0:
            k += 1
        phi_final = cv2.GaussianBlur(phi_raw, (k, k), sigmaX=0)
    else:
        phi_final = phi_raw.copy()

    phi_final = np.clip(phi_final, 0.0, 1.0).astype(np.float32)
    return phi_raw, phi_final


def save_utility_field(phi: np.ndarray, out_path: Optional[Path] = None) -> Path:
    out_path = out_path or (DATA_DIR / "ndvi_field.npy")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, phi)
    return out_path


def load_utility_field(path: Optional[Path] = None) -> np.ndarray:
    path = path or (DATA_DIR / "ndvi_field.npy")
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Utility field not cached at {path}. "
            "Call build_utility_field() or run the prep step first."
        )
    return np.load(path).astype(np.float32)


def build_utility_field(
    image_path: Optional[Path] = None,
    cache: bool = True,
) -> np.ndarray:
    """High-level helper: load image -> compute field -> (optionally) cache."""
    image_path = image_path or (DATA_DIR / "field_satellite.jpg")
    rgb = load_rgb_image(image_path)
    _, phi = compute_vari_field(rgb)
    if cache:
        save_utility_field(phi)
    return phi

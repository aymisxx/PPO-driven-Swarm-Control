# VARI UTILITY FIELD

from __future__ import annotations

import numpy as np
import cv2

from src.config import CFG


# MAIN FUNCTION

def compute_vari_field(rgb: np.ndarray) -> np.ndarray:
    """
    Compute normalized VARI-based utility field from RGB image.

    Input:
        rgb: (H, W, 3) uint8 image

    Output:
        phi: (H, W) float32 in [0, 1]
    """

    assert rgb.ndim == 3 and rgb.shape[2] == 3, "RGB image required"

    # Normalize to [0,1]
    rgb_float = rgb.astype(np.float32) / 255.0

    R = rgb_float[:, :, 0]
    G = rgb_float[:, :, 1]
    B = rgb_float[:, :, 2]

    eps = CFG.vari_epsilon

    # VARI computation
    
    vari_raw = (G - R) / (G + R - B + eps)

    vari_clipped = np.clip(vari_raw, -1.0, 1.0)

    vmin = float(vari_clipped.min())
    vmax = float(vari_clipped.max())

    # Normalize to [0,1]

    phi_raw = (vari_clipped - vmin) / (vmax - vmin + eps)
    phi_raw = np.clip(phi_raw, 0.0, 1.0).astype(np.float32)

    # Optional smoothing

    if CFG.use_gaussian_smoothing:

        k = int(CFG.gaussian_blur_kernel)

        if k < 3:
            k = 3
        if k % 2 == 0:
            k += 1

        phi_smooth = cv2.GaussianBlur(phi_raw, (k, k), sigmaX=0)

    else:
        phi_smooth = phi_raw

    phi = np.clip(phi_smooth, 0.0, 1.0).astype(np.float32)

    return phi

# IMAGE LOADER (separate utility)

def load_rgb_image(path: str) -> np.ndarray:
    """
    Load image using OpenCV and convert BGR → RGB
    """
    bgr = cv2.imread(path)

    if bgr is None:
        raise RuntimeError(f"Failed to load image: {path}")

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    return rgb
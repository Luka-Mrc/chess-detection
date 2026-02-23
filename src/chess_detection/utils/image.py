from __future__ import annotations
from pathlib import Path
import cv2
import numpy as np


def load_image(path: str | Path) -> np.ndarray:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"OpenCV could not decode image: {path}")
    return image


def resize(
    image: np.ndarray,
    width: int,
    height: int,
    keep_aspect: bool = False,
) -> np.ndarray:
  
    if not keep_aspect:
        return cv2.resize(image, (width, height))

    h, w = image.shape[:2]
    scale = min(width / w, height / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(image, (new_w, new_h))

    canvas = np.zeros((height, width, image.shape[2] if image.ndim == 3 else 1), dtype=image.dtype)
    if image.ndim == 2:
        canvas = np.zeros((height, width), dtype=image.dtype)
    pad_top = (height - new_h) // 2
    pad_left = (width - new_w) // 2
    canvas[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized
    return canvas


def bgr_to_gray(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def warp_perspective(
    image: np.ndarray,
    homography: np.ndarray,
    output_size: tuple[int, int] = (512, 512),
) -> np.ndarray:
     return cv2.warpPerspective(image, homography, output_size)

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
import yaml


# 90° CW rotation matrices in the 512×512 canonical space.
# rotation_steps=1 → 90° CW, 2 → 180°, 3 → 270° CW
CANONICAL_ROTATIONS = [
    np.eye(3, dtype=np.float64),                                                    # 0°
    np.array([[0, 1, 0], [-1, 0, 512], [0, 0, 1]], dtype=np.float64),              # 90° CW
    np.array([[-1, 0, 512], [0, -1, 512], [0, 0, 1]], dtype=np.float64),           # 180°
    np.array([[0, -1, 512], [1, 0, 0], [0, 0, 1]], dtype=np.float64),              # 270° CW
]


@dataclass
class BoardResult:
    homography: Optional[np.ndarray] = None   # (3,3) image → 512×512 canonical
    quad_image: Optional[np.ndarray] = None   # (4,2) outer corners in image space [TL,TR,BR,BL]
    success: bool = False
    metadata: dict = field(default_factory=dict)


_CANONICAL_DST = np.array([[0, 0], [512, 0], [512, 512], [0, 512]], dtype=np.float32)


def _quad_to_canonical_homography(quad: np.ndarray) -> np.ndarray:
    return cv2.getPerspectiveTransform(quad.astype(np.float32), _CANONICAL_DST)


def order_points(pts):
    """Order 4 points as [TL, TR, BR, BL].

    Uses sort-by-y then sort-by-x within each pair, which is robust for
    perspective-distorted quads where the sum/diff heuristic breaks down.
    """
    pts = np.array(pts, dtype=np.float32)
    by_y = pts[np.argsort(pts[:, 1])]   # top two have smaller y
    tl, tr = sorted(by_y[:2], key=lambda p: p[0])
    bl, br = sorted(by_y[2:], key=lambda p: p[0])
    return np.array([tl, tr, br, bl], dtype=np.float32)


def detect_chessboard_quad(
    img_bgr,
    canny1=50,
    canny2=150,
    blur_ksize=5,
    morph_ksize=7,
    min_area_ratio=0.08,
):
    h, w = img_bgr.shape[:2]
    img_area = h * w

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
    edges = cv2.Canny(gray, canny1, canny2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_ksize, morph_ksize))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    closed = cv2.dilate(closed, kernel, iterations=1)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    best_score = -1
    best_approx = None

    min_area = min_area_ratio * img_area

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        peri = cv2.arcLength(cnt, True)
        for eps_ratio in [0.01, 0.015, 0.02, 0.03, 0.04]:
            approx = cv2.approxPolyDP(cnt, eps_ratio * peri, True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                approx_pts = approx.reshape(4, 2)
                rect = cv2.minAreaRect(approx_pts.astype(np.float32))
                box = cv2.boxPoints(rect)
                box_area = cv2.contourArea(box)

                if box_area <= 1:
                    continue

                rectangularity = float(area) / float(box_area)
                (rw, rh) = rect[1]
                if rw <= 1 or rh <= 1:
                    continue
                aspect = max(rw, rh) / min(rw, rh)
                aspect_penalty = 1.0 / (1.0 + max(0.0, aspect - 2.2))
                score = area * rectangularity * aspect_penalty

                if score > best_score:
                    best_score = score
                    best = cnt
                    best_approx = approx_pts

    debug = {
        "edges": edges,
        "closed": closed,
        "contours": contours,
        "best_contour": best,
        "best_approx": best_approx,
    }

    if best_approx is None:
        return None, debug

    quad = order_points(best_approx)
    return quad, debug


def line_from_segment(x1, y1, x2, y2):
    a = float(y2 - y1)
    b = float(x1 - x2)
    c = -(a * x1 + b * y1)
    norm = np.hypot(a, b)
    if norm < 1e-8:
        return None

    a /= norm
    b /= norm
    c /= norm

    if abs(a) > abs(b):
        if a < 0:
            a, b, c = -a, -b, -c
    else:
        if b < 0:
            a, b, c = -a, -b, -c

    rho = -c
    return (a, b, c, rho)


def intersect_lines(L1, L2):
    a1, b1, c1 = L1
    a2, b2, c2 = L2
    d = a1 * b2 - a2 * b1
    if abs(d) < 1e-8:
        return None
    x = (b1 * c2 - b2 * c1) / d
    y = (c1 * a2 - c2 * a1) / d
    return np.array([x, y], dtype=np.float32)


def detect_chessboard_quad_hough(
    img_bgr,
    blur_ksize=5,
    adapt_block=31,
    adapt_C=7,
    morph_ksize=7,
    morph_iter=2,
    hough_rho=1,
    hough_theta=np.pi / 180,
    hough_thresh=120,
    min_line_length_ratio=0.20,
    max_line_gap=20,
    angle_tol_deg=20.0,
    min_area_ratio=0.08,
):
    h, w = img_bgr.shape[:2]
    img_area = h * w
    min_area = min_area_ratio * img_area

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)

    thr = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        adapt_block,
        adapt_C,
    )

    minLineLength = int(min(h, w) * min_line_length_ratio)
    lines = cv2.HoughLinesP(
        thr,
        rho=hough_rho,
        theta=hough_theta,
        threshold=hough_thresh,
        minLineLength=minLineLength,
        maxLineGap=max_line_gap,
    )

    debug = {"gray": gray, "thr": thr, "raw_lines": lines}

    if lines is None or len(lines) < 10:
        return None, debug

    angle_tol = np.deg2rad(angle_tol_deg)
    horizontals = []
    verticals = []

    for l in lines.reshape(-1, 4):
        x1, y1, x2, y2 = map(int, l)
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0 and dy == 0:
            continue
        theta = np.arctan2(dy, dx)
        abs_theta = abs(theta)

        if abs_theta < angle_tol or abs(abs_theta - np.pi) < angle_tol:
            L = line_from_segment(x1, y1, x2, y2)
            if L:
                horizontals.append(L)
        elif abs(abs_theta - np.pi / 2) < angle_tol:
            L = line_from_segment(x1, y1, x2, y2)
            if L:
                verticals.append(L)

    debug["horiz_count"] = len(horizontals)
    debug["vert_count"] = len(verticals)

    if len(horizontals) < 2 or len(verticals) < 2:
        return None, debug

    horizontals_sorted = sorted(horizontals, key=lambda t: t[3])
    verticals_sorted = sorted(verticals, key=lambda t: t[3])

    top = horizontals_sorted[0]
    bottom = horizontals_sorted[-1]
    left = verticals_sorted[0]
    right = verticals_sorted[-1]

    topL = intersect_lines(top[:3], left[:3])
    topR = intersect_lines(top[:3], right[:3])
    bottomR = intersect_lines(bottom[:3], right[:3])
    bottomL = intersect_lines(bottom[:3], left[:3])

    if any(p is None for p in [topL, topR, bottomR, bottomL]):
        return None, debug

    quad = order_points([topL, topR, bottomR, bottomL])

    quad_clipped = quad.copy()
    quad_clipped[:, 0] = np.clip(quad_clipped[:, 0], 0, w - 1)
    quad_clipped[:, 1] = np.clip(quad_clipped[:, 1], 0, h - 1)

    area = cv2.contourArea(quad_clipped.reshape(-1, 1, 2).astype(np.float32))
    if area < min_area:
        return None, debug

    debug["chosen_lines"] = {"top": top, "bottom": bottom, "left": left, "right": right}
    return quad_clipped, debug


class CannyBoardDetector:
    def __init__(
        self,
        thresh1: int = 50,
        thresh2: int = 150,
        blur_ksize: int = 5,
        morph_ksize: int = 7,
        min_area_ratio: float = 0.08,
        rotation_steps: int = 0,
    ):
        self.thresh1 = thresh1
        self.thresh2 = thresh2
        self.blur_ksize = blur_ksize
        self.morph_ksize = morph_ksize
        self.min_area_ratio = min_area_ratio
        self.rotation_steps = rotation_steps % 4

    def detect(self, image: np.ndarray) -> BoardResult:
        quad, _ = detect_chessboard_quad(
            image,
            canny1=self.thresh1,
            canny2=self.thresh2,
            blur_ksize=self.blur_ksize,
            morph_ksize=self.morph_ksize,
            min_area_ratio=self.min_area_ratio,
        )
        if quad is None:
            return BoardResult(success=False, metadata={"method": "canny"})
        H = _quad_to_canonical_homography(quad)
        if self.rotation_steps:
            H = CANONICAL_ROTATIONS[self.rotation_steps] @ H
        return BoardResult(homography=H, quad_image=quad, success=True, metadata={"method": "canny"})

    @classmethod
    def from_config(cls, config_path: str) -> "CannyBoardDetector":
        cfg = yaml.safe_load(open(config_path))["canny"]
        return cls(**cfg)


class HoughBoardDetector:
    def __init__(
        self,
        blur_ksize: int = 5,
        adapt_block: int = 31,
        adapt_C: int = 7,
        hough_thresh: int = 120,
        min_line_length_ratio: float = 0.20,
        max_line_gap: int = 20,
        angle_tol_deg: float = 20.0,
        min_area_ratio: float = 0.08,
        rotation_steps: int = 0,
    ):
        self.blur_ksize = blur_ksize
        self.adapt_block = adapt_block
        self.adapt_C = adapt_C
        self.hough_thresh = hough_thresh
        self.min_line_length_ratio = min_line_length_ratio
        self.max_line_gap = max_line_gap
        self.angle_tol_deg = angle_tol_deg
        self.min_area_ratio = min_area_ratio
        self.rotation_steps = rotation_steps % 4

    def detect(self, image: np.ndarray) -> BoardResult:
        quad, _ = detect_chessboard_quad_hough(
            image,
            blur_ksize=self.blur_ksize,
            adapt_block=self.adapt_block,
            adapt_C=self.adapt_C,
            hough_thresh=self.hough_thresh,
            min_line_length_ratio=self.min_line_length_ratio,
            max_line_gap=self.max_line_gap,
            angle_tol_deg=self.angle_tol_deg,
            min_area_ratio=self.min_area_ratio,
        )
        if quad is None:
            return BoardResult(success=False, metadata={"method": "hough"})
        H = _quad_to_canonical_homography(quad)
        if self.rotation_steps:
            H = CANONICAL_ROTATIONS[self.rotation_steps] @ H
        return BoardResult(homography=H, quad_image=quad, success=True, metadata={"method": "hough"})

    @classmethod
    def from_config(cls, config_path: str) -> "HoughBoardDetector":
        cfg = yaml.safe_load(open(config_path))["hough"]
        return cls(**cfg)

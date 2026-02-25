from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from chess_detection.board.classical import BoardResult, order_points, CANONICAL_ROTATIONS


# ---------------------------------------------------------------------------
# Saddle / X-corner detector
# ---------------------------------------------------------------------------

def getSaddle(gray_img, ksize=3):
    img = gray_img.astype(np.float32)
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=ksize)
    gxx = cv2.Sobel(gx, cv2.CV_32F, 1, 0, ksize=ksize)
    gyy = cv2.Sobel(gy, cv2.CV_32F, 0, 1, ksize=ksize)
    gxy = cv2.Sobel(gx, cv2.CV_32F, 0, 1, ksize=ksize)

    S = -(gxx * gyy - gxy * gxy)
    denom = gxx * gyy - gxy * gxy
    sub_s = np.divide(gy * gxy - gx * gyy, denom, out=np.zeros_like(denom), where=denom != 0)
    sub_t = np.divide(gx * gxy - gy * gxx, denom, out=np.zeros_like(denom), where=denom != 0)
    return S, sub_s, sub_t, gx, gy


def nonmax_suppression(score, win=11):
    k = win if win % 2 == 1 else win + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    dil = cv2.dilate(score, kernel)
    keep = score == dil
    out = np.zeros_like(score)
    out[keep] = score[keep]
    return out


def clipBoundingPoints(pts, img_shape, WINSIZE=10):
    h, w = img_shape[:2]
    x, y = pts[:, 0], pts[:, 1]
    ok = (x > WINSIZE) & (y > WINSIZE) & (x < (w - WINSIZE - 1)) & (y < (h - WINSIZE - 1))
    return pts[ok]


def getFinalSaddlePoints(
    gray,
    blur_ksize=5,
    sobel_ksize=3,
    nms_win=11,
    winsize_edge=10,
    score_percentile=98.8,
    max_points=800,
    min_grad_percentile=70,
    subpix_clip=0.75,
):
    if blur_ksize and blur_ksize > 1:
        gray_bl = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
    else:
        gray_bl = gray.copy()

    S, sub_s, sub_t, gx, gy = getSaddle(gray_bl, ksize=sobel_ksize)
    S[S < 0] = 0
    S_nms = nonmax_suppression(S, win=nms_win)

    grad = np.sqrt(gx * gx + gy * gy)
    gthr = np.percentile(grad[grad > 0], min_grad_percentile) if np.any(grad > 0) else 0
    S_nms[grad < gthr] = 0

    cand = S_nms[S_nms > 0]
    if cand.size == 0:
        return np.zeros((0, 2), dtype=np.float32)

    thr = np.percentile(cand, score_percentile)
    ys, xs = np.where(S_nms >= thr)

    vals = S_nms[ys, xs]
    if vals.size > max_points:
        idx = np.argsort(vals)[::-1][:max_points]
        xs, ys = xs[idx], ys[idx]

    pts = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)
    off = np.stack([sub_s[ys, xs], sub_t[ys, xs]], axis=1).astype(np.float32)
    off = np.clip(off, -subpix_clip, subpix_clip)
    pts = pts + off
    pts = clipBoundingPoints(pts, gray.shape, WINSIZE=winsize_edge)
    return pts


# ---------------------------------------------------------------------------
# Patch extraction
# ---------------------------------------------------------------------------

def extract_patches(gray, pts_xy, patch_size=32):
    r = patch_size // 2
    patches = []
    kept_pts = []
    h, w = gray.shape[:2]
    for x, y in pts_xy:
        x = int(round(float(x)))
        y = int(round(float(y)))
        if x - r < 0 or y - r < 0 or x + r >= w or y + r >= h:
            continue
        patch = gray[y - r:y + r, x - r:x + r].astype(np.float32) / 255.0
        patches.append(patch)
        kept_pts.append([x, y])
    if len(patches) == 0:
        return np.zeros((0, patch_size, patch_size), np.float32), np.zeros((0, 2), np.float32)
    return np.stack(patches, axis=0), np.array(kept_pts, dtype=np.float32)


def filter_points_with_dnn(img_bgr, pts_xy, model, device, thr=0.5, patch_size=32):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    patches, kept_pts = extract_patches(gray, pts_xy, patch_size=patch_size)
    if len(patches) == 0:
        return np.zeros((0, 2), np.float32), np.zeros((0,), np.float32)

    xb = torch.tensor(patches[:, None, :, :], dtype=torch.float32).to(device)

    model.eval()
    with torch.no_grad():
        probs = torch.softmax(model(xb), dim=1)[:, 1].cpu().numpy()

    keep = probs >= thr
    return kept_pts[keep], probs


# ---------------------------------------------------------------------------
# RANSAC + homography
# ---------------------------------------------------------------------------

def count_hits_on_grid(grid_xy):
    rounded = np.rint(grid_xy).astype(np.int32)
    in_bounds = (
        (rounded[:, 0] >= 0) & (rounded[:, 0] <= 6) &
        (rounded[:, 1] >= 0) & (rounded[:, 1] <= 6)
    )
    rounded = rounded[in_bounds]
    if len(rounded) == 0:
        return 0
    uniq = set((p[0], p[1]) for p in rounded)
    return len(uniq)


def score_H(H, pts_xy, max_reproj_err=0.55):
    pts = np.array(pts_xy, dtype=np.float32).reshape(-1, 1, 2)
    grid = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
    nearest = np.rint(grid)
    err = np.linalg.norm(grid - nearest, axis=1)
    in_bounds = (
        (nearest[:, 0] >= 0) & (nearest[:, 0] <= 6) &
        (nearest[:, 1] >= 0) & (nearest[:, 1] <= 6)
    )
    inliers = (err < max_reproj_err) & in_bounds
    score = count_hits_on_grid(grid[inliers])
    return score, inliers


def ransac_chessboard_from_hull(pts_xy, iters=5000, max_reproj_err=0.55, min_quad_area_ratio=0.22, seed=1):
    rng = np.random.default_rng(seed)
    pts_xy = np.array(pts_xy, dtype=np.float32)
    if len(pts_xy) < 20:
        raise ValueError("Too few X-points for RANSAC.")

    hull = cv2.convexHull(pts_xy.reshape(-1, 1, 2)).reshape(-1, 2)
    if len(hull) < 4:
        raise ValueError("Convex hull too small.")

    hull_area = cv2.contourArea(hull.reshape(-1, 1, 2))
    min_area = max(1.0, min_quad_area_ratio * hull_area)

    dst = np.array([[0, 0], [6, 0], [6, 6], [0, 6]], dtype=np.float32)
    best = {"score": -1, "H": None, "quad_xy": None, "inliers": None}

    for _ in range(iters):
        idx = rng.choice(len(hull), size=4, replace=False)
        quad = order_points(hull[idx])
        area = cv2.contourArea(quad.reshape(-1, 1, 2))
        if area < min_area:
            continue
        H = cv2.getPerspectiveTransform(quad, dst)
        score, inliers = score_H(H, pts_xy, max_reproj_err=max_reproj_err)
        if score > best["score"]:
            best.update({"score": score, "H": H, "quad_xy": quad, "inliers": inliers})

    if best["H"] is None:
        raise ValueError("RANSAC failed to find a valid homography.")
    return best


def refine_homography(best, pts_xy, ransac_thresh=0.9, strict_reproj=0.45):
    pts_xy = np.array(pts_xy, dtype=np.float32)
    inlier_img = pts_xy[best["inliers"]]
    if len(inlier_img) < 8:
        return best

    grid = cv2.perspectiveTransform(inlier_img.reshape(-1, 1, 2), best["H"]).reshape(-1, 2)
    grid_int = np.rint(grid).astype(np.float32)

    mask_ok = (
        (grid_int[:, 0] >= 0) & (grid_int[:, 0] <= 6) &
        (grid_int[:, 1] >= 0) & (grid_int[:, 1] <= 6)
    )
    inlier_img2 = inlier_img[mask_ok]
    grid_int2 = grid_int[mask_ok]
    if len(inlier_img2) < 8:
        return best

    H_ref, _ = cv2.findHomography(
        inlier_img2, grid_int2, method=cv2.RANSAC, ransacReprojThreshold=ransac_thresh
    )
    if H_ref is not None:
        best["H"] = H_ref

    score_ref, inliers_ref = score_H(best["H"], pts_xy, max_reproj_err=strict_reproj)
    best["score"] = score_ref
    best["inliers"] = inliers_ref
    return best


# ---------------------------------------------------------------------------
# CNN model
# ---------------------------------------------------------------------------

class SmallCNN(nn.Module):
    def __init__(self, patch_size=32):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        s = patch_size // 2 // 2 // 2
        self.fc1 = nn.Linear(64 * s * s, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# ---------------------------------------------------------------------------
# DNNBoardDetector
# ---------------------------------------------------------------------------

class DNNBoardDetector:
    def __init__(
        self,
        model_path: str,
        threshold: float = 0.5,
        device: str = "cpu",
        patch_size: int = 32,
        ransac_iters: int = 5000,
        ransac_max_reproj: float = 0.55,
        ransac_min_quad_area_ratio: float = 0.22,
        saddle_score_pctl: float = 98.8,
        saddle_nms_win: int = 11,
        saddle_max_points: int = 800,
        rotation_steps: int = 0,
    ):
        self.model_path = model_path
        self.threshold = threshold
        self.device = device
        self.patch_size = patch_size
        self.ransac_iters = ransac_iters
        self.ransac_max_reproj = ransac_max_reproj
        self.ransac_min_quad_area_ratio = ransac_min_quad_area_ratio
        self.saddle_score_pctl = saddle_score_pctl
        self.saddle_nms_win = saddle_nms_win
        self.saddle_max_points = saddle_max_points
        self.rotation_steps = rotation_steps % 4
        self._model = None
        self._device_obj = torch.device(device)

    def _load_model(self):
        ckpt = torch.load(self.model_path, map_location=self.device, weights_only=False)
        patch_size = ckpt.get("patch_size", self.patch_size)
        model = SmallCNN(patch_size=patch_size).to(self._device_obj)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        self._model = model
        self.patch_size = patch_size

    def detect(self, image: np.ndarray) -> BoardResult:
        if self._model is None:
            self._load_model()
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            pts0 = getFinalSaddlePoints(
                gray,
                score_percentile=self.saddle_score_pctl,
                nms_win=self.saddle_nms_win,
                max_points=self.saddle_max_points,
            )
            pts1, _ = filter_points_with_dnn(
                image, pts0, self._model, self._device_obj,
                thr=self.threshold, patch_size=self.patch_size,
            )
            if len(pts1) < 20:
                raise ValueError("Too few filtered points")

            best = ransac_chessboard_from_hull(
                pts1,
                iters=self.ransac_iters,
                max_reproj_err=self.ransac_max_reproj,
                min_quad_area_ratio=self.ransac_min_quad_area_ratio,
            )
            best = refine_homography(best, pts1)

            # Compute outer quad in image space (grid coords -1..7 → image)
            H_inv = np.linalg.inv(best["H"]).astype(np.float32)
            outer_grid = np.array(
                [[-1, -1], [7, -1], [7, 7], [-1, 7]], dtype=np.float32
            ).reshape(-1, 1, 2)
            outer_img = cv2.perspectiveTransform(outer_grid, H_inv).reshape(-1, 2)

            # Convert notebook H (image→grid 0..6) to canonical H (image→512×512)
            # Inner corner (i,j) → canonical ((i+1)*64, (j+1)*64)
            S = np.array([[64, 0, 64], [0, 64, 64], [0, 0, 1]], dtype=np.float64)
            H_canonical = S @ best["H"].astype(np.float64)

            # Optional rotation of the canonical frame (fixes board orientation)
            if self.rotation_steps:
                H_canonical = CANONICAL_ROTATIONS[self.rotation_steps] @ H_canonical

        except (ValueError, cv2.error) as e:
            return BoardResult(
                success=False,
                metadata={"method": "dnn_ransac", "error": str(e)},
            )

        return BoardResult(
            homography=H_canonical,
            quad_image=outer_img,
            success=True,
            metadata={"method": "dnn_ransac", "score": int(best["score"])},
        )

    @classmethod
    def from_config(cls, config_path: str) -> "DNNBoardDetector":
        cfg = yaml.safe_load(open(config_path))
        return cls(
            model_path=cfg["model"]["weights"],
            threshold=cfg["inference"]["threshold"],
            device=cfg["inference"]["device"],
            **cfg["dnn"],
        )

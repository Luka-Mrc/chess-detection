from __future__ import annotations
import cv2
import numpy as np


def draw_board(
    image: np.ndarray,
    corners: np.ndarray,
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    out = image.copy()
    pts = corners.reshape(-1, 1, 2).astype(int)
    cv2.polylines(out, [pts], isClosed=True, color=color, thickness=thickness)
    return out


def draw_pieces(
    image: np.ndarray,
    pieces,  # list[PieceResult]
    color_map: dict[str, tuple[int, int, int]] | None = None,
    show_label: bool = True,
    show_confidence: bool = False,
) -> np.ndarray:
    if color_map is None:
        color_map = {'w': (255, 255, 255), 'b': (0, 0, 0)}

    out = image.copy()
    for piece in pieces:
        x1, y1, x2, y2 = (int(v) for v in piece.bbox)
        prefix = piece.label[0] if piece.label else 'w'
        color = color_map.get(prefix, (0, 255, 0))

        cv2.rectangle(out, (x1, y1), (x2, y2), color=color, thickness=2)

        if show_label:
            text = piece.label
            if show_confidence:
                text += f" {piece.confidence:.2f}"
            cv2.putText(
                out, text, (x1, max(y1 - 4, 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA,
            )
            cv2.putText(
                out, text, (x1, max(y1 - 4, 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA,
            )

    return out


def draw_grid(
    image: np.ndarray,
    homography: np.ndarray,
    color: tuple[int, int, int] = (0, 128, 255),
    thickness: int = 1,
) -> np.ndarray:
    out = image.copy()
    H_inv = np.linalg.inv(homography)
    coords = [i * 64 for i in range(9)]

    for row_val in coords:
        pts_canonical = np.array(
            [[col_val, row_val] for col_val in coords], dtype=np.float32
        ).reshape(-1, 1, 2)
        pts_image = cv2.perspectiveTransform(pts_canonical, H_inv)
        pts_image = pts_image.reshape(-1, 2).astype(int)
        for i in range(len(pts_image) - 1):
            cv2.line(out, tuple(pts_image[i]), tuple(pts_image[i + 1]), color, thickness)

    for col_val in coords:
        pts_canonical = np.array(
            [[col_val, row_val] for row_val in coords], dtype=np.float32
        ).reshape(-1, 1, 2)
        pts_image = cv2.perspectiveTransform(pts_canonical, H_inv)
        pts_image = pts_image.reshape(-1, 2).astype(int)
        for i in range(len(pts_image) - 1):
            cv2.line(out, tuple(pts_image[i]), tuple(pts_image[i + 1]), color, thickness)

    return out

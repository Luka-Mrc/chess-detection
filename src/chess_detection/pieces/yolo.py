from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import yaml
from chess_detection.board.classical import BoardResult

@dataclass
class PieceResult:
    label: str
    confidence: float
    bbox: np.ndarray
    square: Optional[str] = None
    metadata: dict = field(default_factory=dict)


class YOLOPieceDetector:
    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = 0.25,
        *,
        iou_threshold: float = 0.45,
        device: str = "cpu",
        class_map: dict[int, str] | None = None,
    ) -> None:
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.class_map = class_map
        self._model = None

    def detect(
        self,
        image: np.ndarray,
        board: Optional[BoardResult] = None,
    ) -> list[PieceResult]:
        from ultralytics import YOLO

        if self._model is None:
            self._model = YOLO(self.model_path)

        results = self._model(
            image,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
        )

        pieces: list[PieceResult] = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                cls_idx = int(box.cls[0])
                if self.class_map is not None:
                    if cls_idx not in self.class_map:
                        continue
                    label = self.class_map[cls_idx]
                else:
                    label = result.names[cls_idx]
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                bbox = np.array([x1, y1, x2, y2], dtype=float)
                pieces.append(PieceResult(label=label, confidence=confidence, bbox=bbox))

        if board is not None and board.homography is not None:
            flip = False
            pieces = assign_squares(pieces, board, flip)

        return pieces

    @classmethod
    def from_config(cls, config_path: str) -> "YOLOPieceDetector":
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        class_map = {int(k): v for k, v in cfg["classes"].items()}
        return cls(
            model_path=cfg["model"]["weights"],
            confidence_threshold=cfg["inference"]["confidence_threshold"],
            iou_threshold=cfg["inference"]["iou_threshold"],
            device=cfg["inference"]["device"],
            class_map=class_map,
        )

def assign_squares(
    pieces: list[PieceResult],
    board: BoardResult,
    flip: bool = False,
) -> list[PieceResult]:
    if board.homography is None:
        raise ValueError("board.homography is None — run board detection first.")

    H = board.homography
    for piece in pieces:
        cx = (piece.bbox[0] + piece.bbox[2]) / 2.0
        cy = (piece.bbox[1] + piece.bbox[3]) / 2.0

        pt = H @ np.array([cx, cy, 1.0])
        gx, gy = pt[0] / pt[2], pt[1] / pt[2]

        col = min(max(int(gx / 64), 0), 7)
        row = min(max(int(gy / 64), 0), 7)

        if flip:
            col = 7 - col
            row = 7 - row

        file = chr(ord('a') + col)
        rank = str(8 - row)
        piece.square = file + rank

    return pieces

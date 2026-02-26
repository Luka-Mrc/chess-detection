"""
run_single.py  —  Quick single-image test.

Edit IMAGE_PATH below, then run:
    python scripts/run_single.py

Runs all three board detectors (DNN, Canny, Hough) and saves a separate
annotated output for each, plus prints a comparison summary.
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import cv2

# ── EDIT THIS ────────────────────────────────────────────────────────────────
IMAGE_PATH = r"D:\Faks\chess detection\data\splits\test\49c2afbbe5726160b289f7c0c62cdace_jpg.rf.l9StEZDZtcpa6hYEmbFe.jpg"
# ─────────────────────────────────────────────────────────────────────────────

PIECE_CONFIG = "config/piece_detection.yaml"
BOARD_CONFIG = "config/board_detection.yaml"
OUTPUT_DIR   = Path("data/results")

from chess_detection.board.dnn      import DNNBoardDetector
from chess_detection.board.classical import CannyBoardDetector, HoughBoardDetector
from chess_detection.pieces.yolo    import YOLOPieceDetector
from chess_detection.pipeline       import ChessPositionPipeline
from chess_detection.utils.image    import load_image
from chess_detection.utils.visualization import draw_board, draw_grid

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
stem = Path(IMAGE_PATH).stem

print(f"Image : {IMAGE_PATH}\n")
image     = load_image(IMAGE_PATH)
piece_det = YOLOPieceDetector.from_config(PIECE_CONFIG)   # shared across all runs

DETECTORS = [
    ("dnn",   DNNBoardDetector.from_config(BOARD_CONFIG)),
    ("canny", CannyBoardDetector.from_config(BOARD_CONFIG)),
    ("hough", HoughBoardDetector.from_config(BOARD_CONFIG)),
]

for name, board_det in DETECTORS:
    print(f"--- {name.upper()} ---")
    pipeline = ChessPositionPipeline(board_det, piece_det)
    result   = pipeline.run(image)
    board    = result.board

    status = "OK" if board.success else "FAILED"
    print(f"  Board  : {status}  {board.metadata}")
    print(f"  Pieces : {len(result.pieces)}")
    for p in result.pieces:
        print(f"    {p.label:4s}  conf={p.confidence:.2f}  sq={p.square}")
    print(f"  FEN    : {result.fen}")

    # ── annotate ──────────────────────────────────────────────────────────────
    vis = image.copy()

    if board.success:
        if board.quad_image is not None:
            vis = draw_board(vis, board.quad_image, color=(0, 255, 0), thickness=3)
        if board.homography is not None:
            vis = draw_grid(vis, board.homography, color=(0, 200, 255), thickness=1)

    for p in result.pieces:
        x1, y1, x2, y2 = (int(v) for v in p.bbox)
        color = (255, 255, 255) if p.label.startswith('w') else (30, 30, 30)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness=3)
        text = f"{p.label}@{p.square}" if p.square else p.label
        cv2.putText(vis, text, (x1, max(y1 - 6, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(vis, text, (x1, max(y1 - 6, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color,    1, cv2.LINE_AA)
        cv2.putText(vis, f"{p.confidence:.2f}", (x1, y2 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 2, cv2.LINE_AA)

    # label which detector was used (top-left corner)
    label_text = f"Board: {name.upper()}  {'OK' if board.success else 'FAILED'}"
    cv2.putText(vis, label_text, (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0),   4, cv2.LINE_AA)
    cv2.putText(vis, label_text, (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 0), 2, cv2.LINE_AA)

    out_path = OUTPUT_DIR / f"{stem}_{name}_out.jpg"
    cv2.imwrite(str(out_path), vis)
    print(f"  Saved  : {out_path}\n")

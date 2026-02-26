"""
evaluate_pieces.py - Evaluate YOLO piece detection on the test set.

Two evaluation modes:
  --data PATH      YOLO dataset YAML  → Precision / Recall / mAP / confusion matrix
  --fen-csv PATH   CSV with image_path,fen columns → per-square accuracy

At least one must be provided; both can be used together.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import yaml

# Make sure the package is importable when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALL_SQUARES = [f + r for r in "87654321" for f in "abcdefgh"]

FEN_TO_LABEL = {
    'P': 'wp', 'N': 'wn', 'B': 'wb', 'R': 'wr', 'Q': 'wq', 'K': 'wk',
    'p': 'bp', 'n': 'bn', 'b': 'bb', 'r': 'br', 'q': 'bq', 'k': 'bk',
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fen_to_board(fen: str) -> dict[str, str]:
    """Convert FEN piece-placement string to {square: label} dict."""
    board: dict[str, str] = {}
    ranks = fen.split('/')
    for rank_idx, rank_str in enumerate(ranks):
        rank_num = 8 - rank_idx
        file_idx = 0
        for ch in rank_str:
            if ch.isdigit():
                file_idx += int(ch)
            else:
                square = chr(ord('a') + file_idx) + str(rank_num)
                board[square] = FEN_TO_LABEL[ch]
                file_idx += 1
    return board


def square_accuracy(pred_board: dict[str, str], gt_board: dict[str, str]) -> float:
    """Fraction of 64 squares that match between prediction and ground truth."""
    correct = sum(
        pred_board.get(sq) == gt_board.get(sq)
        for sq in ALL_SQUARES
    )
    return correct / 64.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate piece detection — YOLO metrics and/or per-square accuracy."
    )
    parser.add_argument(
        "--piece-config",
        default="config/piece_detection.yaml",
        help="Piece detection config YAML (default: config/piece_detection.yaml)",
    )
    parser.add_argument(
        "--board-config",
        default="config/board_detection.yaml",
        help="Board detection config YAML (default: config/board_detection.yaml)",
    )
    parser.add_argument(
        "--data",
        default=None,
        help="YOLO dataset YAML — enables YOLO metrics (Precision/Recall/mAP/CM)",
    )
    parser.add_argument(
        "--fen-csv",
        default=None,
        help="CSV with columns image_path,fen — enables per-square accuracy",
    )
    parser.add_argument(
        "--output-dir",
        default="src/chess_detection/data/results/",
        help="Directory for result CSVs / PNG (default: src/chess_detection/data/results/)",
    )
    parser.add_argument(
        "--save-cm",
        action="store_true",
        help="Save confusion matrix as confusion_matrix.png in output-dir",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="cpu or cuda (default: taken from piece-config)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Part 1 — YOLO-level metrics
# ---------------------------------------------------------------------------

def run_yolo_metrics(args: argparse.Namespace, piece_cfg: dict) -> None:
    from ultralytics import YOLO

    weights = piece_cfg["model"]["weights"]
    conf    = piece_cfg["inference"]["confidence_threshold"]
    iou     = piece_cfg["inference"]["iou_threshold"]
    device  = args.device or piece_cfg["inference"].get("device", "cpu")

    print("Loading YOLO model …")
    model = YOLO(weights)

    print(f"Running validation on: {args.data}")
    metrics = model.val(
        data=args.data,
        conf=conf,
        iou=iou,
        device=device,
        verbose=False,
    )

    # Build class name list in index order
    class_map: dict[int, str] = {int(k): v for k, v in piece_cfg["classes"].items()}
    class_names = [class_map[i] for i in sorted(class_map)]

    p_arr   = np.asarray(metrics.box.p)
    r_arr   = np.asarray(metrics.box.r)
    ap50    = np.asarray(metrics.box.ap50)
    ap5095  = np.asarray(metrics.box.ap)

    # Clamp lengths in case validator returns different nc
    n = min(len(class_names), len(p_arr))
    class_names = class_names[:n]

    # ---- stdout table ----
    header = f"{'class':<14}{'precision':>10}{'recall':>10}{'mAP50':>10}{'mAP50-95':>12}"
    print("\nYOLO Piece Detection Metrics")
    print(header)
    print("-" * len(header))
    for i, name in enumerate(class_names):
        print(f"{name:<14}{p_arr[i]:>10.3f}{r_arr[i]:>10.3f}{ap50[i]:>10.3f}{ap5095[i]:>12.3f}")
    print("-" * len(header))
    print(
        f"{'all (mean)':<14}"
        f"{float(metrics.box.mp):>10.3f}"
        f"{float(metrics.box.mr):>10.3f}"
        f"{float(metrics.box.map50):>10.3f}"
        f"{float(metrics.box.map):>12.3f}"
    )

    # ---- save CSV ----
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "piece_evaluation_yolo.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "precision", "recall", "map50", "map50_95"])
        for i, name in enumerate(class_names):
            writer.writerow([name, f"{p_arr[i]:.4f}", f"{r_arr[i]:.4f}",
                             f"{ap50[i]:.4f}", f"{ap5095[i]:.4f}"])
        writer.writerow(["all", f"{metrics.box.mp:.4f}", f"{metrics.box.mr:.4f}",
                         f"{metrics.box.map50:.4f}", f"{metrics.box.map:.4f}"])
    print(f"Saved: {csv_path}")

    # ---- confusion matrix PNG ----
    if args.save_cm:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import seaborn as sns

            cm = np.asarray(metrics.confusion_matrix.matrix)
            labels = class_names + ["background"]

            fig, ax = plt.subplots(figsize=(max(8, len(labels)), max(6, len(labels) - 2)))
            sns.heatmap(
                cm,
                annot=True,
                fmt=".0f",
                xticklabels=labels,
                yticklabels=labels,
                cmap="Blues",
                ax=ax,
            )
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            ax.set_title("Confusion Matrix — Piece Detection")
            plt.tight_layout()
            cm_path = out_dir / "confusion_matrix.png"
            plt.savefig(cm_path, dpi=150)
            plt.close(fig)
            print(f"Saved: {cm_path}")
        except ImportError as exc:
            print(f"Warning: could not save confusion matrix ({exc}). Install matplotlib and seaborn.")


# ---------------------------------------------------------------------------
# Part 2 — Per-square accuracy
# ---------------------------------------------------------------------------

def run_per_square(args: argparse.Namespace, piece_cfg: dict) -> None:
    from chess_detection.board.dnn import DNNBoardDetector
    from chess_detection.pieces.yolo import YOLOPieceDetector
    from chess_detection.pipeline import ChessPositionPipeline
    from chess_detection.utils.image import load_image

    board_det = DNNBoardDetector.from_config(args.board_config)
    piece_det = YOLOPieceDetector.from_config(args.piece_config)
    if args.device:
        piece_det.device = args.device
    pipeline = ChessPositionPipeline(board_det, piece_det)

    # Read CSV
    rows: list[dict] = []
    with open(args.fen_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    results: list[dict] = []
    board_failures = 0

    for row in rows:
        img_path = row["image_path"]
        fen = row["fen"].strip()

        try:
            image = load_image(img_path)
        except (FileNotFoundError, ValueError) as e:
            print(f"  [skip] {img_path}: {e}", file=sys.stderr)
            results.append({"image": img_path, "board_ok": False, "square_accuracy": None})
            board_failures += 1
            continue

        result = pipeline.run(image)

        if not result.board.success:
            results.append({"image": img_path, "board_ok": False, "square_accuracy": None})
            board_failures += 1
            continue

        gt_board = fen_to_board(fen)
        pred_board = {p.square: p.label for p in result.pieces if p.square}
        acc = square_accuracy(pred_board, gt_board)
        results.append({"image": img_path, "board_ok": True, "square_accuracy": acc})

    # Compute summary stats
    valid_accs = [r["square_accuracy"] for r in results if r["square_accuracy"] is not None]
    n_total = len(results)

    print(f"\nPer-square accuracy — {n_total} images evaluated")
    if valid_accs:
        arr = np.array(valid_accs)
        print(f"  Mean:    {arr.mean():.3f}")
        print(f"  Median:  {np.median(arr):.3f}")
        print(f"  Min:     {arr.min():.3f}")
        print(f"  Max:     {arr.max():.3f}")
    else:
        print("  No successful evaluations.")
    print(f"  Board detection failures: {board_failures}")

    # Save CSV
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "piece_evaluation_per_square.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "board_ok", "square_accuracy"])
        for r in results:
            acc_val = f"{r['square_accuracy']:.4f}" if r["square_accuracy"] is not None else ""
            writer.writerow([r["image"], r["board_ok"], acc_val])
    print(f"Saved: {csv_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    if not args.data and not args.fen_csv:
        sys.exit("Error: provide --data and/or --fen-csv")

    with open(args.piece_config) as f:
        piece_cfg = yaml.safe_load(f)

    if args.data:
        run_yolo_metrics(args, piece_cfg)

    if args.fen_csv:
        run_per_square(args, piece_cfg)


if __name__ == "__main__":
    main()

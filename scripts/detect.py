from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect chess pieces in an image.")
    parser.add_argument("--image", required=True, help="Path to input image.")
    parser.add_argument(
        "--piece-config",
        default="config/piece_detection.yaml",
        help="Piece detection config file.",
    )
    parser.add_argument(
        "--board-config",
        default="config/board_detection.yaml",
        help="Board detection config file.",
    )
    parser.add_argument("--output", default=None, help="Path to save annotated image.")
    parser.add_argument(
        "--weights",
        default=None,
        help="Override weights path from piece-config.",
    )
    parser.add_argument(
        "--no-board",
        action="store_true",
        help="Skip board detection; detect pieces only (no FEN, no square assignment).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from chess_detection.utils.image import load_image
    from chess_detection.pieces.yolo import YOLOPieceDetector
    from chess_detection.utils.visualization import draw_pieces, draw_board, draw_grid

    image = load_image(args.image)

    piece_detector = YOLOPieceDetector.from_config(args.piece_config)
    if args.weights:
        piece_detector.model_path = args.weights

    if args.no_board:
        pieces = piece_detector.detect(image)
        print(f"Detected {len(pieces)} piece(s):")
        for p in pieces:
            print(f"  {p.label}  conf={p.confidence:.2f}")

        if args.output:
            import cv2

            vis = draw_pieces(image.copy(), pieces, show_confidence=True)
            cv2.imwrite(args.output, vis)
            print(f"Saved → {args.output}")
    else:
        from chess_detection.board.dnn import DNNBoardDetector
        from chess_detection.pipeline import ChessPositionPipeline

        board_detector = DNNBoardDetector.from_config(args.board_config)
        pipeline = ChessPositionPipeline(board_detector, piece_detector)
        result = pipeline.run(image)

        board = result.board
        print(
            f"Board detection: {'OK' if board.success else 'FAILED'}"
            f" (method={board.metadata.get('method', '?')})"
        )

        print(f"Detected {len(result.pieces)} piece(s):")
        for p in result.pieces:
            sq = f"  sq={p.square}" if p.square else ""
            print(f"  {p.label}  conf={p.confidence:.2f}{sq}")

        if result.fen:
            print(f"\nFEN: {result.fen}")

        if args.output:
            import cv2

            vis = image.copy()
            if board.success and board.quad_image is not None:
                vis = draw_board(vis, board.quad_image)
            if board.success and board.homography is not None:
                vis = draw_grid(vis, board.homography)
            vis = draw_pieces(vis, result.pieces, show_confidence=True)
            cv2.imwrite(args.output, vis)
            print(f"Saved → {args.output}")


if __name__ == "__main__":
    main()

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from chess_detection.board.classical import BoardResult
from chess_detection.pieces.yolo import PieceResult, YOLOPieceDetector

@dataclass
class PositionResult:
    board: BoardResult
    pieces: list[PieceResult]
    fen: str | None = None
    metadata: dict = field(default_factory=dict)


class ChessPositionPipeline:
    def __init__(
        self,
        board_detector,
        piece_detector: YOLOPieceDetector,
        generate_fen: bool = True,
    ) -> None:
        self.board_detector = board_detector
        self.piece_detector = piece_detector
        self.generate_fen = generate_fen

    def run(self, image: np.ndarray) -> PositionResult:
        board_result = self.board_detector.detect(image)
        pieces = self.piece_detector.detect(image, board_result)
        fen = self._to_fen(pieces) if self.generate_fen else None
        return PositionResult(board=board_result, pieces=pieces, fen=fen)

    @staticmethod
    def _to_fen(pieces: list[PieceResult]) -> str:
        piece_type_map = {'p': 'p', 'r': 'r', 'n': 'n', 'b': 'b', 'q': 'q', 'k': 'k'}
        board: dict[str, str] = {}
        for piece in pieces:
            if piece.square is None:
                continue
            label = piece.label  # e.g. 'wp', 'bk'
            if len(label) < 2:
                continue
            color = label[0]   # 'w' or 'b'
            ptype = label[1]   # 'p', 'r', 'n', 'b', 'q', 'k'
            fen_char = piece_type_map.get(ptype, ptype)
            if color == 'w':
                fen_char = fen_char.upper()
            board[piece.square] = fen_char

        rank_strings: list[str] = []
        for rank_num in range(8, 0, -1):
            rank_str = ''
            empty_count = 0
            for file_char in 'abcdefgh':
                square = file_char + str(rank_num)
                if square in board:
                    if empty_count:
                        rank_str += str(empty_count)
                        empty_count = 0
                    rank_str += board[square]
                else:
                    empty_count += 1
            if empty_count:
                rank_str += str(empty_count)
            rank_strings.append(rank_str)

        return '/'.join(rank_strings)

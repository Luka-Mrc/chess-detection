from abc import ABC, abstractmethod

import numpy as np

from chess_detection.board.classical import BoardResult


class BoardDetector(ABC):
    @abstractmethod
    def detect(self, image: np.ndarray) -> BoardResult: ...

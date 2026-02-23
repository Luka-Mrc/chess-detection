from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class BoardResult:
    corners: np.ndarray
    homography: Optional[np.ndarray] = None
    confidence: float = 1.0
    method: str = "unknown"
    metadata: dict = field(default_factory=dict)

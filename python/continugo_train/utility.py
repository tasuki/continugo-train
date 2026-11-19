"""Static score utility, in normalized-board-area units (no dynamic center).

The arctan scale is 2 / board_size: two sqrt(board-area) points in a
board_size**2 point-equivalent area convention. This is independent of raster
resolution. Gaussian uncertainty is integrated, not applied after the curve.
"""

import math

# Simpson integration of a standard Gaussian on [-8,8]. Pairing nodes preserves
# exact odd symmetry. The omitted tail probability is below 1.3e-15.
_NORMAL_WEIGHTS = tuple(
    (1 if i == 128 else (4 if i % 2 else 2))
    * math.exp(-0.5 * (i / 16) ** 2)
    / (48 * math.sqrt(2 * math.pi))
    for i in range(129)
)


def score_utility(mean: float, stdev: float, board_size: int) -> float:
    scale = 2.0 / board_size
    if stdev == 0:
        return (2 / math.pi) * math.atan(mean / scale)
    result = _NORMAL_WEIGHTS[0] * math.atan(mean / scale)
    for i in range(1, 129):
        delta = stdev * i / 16
        result += _NORMAL_WEIGHTS[i] * (
            math.atan((mean + delta) / scale) + math.atan((mean - delta) / scale)
        )
    return max(-1.0, min(1.0, (2 / math.pi) * result))

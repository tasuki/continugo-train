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


def _adaptive_simpson(f, a, b, values, whole, tolerance, depth):
    fa, fm, fb = values
    m = (a + b) / 2
    fl, fr = f((a + m) / 2), f((m + b) / 2)
    left = (m - a) / 6 * (fa + 4 * fl + fm)
    right = (b - m) / 6 * (fm + 4 * fr + fb)
    error = left + right - whole
    if depth == 0 or abs(error) <= 15 * tolerance:
        return left + right + error / 15
    return _adaptive_simpson(f, a, m, (fa, fl, fm), left, tolerance / 2, depth - 1) + _adaptive_simpson(
        f, m, b, (fm, fr, fb), right, tolerance / 2, depth - 1
    )


def score_utility(mean: float, stdev: float, board_size: int) -> float:
    scale = 2.0 / board_size
    if stdev == 0:
        return (2 / math.pi) * math.atan(mean / scale)
    if stdev > 1:
        def f(z):
            return math.exp(-0.5 * z * z) / math.sqrt(2 * math.pi) * (
                math.atan((mean + stdev * z) / scale)
                + math.atan((mean - stdev * z) / scale)
            )

        fa, fm, fb = f(0), f(4), f(8)
        whole = 8 / 6 * (fa + 4 * fm + fb)
        result = _adaptive_simpson(f, 0, 8, (fa, fm, fb), whole, 1e-7, 20)
        return max(-1.0, min(1.0, (2 / math.pi) * result))
    result = _NORMAL_WEIGHTS[0] * math.atan(mean / scale)
    for i in range(1, 129):
        delta = stdev * i / 16
        result += _NORMAL_WEIGHTS[i] * (
            math.atan((mean + delta) / scale) + math.atan((mean - delta) / scale)
        )
    return max(-1.0, min(1.0, (2 / math.pi) * result))

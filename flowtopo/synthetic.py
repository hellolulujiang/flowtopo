"""A synthetic D8 network, for benchmarking on grids larger than the example.

Steepest descent on smoothed noise gives a dendritic network with realistic
layer sizes: many basins, a wide layer 0 of headwaters, and a long thin tail.
"""

import numpy as np

_CODES = np.array([1, 2, 4, 8, 16, 32, 64, 128], dtype=np.uint8)
_OFFSETS = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]


def synthetic_d8(nrow, ncol=None, seed=0, smooth=12, passes=3):
    """Return a ``(nrow, ncol)`` uint8 D8 raster with no nodata cells."""
    ncol = nrow if ncol is None else ncol
    rng = np.random.default_rng(seed)
    z = rng.random((nrow, ncol))

    width = 2 * smooth + 1
    for _ in range(passes):
        pad = np.pad(z, smooth, mode="reflect")
        cum = np.cumsum(np.cumsum(pad, 0), 1)
        cum = np.pad(cum, ((1, 0), (1, 0)))
        z = (
            cum[width:, width:]
            - cum[:-width, width:]
            - cum[width:, :-width]
            + cum[:-width, :-width]
        )[:nrow, :ncol] / (width * width)
    z = z + 1e-6 * rng.random((nrow, ncol))

    best = np.zeros((nrow, ncol))
    out = np.zeros((nrow, ncol), dtype=np.uint8)
    for code, (dr, dc) in zip(_CODES, _OFFSETS):
        neighbour = np.full((nrow, ncol), np.inf)
        r0, r1 = max(0, -dr), nrow - max(0, dr)
        c0, c1 = max(0, -dc), ncol - max(0, dc)
        neighbour[r0:r1, c0:c1] = z[r0 + dr : r1 + dr, c0 + dc : c1 + dc]
        drop = (z - neighbour) / np.hypot(dr, dc)
        take = drop > best
        best[take] = drop[take]
        out[take] = code
    return out

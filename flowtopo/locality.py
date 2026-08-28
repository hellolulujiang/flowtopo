"""Memory-locality metrics for orderings and layerings.

A traversal does two reads per cell: the cell itself, and its receiver.  Those
two streams compete for the same cache, so the metrics here feed both through
one simulated N-way set-associative LRU cache, with the two arrays offset far
enough apart that their lines cannot alias.

The default cache geometry is the machine the companion paper measured on:
32 KB 8-way L1, 1 MB 16-way L2, 35.75 MB 11-way L3.

The simulator is a scalar loop.  Under numba it is instant; without numba it
runs but is slow, so pass ``levels=("L1",)`` on a large grid if numba is not
installed.
"""

import numpy as np

from ._compat import njit
from .layering import Decomposition

CACHE_LINE_BYTES = 64

CACHE_LEVELS = {
    # name: (capacity in cache lines, associativity)
    "L1": (512, 8),
    "L2": (16384, 16),
    "L3": (585728, 11),
}

_ARRAY_OFFSET = 1 << 40  # keep the two arrays out of each other's tags


@njit(cache=True)
def _simulate_lru(seq, idxs_ds, elem_bytes, capacity_lines, n_ways, array_offset):
    """N-way set-associative LRU miss rate over the interleaved access stream."""
    if seq.size == 0 or capacity_lines <= 0 or n_ways <= 0:
        return 1.0
    n_sets = capacity_lines // n_ways
    if n_sets <= 0:
        return 1.0

    tags = np.full(n_sets * n_ways, -1, dtype=np.int64)
    ages = np.zeros(n_sets * n_ways, dtype=np.int64)

    misses = 0
    total = 0
    clock = 0
    shift = 6  # 64-byte cache line

    for i in range(seq.size):
        idx = seq[i]
        idx_ds = idxs_ds[idx]

        for access in range(2):
            if access == 0:
                byte_addr = np.int64(idx) * elem_bytes
            else:
                if idx_ds < 0 or idx_ds == idx:
                    continue
                byte_addr = array_offset + np.int64(idx_ds) * elem_bytes

            cl_addr = byte_addr >> shift
            base = int(cl_addr % n_sets) * n_ways

            hit = False
            for w in range(n_ways):
                if tags[base + w] == cl_addr:
                    ages[base + w] = clock
                    clock += 1
                    hit = True
                    break
            if not hit:
                misses += 1
                lru_w = 0
                lru_age = ages[base]
                for w in range(1, n_ways):
                    if ages[base + w] < lru_age:
                        lru_age = ages[base + w]
                        lru_w = w
                tags[base + lru_w] = cl_addr
                ages[base + lru_w] = clock
                clock += 1
            total += 1

    return misses / total if total > 0 else 1.0


def miss_rates(seq, idxs_ds, elem_bytes=4, levels=("L1", "L2", "L3")):
    """Simulated cache miss rate per level for one visitation order."""
    seq = np.ascontiguousarray(seq, dtype=np.int32)
    idxs_ds = np.ascontiguousarray(idxs_ds, dtype=np.int32)
    out = {}
    for name in levels:
        capacity, ways = CACHE_LEVELS[name]
        out[name] = _simulate_lru(seq, idxs_ds, elem_bytes, capacity, ways,
                                  _ARRAY_OFFSET)
    return out


def _reuse_distances(cell_lines):
    """Accesses elapsed since the previous access to the same cache line."""
    if cell_lines.size < 2:
        return np.empty(0, dtype=np.int64)
    position = np.arange(cell_lines.size, dtype=np.int64)
    order = np.argsort(cell_lines, kind="stable")
    lines_sorted = cell_lines[order]
    pos_sorted = position[order]
    same = lines_sorted[1:] == lines_sorted[:-1]
    return (pos_sorted[1:] - pos_sorted[:-1])[same]


def serial_locality(seq_d2u, idxs_ds, ncol, elem_bytes=4,
                    levels=("L1", "L2", "L3")):
    """Locality metrics for one serial ordering.

    Parameters
    ----------
    seq_d2u : ndarray
        Traversal order, downstream to upstream.
    idxs_ds : ndarray
    ncol : int
        Needed for the row-jump fraction.

    Returns
    -------
    dict
        ``stride_*``, ``row_jump_frac``, ``reuse_dist_*``, ``ds_stride_*`` and
        one ``miss_rate_<level>`` per requested cache level.
    """
    seq = np.ascontiguousarray(seq_d2u, dtype=np.int64)
    idxs_ds = np.ascontiguousarray(idxs_ds, dtype=np.int32)
    cl_elems = CACHE_LINE_BYTES // elem_bytes

    stride = np.abs(np.diff(seq))
    rows = seq // ncol
    row_jump = np.abs(np.diff(rows)) > 0

    valid = idxs_ds >= 0
    cells = np.nonzero(valid)[0]
    ds_stride = np.abs(cells - idxs_ds[cells].astype(np.int64))

    reuse = _reuse_distances(seq // cl_elems)

    out = {
        "stride_mean": float(stride.mean()) if stride.size else 0.0,
        "stride_median": float(np.median(stride)) if stride.size else 0.0,
        "stride_p95": float(np.percentile(stride, 95)) if stride.size else 0.0,
        "stride_p99": float(np.percentile(stride, 99)) if stride.size else 0.0,
        "stride_max": int(stride.max()) if stride.size else 0,
        "row_jump_frac": float(row_jump.mean()) if row_jump.size else 0.0,
        "reuse_dist_mean": float(reuse.mean()) if reuse.size else 0.0,
        "reuse_dist_median": float(np.median(reuse)) if reuse.size else 0.0,
        "ds_stride_mean": float(ds_stride.mean()) if ds_stride.size else 0.0,
        "ds_stride_median": float(np.median(ds_stride)) if ds_stride.size else 0.0,
        "ds_stride_max": int(ds_stride.max()) if ds_stride.size else 0,
    }
    for name, rate in miss_rates(seq.astype(np.int32), idxs_ds, elem_bytes,
                                 levels).items():
        out[f"miss_rate_{name}"] = rate
    return out


def parallel_locality(layers, idxs_ds, msk, ncol, elem_bytes=4,
                      levels=("L1", "L2", "L3")):
    """Locality metrics for one layering.

    The cache simulation flattens the layers into a single visitation order
    (layer 0 first, ascending cell index inside a layer) and feeds it through
    the same simulator as :func:`serial_locality`, so the two are comparable.
    """
    layers = np.ascontiguousarray(layers, dtype=np.int64)
    idxs_ds = np.ascontiguousarray(idxs_ds, dtype=np.int32)
    msk = np.asarray(msk, dtype=bool)
    cl_elems = CACHE_LINE_BYTES // elem_bytes

    masked = np.where(msk, layers, -1)
    decomp = Decomposition.from_layers(masked)

    spans = np.zeros(decomp.nlayers, dtype=np.int64)
    gap_medians = np.zeros(decomp.nlayers)
    gap_hits = np.zeros(decomp.nlayers)
    footprints = np.zeros(decomp.nlayers, dtype=np.int64)

    for layer_index, members in enumerate(decomp):
        if members.size == 0:
            continue
        cells = members.astype(np.int64)
        spans[layer_index] = int(cells.max() - cells.min())
        footprints[layer_index] = int(np.unique(cells // cl_elems).size)
        if cells.size > 1:
            gaps = np.diff(cells)
            gap_medians[layer_index] = float(np.median(gaps))
            gap_hits[layer_index] = float((gaps <= cl_elems).mean())
        else:
            gap_hits[layer_index] = 1.0

    cells = np.nonzero(msk & (idxs_ds >= 0))[0]
    ds_stride = np.abs(cells - idxs_ds[cells].astype(np.int64))

    out = {
        "nlayers": decomp.nlayers,
        "ncells": decomp.ncells,
        "span_mean": float(spans.mean()) if spans.size else 0.0,
        "span_median": float(np.median(spans)) if spans.size else 0.0,
        "span_max": int(spans.max()) if spans.size else 0,
        "gap_median_mean": float(gap_medians.mean()) if gap_medians.size else 0.0,
        "gap_cl_hit_frac": float(gap_hits.mean()) if gap_hits.size else 0.0,
        "cl_footprint_mean": float(footprints.mean()) if footprints.size else 0.0,
        "cl_footprint_max": int(footprints.max()) if footprints.size else 0,
        "cl_footprint_sum": int(footprints.sum()) if footprints.size else 0,
        "ds_stride_mean": float(ds_stride.mean()) if ds_stride.size else 0.0,
        "ds_stride_median": float(np.median(ds_stride)) if ds_stride.size else 0.0,
        "ds_stride_max": int(ds_stride.max()) if ds_stride.size else 0,
    }
    for name, rate in miss_rates(decomp.flattened(), idxs_ds, elem_bytes,
                                 levels).items():
        out[f"miss_rate_{name}"] = rate
    return out

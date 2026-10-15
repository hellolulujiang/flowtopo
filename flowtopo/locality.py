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

from .core import _checked_cells, _checked_downstream, _checked_sequence
from ._compat import njit
from .layering import Decomposition

CACHE_LINE_BYTES = 64

CACHE_LEVELS = {
    # name: (capacity in cache lines, associativity)
    "L1": (512, 8),
    "L2": (16384, 16),
    "L3": (585728, 11),
}

# Where the receiver array sits relative to the cell array. Any large offset
# keeps their tags apart, which is all the simulation needs to tell the two
# streams apart. This particular value is a whole number of L1 and L2 sets, so
# cell i of one array and cell i of the other always land in the same set:
# the worst case for conflict misses, and a real program's allocator gives an
# arbitrary alignment instead. It moves the absolute miss rates by a few
# points on the example basin (depth-first 8.8% to 10.5%, breadth-first 14.2%
# to 21.8%, topological 36.9% to 37.0%) and does not change their order, which
# holds at every offset tried.
_ARRAY_OFFSET = 1 << 40


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
    # Ages start at 1, never 0. An untouched way keeps age 0 and so always
    # looks older than any line that has been used, which is what makes the
    # eviction scan pick an empty way before it evicts anything real.
    clock = 1
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


def _check_elem_bytes(elem_bytes):
    """An element of no size has no cache line; the C code's parallel measure refuses it too."""
    if int(elem_bytes) <= 0:
        raise ValueError("elem_bytes must be positive, got %r" % (elem_bytes,))
    return int(elem_bytes)


def miss_rates(seq, idxs_ds, elem_bytes=4, levels=("L1", "L2", "L3")):
    """Simulated cache miss rate per level for one visitation order."""
    elem_bytes = _check_elem_bytes(elem_bytes)
    idxs_ds = _checked_downstream(idxs_ds)
    seq = _checked_sequence("seq", seq, idxs_ds.size)
    out = {}
    for name in levels:
        capacity, ways = CACHE_LEVELS[name]
        out[name] = _simulate_lru(seq, idxs_ds, elem_bytes, capacity, ways,
                                  _ARRAY_OFFSET)
    return out


def _cache_line_of(indices, elem_bytes):
    """The cache line each element sits on: its first byte over the line size, which is the C code's
    rule.  Dividing the index by "elements per line" is the same thing only when the element
    size divides 64: with three-byte elements, 20 and 21 start in line 0 and 1 by that formula and in
    line 0 and 0 by this one."""
    if elem_bytes <= 0:
        raise ValueError("elem_bytes must be positive, got %r" % (elem_bytes,))
    return (np.asarray(indices, dtype=np.int64) * int(elem_bytes)) // CACHE_LINE_BYTES


_REUSE_TABLE_SLOTS = 1 << 20


@njit(cache=True)
def _reuse_intervals(cell_lines):
    """Accesses between two touches of the same cache line, as the C code measures them.

    This is the reuse interval, not the reuse distance of the cache literature: that one counts how many
    *distinct* lines were touched in between (the LRU stack distance); this counts accesses, which is cheaper
    and is what the miss rates here are read alongside.

    The last access of a line is kept in a table of 2**20 slots, one line a slot (the line's number modulo the
    table): a line whose slot was taken by another line in between has lost its last access, and its next
    access counts as a first one.  On a grid of fewer than 2**20 lines (16.7 million four-byte cells) no two
    lines share a slot and every reuse is counted; beyond that some are missed, as in the C code, whose numbers
    the companion paper reports."""
    count = cell_lines.size
    intervals = np.empty(count, dtype=np.int64)
    found = 0
    slot_line = np.full(_REUSE_TABLE_SLOTS, -1, dtype=np.int64)
    slot_time = np.zeros(_REUSE_TABLE_SLOTS, dtype=np.int64)
    mask = _REUSE_TABLE_SLOTS - 1
    for position in range(count):
        line = cell_lines[position]
        slot = line & mask
        if slot_line[slot] == line:
            intervals[found] = position - slot_time[slot]
            found += 1
        slot_line[slot] = line
        slot_time[slot] = position
    return intervals[:found]


def _upper_median(values):
    """The middle value of the sorted values, the upper of the two when their number is even: the C code's
    median (sorted[n // 2]).  ``numpy.median`` averages the two."""
    ordered = np.sort(np.asarray(values))
    return ordered[ordered.size // 2]


def _rank_percentile(values, percent):
    """The value at rank round(percent / 100 * (n - 1)) of the sorted values, as the C code takes a percentile;
    ``numpy.percentile`` interpolates between two ranks."""
    ordered = np.sort(np.asarray(values))
    rank = int(percent / 100.0 * (ordered.size - 1) + 0.5)
    return ordered[min(rank, ordered.size - 1)]


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
    elem_bytes : int
        The size of one element in the simulated caches (4: Int32 or Float32).
    levels : sequence of str
        The cache levels to simulate, of "L1", "L2", "L3" (``CACHE_LEVELS``).

    Returns
    -------
    dict
        ``stride_*``, ``row_jump_frac``, ``reuse_interval_*``, ``ds_stride_*`` and
        one ``miss_rate_<level>`` per requested cache level.
    """
    elem_bytes = _check_elem_bytes(elem_bytes)
    idxs_ds = _checked_downstream(idxs_ds)
    seq = _checked_sequence("seq_d2u", seq_d2u, idxs_ds.size).astype(np.int64)

    stride = np.abs(np.diff(seq))
    rows = seq // ncol
    row_jump = np.abs(np.diff(rows)) > 0

    # the read of the receiver: the cells of the sequence that have one (a pit or a cell that drains off the
    # network reads none), as the C code counts them.  Until 1.0.5 every cell with idxs_ds >= 0 was counted, a
    # pit with a stride of 0
    receiver = idxs_ds[seq].astype(np.int64)
    reads = (receiver >= 0) & (receiver != seq)
    ds_stride = np.abs(seq[reads] - receiver[reads])

    reuse = _reuse_intervals(_cache_line_of(seq, elem_bytes))

    # the medians and percentiles are order statistics, as in the C code (the upper of two middle values; the
    # value at the rounded rank), not numpy's interpolated ones.  Without a reuse the interval is the length of
    # the sequence, as in the C code
    out = {
        "stride_mean": float(stride.mean()) if stride.size else 0.0,
        "stride_median": float(_upper_median(stride)) if stride.size else 0.0,
        "stride_p95": float(_rank_percentile(stride, 95)) if stride.size else 0.0,
        "stride_p99": float(_rank_percentile(stride, 99)) if stride.size else 0.0,
        "stride_max": int(stride.max()) if stride.size else 0,
        "row_jump_frac": float(row_jump.mean()) if row_jump.size else 0.0,
        "reuse_interval_mean": float(reuse.mean()) if reuse.size else float(seq.size),
        "reuse_interval_median": float(_upper_median(reuse)) if reuse.size else float(seq.size),
        "ds_stride_mean": float(ds_stride.mean()) if ds_stride.size else 0.0,
        "ds_stride_median": float(_upper_median(ds_stride)) if ds_stride.size else 0.0,
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
    elem_bytes = _check_elem_bytes(elem_bytes)
    idxs_ds = _checked_downstream(idxs_ds)
    layers = _checked_cells("layers", layers, idxs_ds.size, np.int64)     # one value per cell
    msk = _checked_cells("msk", msk, idxs_ds.size, bool)

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
        footprints[layer_index] = int(np.unique(_cache_line_of(cells, elem_bytes)).size)
        if cells.size > 1:
            gaps = np.diff(cells)
            gap_medians[layer_index] = float(_upper_median(gaps))
            # two consecutive cells share a cache line when they have the same line number, not when
            # their indices are within one line's worth of each other: with 16 elements to a line,
            # 15 and 16 are one apart and on different lines (the C code's criterion)
            lines = _cache_line_of(cells, elem_bytes)
            gap_hits[layer_index] = float((np.diff(lines) == 0).mean())
        else:
            gap_hits[layer_index] = 1.0

    # the read of the receiver: the cells that have a layer and a receiver other than themselves, as the C
    # code counts them (until 1.0.5 every masked cell with idxs_ds >= 0, a pit with a stride of 0)
    cells = np.nonzero((masked >= 0) & (idxs_ds >= 0) & (idxs_ds != np.arange(idxs_ds.size)))[0]
    ds_stride = np.abs(cells - idxs_ds[cells].astype(np.int64))

    out = {
        "nlayers": decomp.nlayers,
        "ncells": decomp.ncells,
        "span_mean": float(spans.mean()) if spans.size else 0.0,
        "span_median": float(_upper_median(spans)) if spans.size else 0.0,
        "span_max": int(spans.max()) if spans.size else 0,
        "gap_median_mean": float(gap_medians.mean()) if gap_medians.size else 0.0,
        "gap_cl_hit_frac": float(gap_hits.mean()) if gap_hits.size else 0.0,
        "cl_footprint_mean": float(footprints.mean()) if footprints.size else 0.0,
        "cl_footprint_max": int(footprints.max()) if footprints.size else 0,
        "cl_footprint_sum": int(footprints.sum()) if footprints.size else 0,
        "ds_stride_mean": float(ds_stride.mean()) if ds_stride.size else 0.0,
        "ds_stride_median": float(_upper_median(ds_stride)) if ds_stride.size else 0.0,
        "ds_stride_max": int(ds_stride.max()) if ds_stride.size else 0,
    }
    for name, rate in miss_rates(decomp.flattened(), idxs_ds, elem_bytes,
                                 levels).items():
        out[f"miss_rate_{name}"] = rate
    return out

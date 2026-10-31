"""Topological representations of a D8 flow network.

One input to everything here: ``idxs_ds``, the linear index of the cell each
cell drains into, or the cell itself at a pit.  From it this module builds

* three serial orderings, each a topologically sorted list of cell indices;
* three parallel layerings, each a per-cell layer number where the members of
  one layer are mutually independent.

The layer index of a layering grows toward the pit; layer 0 holds headwaters
(every one of them in the as-soon-as-possible form).  A cell caught in a cycle is left out of the sequences and marked
``-1`` in the layerings, so a sequence shorter than the valid-cell count means
the input has a cycle.

The representation these are built on -- one flat array holding, for every
cell, the linear index of the cell it drains into -- and the D8 decoding
conventions in this module follow pyflwdir (D. Eilander, Deltares and IVM,
Vrije Universiteit Amsterdam; MIT licence; doi:10.5281/zenodo.4287337).  No
pyflwdir source is included; see the README's acknowledgements and
``docs/methods.md`` for which construction came from where.

"""

import numpy as np

from ._compat import njit

MV = -1
"""Missing linear index."""

D8_NODATA = np.uint8(247)
"""Flow-direction code for nodata in the MERIT Hydro convention."""

RANK_NODATA = -9999
RANK_CYCLE = -1
_RANK_BUSY = -8888

LAYER_NODATA = -1

D8_CODES = frozenset({0, 1, 2, 4, 8, 16, 32, 64, 128, 255})
"""Every code this convention defines, the two terminals included."""



def _checked_downstream(idxs_ds):
    """The downstream indices as int32, checked on the type they came in.

    Every public function of this module and of kernels takes its indices through this: the cast to int32
    wraps round (2**32 becomes cell 0, a receiver that exists) and truncates a float.
    A value must be a whole number in -1 .. size - 1, and the grid no larger than int32 counts.
    """
    given = np.ascontiguousarray(idxs_ds)
    size = given.size
    if size > 2147483647:
        raise ValueError("the grid holds more cells than the int32 the indices are kept in")
    if given.dtype == np.int32:
        values = given
    elif np.issubdtype(given.dtype, np.integer):
        values = given
    else:
        if size and not np.all(np.isfinite(given) & (np.mod(given, 1.0) == 0.0)):
            raise ValueError("a downstream index is not a whole number")
        values = given
    if size and (values.min() < -1 or values.max() >= size):
        raise ValueError("a downstream index is outside the grid: they run from -1 (no receiver) to %d" % (size - 1))
    return np.ascontiguousarray(values, dtype=np.int32)


def _checked_cells(name, values, size, dtype):
    """One value per cell of the network, flat.  The compiled kernels do not check bounds, so a
    shorter array would be read and written past its end; every module takes its arrays through
    this one check"""
    array = np.ascontiguousarray(values, dtype=dtype).reshape(-1)
    if array.size != size:
        raise ValueError(f"{name} holds {array.size} values; the network has {size} cells, one value each")
    return array


def _checked_sequence(name, values, size):
    """An ordering: cell indices 0 .. size - 1, checked on the type they came in before they are made int32
    (an int64 2**32 would wrap to cell 0)"""
    given = np.asarray(values)
    if given.ndim != 1 or (given.size and given.dtype.kind not in "iu"):
        raise ValueError(f"{name} must be a flat array of cell indices")
    if given.size and (int(given.min()) < 0 or int(given.max()) >= size):
        raise ValueError(f"{name} holds a cell index outside 0 .. {size - 1}")
    return np.ascontiguousarray(given, dtype=np.int32)


def _as_d8_codes(dir_flat, nodata=None):
    """The grid as uint8 codes, its nodata as D8_NODATA.

    The nodata is found on the grid's own type before the cast: a UInt16 grid
    whose nodata is 65535 would turn it into 255, a valid terminal, and a value
    outside 0..255 would wrap round into some code. A NaN
    nodata is found with isnan.  Every cell of the nodata becomes D8_NODATA,
    on a uint8 grid too, so nothing downstream has to exclude the nodata again.
    Compare nodata without truncation: int(1.5) would exclude the valid code 1.
    A value that is neither nodata nor a whole number in 0..255 is refused.
    """
    raw = np.asarray(dir_flat).ravel()
    is_nodata = np.zeros(raw.shape, dtype=bool)
    if nodata is not None:
        if np.isnan(float(nodata)):
            is_nodata = np.isnan(raw) if raw.dtype.kind == "f" else is_nodata
        else:
            is_nodata = raw == nodata
    values = raw[~is_nodata]
    if values.size:
        if raw.dtype.kind == "f" and not np.all(np.isfinite(values) & (np.mod(values, 1.0) == 0.0)):
            raise ValueError("the flow directions hold values that are not whole numbers")
        if values.min() < 0 or values.max() > 255:
            raise ValueError("the flow directions hold values outside 0..255 that are not the nodata")
    codes = np.where(is_nodata, 0, raw).astype(np.uint8)
    codes[is_nodata] = D8_NODATA
    return np.ascontiguousarray(codes)


def unknown_codes(dir_flat, nodata=None):
    """Codes in the grid that this convention does not define.

    Returns ``{code: count}``. A non-empty result almost always means the grid
    uses another D8 convention: TauDEM numbers its directions 1 to 8
    anticlockwise from east, GRASS 1 to 8 anticlockwise from north-east, and
    neither survives being read as powers of two. Such a code decodes to no
    direction at all, so the cell becomes a pit and the network quietly falls
    apart into fragments.
    """
    dir_flat = _as_d8_codes(dir_flat, nodata)       # nodata found on the grid's own type
    codes, counts = np.unique(dir_flat, return_counts=True)
    skip = set(D8_CODES) | {int(D8_NODATA)}      # the grid's nodata is D8_NODATA already (_as_d8_codes)
    return {int(c): int(n) for c, n in zip(codes, counts) if int(c) not in skip}


# ---------------------------------------------------------------------------
# D8 decoding
# ---------------------------------------------------------------------------

# Powers of two, clockwise from east.  0 and 255 are terminals.
#
#     32  64 128        NW  N  NE
#     16   0   1   =>    W  .   E
#      8   4   2        SW  S  SE
_D8_DR = np.zeros(256, dtype=np.int8)
_D8_DC = np.zeros(256, dtype=np.int8)
for _code, (_dr, _dc) in {
    1: (0, 1),
    2: (1, 1),
    4: (1, 0),
    8: (1, -1),
    16: (0, -1),
    32: (-1, -1),
    64: (-1, 0),
    128: (-1, 1),
}.items():
    _D8_DR[_code] = _dr
    _D8_DC[_code] = _dc


def d8_to_downstream(dir_flat, nrow, ncol, nodata=None):
    """Decode a D8 direction raster into a downstream-pointer array.

    Parameters
    ----------
    dir_flat : ndarray of uint8, shape (nrow * ncol,)
        D8 codes, row-major.
    nrow, ncol : int
        Grid shape.
    nodata : int, optional
        The grid's nodata code. 247, the MERIT Hydro value, is always treated
        as nodata; pass another one and it is excluded as well. Note that 255
        is a valid endorheic terminal in this convention, so a grid that uses
        255 for nodata has to say so.

    Returns
    -------
    idxs_ds : ndarray of int32, shape (nrow * ncol,)
        Downstream linear index per cell; ``-1`` outside the network.  A cell
        that drains off the grid, into nodata, or nowhere points at itself and
        is treated as a pit.
    """
    dir_flat = _as_d8_codes(dir_flat, nodata)       # nodata found on the grid's own type
    size = nrow * ncol
    if dir_flat.size != size:
        raise ValueError(f"dir_flat has {dir_flat.size} cells, expected {size}")

    idxs_ds = np.full(size, MV, dtype=np.int32)
    valid = dir_flat != D8_NODATA          # the grid's nodata is D8_NODATA already (_as_d8_codes)
    idx = np.nonzero(valid)[0]

    codes = dir_flat[idx]
    dr = _D8_DR[codes].astype(np.int64)
    dc = _D8_DC[codes].astype(np.int64)

    r_ds = idx // ncol + dr
    c_ds = idx % ncol + dc
    pit = (dr == 0) & (dc == 0)
    outside = (r_ds < 0) | (r_ds >= nrow) | (c_ds < 0) | (c_ds >= ncol)

    # Clamp before indexing; the clamped values are discarded by the masks.
    r_safe = np.clip(r_ds, 0, nrow - 1)
    c_safe = np.clip(c_ds, 0, ncol - 1)
    idx_ds = (c_safe + r_safe * ncol).astype(np.int64)

    into_nodata = np.zeros(idx.size, dtype=bool)
    ok = ~(pit | outside)
    into_nodata[ok] = ~valid[idx_ds[ok]]

    self_pointing = pit | outside | into_nodata
    idxs_ds[idx] = np.where(self_pointing, idx, idx_ds).astype(np.int32)
    return idxs_ds


# ---------------------------------------------------------------------------
# Upstream topology
# ---------------------------------------------------------------------------


@njit(cache=True)
def _upstream_count(idxs_ds, msk, use_msk):
    size = idxs_ds.size
    n_up = np.full(size, -9, dtype=np.int32)
    for idx0 in range(size):
        idx_ds = idxs_ds[idx0]
        if idx_ds == MV or idx_ds < 0 or idx_ds >= size:
            continue
        if n_up[idx0] < 0:
            n_up[idx0] = 0
        valid = (not use_msk) or (msk[idx0] != 0)
        if idx0 != idx_ds and valid:
            if n_up[idx_ds] < 0:
                n_up[idx_ds] = 0
            n_up[idx_ds] += 1
    max_val = 0
    for i in range(size):
        if n_up[i] > max_val:
            max_val = n_up[i]
    return n_up, max_val


def upstream_count(idxs_ds, msk=None):
    """Donors per cell.

    Returns
    -------
    n_up : ndarray of int32
        Donor count per cell; ``-9`` outside the network.
    max_donors : int
        Largest donor count on the grid (at most 8 for D8).
    """
    idxs_ds = _checked_downstream(idxs_ds)
    use_msk = msk is not None
    msk_arr = (
        _checked_cells("msk", msk, idxs_ds.size, np.uint8)       # one value per cell
        if use_msk
        else np.zeros(1, dtype=np.uint8)
    )
    return _upstream_count(idxs_ds, msk_arr, use_msk)


@njit(cache=True)
def _upstream_table(idxs_ds, msk, use_msk, n_up, max_donors):
    size = idxs_ds.size
    cursor = n_up.copy()
    table = np.full((size, max_donors), MV, dtype=np.int32)
    for idx in range(size):
        if use_msk and msk[idx] == 0:
            continue
        idx_ds = idxs_ds[idx]
        if idx_ds == MV or idx_ds < 0 or idx_ds >= size or idx_ds == idx:
            continue
        if use_msk and msk[idx_ds] == 0:
            continue
        cursor[idx_ds] -= 1
        slot = cursor[idx_ds]
        if 0 <= slot < max_donors:
            table[idx_ds, slot] = idx
    return table


def upstream_table(idxs_ds, msk=None):
    """Upstream adjacency table and donor counts, ``(table, n_up)``: ``table`` has a
    row per grid position, ``(idxs_ds.size, max_donors)``, -1 in an empty slot.

    Row ``i`` lists the cells that drain into ``i``, padded with ``-1``.  This
    is what the pull manner reads: a receiver gathers from its donors instead
    of each donor writing into the receiver, so concurrent writes cannot
    collide.  The cost is building and storing the table.
    """
    idxs_ds = _checked_downstream(idxs_ds)
    n_up, max_donors = upstream_count(idxs_ds, msk)
    max_donors = max(int(max_donors), 1)
    use_msk = msk is not None
    msk_arr = (
        _checked_cells("msk", msk, idxs_ds.size, np.uint8)       # one value per cell
        if use_msk
        else np.zeros(1, dtype=np.uint8)
    )
    table = _upstream_table(idxs_ds, msk_arr, use_msk, n_up, max_donors)
    return table, n_up


# ---------------------------------------------------------------------------
# Rank
# ---------------------------------------------------------------------------


@njit(cache=True)
def _rank(idxs_ds):
    size = idxs_ds.size
    ranks = np.full(size, RANK_NODATA, dtype=np.int32)
    stack = np.empty(size, dtype=np.int32)

    for idx0 in range(size):
        if idxs_ds[idx0] == MV or ranks[idx0] != RANK_NODATA:
            continue

        sp = 0
        stack[sp] = idx0
        sp += 1
        ranks[idx0] = _RANK_BUSY

        current = idx0
        rnk = 0
        while True:
            idx_ds = idxs_ds[current]
            if idx_ds == MV or idx_ds < 0 or idx_ds >= size:
                rnk = RANK_CYCLE
                break
            dr = ranks[idx_ds]
            if dr >= 0:
                rnk = dr
                break
            if idx_ds == current:
                rnk = RANK_CYCLE
                break
            if dr == RANK_CYCLE or dr == _RANK_BUSY:
                for i in range(sp):
                    ranks[stack[i]] = RANK_CYCLE
                sp = 0
                break
            current = idx_ds
            stack[sp] = current
            sp += 1
            ranks[current] = _RANK_BUSY

        while sp > 0:
            rnk += 1
            sp -= 1
            ranks[stack[sp]] = rnk
    return ranks


def rank_to_pit(idxs_ds):
    """Downstream hops from each cell to its pit.

    ``rank[pit] == 0``, and the headwater farthest from the pit in hops holds
    the largest value of its basin; another headwater can hold less.  ``-9999`` outside the network, ``-1`` in a cycle or on a path that enters one.  Used here
    only as the intermediate of the as-late-as-possible layering.
    """
    return _rank(_checked_downstream(idxs_ds))


# ---------------------------------------------------------------------------
# Serial orderings
# ---------------------------------------------------------------------------


@njit(cache=True)
def _seq_dfs(idxs_ds):
    """Depth-first from the pits (Braun-Willett stack ordering)."""
    size = idxs_ds.size
    ndon = np.zeros(size, dtype=np.int32)
    n_valid = 0
    for i in range(size):
        ds = idxs_ds[i]
        if ds == MV or ds < 0 or ds >= size:
            continue
        n_valid += 1
        if ds == i:
            continue
        ndon[ds] += 1

    delta = np.zeros(size + 1, dtype=np.int64)
    for i in range(size):
        delta[i + 1] = delta[i] + ndon[i]

    donors = np.empty(delta[size], dtype=np.int32)
    fill = np.zeros(size, dtype=np.int32)
    for i in range(size):
        ds = idxs_ds[i]
        if ds == MV or ds < 0 or ds >= size or ds == i:
            continue
        donors[delta[ds] + fill[ds]] = i
        fill[ds] += 1

    seq = np.full(size, MV, dtype=np.int32)
    stack = np.empty(n_valid if n_valid > 0 else 1, dtype=np.int32)
    top = 0
    n_out = 0
    for i in range(size):
        ds = idxs_ds[i]
        if ds != MV and 0 <= ds < size and ds == i:
            stack[top] = i
            top += 1
    while top > 0:
        top -= 1
        c = stack[top]
        seq[n_out] = c
        n_out += 1
        for k in range(delta[c + 1] - 1, delta[c] - 1, -1):
            stack[top] = donors[k]
            top += 1
    return seq, n_out


@njit(cache=True)
def _seq_bfs(idxs_ds):
    """Breadth-first from the pits: one ring of donors at a time."""
    size = idxs_ds.size
    n_up = np.zeros(size, dtype=np.int32)
    for i in range(size):
        ds = idxs_ds[i]
        if ds == MV or ds == i or ds < 0 or ds >= size:
            continue
        n_up[ds] += 1

    offset = np.zeros(size + 1, dtype=np.int64)
    for i in range(size):
        offset[i + 1] = offset[i] + n_up[i]

    us = np.empty(offset[size], dtype=np.int32)
    fill = np.zeros(size, dtype=np.int32)
    for i in range(size):
        ds = idxs_ds[i]
        if ds == MV or ds == i or ds < 0 or ds >= size:
            continue
        us[offset[ds] + fill[ds]] = i
        fill[ds] += 1

    visited = np.zeros(size, dtype=np.uint8)
    seq = np.full(size, MV, dtype=np.int32)
    frontier = np.empty(size, dtype=np.int32)
    nxt = np.empty(size, dtype=np.int32)

    n_out = 0
    front = 0
    for i in range(size):
        ds = idxs_ds[i]
        if ds != MV and 0 <= ds < size and ds == i:
            seq[n_out] = i
            n_out += 1
            frontier[front] = i
            front += 1
            visited[i] = 1

    while front > 0:
        nxt_n = 0
        for f in range(front):
            cell = frontier[f]
            for k in range(offset[cell], offset[cell + 1]):
                uc = us[k]
                if visited[uc] == 0:
                    visited[uc] = 1
                    nxt[nxt_n] = uc
                    nxt_n += 1
        for j in range(nxt_n):
            seq[n_out + j] = nxt[j]
        n_out += nxt_n
        for j in range(nxt_n):
            frontier[j] = nxt[j]
        front = nxt_n
    return seq, n_out


@njit(cache=True)
def _seq_topo(idxs_ds, n_up_in):
    """Kahn's algorithm from the sources: headwaters first, pit last."""
    size = idxs_ds.size
    n_up = n_up_in.copy()
    seq = np.full(size, MV, dtype=np.int32)
    q_curr = np.empty(size, dtype=np.int32)
    q_next = np.empty(size, dtype=np.int32)

    n_out = 0
    curr = 0
    for idx in range(size):
        if idxs_ds[idx] != MV and n_up[idx] == 0:
            seq[n_out] = idx
            n_out += 1
            q_curr[curr] = idx
            curr += 1

    while curr > 0:
        nxt = 0
        for i in range(curr):
            idx = q_curr[i]
            idx_ds = idxs_ds[idx]
            if idx_ds < 0 or idx_ds >= size or idx_ds == idx:
                continue
            n_up[idx_ds] -= 1
            if n_up[idx_ds] == 0:
                seq[n_out] = idx_ds
                n_out += 1
                q_next[nxt] = idx_ds
                nxt += 1
        for j in range(nxt):
            q_curr[j] = q_next[j]
        curr = nxt
    return seq, n_out


@njit(cache=True)
def _seq_dfs_sorted(idxs_ds, upa):
    """Depth-first from the pits with the donors of a cell taken by upstream area, largest first."""
    size = idxs_ds.size
    ndon = np.zeros(size, dtype=np.int32)
    n_valid = 0
    for i in range(size):
        ds = idxs_ds[i]
        if ds == MV or ds < 0 or ds >= size:
            continue
        n_valid += 1
        if ds == i:
            continue
        ndon[ds] += 1

    delta = np.zeros(size + 1, dtype=np.int64)
    for i in range(size):
        delta[i + 1] = delta[i] + ndon[i]

    donors = np.empty(delta[size], dtype=np.int32)
    fill = np.zeros(size, dtype=np.int32)
    for i in range(size):
        ds = idxs_ds[i]
        if ds == MV or ds < 0 or ds >= size or ds == i:
            continue
        donors[delta[ds] + fill[ds]] = i
        fill[ds] += 1

    # insertion sort by descending area; the donors were filled in ascending index order and
    # the sort is stable, so donors of equal area keep the smaller index first
    for i in range(size):
        count = fill[i]
        if count <= 1:
            continue
        base = delta[i]
        for a in range(1, count):
            key = donors[base + a]
            key_upa = upa[key]
            b = a - 1
            while b >= 0 and upa[donors[base + b]] < key_upa:
                donors[base + b + 1] = donors[base + b]
                b -= 1
            donors[base + b + 1] = key

    seq = np.full(size, MV, dtype=np.int32)
    stack = np.empty(n_valid if n_valid > 0 else 1, dtype=np.int32)
    top = 0
    n_out = 0
    for i in range(size):
        ds = idxs_ds[i]
        if ds != MV and 0 <= ds < size and ds == i:
            stack[top] = i
            top += 1
    while top > 0:
        top -= 1
        c = stack[top]
        seq[n_out] = c
        n_out += 1
        for k in range(delta[c + 1] - 1, delta[c] - 1, -1):
            stack[top] = donors[k]
            top += 1
    return seq, n_out


def seq_dfs_from_pit(idxs_ds, upa=None):
    """Depth-first order, downstream to upstream (position 0 is a pit).

    A tributary subtree is finished before the next one starts, so a cell's
    receiver sits only a few positions back.  This is the most cache-friendly
    of the three orderings.

    ``upa`` takes the donors of a cell largest first by upstream area instead
    of by index, which puts the main stem straight after the outlet and gives
    it consecutive positions. This is the rule of the released ``seq_dfs``
    layers of FullTopo v1.1. Without ``upa`` the donors are
    taken in index order. Areas are read as float32; equal areas take the
    smaller cell index first. Any kernel gives the same answer under
    either, up to the order floating-point sums are added in, since both are
    topological orders: as returned (d2u) a receiver
    comes before its donors, which the flow length downstream takes; reversed
    (u2d) a donor comes before its receiver, which the upstream area, the
    flow length upstream and the Strahler stream order take.
    """
    idxs_ds = _checked_downstream(idxs_ds)
    if upa is None:
        seq, n = _seq_dfs(idxs_ds)
    else:
        upa = np.ascontiguousarray(upa, dtype=np.float32).reshape(-1)
        if upa.size != idxs_ds.size:
            raise ValueError("upa does not match the grid")
        # every network cell needs a finite, positive area: a NaN interrupts the insertion sort,
        # and an infinity (a no-data of 1.79e308 cast to float32) would go first
        network_upa = upa[idxs_ds >= 0]
        if network_upa.size and not np.all(np.isfinite(network_upa) & (network_upa > 0)):
            raise ValueError("upa holds a value that is not a finite, positive area on a cell of the network")
        seq, n = _seq_dfs_sorted(idxs_ds, upa)
    return seq[:n]


def seq_bfs_from_pit(idxs_ds):
    """Breadth-first order, downstream to upstream (position 0 is a pit).

    Cells are grouped by downstream hop count: first the pits, then everything
    one hop above a pit, and so on out to the headwaters.
    """
    seq, n = _seq_bfs(_checked_downstream(idxs_ds))
    return seq[:n]


def seq_topo_from_source(idxs_ds):
    """Topological order, upstream to downstream (position 0 is a headwater)."""
    idxs_ds = _checked_downstream(idxs_ds)
    n_up, _ = upstream_count(idxs_ds)
    seq, n = _seq_topo(idxs_ds, n_up)
    return seq[:n]


# ---------------------------------------------------------------------------
# Basin labels
# ---------------------------------------------------------------------------


@njit(cache=True)
def _basin_labels(idxs_ds, seq_d2u):
    size = idxs_ds.size
    bsn = np.zeros(size, dtype=np.uint32)
    nxt = np.uint32(1)
    for k in range(seq_d2u.size):
        i = seq_d2u[k]
        q = idxs_ds[i]
        if q < 0 or q == i:
            bsn[i] = nxt
            nxt += np.uint32(1)
        else:
            bsn[i] = bsn[q]
    return bsn, int(nxt) - 1


def basin_labels(idxs_ds, seq_d2u=None):
    """``(labels, count)``: one-based basin id per cell (uint32), ``0`` outside the
    network and on a cell whose path reaches no pit; ``count`` the basins.

    A downstream-to-upstream pass visits every receiver before its donors, so
    a pit opens a new basin and every other cell copies its receiver's label.
    Required by the as-late-as-possible layering.
    """
    idxs_ds = _checked_downstream(idxs_ds)
    if seq_d2u is None:
        seq_d2u = seq_dfs_from_pit(idxs_ds)
    # the ordering checked on the type it came in, as the downstream indices (an int64
    # 2**32 wrapped to cell 0)
    return _basin_labels(idxs_ds, _checked_sequence("seq_d2u", seq_d2u, idxs_ds.size))


# ---------------------------------------------------------------------------
# Depth-first intervals
# ---------------------------------------------------------------------------

_INTERVAL_CHUNK = 1 << 24


def _basin_blocks(last, count, n):
    """The blocks of the basins in the upstream-to-downstream depth-first sequence.

    ``last`` holds the position of each basin's outlet in the sequence and ``count``
    its upstream cell count, the number of cells in the basin.  A depth-first
    traversal finishes one basin before it starts the next, so basin ``k`` takes
    the positions ``first[k]`` to ``last[k]``, its outlet last, and the blocks
    follow one another from 0 to ``n - 1``.  Returns ``(first, last)`` as int64,
    sorted by position; outlets that do not make such blocks raise a ``ValueError``.
    """
    last = np.asarray(last, dtype=np.int64).reshape(-1)
    count = np.asarray(count).reshape(-1)
    if last.size == 0:
        if n:
            raise ValueError(f"no outlets are given for a sequence of {n} cells")
        return last, last.copy()
    if int(last.min()) < 0:
        raise ValueError(
            f"{int(np.count_nonzero(last < 0))} outlets are off the network (seq_dfs -1): "
            f"seq_dfs and the outlets do not belong together"
        )
    # compared before the cast, which would wrap a huge unsigned upg
    if int(count.min()) < 1 or int(count.max()) > n:
        raise ValueError(
            f"{int(np.count_nonzero((count < 1) | (count > n)))} outlets have upg below 1 or above "
            f"{n}, the cells in the sequence: upg and the outlets do not belong together"
        )
    count = count.astype(np.int64)
    order = np.argsort(last, kind="stable")
    last, count = last[order], count[order]
    if last.size > 1 and not (np.diff(last) > 0).all():
        raise ValueError("an outlet is given twice")
    first = last - count + 1
    # each block starts one past the end of the one before it, from 0 to n - 1
    gap = np.flatnonzero(first[1:] != last[:-1] + 1)
    if first[0] != 0 or gap.size or last[-1] != n - 1:
        where = (f"position {int(last[gap[0]]) + 1}" if gap.size
                 else "position 0" if first[0] != 0 else f"position {n - 1}")
        raise ValueError(
            f"the {last.size} basins of these outlets do not take positions 0 to {n - 1} of seq_dfs "
            f"one after another (first mismatch near {where}): the outlets are not all the "
            f"outlets of this region, or seq_dfs, upg and the outlets do not belong together"
        )
    return first, last


def _new_seen(n):
    """One bit for each position of a sequence of ``n`` cells, all clear."""
    return np.zeros((n + 7) >> 3, dtype=np.uint8)


def _first_missing(seen, n):
    """The first of the ``n`` positions whose bit is clear in ``seen``, or ``-1``."""
    full = n >> 3
    clear = np.flatnonzero(seen[:full] != 0xFF)
    if clear.size:
        k = int(clear[0])
    elif n & 7 and int(seen[full]) != (1 << (n & 7)) - 1:
        k = full
    else:
        return -1
    free = ~int(seen[k]) & 0xFF
    return 8 * k + (free & -free).bit_length() - 1


def _intervals(seq, up, first, last, seen=None):
    """``dfs_in`` and ``dfs_out`` of the cells ``seq`` and ``up`` (any shape), in the
    basin blocks ``first`` and ``last`` of :func:`_basin_blocks`.  The position of
    each cell is marked in ``seen`` (:func:`_new_seen`) when it is given."""
    flat_seq = seq.reshape(-1)
    flat_up = up.reshape(-1)
    dfs_in = np.full(flat_seq.shape, -1, dtype=seq.dtype)
    dfs_out = np.full(flat_seq.shape, -1, dtype=seq.dtype)
    top = int(last[-1]) if last.size else -1
    # in chunks, so the int64 temporaries stay small on a region of two billion cells
    for start in range(0, flat_seq.size, _INTERVAL_CHUNK):
        part = slice(start, start + _INTERVAL_CHUNK)
        on = flat_seq[part] >= 0
        if not on.any():
            continue
        pos = flat_seq[part][on].astype(np.int64)
        count = flat_up[part][on].astype(np.int64)
        if int(pos.max()) > top:
            raise ValueError(f"seq_dfs holds a position past {top}, the last position of the outlets")
        basin = np.searchsorted(last, pos, side="left")
        lo, hi = first[basin], last[basin]
        # the cell and its upstream cells take positions pos - count + 1 to pos, which must
        # stay in the cell's basin; compared before any sum, which a huge upg would overflow
        bad = (count < 1) | (count > pos - lo + 1)
        if bad.any():
            raise ValueError(
                "some cells of the network have upg below 1 or more upstream cells than their "
                "basin holds before them: seq_dfs, upg and the outlets do not belong together"
            )
        if seen is not None:
            np.bitwise_or.at(seen, pos >> 3, np.left_shift(1, pos & 7).astype(np.uint8))
        start_in = lo + hi - pos
        dfs_in[part][on] = start_in
        dfs_out[part][on] = start_in + count - 1
    return dfs_in.reshape(seq.shape), dfs_out.reshape(seq.shape)


def _sequence_length(flat_seq):
    """``n``, the number of cells in a whole sequence; a sequence with gaps raises."""
    top, count = -1, 0
    for start in range(0, flat_seq.size, _INTERVAL_CHUNK):
        part = flat_seq[start:start + _INTERVAL_CHUNK]
        if part.size:
            top = max(top, int(part.max()))
            count += int(np.count_nonzero(part >= 0))
    if count != top + 1:
        raise ValueError(
            f"seq_dfs holds {count} cells of the network and positions up to {top}: "
            f"it is not the whole sequence of a region"
        )
    return top + 1


def dfs_interval(seq_dfs, upg, outlets):
    """``(dfs_in, dfs_out)``: the depth-first interval of every cell, numbered within
    its basin, from its position in the depth-first sequence, its upstream cell
    count and the outlets of the basins.

    ``seq_dfs`` holds at every cell its position in the upstream-to-downstream
    depth-first sequence of a whole region, ``-1`` off the network: the
    ``seq_dfs`` layer of FullTopo v1.1, or the inverse of
    ``topo.ordering("dfs", "u2d")``.  ``upg`` holds at every cell the number of
    cells that drain through it, the cell itself included: MERIT Hydro's ``upg``
    layer on the same cells.  ``outlets`` are the outlet cells of all the basins
    of the region (the cells that drain into themselves, or out of the region):
    flat indices into ``seq_dfs``, or a tuple of index arrays, one per axis, such
    as ``(rows, cols)``.  For a FullBasin v1.1 region they are the
    ``local_outlet_row`` and ``local_outlet_col`` columns of its basin table.

    The traversal finishes one basin before it starts the next, so each basin
    takes one block of positions, from ``a`` to ``b``: ``b`` is ``seq_dfs`` at the
    outlet, and ``a = b - upg + 1`` with ``upg`` at the outlet.  ``dfs_in`` is a
    cell's position in its basin's block read downstream to upstream,
    ``dfs_in = a + b - seq_dfs``, so the outlet has ``dfs_in = a``.  A
    depth-first traversal visits a cell and then all its upstream cells before
    any other cell, so these take the ``upg`` positions from ``dfs_in`` on and
    ``dfs_out = dfs_in + upg - 1`` is the last of them.  Cell ``q`` drains
    through cell ``p`` exactly when ``dfs_in[p] <= dfs_in[q] <= dfs_out[p]``, and
    ``dfs_out - dfs_in + 1`` is the upstream cell count.  The whole sequence read
    backward, ``n - 1 - seq_dfs`` (``topo.ordering("dfs", "d2u")``), differs from
    ``dfs_in`` by a constant within each basin and gives the same upstream cells.

    Both come back in the type and shape of ``seq_dfs``, ``-1`` off the network.
    Inputs that do not belong together raise a ``ValueError``: blocks that do not
    run from 0 to ``n - 1`` one after another (an outlet missing, or one that is
    not an outlet), a network cell with ``upg`` below 1, or a cell with more
    upstream cells than its basin holds before it (another grid, or a window
    shifted by some cells), or two cells at the same position, which leaves
    another position empty.  These checks cannot prove that ``seq_dfs`` is a
    depth-first order of the flow network, which needs the flow directions:
    ``seq_dfs`` and ``upg`` must come from the same flow network.
    """
    seq = np.asarray(seq_dfs)
    up = np.asarray(upg)
    if seq.dtype.kind != "i":
        raise ValueError("seq_dfs must be signed integers, -1 off the network")
    if up.dtype.kind not in "iu":
        raise ValueError("upg must be integer cell counts")
    if up.shape != seq.shape:
        raise ValueError(f"upg has shape {up.shape} and seq_dfs {seq.shape}")
    if isinstance(outlets, tuple):
        if len(outlets) != seq.ndim:
            raise ValueError(f"outlets has {len(outlets)} index arrays and seq_dfs {seq.ndim} axes")
        index = tuple(np.asarray(axis).reshape(-1) for axis in outlets)
        shape = seq.shape
    else:
        index = (np.asarray(outlets).reshape(-1),)
        shape = (seq.size,)
    for axis, size in zip(index, shape):
        if axis.size and axis.dtype.kind not in "iu":
            raise ValueError("outlets must be integer indices")
        if axis.size and (int(axis.min()) < 0 or int(axis.max()) >= size):
            raise ValueError("an outlet lies outside seq_dfs")
    if len({axis.size for axis in index}) > 1:
        raise ValueError("the index arrays of outlets differ in length")
    index = tuple(axis.astype(np.intp) for axis in index)
    flat = np.ravel_multi_index(index, shape) if len(index) > 1 else index[0]
    flat_seq = seq.reshape(-1)
    n = _sequence_length(flat_seq)
    first, last = _basin_blocks(flat_seq[flat], up.reshape(-1)[flat], n)
    seen = _new_seen(n)
    dfs_in, dfs_out = _intervals(seq, up, first, last, seen)
    missing = _first_missing(seen, n)
    if missing >= 0:
        raise ValueError(
            f"seq_dfs has no cell at position {missing} and another position twice: "
            f"it is not the whole sequence of a region"
        )
    return dfs_in, dfs_out


# ---------------------------------------------------------------------------
# Layerings
# ---------------------------------------------------------------------------


@njit(cache=True)
def _layering_asap(idxs_ds, n_up_in):
    size = idxs_ds.size
    n_up = n_up_in.copy()
    layers = np.full(size, LAYER_NODATA, dtype=np.int32)
    q_curr = np.empty(size, dtype=np.int32)
    q_next = np.empty(size, dtype=np.int32)

    order = 0
    curr = 0
    for idx in range(size):
        if idxs_ds[idx] != MV and n_up[idx] == 0:
            layers[idx] = order
            q_curr[curr] = idx
            curr += 1

    while curr > 0:
        order += 1
        nxt = 0
        for i in range(curr):
            idx = q_curr[i]
            idx_ds = idxs_ds[idx]
            if idx_ds < 0 or idx_ds >= size or idx_ds == idx:
                continue
            n_up[idx_ds] -= 1
            if n_up[idx_ds] == 0:
                layers[idx_ds] = order
                q_next[nxt] = idx_ds
                nxt += 1
        for j in range(nxt):
            q_curr[j] = q_next[j]
        curr = nxt
    return layers, order


@njit(cache=True)
def _layering_cfds(idxs_ds, n_up_in, ncells):
    size = idxs_ds.size
    n_up = n_up_in.copy()
    layers = np.full(size, LAYER_NODATA, dtype=np.int32)
    done_flg = np.zeros(size, dtype=np.uint8)
    flg_ds = np.zeros(size, dtype=np.uint8)
    q_a = np.empty(size, dtype=np.int32)
    q_b = np.empty(size, dtype=np.int32)
    occupied = np.empty(size, dtype=np.int32)

    curr_q = q_a
    next_q = q_b
    curr_sz = 0
    next_sz = 0

    for idx in range(size):
        if n_up[idx] == 0 and idxs_ds[idx] != MV:
            curr_q[curr_sz] = idx
            curr_sz += 1

    order = 0
    done = 0
    while done < ncells:
        if curr_sz == 0:
            # The queue drained with cells still unscheduled. What is left sits
            # in a cycle and has no layer that satisfies its dependencies. Leave those at LAYER_NODATA, the same thing the
            # other two layerings do. Sweeping them into one final layer would
            # put a cell and its own receiver in the same layer and break the
            # conflict-free promise this layering exists to keep.
            break

        occ_sz = 0
        for k in range(curr_sz):
            idx = curr_q[k]
            if done_flg[idx] != 0:
                continue
            idx_ds = idxs_ds[idx]

            if idx_ds == MV or idx_ds == idx:
                layers[idx] = order
                done_flg[idx] = 1
                done += 1
                continue
            if flg_ds[idx_ds] == 1:
                # receiver already claimed in this layer: wait for the next one
                next_q[next_sz] = idx
                next_sz += 1
                continue

            layers[idx] = order
            done_flg[idx] = 1
            done += 1
            flg_ds[idx_ds] = 1
            occupied[occ_sz] = idx_ds
            occ_sz += 1

            if n_up[idx_ds] > 0:
                n_up[idx_ds] -= 1
                if n_up[idx_ds] == 0 and done_flg[idx_ds] == 0:
                    next_q[next_sz] = idx_ds
                    next_sz += 1

        for i in range(occ_sz):
            flg_ds[occupied[i]] = 0

        tmp = curr_q
        curr_q = next_q
        next_q = tmp
        curr_sz = next_sz
        next_sz = 0
        order += 1
    return layers, order


def layering_asap(idxs_ds):
    """As-soon-as-possible layering.

    Every headwater lands in layer 0 and every other cell one layer beyond its
    last-scheduled donor, so each cell gets the earliest layer its dependencies
    allow and most cells pile into the first layers.  Hits the minimum layer
    count set by the longest flow path.

    Returns
    -------
    layers : ndarray of int32
        Layer index per cell.  ``-1`` means the cell has no layer, for one of
        two reasons: it is outside the network, or it sits in a cycle, which
        no layering can schedule (a cell that drains into a cycle
        without being in one is scheduled).
        ``(layers < 0) & mask`` counts the second kind and is how you detect a
        cycle.
    nlayers : int
    """
    idxs_ds = _checked_downstream(idxs_ds)
    n_up, _ = upstream_count(idxs_ds)
    return _layering_asap(idxs_ds, n_up)


def layering_cfds(idxs_ds):
    """Conflict-free downstream layering.

    As-soon-as-possible with one extra rule: if a ready cell's receiver has
    already been claimed by another cell in the current layer, the cell waits
    for the next layer.  No two cells of a layer then write to the same
    receiver, so a scatter-add over a layer touches each target exactly once.

    That is what makes plain push safe without atomics, and it is the only one
    of the three that makes Strahler stream order safe under push at all: the
    confluence rule is a comparison, not an addition, so no atomic can fix it.
    It can take more layers than the plain form: a held-back cell can hold
    back the next, confluence after confluence (none on the example basin).

    Cells caught in a cycle cannot be scheduled by their dependencies and
    keep ``-1``, as in the as-soon-as-possible form
    (``layering_cfds([1, 0])`` gives ``[-1, -1]``).  A cell
    that drains into a cycle without being in one is still placed, as the
    as-soon-as-possible form places it.

    Returns
    -------
    layers : ndarray of int32
        Layer index per cell.  ``-1`` means the cell has no layer, for one of
        two reasons: it is outside the network, or it sits in a cycle, which
        no layering can schedule.  ``(layers < 0) & mask`` counts the second
        kind and is how you detect a cycle.
    nlayers : int
    """
    idxs_ds = _checked_downstream(idxs_ds)
    n_up, _ = upstream_count(idxs_ds)
    ncells = int(np.count_nonzero(idxs_ds != MV))
    return _layering_cfds(idxs_ds, n_up, ncells)


def layering_alap(idxs_ds, bsn=None):
    """As-late-as-possible layering.

    Per basin, ``layer = max_rank_in_basin - rank``, so short tributaries are
    held back to later layers instead of crowding into the first ones.  Hits
    the same minimum layer count as the as-soon-as-possible form but spreads
    the work differently, which changes the memory-access pattern.

    Returns
    -------
    layers : ndarray of int32
        Layer index per cell.  ``-1`` means the cell has no layer, for one of
        two reasons: it is outside the network, or it sits in a cycle or
        drains into one, which this layering does not schedule (the
        as-soon-as-possible and conflict-free layerings do schedule a cell
        that drains into a cycle, not one in it).  ``(layers < 0) & mask``
        counts the second kind and is how you detect a cycle.
    nlayers : int
    """
    idxs_ds = _checked_downstream(idxs_ds)
    if bsn is None:
        bsn, _ = basin_labels(idxs_ds)
    # the basin labels checked on the type they came in, one per cell, whole and within uint32, before they
    # are made uint32: an int64 2**32 wrapped to 0 and took the whole network out
    given_bsn = np.asarray(bsn).reshape(-1)
    if given_bsn.size != idxs_ds.size or (given_bsn.size and given_bsn.dtype.kind not in "iu"):
        raise ValueError("bsn must hold one whole basin label per cell")
    if given_bsn.size and (int(given_bsn.min()) < 0 or int(given_bsn.max()) > 4294967295):
        raise ValueError("a basin label is outside 0 .. 2**32 - 1")
    bsn = np.ascontiguousarray(given_bsn, dtype=np.uint32)

    rnk = rank_to_pit(idxs_ds)
    max_bsn = int(bsn.max()) if bsn.size else 0
    if max_bsn == 0:
        # Nothing reaches a pit: an empty grid, or a network that is all cycle.
        # The other two layerings return an empty layering here rather than
        # failing, so this one does too.
        return np.full(idxs_ds.size, LAYER_NODATA, dtype=np.int32), 0

    max_rnk = np.zeros(max_bsn + 1, dtype=np.int64)
    inside = bsn != 0
    np.maximum.at(max_rnk, bsn[inside].astype(np.int64), rnk[inside].astype(np.int64))

    layers = np.full(idxs_ds.size, LAYER_NODATA, dtype=np.int32)
    ok = inside & (rnk >= 0)
    layers[ok] = (max_rnk[bsn[ok].astype(np.int64)] - rnk[ok]).astype(np.int32)
    nlayers = int(layers.max()) + 1 if np.any(ok) else 0
    return layers, nlayers

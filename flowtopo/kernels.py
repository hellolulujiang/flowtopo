"""Four flow-network kernels, each under the propagation manners it supports.

=====================  ====================================================
upstream drainage area many-to-one accumulation, linear
flow length downstream one-to-one propagation
flow length upstream   many-to-one maximum, linear
Strahler stream order  many-to-one, non-linear
=====================  ====================================================

Three manners:

``serial``
    Walk a topological ordering one cell at a time.

``pull``
    Per layer, every cell gathers from its donors.  Each cell writes only to
    itself, so nothing can collide.  The cost is building and storing the
    upstream adjacency table.

``push``
    Per layer, every cell writes into its receiver, vectorised with fancy
    indexing.  **In numpy ``a[idx] += v`` keeps only the last write when
    ``idx`` repeats**, which is exactly what a non-atomic parallel push does
    when two donors of one receiver land in the same layer.  Under the
    conflict-free downstream layering no layer ever repeats a receiver, so
    this is correct *and* it is the fastest manner here.  Under the
    as-soon-as-possible or as-late-as-possible layering it silently loses
    writes.

``atomic_push``
    The same scatter, but through ``np.add.at`` / ``np.maximum.at``, which
    handle repeated indices correctly.  Always right, and much slower than
    plain fancy indexing.

That last pair is the practical point of the conflict-free layering in a numpy
setting: it buys the correctness of the slow scatter at the speed of the fast
one, without threads and without atomics.  Strahler order cannot be rescued by
an atomic at all -- its confluence rule is a comparison and a count, not an
addition -- so the layering is the only way to run it under push.
"""

import numpy as np

from ._compat import njit

UPA_NODATA = np.float32(-9999.0)
FLN_NODATA = np.float32(-9999.0)

MANNERS = ("serial", "pull", "push", "atomic_push")


def _check_manner(manner, allowed):
    if manner not in allowed:
        raise ValueError(
            f"manner {manner!r} is not available here; use one of {allowed}"
        )


# ---------------------------------------------------------------------------
# upstream drainage area
# ---------------------------------------------------------------------------


@njit(cache=True)
def _upa_serial(idxs_ds, seq_u2d, upa, nodata):
    for k in range(seq_u2d.size):
        idx = seq_u2d[k]
        idx_ds = idxs_ds[idx]
        if idx_ds < 0 or idx_ds == idx:
            continue
        v = upa[idx]
        if v == nodata:
            continue
        if upa[idx_ds] == nodata:
            continue
        upa[idx_ds] += v


def upstream_area(idxs_ds, cell_area, *, seq_u2d=None, decomp=None,
                  us_table=None, n_up=None, manner="serial",
                  nodata=UPA_NODATA, mask=None):
    """Accumulate cell area downstream.

    Parameters
    ----------
    idxs_ds : ndarray of int32
    cell_area : ndarray
        Per-cell area; the accumulator starts from this, and keeps its
        precision. Pass a float64 array on a large network: accumulating in
        float32 stops resolving one 90 m cell (about 0.007 km2) once the
        running total passes roughly 1e5 km2, so a continental basin loses the
        contribution of individual cells near its outlet.
    seq_u2d : ndarray, optional
        Upstream-to-downstream ordering, required for ``manner="serial"``.
    decomp : Decomposition, optional
        Layering, required for every parallel manner.  Must be a u2d layering.
    us_table, n_up : optional
        Upstream adjacency, required for ``manner="pull"``.
    mask : ndarray of bool, optional
        Cells outside the mask are set to ``nodata`` and never accumulate.

    Returns
    -------
    upa : ndarray
        float32 when ``cell_area`` is float32, float64 otherwise.  Integer
        input accumulates in float64 so that counting cells stays exact.
    """
    _check_manner(manner, MANNERS)
    idxs_ds = np.ascontiguousarray(idxs_ds, dtype=np.int32)
    # float32 only when that is what came in. Anything else accumulates in
    # float64: an integer cell_area, the natural way to count upstream cells,
    # stops being exact above 2**24 in float32, which a 90 m network passes
    # long before it runs out of cells.
    dtype = np.float32 if np.asarray(cell_area).dtype == np.float32 else np.float64
    upa = np.ascontiguousarray(cell_area, dtype=dtype).copy()
    nodata = dtype(nodata)
    zero = dtype(0.0)
    if mask is not None:
        upa[~np.asarray(mask, dtype=bool)] = nodata

    if manner == "serial":
        if seq_u2d is None:
            raise ValueError("manner='serial' needs seq_u2d")
        _upa_serial(idxs_ds, np.ascontiguousarray(seq_u2d, dtype=np.int32),
                    upa, nodata)
        return upa

    if decomp is None:
        raise ValueError(f"manner={manner!r} needs a layering decomposition")

    if manner == "pull":
        if us_table is None or n_up is None:
            raise ValueError("manner='pull' needs us_table and n_up")
        max_donors = us_table.shape[1]
        for idx in decomp:
            cur = upa[idx]
            alive = cur != nodata
            total = cur.copy()
            for k in range(max_donors):
                u = us_table[idx, k]
                has = u >= 0
                if not has.any():
                    continue
                val = upa[np.where(has, u, 0)]
                total += np.where(has & (val != nodata), val, zero)
            upa[idx] = np.where(alive, total, cur)
        return upa

    for idx in decomp:
        ds = idxs_ds[idx]
        keep = (ds >= 0) & (ds != idx)
        i, d = idx[keep], ds[keep]
        if i.size == 0:
            continue
        v = upa[i]
        keep = (v != nodata) & (upa[d] != nodata)
        i, d, v = i[keep], d[keep], v[keep]
        if manner == "push":
            # repeated receivers in d lose all but the last write
            upa[d] += v
        else:
            np.add.at(upa, d, v)
    return upa


# ---------------------------------------------------------------------------
# flow length downstream (distance to outlet)
# ---------------------------------------------------------------------------


@njit(cache=True)
def _ldn_serial(idxs_ds, seq_d2u, plen, dist, mv):
    for k in range(seq_d2u.size):
        idx = seq_d2u[k]
        idx_ds = idxs_ds[idx]
        if idx_ds < 0 or idx_ds == idx:
            continue
        dds = dist[idx_ds]
        seg = plen[idx]
        if dds == mv or seg == mv:
            continue
        dist[idx] = dds + seg


def distance_to_outlet(idxs_ds, plen, *, seq_d2u=None, decomp=None,
                       manner="serial", nodata=FLN_NODATA, mask=None):
    """Cumulative along-stream distance from each cell down to its pit.

    One-to-one propagation: every cell writes only to itself, so the parallel
    form is safe under any layering.  ``decomp`` must be a *downstream to
    upstream* layering, since a receiver has to be visited before its donors.
    """
    _check_manner(manner, ("serial", "push", "atomic_push", "pull"))
    idxs_ds = np.ascontiguousarray(idxs_ds, dtype=np.int32)
    plen = np.ascontiguousarray(plen, dtype=np.float32)

    dist = np.zeros(idxs_ds.size, dtype=np.float32)
    if mask is not None:
        dist[~np.asarray(mask, dtype=bool)] = nodata

    if manner == "serial":
        if seq_d2u is None:
            raise ValueError("manner='serial' needs seq_d2u")
        _ldn_serial(idxs_ds, np.ascontiguousarray(seq_d2u, dtype=np.int32),
                    plen, dist, np.float32(nodata))
        return dist

    if decomp is None:
        raise ValueError("the parallel form needs a d2u layering decomposition")

    for idx in decomp:
        ds = idxs_ds[idx]
        keep = (ds >= 0) & (ds != idx)
        i, d = idx[keep], ds[keep]
        if i.size == 0:
            continue
        dds, seg = dist[d], plen[i]
        keep = (dds != nodata) & (seg != nodata)
        dist[i[keep]] = dds[keep] + seg[keep]
    return dist


# ---------------------------------------------------------------------------
# flow length upstream (longest upstream path)
# ---------------------------------------------------------------------------


@njit(cache=True)
def _lup_serial(idxs_ds, seq_u2d, height, mv):
    for k in range(seq_u2d.size):
        idx = seq_u2d[k]
        hu = height[idx]
        if hu == mv:
            continue
        ds = idxs_ds[idx]
        if ds < 0 or ds == idx:
            continue
        hds = height[ds]
        if hds == mv or hu > hds:
            height[ds] = hu


def _lup_finalise(height, ldn, mv):
    out = np.full(height.size, mv, dtype=np.float32)
    ok = (height != mv) & (ldn != mv)
    val = height[ok] - ldn[ok]
    val = np.where((val < 0.0) & (val > -1e-4), np.float32(0.0), val)
    out[ok] = val.astype(np.float32)
    return out


def longest_upstream_path(idxs_ds, ldn, *, seq_u2d=None, decomp=None,
                          us_table=None, n_up=None, manner="serial",
                          nodata=FLN_NODATA, mask=None):
    """Distance from each cell up to the furthest headwater above it.

    Propagates the maximum distance-to-outlet seen upstream, then subtracts the
    cell's own distance to outlet.  ``ldn`` is the output of
    :func:`distance_to_outlet`.
    """
    _check_manner(manner, MANNERS)
    idxs_ds = np.ascontiguousarray(idxs_ds, dtype=np.int32)
    ldn = np.ascontiguousarray(ldn, dtype=np.float32)
    mv = np.float32(nodata)

    height = ldn.copy()
    if mask is not None:
        height[~np.asarray(mask, dtype=bool)] = mv

    if manner == "serial":
        if seq_u2d is None:
            raise ValueError("manner='serial' needs seq_u2d")
        _lup_serial(idxs_ds, np.ascontiguousarray(seq_u2d, dtype=np.int32),
                    height, mv)
        return _lup_finalise(height, ldn, mv)

    if decomp is None:
        raise ValueError(f"manner={manner!r} needs a layering decomposition")

    if manner == "pull":
        if us_table is None or n_up is None:
            raise ValueError("manner='pull' needs us_table and n_up")
        max_donors = us_table.shape[1]
        for idx in decomp:
            best = height[idx]
            alive = best != mv
            acc = best.copy()
            for k in range(max_donors):
                u = us_table[idx, k]
                has = u >= 0
                if not has.any():
                    continue
                hu = height[np.where(has, u, 0)]
                take = has & (hu != mv) & (hu > acc)
                acc = np.where(take, hu, acc)
            height[idx] = np.where(alive, acc, best)
        return _lup_finalise(height, ldn, mv)

    for idx in decomp:
        hu = height[idx]
        ds = idxs_ds[idx]
        keep = (hu != mv) & (ds >= 0) & (ds != idx)
        i, d, hu = idx[keep], ds[keep], hu[keep]
        if i.size == 0:
            continue
        if manner == "push":
            hds = height[d]
            take = (hds == mv) | (hu > hds)
            height[d[take]] = hu[take]
        else:
            np.maximum.at(height, d, hu)
    return _lup_finalise(height, ldn, mv)


# ---------------------------------------------------------------------------
# Strahler stream order
# ---------------------------------------------------------------------------


@njit(cache=True)
def _strahler_serial(idxs_ds, seq_u2d, strord, strmax):
    for t in range(seq_u2d.size):
        idx0 = seq_u2d[t]
        if idx0 < 0:
            continue
        if strord[idx0] == 0:
            strord[idx0] = 1
        idx_ds = idxs_ds[idx0]
        if idx_ds < 0 or idx_ds == idx0:
            continue
        sto = strord[idx0]
        sto_ds = strord[idx_ds]
        sto_up = strmax[idx_ds]
        if sto_ds < sto:
            strord[idx_ds] = sto
        elif sto == sto_ds and sto_up == sto:
            if strord[idx_ds] < 255:
                strord[idx_ds] += 1
        if sto_up < sto:
            strmax[idx_ds] = sto


def strahler_order(idxs_ds, *, seq_u2d=None, decomp=None, us_table=None,
                   n_up=None, manner="serial", mask=None):
    """Strahler stream order.

    Two branches of equal order meeting raises the order by one; otherwise the
    largest incoming order carries through.  That rule is a comparison and a
    count, not an addition, so there is no atomic that makes a parallel push
    safe.  ``manner="push"`` is therefore correct **only** under the
    conflict-free downstream layering, which is the point the companion paper
    makes about it.

    ``mask`` restricts the computation to the channel network; cells outside it
    keep order 0.
    """
    _check_manner(manner, ("serial", "pull", "push"))
    idxs_ds = np.ascontiguousarray(idxs_ds, dtype=np.int32)
    size = idxs_ds.size
    strord = np.zeros(size, dtype=np.uint8)
    strmax = np.zeros(size, dtype=np.uint8)
    msk = None if mask is None else np.asarray(mask, dtype=bool)

    if manner == "serial":
        if seq_u2d is None:
            raise ValueError("manner='serial' needs seq_u2d")
        seq = np.ascontiguousarray(seq_u2d, dtype=np.int32)
        if msk is not None:
            seq = np.ascontiguousarray(seq[msk[seq]], dtype=np.int32)
        _strahler_serial(idxs_ds, seq, strord, strmax)
        return strord

    if decomp is None:
        raise ValueError(f"manner={manner!r} needs a layering decomposition")

    if manner == "pull":
        if us_table is None or n_up is None:
            raise ValueError("manner='pull' needs us_table and n_up")
        max_donors = us_table.shape[1]
        for idx in decomp:
            if msk is not None:
                idx = idx[msk[idx]]
                if idx.size == 0:
                    continue
            max_up = np.zeros(idx.size, dtype=np.int16)
            cnt_max = np.zeros(idx.size, dtype=np.int16)
            for k in range(max_donors):
                u = us_table[idx, k]
                has = u >= 0
                if msk is not None:
                    has = has & msk[np.where(has, u, 0)]
                if not has.any():
                    continue
                su = strord[np.where(has, u, 0)].astype(np.int16)
                su = np.where(su == 0, np.int16(1), su)
                gt = has & (su > max_up)
                eq = has & (su == max_up) & ~gt
                cnt_max = np.where(gt, np.int16(1),
                                   np.where(eq, np.minimum(cnt_max + 1, 255),
                                            cnt_max))
                max_up = np.where(gt, su, max_up)

            deg = n_up[idx].astype(np.int32)
            own = strord[idx].astype(np.int16)
            head = (deg <= 0) | (max_up == 0)
            confluence = np.minimum(max_up + 1, 255)
            value = np.where(
                head,
                np.where(own == 0, np.int16(1), own),
                np.where(cnt_max >= 2, confluence, max_up),
            )
            strord[idx] = value.astype(np.uint8)
            strmax[idx] = max_up.astype(np.uint8)
        return strord

    # push: exclusive only under the conflict-free downstream layering
    for idx in decomp:
        if msk is not None:
            idx = idx[msk[idx]]
            if idx.size == 0:
                continue
        own = strord[idx]
        strord[idx] = np.where(own == 0, np.uint8(1), own)

        ds = idxs_ds[idx]
        keep = (ds >= 0) & (ds != idx)
        if msk is not None:
            keep = keep & msk[np.where(keep, ds, 0)]
        i, d = idx[keep], ds[keep]
        if i.size == 0:
            continue

        sto = strord[i].astype(np.int16)
        sto_ds = strord[d].astype(np.int16)
        sto_up = strmax[d].astype(np.int16)

        value = np.where(
            sto_ds < sto,
            sto,
            np.where((sto == sto_ds) & (sto_up == sto),
                     np.minimum(sto_ds + 1, 255), sto_ds),
        )
        strord[d] = value.astype(np.uint8)
        strmax[d] = np.where(sto_up < sto, sto, sto_up).astype(np.uint8)
    return strord

"""Split a flow network into subregions that can run on separate processors.

Both strategies cut along the drainage hierarchy, so no value has to cross a
subregion boundary while a kernel runs.

``basin``
    Whole basins are assigned to subregions, each weighted by its cell count.
    Simple, but it balances badly when one basin dominates: a basin cannot be
    split, so one subregion sits nearly idle.

``subbasin``
    A basin too large for one subregion is decomposed along its mainstem.
    Walking upstream from the outlet and taking the larger tributary at each
    confluence isolates the mainstem; the tributary subtrees hanging off it
    become items that can be dealt to the lighter subregions. The mainstem
    itself depends on those subtrees, so it runs in a second stage, after they
    finish.

Assignment uses longest-processing-time-first: items are placed largest first,
each into the lightest subregion so far.
"""

import numpy as np

from .core import seq_dfs_from_pit

MAINSTEM = -2
"""Marker for cells held back to the second stage."""


def _upstream_cells(idxs_ds, mask):
    """Number of cells draining into each cell, itself included."""
    seq = seq_dfs_from_pit(idxs_ds)[::-1]          # upstream to downstream
    count = np.where(mask, 1.0, 0.0).astype(np.float64)
    for idx in seq:
        ds = idxs_ds[idx]
        if ds >= 0 and ds != idx and mask[idx] and mask[ds]:
            count[ds] += count[idx]
    return count


def _lpt(weights, n_parts):
    """Longest-processing-time-first assignment; returns a part per item."""
    order = np.argsort(-np.asarray(weights, dtype=np.float64))
    load = np.zeros(n_parts, dtype=np.float64)
    part = np.empty(len(weights), dtype=np.int32)
    for item in order:
        lightest = int(np.argmin(load))
        part[item] = lightest
        load[lightest] += weights[item]
    return part, load


def _mainstem(idxs_ds, pit, upstream, donors):
    """Cells from a pit up to the head, taking the larger tributary each time."""
    stem = [pit]
    cell = pit
    while True:
        ups = donors.get(cell)
        if not ups:
            break
        cell = max(ups, key=lambda u: upstream[u])
        stem.append(cell)
    return stem


def _donor_map(idxs_ds, cells):
    donors = {}
    for idx in cells:
        ds = int(idxs_ds[idx])
        if ds >= 0 and ds != idx:
            donors.setdefault(ds, []).append(int(idx))
    return donors


def partition(topo, n_parts=4, level="subbasin"):
    """Assign every valid cell to one of ``n_parts`` subregions.

    Parameters
    ----------
    topo : FlowTopo
    n_parts : int
        Number of subregions, normally the number of processors.
    level : {"basin", "subbasin"}
        ``"basin"`` keeps every basin whole. ``"subbasin"`` decomposes any
        basin larger than one subregion's share along its mainstem.

    Returns
    -------
    part : ndarray of int32
        Subregion index per cell; ``-1`` outside the network, and ``-2`` on a
        mainstem held back to the second stage (``subbasin`` only).
    load : ndarray of float64
        Cells assigned to each subregion, the mainstem cells excluded.
    """
    if level not in ("basin", "subbasin"):
        raise ValueError("level must be 'basin' or 'subbasin'")
    if n_parts < 1:
        raise ValueError("n_parts must be at least 1")

    idxs_ds = topo.idxs_ds
    mask = topo.mask
    basins = topo.basins
    part = np.full(idxs_ds.size, -1, dtype=np.int32)

    labels = np.unique(basins[mask])
    sizes = {int(b): int(np.count_nonzero(basins == b)) for b in labels}

    if level == "basin":
        items = list(sizes)
        assign, load = _lpt([sizes[b] for b in items], n_parts)
        for item, basin in enumerate(items):
            part[(basins == basin) & mask] = assign[item]
        return part, load

    # subbasin: decompose any basin that will not fit in one subregion
    target = sum(sizes.values()) / float(n_parts)
    upstream = _upstream_cells(idxs_ds, mask)

    items, members = [], []
    for basin, size in sizes.items():
        cells = np.nonzero((basins == basin) & mask)[0]
        if size <= target or cells.size < 3:
            items.append(size)
            members.append(cells)
            continue

        pit = int(cells[np.argmax(upstream[cells])])
        donors = _donor_map(idxs_ds, cells)
        stem = _mainstem(idxs_ds, pit, upstream, donors)
        on_stem = np.zeros(idxs_ds.size, dtype=bool)
        on_stem[stem] = True
        part[stem] = MAINSTEM

        # every cell inherits the subtree of the cell it drains into
        root = np.full(idxs_ds.size, -1, dtype=np.int64)
        for idx in seq_dfs_from_pit(idxs_ds):        # receivers before donors
            if not mask[idx] or basins[idx] != basin or on_stem[idx]:
                continue
            ds = int(idxs_ds[idx])
            root[idx] = idx if on_stem[ds] else root[ds]

        for subtree in np.unique(root[root >= 0]):
            cells = np.nonzero(root == subtree)[0]
            items.append(cells.size)
            members.append(cells)

    assign, load = _lpt(items, n_parts)
    for item, cells in enumerate(members):
        part[cells] = assign[item]
    return part, load

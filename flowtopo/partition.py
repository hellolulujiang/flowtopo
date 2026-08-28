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

from ._compat import njit
from .core import seq_dfs_from_pit
from .kernels import upstream_area

MAINSTEM = -2
"""Marker for cells held back to the second stage."""


def _basin_sizes(basins, mask):
    """Cells per basin label, without grouping the cells themselves."""
    counts = np.bincount(basins[mask])
    labels = np.nonzero(counts)[0]
    return labels, counts


def _lpt(weights, n_parts):
    """Longest-processing-time-first assignment; returns a part per item."""
    weights = np.asarray(weights, dtype=np.float64)
    load = np.zeros(n_parts, dtype=np.float64)
    part = np.empty(weights.size, dtype=np.int32)
    for item in np.argsort(-weights):
        lightest = int(np.argmin(load))
        part[item] = lightest
        load[lightest] += weights[item]
    return part, load


def _mainstem(idxs_ds, pit, upstream, us_table, n_up):
    """Cells from a pit up to the head, taking the larger tributary each time."""
    stem = [int(pit)]
    seen = {int(pit)}
    cell = int(pit)
    while True:
        donors = us_table[cell][: max(int(n_up[cell]), 0)]
        donors = donors[donors >= 0]
        if donors.size == 0:
            break
        cell = int(donors[np.argmax(upstream[donors])])
        if cell in seen:          # a cycle: walking on would never end
            break
        seen.add(cell)
        stem.append(cell)
    return np.array(stem, dtype=np.int64)


@njit(cache=True)
def _label_subtrees(idxs_ds, seq_d2u, on_stem, eligible, root):
    """Every cell inherits the subtree of the cell it drains into.

    ``seq_d2u`` visits a receiver before its donors, so one pass is enough.
    A cell draining straight onto the mainstem roots its own subtree.
    """
    for k in range(seq_d2u.size):
        idx = seq_d2u[k]
        if eligible[idx] == 0 or on_stem[idx] == 1:
            continue
        ds = idxs_ds[idx]
        if ds < 0 or ds == idx:
            continue
        root[idx] = idx if on_stem[ds] == 1 else root[ds]


def partition(topo, n_parts=4, level="subbasin"):
    """Assign every valid cell to one of ``n_parts`` subregions.

    Parameters
    ----------
    topo : FlowTopo
    n_parts : int
        Number of subregions. Set it to the number of processors, one subregion
        each, so a subregion's working set stays in that processor's own
        memory; threads then divide the work inside a subregion.
    level : {"basin", "subbasin"}
        ``"basin"`` keeps every basin whole. ``"subbasin"`` decomposes any
        basin larger than one subregion's share along its mainstem.

    Returns
    -------
    part : ndarray of int32
        Subregion index per cell; ``-1`` outside the network, and ``-2``
        (:data:`MAINSTEM`) on a mainstem held back to the second stage.
    load : ndarray of float64
        Work in each subregion, counted in cells.  Excludes the mainstem cells
        and the cells caught in a cycle, neither of which a kernel processes
        in this stage, so ``load.sum()`` is below the number of valid cells
        whenever either is present.

    Notes
    -----
    Cells caught in a cycle have no place in the drainage hierarchy. They get
    a subregion like any other cell, so that every cell has one, but they do
    not count towards its load: a sequence skips them and a layering leaves
    them unnumbered, so nothing is computed for them.
    """
    if level not in ("basin", "subbasin"):
        raise ValueError("level must be 'basin' or 'subbasin'")
    if n_parts < 1:
        raise ValueError("n_parts must be at least 1")

    idxs_ds = topo.idxs_ds
    mask = topo.mask
    part = np.full(idxs_ds.size, -1, dtype=np.int32)
    if not mask.any():
        return part, np.zeros(n_parts, dtype=np.float64)

    basins = topo.basins
    labels, counts = _basin_sizes(basins, mask)
    # Label 0 is not a basin: it collects the cells that never reach a pit,
    # the ones caught in a cycle. They are assigned like any other cell but
    # must never be decomposed, since a mainstem walk through a cycle does
    # not terminate.
    labels = labels[labels != 0]
    sizes = counts[labels]
    if labels.size == 0:
        # Every cell is label 0, so nothing reaches a pit. They all go to one
        # subregion to keep the cover complete, and its load stays zero:
        # there is no work here for any kernel to do.
        part[mask] = 0
        return part, np.zeros(n_parts, dtype=np.float64)

    def by_whole_basin():
        """Assign each basin as one item, then look the answer up per cell."""
        assign, load = _lpt(sizes, n_parts)
        lookup = np.full(counts.size, -1, dtype=np.int32)
        lookup[labels] = assign
        part[mask] = lookup[basins[mask]]
        stranded = mask & (part == -1)
        if stranded.any():
            # Cells that reach no pit. They need a subregion so that every
            # cell has one, but no kernel will process them: a sequence skips
            # them and a layering leaves them unnumbered. Keeping them out of
            # load is what makes load the amount of work in a subregion.
            part[stranded] = int(np.argmin(load))
        return part, load

    if level == "basin":
        return by_whole_basin()

    # ---- subbasin: decompose any basin that will not fit in one subregion --
    target = sizes.sum() / float(n_parts)
    big = labels[(sizes > target) & (sizes >= 3)]
    if big.size == 0:
        return by_whole_basin()

    seq_d2u = seq_dfs_from_pit(idxs_ds)
    ones = np.ones(idxs_ds.size, dtype=np.float32)
    upstream = upstream_area(idxs_ds, ones, seq_u2d=seq_d2u[::-1],
                             manner="serial", mask=mask)
    us_table, n_up = topo.upstream()

    decomposed = np.isin(basins, big) & mask
    on_stem = np.zeros(idxs_ds.size, dtype=np.uint8)
    for basin in big:
        members = np.nonzero(basins == basin)[0]
        pit = members[np.argmax(upstream[members])]
        on_stem[_mainstem(idxs_ds, pit, upstream, us_table, n_up)] = 1

    part[on_stem == 1] = MAINSTEM

    root = np.full(idxs_ds.size, -1, dtype=np.int64)
    _label_subtrees(idxs_ds, np.ascontiguousarray(seq_d2u, dtype=np.int32),
                    on_stem, decomposed.view(np.uint8), root)

    # items: every basin left whole, then every tributary subtree
    whole = labels[~np.isin(labels, big)]
    items = list(counts[whole])
    members = [("basin", b) for b in whole]

    labelled = np.nonzero(root >= 0)[0]
    if labelled.size:
        keys = root[labelled]
        order = np.argsort(keys, kind="stable")
        labelled, keys = labelled[order], keys[order]
        _, starts = np.unique(keys, return_index=True)
        starts = np.append(starts, labelled.size)
        for k in range(starts.size - 1):
            group = labelled[starts[k] : starts[k + 1]]
            items.append(group.size)
            members.append(("cells", group))

    assign, load = _lpt(items, n_parts)
    lookup = np.full(counts.size, -1, dtype=np.int32)
    for item, (kind, value) in enumerate(members):
        if kind == "basin":
            lookup[value] = assign[item]
        else:
            part[value] = assign[item]
    keep = mask & ~decomposed
    part[keep] = lookup[basins[keep]]
    # Cells that reach no pit carry label 0, which has no entry in the
    # assignment; give them the lightest subregion rather than leaving them
    # out. They do not count towards load, since no kernel processes them.
    stranded = mask & (part == -1)
    if stranded.any():
        part[stranded] = int(np.argmin(load))
    return part, load

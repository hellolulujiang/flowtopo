"""What a structure is, checked rather than assumed.

An ordering is a topological sort; a layering's members are mutually
independent; a decomposition covers every cell once and keeps the layers in
dependency order. These are the properties every kernel relies on, so they are
asserted directly instead of being inferred from a kernel's result.
"""

import numpy as np
import pytest

import flowtopo
from flowtopo.layering import reverse_layers
from flowtopo.synthetic import synthetic_d8

TRANSFORM = (0.0, 1 / 1200, 0.0, 35.0, 0.0, -1 / 1200)
ORDERINGS = ["dfs", "bfs", "topo"]
LAYERINGS = ["asap", "cfds", "alap"]


@pytest.fixture(scope="module", params=[1, 6])
def topo(request):
    return flowtopo.FlowTopo.from_d8(synthetic_d8(120, seed=request.param),
                                     transform=TRANSFORM)


def _layer_of(topo, decomp):
    layer = np.full(topo.idxs_ds.size, -1, dtype=np.int64)
    for index in range(decomp.nlayers):
        layer[decomp.layer(index)] = index
    return layer


# ---------------------------------------------------------------------------
# Orderings
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ORDERINGS)
def test_an_ordering_is_a_topological_sort(topo, name):
    """Every cell comes after the cell it drains into, walking d2u."""
    sequence = topo.ordering(name, "d2u")
    position = np.full(topo.idxs_ds.size, -1, dtype=np.int64)
    position[sequence] = np.arange(sequence.size)

    cells = np.asarray(sequence, dtype=np.int64)
    receivers = topo.idxs_ds[cells]
    moving = (receivers >= 0) & (receivers != cells)
    assert np.all(position[receivers[moving]] < position[cells[moving]])


@pytest.mark.parametrize("name", ORDERINGS)
def test_the_two_directions_are_reverses(topo, name):
    assert np.array_equal(topo.ordering(name, "d2u"),
                          topo.ordering(name, "u2d")[::-1])


@pytest.mark.parametrize("name", ORDERINGS)
def test_an_ordering_holds_every_valid_cell_once(topo, name):
    sequence = topo.ordering(name, "d2u")
    assert sequence.size == topo.ncells
    assert np.unique(sequence).size == sequence.size
    assert topo.mask[sequence].all()


# ---------------------------------------------------------------------------
# Layerings
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", LAYERINGS)
def test_a_layer_is_an_antichain(topo, name):
    """No cell in a layer drains into another cell of the same layer."""
    decomp = topo.decomposition(name, "u2d")
    layer = _layer_of(topo, decomp)

    cells = decomp.cells.astype(np.int64)
    receivers = topo.idxs_ds[cells]
    moving = (receivers >= 0) & (receivers != cells)
    cells, receivers = cells[moving], receivers[moving]
    assert not np.any(layer[receivers] == layer[cells])


@pytest.mark.parametrize("name", LAYERINGS)
def test_a_receiver_comes_after_its_donors(topo, name):
    decomp = topo.decomposition(name, "u2d")
    layer = _layer_of(topo, decomp)

    cells = decomp.cells.astype(np.int64)
    receivers = topo.idxs_ds[cells]
    moving = (receivers >= 0) & (receivers != cells)
    assert np.all(layer[receivers[moving]] > layer[cells[moving]])


@pytest.mark.parametrize("name", LAYERINGS)
def test_layer_zero_holds_the_headwaters(topo, name):
    n_up, _ = flowtopo.upstream_count(topo.idxs_ds)
    assert np.all(n_up[topo.decomposition(name, "u2d").layer(0)] <= 0)


def test_only_the_conflict_free_layering_promises_exclusive_receivers(topo):
    counts = {}
    for name in LAYERINGS:
        total = 0
        for members in topo.decomposition(name, "u2d"):
            receivers = topo.idxs_ds[members]
            receivers = receivers[(receivers >= 0) & (receivers != members)]
            if receivers.size:
                total += receivers.size - np.unique(receivers).size
        counts[name] = total
    assert counts["cfds"] == 0
    assert counts["asap"] > 0 or counts["alap"] > 0


# ---------------------------------------------------------------------------
# Decompositions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", LAYERINGS)
@pytest.mark.parametrize("direction", ["u2d", "d2u"])
def test_a_decomposition_covers_every_cell_once(topo, name, direction):
    decomp = topo.decomposition(name, direction)
    assert decomp.cells.size == topo.ncells
    assert np.unique(decomp.cells).size == decomp.cells.size
    assert decomp.offsets[0] == 0
    assert decomp.offsets[-1] == decomp.cells.size
    assert np.all(np.diff(decomp.offsets) >= 0)


@pytest.mark.parametrize("name", LAYERINGS)
def test_cells_inside_a_layer_are_in_ascending_order(topo, name):
    """A static loop schedule walks them this way, and the locality metrics
    are measured on that assumption."""
    for members in topo.decomposition(name, "u2d"):
        if members.size > 1:
            assert np.all(np.diff(members) > 0)


@pytest.mark.parametrize("name", LAYERINGS)
def test_flattening_a_decomposition_concatenates_its_layers(topo, name):
    decomp = topo.decomposition(name, "u2d")
    rebuilt = np.concatenate([decomp.layer(i) for i in range(decomp.nlayers)])
    assert np.array_equal(rebuilt, decomp.flattened())


def test_reversing_a_layering_mirrors_it(topo):
    layers, _ = topo.layering("cfds")
    flipped = reverse_layers(layers, topo.mask)
    inside = topo.mask
    assert np.array_equal(flipped[inside], layers[inside].max() - layers[inside])
    assert np.all(flipped[~inside] == -1)


# ---------------------------------------------------------------------------
# Clipping a basin out of a region keeps the structures usable, which is what
# lets someone take a released region and work on part of it
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def clipped(topo):
    """A cell and everything upstream of it: a whole subbasin."""
    area = topo.upstream_area(ordering="dfs")
    inside_range = topo.mask & (area > area[topo.mask].max() * 0.05)
    root = int(np.nonzero(inside_range)[0][0])

    inside = np.zeros(topo.idxs_ds.size, dtype=bool)
    inside[root] = True
    for cell in topo.ordering("dfs", "d2u"):      # receivers before donors
        receiver = topo.idxs_ds[cell]
        if receiver >= 0 and receiver != cell and inside[receiver]:
            inside[cell] = True
    assert inside.sum() > 100
    return inside


@pytest.mark.parametrize("name", ORDERINGS)
def test_an_ordering_filtered_to_a_subbasin_is_still_a_topological_sort(
        topo, clipped, name):
    kept = topo.ordering(name, "d2u")
    kept = kept[clipped[kept]]

    position = np.full(topo.idxs_ds.size, -1, dtype=np.int64)
    position[kept] = np.arange(kept.size)
    cells = kept.astype(np.int64)
    receivers = topo.idxs_ds[cells]
    moving = (receivers >= 0) & (receivers != cells)
    moving &= clipped[np.where(receivers >= 0, receivers, 0)]
    assert np.all(position[receivers[moving]] < position[cells[moving]])


@pytest.mark.parametrize("name", LAYERINGS)
def test_a_layering_filtered_to_a_subbasin_keeps_its_layers_independent(
        topo, clipped, name):
    layers, _ = topo.layering(name)
    for level in np.unique(layers[clipped]):
        members = np.nonzero(clipped & (layers == level))[0]
        receivers = topo.idxs_ds[members]
        moving = (receivers >= 0) & (receivers != members)
        moving &= clipped[np.where(receivers >= 0, receivers, 0)]
        assert not np.any(layers[receivers[moving]] == level)


def test_the_conflict_free_guarantee_survives_clipping(topo, clipped):
    layers, _ = topo.layering("cfds")
    for level in np.unique(layers[clipped]):
        members = np.nonzero(clipped & (layers == level))[0]
        receivers = topo.idxs_ds[members]
        moving = (receivers >= 0) & (receivers != members)
        moving &= clipped[np.where(receivers >= 0, receivers, 0)]
        receivers = receivers[moving]
        assert receivers.size == np.unique(receivers).size

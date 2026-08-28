"""Check the package against precomputed expected outputs.

``tests/reference/expected.npz`` holds the input grid and the expected outputs
for the bundled example basin, so this runs on a fresh clone with numpy alone.

The structures and the Strahler order must match exactly.  The two
floating-point kernels are allowed a small tolerance, because summation order
at a confluence differs between methods.
"""

import os

import numpy as np
import pytest

import flowtopo

HERE = os.path.dirname(os.path.abspath(__file__))
_REFERENCE_ARRAYS = np.load(os.path.join(HERE, "reference", "expected.npz"))


def _ref(name):
    return _REFERENCE_ARRAYS[name]


@pytest.fixture(scope="module")
def topo():
    return flowtopo.FlowTopo.from_d8(
        _ref("dir"),
        shape=tuple(_ref("shape")),
        transform=tuple(_ref("transform")),
    )


@pytest.fixture(scope="module")
def ldn(topo):
    return topo.distance_to_outlet(ordering="dfs")


@pytest.fixture(scope="module")
def upa(topo):
    return topo.upstream_area(ordering="dfs")


def _visit_position(seq, size):
    pos = np.zeros(size, dtype=np.int32)
    pos[seq] = np.arange(seq.size, dtype=np.int32)
    return pos


@pytest.mark.parametrize("name", ["dfs", "bfs", "topo"])
def test_ordering_matches_c(topo, name):
    # the reference stores the u2d visit position: 0 at a headwater
    seq = topo.ordering(name, "u2d")
    mine = _visit_position(seq, topo.idxs_ds.size)
    theirs = _ref(f"seq_{name}")
    assert np.array_equal(mine[topo.mask], theirs[topo.mask])


@pytest.mark.parametrize("name", ["asap", "cfds", "alap"])
def test_layering_matches_c(topo, name):
    layers, _ = topo.layering(name)
    theirs = _ref(f"lyr_{name}")
    assert np.array_equal(layers[topo.mask], theirs[topo.mask])


def test_distance_to_outlet_matches_c(topo, ldn):
    theirs = _ref("ldn")
    assert np.allclose(ldn[topo.mask], theirs[topo.mask], atol=1e-3)


def test_upstream_area_matches_c(topo, upa):
    theirs = _ref("upa")
    assert np.allclose(upa[topo.mask], theirs[topo.mask], rtol=1e-5, atol=1e-2)


def test_longest_upstream_path_matches_c(topo, ldn):
    mine = topo.longest_upstream_path(ldn, ordering="dfs")
    theirs = _ref("lup")
    assert np.allclose(mine[topo.mask], theirs[topo.mask], atol=1e-3)


def test_strahler_matches_c(topo, upa):
    channel = topo.channel_mask(upa, 10.0)
    mine = topo.strahler_order(ordering="dfs", channel_mask=channel)
    theirs = _ref("ord")
    assert np.array_equal(mine[channel], theirs[channel])


# ---------------------------------------------------------------------------
# Internal agreement: every ordering, layering and manner on the same input
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ordering", ["dfs", "bfs", "topo"])
def test_orderings_agree(topo, ordering, upa):
    mine = topo.upstream_area(ordering=ordering)
    assert np.allclose(mine[topo.mask], upa[topo.mask], rtol=1e-5, atol=1e-2)


@pytest.mark.parametrize("layering", ["asap", "cfds", "alap"])
@pytest.mark.parametrize("manner", ["pull", "atomic_push"])
def test_safe_manners_agree(topo, layering, manner, upa):
    mine = topo.upstream_area(layering=layering, manner=manner)
    assert np.allclose(mine[topo.mask], upa[topo.mask], rtol=1e-5, atol=1e-2)


def test_push_is_correct_under_cfds(topo, upa):
    mine = topo.upstream_area(layering="cfds", manner="push")
    assert np.allclose(mine[topo.mask], upa[topo.mask], rtol=1e-5, atol=1e-2)


@pytest.mark.parametrize("layering", ["asap", "alap"])
def test_push_loses_writes_without_cfds(topo, layering, upa):
    """The whole point of the conflict-free layering, shown as a failure."""
    mine = topo.upstream_area(layering=layering, manner="push")
    assert not np.allclose(mine[topo.mask], upa[topo.mask], rtol=1e-5, atol=1e-2)


def _duplicate_receivers(decomp, idxs_ds, sub=None):
    """How many times a layer writes to a receiver another cell already wrote."""
    total = 0
    for members in decomp:
        cells = members if sub is None else members[sub[members]]
        if cells.size == 0:
            continue
        ds = idxs_ds[cells]
        keep = (ds >= 0) & (ds != cells)
        if sub is not None:
            keep &= sub[np.where(keep, ds, 0)]
        ds = ds[keep]
        if ds.size:
            total += ds.size - np.unique(ds).size
    return total


@pytest.mark.parametrize("layering,expect_conflicts",
                         [("asap", True), ("cfds", False), ("alap", True)])
def test_only_cfds_has_no_write_conflicts(topo, layering, expect_conflicts):
    """The guarantee, measured directly rather than inferred from a result."""
    decomp = topo.decomposition(layering, "u2d")
    duplicates = _duplicate_receivers(decomp, topo.idxs_ds)
    assert (duplicates > 0) is expect_conflicts
    if not expect_conflicts:
        assert duplicates == 0


def test_strahler_push_needs_cfds(topo, upa):
    channel = topo.channel_mask(upa, 10.0)
    reference = topo.strahler_order(ordering="dfs", channel_mask=channel)

    good = topo.strahler_order(layering="cfds", manner="push",
                               channel_mask=channel)
    assert np.array_equal(good[channel], reference[channel])

    pulled = topo.strahler_order(layering="cfds", manner="pull",
                                 channel_mask=channel)
    assert np.array_equal(pulled[channel], reference[channel])

    # A layering without the guarantee is wrong exactly when the channel
    # network actually gives it a conflict to trip over.  On this small basin
    # the as-soon-as-possible layering happens to have none once the network is
    # thinned to channels, so test the one that does.
    for layering in ("asap", "alap"):
        decomp = topo.decomposition(layering, "u2d")
        conflicts = _duplicate_receivers(decomp, topo.idxs_ds, channel)
        result = topo.strahler_order(layering=layering, manner="push",
                                     channel_mask=channel)
        if conflicts > 0:
            assert not np.array_equal(result[channel], reference[channel])


def test_cfds_needs_no_more_layers_than_a_few(topo):
    _, n_asap = topo.layering("asap")
    _, n_cfds = topo.layering("cfds")
    assert n_cfds >= n_asap
    assert n_cfds <= n_asap * 1.5


# ---------------------------------------------------------------------------
# Spatial partitions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("level", ["basin", "subbasin"])
def test_partition_covers_every_cell_exactly_once(topo, level):
    part, _ = topo.partition(n_parts=4, level=level)
    covered = (part >= 0) | (part == flowtopo.MAINSTEM)
    assert np.array_equal(covered, topo.mask)


@pytest.mark.parametrize("level", ["basin", "subbasin"])
def test_partition_is_communication_free(topo, level):
    """No cell drains into a different subregion, except onto the mainstem."""
    part, _ = topo.partition(n_parts=4, level=level)
    cells = np.nonzero(part >= 0)[0]
    ds = topo.idxs_ds[cells]
    moving = (ds >= 0) & (ds != cells)
    cells, ds = cells[moving], ds[moving]
    crossing = (part[ds] != part[cells]) & (part[ds] != flowtopo.MAINSTEM)
    assert not crossing.any()


def test_subbasin_balances_what_basin_cannot(topo):
    """The bundled basin is a single basin: whole-basin assignment cannot split it."""
    _, basin_load = topo.partition(n_parts=4, level="basin")
    _, sub_load = topo.partition(n_parts=4, level="subbasin")
    assert np.count_nonzero(basin_load) == 1          # one processor does everything
    assert sub_load.max() / sub_load.mean() < 1.05    # subbasin spreads it evenly


def test_mainstem_runs_after_its_tributaries(topo):
    """Every mainstem cell has a donor outside the mainstem, so it must wait."""
    part, _ = topo.partition(n_parts=4, level="subbasin")
    stem = np.nonzero(part == flowtopo.MAINSTEM)[0]
    assert stem.size > 0
    us_table, n_up = topo.upstream()
    fed_from_parts = 0
    for cell in stem:
        donors = us_table[cell][: n_up[cell]]
        donors = donors[donors >= 0]
        if np.any(part[donors] >= 0):
            fed_from_parts += 1
    assert fed_from_parts > 0


# ---------------------------------------------------------------------------
# Cell area against the closed-form spherical value
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("nrow, ncol", [
    (180, 360),      # square, one degree each way
    (180, 720),      # wider than tall
    (360, 360),      # taller than wide
    (90, 360),       # twice as tall as wide
])
def test_cell_area_matches_the_spherical_formula(nrow, ncol):
    """The two sides of a pixel have to be read separately.

    A band of latitude between lat1 and lat2, spanning dlon of longitude, has
    area R^2 * dlon * (sin lat2 - sin lat1). Reading only the pixel width and
    using it for the height as well halves or doubles that on any grid whose
    pixels are not square, and gets the right answer on the square grid that
    would catch it.
    """
    radius_km = 6371.0088
    dlat, dlon = 180.0 / nrow, 360.0 / ncol
    transform = (-180.0, dlon, 0.0, 90.0, 0.0, -dlat)
    topo = flowtopo.FlowTopo.from_d8(np.zeros((nrow, ncol), np.uint8),
                                     transform=transform)
    area = np.asarray(topo.cell_area).reshape(nrow, ncol)

    for row in (0, nrow // 3, nrow // 2, nrow - 1):
        top = np.radians(90.0 - row * dlat)
        bottom = np.radians(90.0 - (row + 1) * dlat)
        expected = (radius_km ** 2 * np.radians(dlon)
                    * (np.sin(top) - np.sin(bottom)))
        assert area[row, 0] == pytest.approx(expected, rel=1e-4)

    # The whole grid is the whole sphere, whatever the pixel shape.
    assert area.sum() == pytest.approx(4 * np.pi * radius_km ** 2, rel=1e-4)

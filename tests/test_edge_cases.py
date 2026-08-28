"""Degenerate inputs: an empty grid, a lone cell, and networks with cycles.

These are not exotic. Tiling a global dataset produces all-nodata tiles, and a
conditioned flow-direction grid can still carry a cycle. Nothing here should
raise; the structures should simply report that some cells cannot be ordered.
"""

import numpy as np
import pytest

import flowtopo
from flowtopo.core import D8_NODATA

TRANSFORM = (0.0, 1 / 1200, 0.0, 40.0, 0.0, -1 / 1200)

LAYERINGS = ["asap", "cfds", "alap"]
ORDERINGS = ["dfs", "bfs", "topo"]


def build(d8):
    return flowtopo.FlowTopo.from_d8(np.asarray(d8, dtype=np.uint8),
                                     transform=TRANSFORM)


@pytest.fixture
def empty():
    return build(np.full((3, 3), D8_NODATA))


@pytest.fixture
def lone_pit():
    grid = np.full((3, 3), D8_NODATA, dtype=np.uint8)
    grid[1, 1] = 0
    return build(grid)


@pytest.fixture
def two_cell_cycle():
    """Left cell drains east, right cell drains west: neither reaches a pit."""
    return build([[1, 16]])


@pytest.fixture
def cycle_beside_basin():
    """A four-cell chain to a pit, plus a detached two-cell cycle."""
    grid = np.full((5, 5), D8_NODATA, dtype=np.uint8)
    grid[:4, 0] = 4          # south
    grid[4, 0] = 0           # pit
    grid[0, 3] = 1           # east
    grid[0, 4] = 16          # west
    return build(grid)


# ---------------------------------------------------------------------------
# Nothing raises
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ORDERINGS)
def test_orderings_survive_an_empty_grid(empty, name):
    assert empty.ordering(name, "u2d").size == 0


@pytest.mark.parametrize("name", LAYERINGS)
def test_layerings_survive_an_empty_grid(empty, name):
    layers, nlayers = empty.layering(name)
    assert nlayers == 0
    assert np.all(layers == flowtopo.LAYER_NODATA)


@pytest.mark.parametrize("name", LAYERINGS)
def test_layerings_survive_a_pure_cycle(two_cell_cycle, name):
    """A network that never reaches a pit must not raise."""
    layers, nlayers = two_cell_cycle.layering(name)
    assert nlayers >= 0
    assert layers.size == two_cell_cycle.idxs_ds.size


@pytest.mark.parametrize("level", ["basin", "subbasin"])
def test_partition_survives_an_empty_grid(empty, level):
    part, load = empty.partition(n_parts=4, level=level)
    assert np.all(part == -1)
    assert load.sum() == 0


def test_lone_pit_is_one_cell_one_basin(lone_pit):
    assert lone_pit.ncells == 1
    assert lone_pit.nbasins == 1
    assert lone_pit.upstream_area(ordering="dfs")[lone_pit.mask].size == 1


# ---------------------------------------------------------------------------
# A cycle is reported, not hidden
# ---------------------------------------------------------------------------


def test_a_sequence_shorter_than_the_cell_count_means_a_cycle(cycle_beside_basin):
    topo = cycle_beside_basin
    for name in ORDERINGS:
        assert topo.ordering(name, "u2d").size < topo.ncells


def test_asap_leaves_cycle_cells_unnumbered(cycle_beside_basin):
    layers, _ = cycle_beside_basin.layering("asap")
    stranded = np.count_nonzero((layers < 0) & cycle_beside_basin.mask)
    assert stranded == 2


# ---------------------------------------------------------------------------
# The conflict-free guarantee holds even with a cycle present
# ---------------------------------------------------------------------------


def _conflicts(topo, layering):
    total = 0
    for members in topo.decomposition(layering, "u2d"):
        ds = topo.idxs_ds[members]
        ds = ds[(ds >= 0) & (ds != members)]
        if ds.size:
            total += ds.size - np.unique(ds).size
    return total


def test_cfds_has_no_conflicts_even_with_a_cycle():
    grid = np.full((4, 6), D8_NODATA, dtype=np.uint8)
    grid[1, 1], grid[1, 2] = 1, 4        # a four-cell cycle
    grid[2, 2], grid[2, 1] = 16, 64
    grid[0, 1], grid[1, 0] = 4, 1        # two cells draining into it
    topo = build(grid)
    assert _conflicts(topo, "cfds") == 0


@pytest.mark.parametrize("n_parts", [1, 3, 64, 500])
def test_partition_covers_everything_at_any_part_count(cycle_beside_basin, n_parts):
    part, _ = cycle_beside_basin.partition(n_parts=n_parts, level="subbasin")
    covered = (part >= 0) | (part == flowtopo.MAINSTEM)
    assert np.array_equal(covered, cycle_beside_basin.mask)


@pytest.mark.parametrize("n_parts", [0, -1])
def test_partition_rejects_a_meaningless_part_count(lone_pit, n_parts):
    with pytest.raises(ValueError):
        lone_pit.partition(n_parts=n_parts)


# ---------------------------------------------------------------------------
# A grid that does not use 247 for nodata
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("nodata_code,terminal", [(255, 0), (0, 255), (5, 0)])
def test_the_grids_own_nodata_code_is_honoured(tmp_path, nodata_code, terminal):
    """255 is a valid terminal in this convention, so a file must be believed.

    The terminal code has to differ from the nodata code: a grid that uses one
    value for both cannot say which cells are outside the network.
    """
    from flowtopo.raster import GridHeader, write_geotiff

    grid = np.full((4, 4), nodata_code, dtype=np.uint8)
    grid[1, :3] = [1, 1, terminal]          # a three-cell chain to a pit

    path = tmp_path / "dir.tif"
    write_geotiff(str(path), grid.ravel(),
                  GridHeader(ncol=4, nrow=4, dtype="uint8",
                             nodata=float(nodata_code), transform=TRANSFORM))
    topo = flowtopo.FlowTopo.from_raster(str(path))
    assert topo.ncells == 3
    assert topo.nbasins == 1


def test_a_cell_draining_into_nodata_becomes_a_pit():
    grid = np.full((3, 3), D8_NODATA, dtype=np.uint8)
    grid[1, 1] = 1                          # east, into nodata
    topo = build(grid)
    centre = 1 * 3 + 1
    assert topo.idxs_ds[centre] == centre

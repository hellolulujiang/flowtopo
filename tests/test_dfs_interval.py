"""The depth-first intervals: from seq_dfs and upg to dfs_in and dfs_out.

The intervals are checked against the upstream sets found by walking every
cell down to its pit, and the GeoTIFF form against the array form, across the
antimeridian included.
"""

import numpy as np
import pytest

import flowtopo
from flowtopo.synthetic import synthetic_d8

TRANSFORM = (0.0, 1 / 1200, 0.0, 40.0, 0.0, -1 / 1200)


def released_layers(topo, upa=None):
    """``seq_dfs`` as FullTopo releases it, and ``upg`` counted by walking every cell to its pit."""
    seq_dfs = np.full(topo.idxs_ds.size, -1, dtype=np.int32)
    order = topo.ordering("dfs", "u2d", upa=upa)
    seq_dfs[order] = np.arange(order.size, dtype=np.int32)
    upg = np.zeros(topo.idxs_ds.size, dtype=np.uint32)
    for cell in order:
        i = int(cell)
        while True:
            upg[i] += 1
            j = int(topo.idxs_ds[i])
            if j == i:
                break
            i = j
    return seq_dfs, upg


def upstream_sets(topo):
    """The cells draining through every cell, the cell itself included."""
    sets = [set() for _ in range(topo.idxs_ds.size)]
    for cell in np.flatnonzero(topo.idxs_ds >= 0):
        i = int(cell)
        while True:
            sets[i].add(int(cell))
            j = int(topo.idxs_ds[i])
            if j == i:
                break
            i = j
    return sets


@pytest.mark.parametrize("seed", [1, 2])
@pytest.mark.parametrize("sorted_donors", [False, True])
def test_intervals_hold_the_upstream_cells(seed, sorted_donors):
    topo = flowtopo.FlowTopo.from_d8(synthetic_d8(24, 32, seed=seed), transform=TRANSFORM)
    upa = topo.upstream_area(ordering="dfs") if sorted_donors else None
    seq_dfs, upg = released_layers(topo, upa)
    dfs_in, dfs_out = flowtopo.dfs_interval(seq_dfs, upg)

    assert dfs_in.dtype == np.int32 and dfs_out.dtype == np.int32
    # dfs_in is the position in the same traversal read downstream to upstream
    d2u = topo.ordering("dfs", "d2u", upa=upa)
    assert np.array_equal(dfs_in[d2u], np.arange(d2u.size))
    assert np.array_equal(dfs_out - dfs_in + 1, upg.astype(np.int64))
    for p, cells in enumerate(upstream_sets(topo)):
        inside = np.flatnonzero((dfs_in >= dfs_in[p]) & (dfs_in <= dfs_out[p]))
        assert set(inside.tolist()) == cells


def test_cells_off_the_network_stay_minus_one():
    d8 = synthetic_d8(12, 16, seed=3)
    d8[0, :5] = 247
    topo = flowtopo.FlowTopo.from_d8(d8, transform=TRANSFORM)
    seq_dfs, upg = released_layers(topo)
    upg[seq_dfs < 0] = 0
    dfs_in, dfs_out = flowtopo.dfs_interval(seq_dfs.reshape(12, 16), upg.reshape(12, 16))
    assert dfs_in.shape == (12, 16)
    off = seq_dfs.reshape(12, 16) < 0
    assert off.sum() == 5
    assert np.all(dfs_in[off] == -1) and np.all(dfs_out[off] == -1)


def test_layers_that_do_not_belong_together_are_refused():
    topo = flowtopo.FlowTopo.from_d8(synthetic_d8(16, 16, seed=4), transform=TRANSFORM)
    seq_dfs, upg = released_layers(topo)
    grid_seq, grid_upg = seq_dfs.reshape(16, 16), upg.reshape(16, 16)
    with pytest.raises(ValueError, match="do not belong together"):
        flowtopo.dfs_interval(grid_seq, np.roll(grid_upg, 1, axis=1))     # shifted by a cell
    holed = grid_upg.copy()
    holed[5, 5] = 0
    with pytest.raises(ValueError, match="do not belong together"):
        flowtopo.dfs_interval(grid_seq, holed)
    with pytest.raises(ValueError, match="not a whole sequence"):
        flowtopo.dfs_interval(grid_seq[:8], grid_upg[:8])
    # a part of the sequence is read when n comes with it
    whole_in, whole_out = flowtopo.dfs_interval(grid_seq, grid_upg)
    part_in, part_out = flowtopo.dfs_interval(grid_seq[:8], grid_upg[:8], n=topo.idxs_ds.size)
    assert np.array_equal(part_in, whole_in[:8]) and np.array_equal(part_out, whole_out[:8])


def write(path, data, transform):
    flowtopo.write_geotiff(str(path), data, flowtopo.GridHeader(
        data.shape[1], data.shape[0], dtype=data.dtype.name, nodata=data.dtype.type(-1 if data.dtype.kind == "i" else 0),
        transform=transform))


def test_geotiff_reads_the_window_of_a_larger_upg(tmp_path):
    rasterio = pytest.importorskip("rasterio")
    topo = flowtopo.FlowTopo.from_d8(synthetic_d8(40, 30, seed=5), transform=TRANSFORM)
    seq_dfs, upg = released_layers(topo)
    seq_grid, upg_grid = seq_dfs.reshape(40, 30), upg.reshape(40, 30)
    big = np.zeros((60, 50), dtype=np.uint32)
    big[7:47, 11:41] = upg_grid
    cell = 1 / 1200
    write(tmp_path / "seq.tif", seq_grid, (10.0, cell, 0.0, 40.0, 0.0, -cell))
    write(tmp_path / "upg.tif", big, (10.0 - 11 * cell, cell, 0.0, 40.0 + 7 * cell, 0.0, -cell))

    n = flowtopo.dfs_interval_geotiff(str(tmp_path / "seq.tif"), str(tmp_path / "upg.tif"),
                                      str(tmp_path / "in.tif"), str(tmp_path / "out.tif"), rows=16)
    assert n == topo.idxs_ds.size
    want_in, want_out = flowtopo.dfs_interval(seq_grid, upg_grid)
    with rasterio.open(tmp_path / "in.tif") as src:
        assert src.nodata == -1 and src.dtypes[0] == "int32"
        assert np.array_equal(src.read(1), want_in)
    with rasterio.open(tmp_path / "out.tif") as src:
        assert np.array_equal(src.read(1), want_out)
    with pytest.raises(FileExistsError):
        flowtopo.dfs_interval_geotiff(str(tmp_path / "seq.tif"), str(tmp_path / "upg.tif"),
                                      str(tmp_path / "in.tif"), str(tmp_path / "out2.tif"))


def test_geotiff_reads_upg_across_the_antimeridian(tmp_path):
    rasterio = pytest.importorskip("rasterio")
    topo = flowtopo.FlowTopo.from_d8(synthetic_d8(6, 8, seed=6), transform=TRANSFORM)
    seq_dfs, upg = released_layers(topo)
    seq_grid, upg_grid = seq_dfs.reshape(6, 8), upg.reshape(6, 8)
    # a globe of 30-degree cells; the region starts at 120 E and ends at 120 W
    globe = np.zeros((6, 12), dtype=np.uint32)
    globe[:, 10:] = upg_grid[:, :2]
    globe[:, :6] = upg_grid[:, 2:]
    write(tmp_path / "seq.tif", seq_grid, (120.0, 30.0, 0.0, 90.0, 0.0, -30.0))
    write(tmp_path / "upg.tif", globe, (-180.0, 30.0, 0.0, 90.0, 0.0, -30.0))

    flowtopo.dfs_interval_geotiff(str(tmp_path / "seq.tif"), str(tmp_path / "upg.tif"),
                                  str(tmp_path / "in.tif"), str(tmp_path / "out.tif"))
    want_in, want_out = flowtopo.dfs_interval(seq_grid, upg_grid)
    with rasterio.open(tmp_path / "in.tif") as src:
        assert np.array_equal(src.read(1), want_in)
    with rasterio.open(tmp_path / "out.tif") as src:
        assert np.array_equal(src.read(1), want_out)


def test_geotiff_refuses_cell_edges_that_do_not_coincide(tmp_path):
    topo = flowtopo.FlowTopo.from_d8(synthetic_d8(8, 8, seed=7), transform=TRANSFORM)
    seq_dfs, upg = released_layers(topo)
    cell = 1 / 1200
    write(tmp_path / "seq.tif", seq_dfs.reshape(8, 8), (10.0, cell, 0.0, 40.0, 0.0, -cell))
    write(tmp_path / "upg.tif", upg.reshape(8, 8), (10.0 + cell / 2, cell, 0.0, 40.0, 0.0, -cell))
    with pytest.raises(ValueError, match="not on whole cells"):
        flowtopo.dfs_interval_geotiff(str(tmp_path / "seq.tif"), str(tmp_path / "upg.tif"),
                                      str(tmp_path / "in.tif"), str(tmp_path / "out.tif"))
    assert not (tmp_path / "in.tif").exists()


def test_a_huge_upg_is_refused_not_wrapped():
    topo = flowtopo.FlowTopo.from_d8(synthetic_d8(8, 8, seed=8), transform=TRANSFORM)
    seq_dfs, upg = released_layers(topo)
    for huge in (np.iinfo(np.int64).max, np.iinfo(np.uint64).max):
        big = upg.astype(np.int64 if huge <= np.iinfo(np.int64).max else np.uint64)
        big[seq_dfs == 0] = huge             # dfs_in = n - 1: dfs_in + upg - 1 would wrap
        with pytest.raises(ValueError, match="do not belong together"):
            flowtopo.dfs_interval(seq_dfs, big)


def leftovers(folder, *names):
    return sorted(p.name for p in folder.iterdir() if p.name.split(".")[0] in names)


def test_geotiff_refuses_paths_that_clash_and_leaves_nothing(tmp_path):
    pytest.importorskip("rasterio")
    topo = flowtopo.FlowTopo.from_d8(synthetic_d8(8, 8, seed=9), transform=TRANSFORM)
    seq_dfs, upg = released_layers(topo)
    cell = 1 / 1200
    write(tmp_path / "seq.tif", seq_dfs.reshape(8, 8), (10.0, cell, 0.0, 40.0, 0.0, -cell))
    write(tmp_path / "upg.tif", upg.reshape(8, 8), (10.0, cell, 0.0, 40.0, 0.0, -cell))
    write(tmp_path / "shifted.tif", np.roll(upg.reshape(8, 8), 1, axis=1), (10.0, cell, 0.0, 40.0, 0.0, -cell))
    seq, upg_path = str(tmp_path / "seq.tif"), str(tmp_path / "upg.tif")

    with pytest.raises(ValueError, match="are both"):
        flowtopo.dfs_interval_geotiff(seq, upg_path, str(tmp_path / "a.tif"), str(tmp_path / "a.tif"))
    # a file at a temporary path, an input for one, is not written over
    (tmp_path / "b.tif.part").write_bytes(b"keep")
    with pytest.raises(FileExistsError, match="b.tif.part"):
        flowtopo.dfs_interval_geotiff(seq, upg_path, str(tmp_path / "b.tif"), str(tmp_path / "c.tif"))
    assert (tmp_path / "b.tif.part").read_bytes() == b"keep"
    # the same file under two spellings, through a link to the folder: neither output is left
    (tmp_path / "link").symlink_to(tmp_path)
    with pytest.raises(FileExistsError, match="the same file"):
        flowtopo.dfs_interval_geotiff(seq, upg_path, str(tmp_path / "d.tif"), str(tmp_path / "link" / "d.tif"))
    # a run that fails leaves neither output nor temporary file
    with pytest.raises(ValueError, match="do not belong together"):
        flowtopo.dfs_interval_geotiff(seq, str(tmp_path / "shifted.tif"),
                                      str(tmp_path / "e.tif"), str(tmp_path / "f.tif"))
    with pytest.raises(ValueError, match="at least 1"):
        flowtopo.dfs_interval_geotiff(seq, upg_path, str(tmp_path / "g.tif"), str(tmp_path / "h.tif"), rows=0)
    assert leftovers(tmp_path, "a", "c", "d", "e", "f", "g", "h") == []

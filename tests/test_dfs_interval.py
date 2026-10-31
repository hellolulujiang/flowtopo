"""The depth-first intervals: from seq_dfs, upg and the outlets to dfs_in and dfs_out.

The intervals are checked against the upstream sets found by walking every
cell down to its pit and against the blocks of the basins, and the GeoTIFF
form against the array form, across the antimeridian included.
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


def pits(topo):
    """The outlet of every basin: the cells that drain into themselves."""
    cells = np.arange(topo.idxs_ds.size)
    return np.flatnonzero(topo.idxs_ds == cells)


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
    # a rough surface, so that the grid holds several basins
    topo = flowtopo.FlowTopo.from_d8(synthetic_d8(24, 32, seed=seed, smooth=2), transform=TRANSFORM)
    upa = topo.upstream_area(ordering="dfs") if sorted_donors else None
    seq_dfs, upg = released_layers(topo, upa)
    outlets = pits(topo)
    assert outlets.size > 1
    dfs_in, dfs_out = flowtopo.dfs_interval(seq_dfs, upg, outlets)

    assert dfs_in.dtype == np.int32 and dfs_out.dtype == np.int32
    assert np.array_equal(dfs_out - dfs_in + 1, upg.astype(np.int64))
    # each basin takes the block a..b of seq_dfs, its outlet at b, and dfs_in reads the
    # block backward: the outlet at a, the block a..b again
    labels, count = flowtopo.basin_labels(topo.idxs_ds)
    assert count == outlets.size
    d2u = topo.ordering("dfs", "d2u", upa=upa)
    whole = np.empty(d2u.size, dtype=np.int64)
    whole[d2u] = np.arange(d2u.size)
    for outlet in outlets:
        cells = np.flatnonzero(labels == labels[outlet])
        b = int(seq_dfs[outlet])
        a = b - int(upg[outlet]) + 1
        assert sorted(seq_dfs[cells].tolist()) == list(range(a, b + 1))
        assert sorted(dfs_in[cells].tolist()) == list(range(a, b + 1))
        assert dfs_in[outlet] == a and dfs_out[outlet] == b
        assert np.array_equal(dfs_in[cells], a + b - seq_dfs[cells])
        # the whole sequence read backward differs by a constant within the basin
        assert np.unique(dfs_in[cells] - whole[cells]).size == 1
    assert not np.array_equal(dfs_in, whole)
    for p, cells in enumerate(upstream_sets(topo)):
        inside = np.flatnonzero((dfs_in >= dfs_in[p]) & (dfs_in <= dfs_out[p]))
        assert set(inside.tolist()) == cells


def test_cells_off_the_network_stay_minus_one():
    d8 = synthetic_d8(12, 16, seed=3)
    d8[0, :5] = 247
    topo = flowtopo.FlowTopo.from_d8(d8, transform=TRANSFORM)
    seq_dfs, upg = released_layers(topo)
    upg[seq_dfs < 0] = 0
    rows, cols = np.divmod(pits(topo), 16)
    dfs_in, dfs_out = flowtopo.dfs_interval(seq_dfs.reshape(12, 16), upg.reshape(12, 16), (rows, cols))
    assert dfs_in.shape == (12, 16)
    off = seq_dfs.reshape(12, 16) < 0
    assert off.sum() == 5
    assert np.all(dfs_in[off] == -1) and np.all(dfs_out[off] == -1)


def test_layers_that_do_not_belong_together_are_refused():
    topo = flowtopo.FlowTopo.from_d8(synthetic_d8(16, 16, seed=4), transform=TRANSFORM)
    seq_dfs, upg = released_layers(topo)
    outlets = pits(topo)
    grid_seq, grid_upg = seq_dfs.reshape(16, 16), upg.reshape(16, 16)
    rows, cols = np.divmod(outlets, 16)
    flat_in, flat_out = flowtopo.dfs_interval(seq_dfs, upg, outlets)
    grid_in, grid_out = flowtopo.dfs_interval(grid_seq, grid_upg, (rows, cols))
    assert np.array_equal(grid_in.reshape(-1), flat_in) and np.array_equal(grid_out.reshape(-1), flat_out)

    with pytest.raises(ValueError, match="do not belong together"):
        flowtopo.dfs_interval(grid_seq, np.roll(grid_upg, 1, axis=1), (rows, cols))     # shifted by a cell
    holed = grid_upg.copy()
    holed[np.unravel_index(int(np.argmax(seq_dfs == 0)), (16, 16))] = 0
    with pytest.raises(ValueError, match="do not belong together"):
        flowtopo.dfs_interval(grid_seq, holed, (rows, cols))
    with pytest.raises(ValueError, match="not the whole sequence"):
        flowtopo.dfs_interval(grid_seq[:8], grid_upg[:8], (rows[rows < 8], cols[rows < 8]))
    # an outlet missing, a cell that is not an outlet, an outlet twice
    biggest = int(np.argmax(upg[outlets]))
    with pytest.raises(ValueError, match="do not belong together"):
        flowtopo.dfs_interval(seq_dfs, upg, np.delete(outlets, biggest))
    inner = int(np.flatnonzero((topo.idxs_ds >= 0) & (topo.idxs_ds != np.arange(256)))[0])
    with pytest.raises(ValueError, match="do not belong together"):
        flowtopo.dfs_interval(seq_dfs, upg, np.append(outlets, inner))
    with pytest.raises(ValueError, match="given twice"):
        flowtopo.dfs_interval(seq_dfs, upg, np.append(outlets, outlets[0]))
    with pytest.raises(ValueError, match="outside seq_dfs"):
        flowtopo.dfs_interval(seq_dfs, upg, np.append(outlets, 256))
    with pytest.raises(ValueError, match="outside seq_dfs"):
        flowtopo.dfs_interval(seq_dfs, upg, np.append(outlets, -1))


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

    outlets = np.divmod(pits(topo), 30)
    assert outlets[0].size > 1
    n = flowtopo.dfs_interval_geotiff(str(tmp_path / "seq.tif"), str(tmp_path / "upg.tif"),
                                      str(tmp_path / "in.tif"), str(tmp_path / "out.tif"), outlets, rows=16)
    assert n == topo.idxs_ds.size
    want_in, want_out = flowtopo.dfs_interval(seq_grid, upg_grid, outlets)
    with rasterio.open(tmp_path / "in.tif") as src:
        assert src.nodata == -1 and src.dtypes[0] == "int32"
        assert np.array_equal(src.read(1), want_in)
    with rasterio.open(tmp_path / "out.tif") as src:
        assert np.array_equal(src.read(1), want_out)
    with pytest.raises(FileExistsError):
        flowtopo.dfs_interval_geotiff(str(tmp_path / "seq.tif"), str(tmp_path / "upg.tif"),
                                      str(tmp_path / "in.tif"), str(tmp_path / "out2.tif"), outlets)


def test_geotiff_reads_upg_across_the_antimeridian(tmp_path):
    rasterio = pytest.importorskip("rasterio")
    topo = flowtopo.FlowTopo.from_d8(synthetic_d8(6, 8, seed=6, smooth=1), transform=TRANSFORM)
    seq_dfs, upg = released_layers(topo)
    seq_grid, upg_grid = seq_dfs.reshape(6, 8), upg.reshape(6, 8)
    # a globe of 30-degree cells; the region starts at 120 E and ends at 120 W
    globe = np.zeros((6, 12), dtype=np.uint32)
    globe[:, 10:] = upg_grid[:, :2]
    globe[:, :6] = upg_grid[:, 2:]
    write(tmp_path / "seq.tif", seq_grid, (120.0, 30.0, 0.0, 90.0, 0.0, -30.0))
    write(tmp_path / "upg.tif", globe, (-180.0, 30.0, 0.0, 90.0, 0.0, -30.0))

    outlets = np.divmod(pits(topo), 8)
    assert outlets[0].size > 1
    flowtopo.dfs_interval_geotiff(str(tmp_path / "seq.tif"), str(tmp_path / "upg.tif"),
                                  str(tmp_path / "in.tif"), str(tmp_path / "out.tif"), outlets)
    want_in, want_out = flowtopo.dfs_interval(seq_grid, upg_grid, outlets)
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
                                      str(tmp_path / "in.tif"), str(tmp_path / "out.tif"),
                                      np.divmod(pits(topo), 8))
    assert not (tmp_path / "in.tif").exists()


def test_a_huge_upg_is_refused_not_wrapped():
    topo = flowtopo.FlowTopo.from_d8(synthetic_d8(8, 8, seed=8), transform=TRANSFORM)
    seq_dfs, upg = released_layers(topo)
    outlets = pits(topo)
    inner = int(np.flatnonzero((topo.idxs_ds >= 0) & (topo.idxs_ds != np.arange(64)))[0])
    for huge in (np.iinfo(np.int64).max, np.iinfo(np.uint64).max):
        for cell in (inner, int(outlets[0])):          # dfs_in + upg - 1 would wrap
            big = upg.astype(np.int64 if huge <= np.iinfo(np.int64).max else np.uint64)
            big[cell] = huge
            with pytest.raises(ValueError, match="do not belong together"):
                flowtopo.dfs_interval(seq_dfs, big, outlets)


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
    outlets = np.divmod(pits(topo), 8)

    with pytest.raises(ValueError, match="are both"):
        flowtopo.dfs_interval_geotiff(seq, upg_path, str(tmp_path / "a.tif"), str(tmp_path / "a.tif"), outlets)
    # a file at a temporary path, an input for one, is not written over
    (tmp_path / "b.tif.part").write_bytes(b"keep")
    with pytest.raises(FileExistsError, match="b.tif.part"):
        flowtopo.dfs_interval_geotiff(seq, upg_path, str(tmp_path / "b.tif"), str(tmp_path / "c.tif"), outlets)
    assert (tmp_path / "b.tif.part").read_bytes() == b"keep"
    # the same file under two spellings, through a link to the folder: neither output is left
    (tmp_path / "link").symlink_to(tmp_path)
    with pytest.raises(FileExistsError, match="the same file"):
        flowtopo.dfs_interval_geotiff(seq, upg_path, str(tmp_path / "d.tif"), str(tmp_path / "link" / "d.tif"), outlets)
    # a run that fails leaves neither output nor temporary file
    with pytest.raises(ValueError, match="do not belong together"):
        flowtopo.dfs_interval_geotiff(seq, str(tmp_path / "shifted.tif"),
                                      str(tmp_path / "e.tif"), str(tmp_path / "f.tif"), outlets)
    with pytest.raises(ValueError, match="at least 1"):
        flowtopo.dfs_interval_geotiff(seq, upg_path, str(tmp_path / "g.tif"), str(tmp_path / "h.tif"), outlets, rows=0)
    assert leftovers(tmp_path, "a", "c", "d", "e", "f", "g", "h") == []


def write_table(path, rows, cols, counts):
    """A basin table with the columns of FullBasin's, in another order and with others between."""
    with open(path, "w") as table:
        table.write("region_id,local_basin_id,basin_cell_count,outlet_lon,local_outlet_col,local_outlet_row\n")
        for k, (r, c, n) in enumerate(zip(rows, cols, counts)):
            table.write(f"43,{k + 1},{n},121.5,{c},{r}\n")


def test_geotiff_reads_the_outlets_from_a_basin_table(tmp_path):
    rasterio = pytest.importorskip("rasterio")
    topo = flowtopo.FlowTopo.from_d8(synthetic_d8(20, 12, seed=10), transform=TRANSFORM)
    seq_dfs, upg = released_layers(topo)
    seq_grid, upg_grid = seq_dfs.reshape(20, 12), upg.reshape(20, 12)
    cell = 1 / 1200
    write(tmp_path / "seq.tif", seq_grid, (10.0, cell, 0.0, 40.0, 0.0, -cell))
    write(tmp_path / "upg.tif", upg_grid, (10.0, cell, 0.0, 40.0, 0.0, -cell))
    rows, cols = np.divmod(pits(topo), 12)
    assert rows.size > 1
    counts = upg_grid[rows, cols]
    write_table(tmp_path / "bat.csv", rows, cols, counts)
    seq, upg_path = str(tmp_path / "seq.tif"), str(tmp_path / "upg.tif")

    flowtopo.dfs_interval_geotiff(seq, upg_path, str(tmp_path / "in.tif"), str(tmp_path / "out.tif"),
                                  str(tmp_path / "bat.csv"), rows=3)
    want_in, want_out = flowtopo.dfs_interval(seq_grid, upg_grid, (rows, cols))
    with rasterio.open(tmp_path / "in.tif") as src:
        assert np.array_equal(src.read(1), want_in)
    with rasterio.open(tmp_path / "out.tif") as src:
        assert np.array_equal(src.read(1), want_out)

    # a basin_cell_count other than upg at its outlet is refused, and nothing is left
    wrong = counts.copy()
    wrong[-1] += 1
    write_table(tmp_path / "wrong.csv", rows, cols, wrong)
    with pytest.raises(ValueError, match="do not belong together"):
        flowtopo.dfs_interval_geotiff(seq, upg_path, str(tmp_path / "i.tif"), str(tmp_path / "j.tif"),
                                      str(tmp_path / "wrong.csv"))
    other = upg_grid.copy()
    other[rows[-1], cols[-1]] += 1
    write(tmp_path / "other.tif", other, (10.0, cell, 0.0, 40.0, 0.0, -cell))
    with pytest.raises(ValueError, match="an outlet has a basin_cell_count other than upg: row"):
        flowtopo.dfs_interval_geotiff(seq, str(tmp_path / "other.tif"), str(tmp_path / "i.tif"),
                                      str(tmp_path / "j.tif"), str(tmp_path / "bat.csv"))
    (tmp_path / "short.csv").write_text("local_outlet_row,local_outlet_col\n0,0\n")
    with pytest.raises(ValueError, match="no column basin_cell_count"):
        flowtopo.dfs_interval_geotiff(seq, upg_path, str(tmp_path / "i.tif"), str(tmp_path / "j.tif"),
                                      str(tmp_path / "short.csv"))
    # a line that is short, not an integer, or off the raster, by its line number
    head = "local_outlet_row,local_outlet_col,basin_cell_count\n"
    for name, body, message in (("few", "0,0,3\n0,0\n", "line 3 of .*few.csv has too few fields"),
                                ("text", "0,x,3\n", "line 2 of .*text.csv has no integer local_outlet_col"),
                                ("big", f"0,0,{1 << 63}\n", "line 2 of .*big.csv has no integer basin_cell_count"),
                                ("off", "0,0,3\n\n20,0,3\n", r"outside seq_dfs \(line 4 of .*off.csv\)"),
                                # each field read is a plain integer: no spaces, quotes, underscores
                                # or other digits, as the C tool reads it
                                ("cut", "0,x\n", "line 2 of .*cut.csv has no integer local_outlet_col"),
                                ("space", "0,0,3 \n", "line 2 of .*space.csv has no integer basin_cell_count"),
                                ("lead", " 0,0,3\n", "line 2 of .*lead.csv has no integer local_outlet_row"),
                                ("quoted", '"0",0,3\n', "line 2 of .*quoted.csv has no integer local_outlet_row"),
                                ("under", "0,0,1_0\n", "line 2 of .*under.csv has no integer basin_cell_count"),
                                ("wide", "0,0,\uff13\n", "line 2 of .*wide.csv has no integer basin_cell_count"),
                                # a NUL byte in a field, and a lone CR before the line end
                                ("nul", "0,0,1\x00junk\n", "line 2 of .*nul.csv has no integer basin_cell_count"),
                                ("cr", "0,0,3\r\r\n", "line 2 of .*cr.csv has no integer basin_cell_count")):
        (tmp_path / f"{name}.csv").write_bytes((head + body).encode("utf-8"))
        with pytest.raises(ValueError, match=message):
            flowtopo.dfs_interval_geotiff(seq, upg_path, str(tmp_path / "i.tif"), str(tmp_path / "j.tif"),
                                          str(tmp_path / f"{name}.csv"))
    # a table with Windows line ends reads the same
    (tmp_path / "crlf.csv").write_bytes((tmp_path / "bat.csv").read_bytes().replace(b"\n", b"\r\n"))
    flowtopo.dfs_interval_geotiff(seq, upg_path, str(tmp_path / "in2.tif"), str(tmp_path / "out2.tif"),
                                  str(tmp_path / "crlf.csv"))
    with rasterio.open(tmp_path / "in2.tif") as src:
        assert np.array_equal(src.read(1), want_in)
    # and so does one with 4300 zeros before every field, more digits than int() reads
    head_line, *body = (tmp_path / "bat.csv").read_bytes().split(b"\n")
    (tmp_path / "zeros.csv").write_bytes(b"\n".join(
        [head_line] + [b",".join(b"0" * 4300 + v for v in line.split(b",")) if line else line for line in body]))
    flowtopo.dfs_interval_geotiff(seq, upg_path, str(tmp_path / "in3.tif"), str(tmp_path / "out3.tif"),
                                  str(tmp_path / "zeros.csv"))
    with rasterio.open(tmp_path / "in3.tif") as src:
        assert np.array_equal(src.read(1), want_in)
    # an outlet missing from the table
    write_table(tmp_path / "missing.csv", rows[1:], cols[1:], counts[1:])
    with pytest.raises(ValueError, match="do not belong together"):
        flowtopo.dfs_interval_geotiff(seq, upg_path, str(tmp_path / "i.tif"), str(tmp_path / "j.tif"),
                                      str(tmp_path / "missing.csv"))
    assert leftovers(tmp_path, "i", "j") == []


def test_a_position_held_twice_is_refused(tmp_path):
    # two cells at position 0 and none at 1: the cell counts and the blocks do not show it
    with pytest.raises(ValueError, match="no cell at position 1 and another position twice"):
        flowtopo.dfs_interval(np.array([0, 0, 2]), np.array([1, 1, 3]), np.array([2]))
    topo = flowtopo.FlowTopo.from_d8(synthetic_d8(16, 16, seed=4), transform=TRANSFORM)
    seq_dfs, upg = released_layers(topo)
    outlets = pits(topo)
    # a source cell, which is not an outlet, moved onto the position before its own
    cell = next(int(c) for c in np.flatnonzero((upg == 1) & (seq_dfs > 0)) if c not in set(outlets))
    moved = seq_dfs.copy()
    moved[cell] -= 1
    at = int(seq_dfs[cell])
    with pytest.raises(ValueError, match=f"no cell at position {at} and another position twice"):
        flowtopo.dfs_interval(moved, upg, outlets)

    rasterio = pytest.importorskip("rasterio")
    cell_size = 1 / 1200
    transform = (10.0, cell_size, 0.0, 40.0, 0.0, -cell_size)
    write(tmp_path / "seq.tif", moved.reshape(16, 16), transform)
    write(tmp_path / "upg.tif", upg.reshape(16, 16), transform)
    with pytest.raises(ValueError, match=f"seq.tif has no cell at position {at} and another position twice"):
        flowtopo.dfs_interval_geotiff(str(tmp_path / "seq.tif"), str(tmp_path / "upg.tif"),
                                      str(tmp_path / "i.tif"), str(tmp_path / "j.tif"),
                                      np.divmod(outlets, 16), rows=5)
    assert leftovers(tmp_path, "i", "j") == []


def test_geotiff_refuses_rasters_that_are_not_integers(tmp_path):
    pytest.importorskip("rasterio")
    topo = flowtopo.FlowTopo.from_d8(synthetic_d8(16, 16, seed=4), transform=TRANSFORM)
    seq_dfs, upg = released_layers(topo)
    seq_grid, upg_grid = seq_dfs.reshape(16, 16), upg.reshape(16, 16)
    outlets = np.divmod(pits(topo), 16)
    cell = 1 / 1200
    transform = (10.0, cell, 0.0, 40.0, 0.0, -cell)
    write(tmp_path / "seq.tif", seq_grid, transform)
    write(tmp_path / "upg.tif", upg_grid, transform)
    write(tmp_path / "seq_float.tif", seq_grid.astype(np.float32), transform)
    write(tmp_path / "seq_unsigned.tif", np.where(seq_grid < 0, 0, seq_grid).astype(np.uint32), transform)
    write(tmp_path / "upg_float.tif", upg_grid.astype(np.float32), transform)
    for seq, up in (("seq_float", "upg"), ("seq_unsigned", "upg"), ("seq", "upg_float")):
        with pytest.raises(ValueError, match="seq_dfs must hold signed integers and upg integers"):
            flowtopo.dfs_interval_geotiff(str(tmp_path / f"{seq}.tif"), str(tmp_path / f"{up}.tif"),
                                          str(tmp_path / "i.tif"), str(tmp_path / "j.tif"), outlets)
    assert leftovers(tmp_path, "i", "j") == []


def test_geotiff_reads_uint64_upg_as_int64(tmp_path):
    # as the C tool reads it through GDAL: values past int64 are held at its largest
    rasterio = pytest.importorskip("rasterio")
    topo = flowtopo.FlowTopo.from_d8(synthetic_d8(16, 16, seed=4), transform=TRANSFORM)
    seq_dfs, upg = released_layers(topo)
    seq_grid, upg_grid = seq_dfs.reshape(16, 16), upg.reshape(16, 16)
    rows, cols = np.divmod(pits(topo), 16)
    cell = 1 / 1200
    transform = (10.0, cell, 0.0, 40.0, 0.0, -cell)
    write(tmp_path / "seq.tif", seq_grid, transform)
    write(tmp_path / "upg.tif", upg_grid.astype(np.uint64), transform)
    write_table(tmp_path / "bat.csv", rows, cols, upg_grid[rows, cols])
    seq, bat = str(tmp_path / "seq.tif"), str(tmp_path / "bat.csv")
    flowtopo.dfs_interval_geotiff(seq, str(tmp_path / "upg.tif"), str(tmp_path / "in.tif"),
                                  str(tmp_path / "out.tif"), bat)
    want_in, want_out = flowtopo.dfs_interval(seq_grid, upg_grid, (rows, cols))
    with rasterio.open(tmp_path / "in.tif") as src:
        assert np.array_equal(src.read(1), want_in)
    with rasterio.open(tmp_path / "out.tif") as src:
        assert np.array_equal(src.read(1), want_out)
    k = int(np.lexsort((cols, rows))[0])
    huge = upg_grid.astype(np.uint64)
    huge[rows[k], cols[k]] = np.iinfo(np.uint64).max
    write(tmp_path / "huge.tif", huge, transform)
    with pytest.raises(ValueError, match=f"row {rows[k]}, column {cols[k]}, {upg_grid[rows[k], cols[k]]} "
                                         f"in the table, 9223372036854775807 in upg"):
        flowtopo.dfs_interval_geotiff(seq, str(tmp_path / "huge.tif"), str(tmp_path / "i.tif"),
                                      str(tmp_path / "j.tif"), bat)
    assert leftovers(tmp_path, "i", "j") == []

# Changelog

## 1.1.3 (2026-10-31)

* `dfs_in` is numbered within each basin: `dfs_in = a + b - seq_dfs`, where
  `b` is `seq_dfs` at the basin's outlet and `a = b - upg + 1` with `upg` at the
  outlet, so a basin keeps the block of numbers it has in `seq_dfs`, its outlet
  at `a`. 1.1.2 read the whole region backward, `dfs_in = n - 1 - seq_dfs`; the
  two differ by a constant within each basin and give the same upstream cells,
  but the numbers differ, and those of 1.1.3 are the ones FullBasin v1.1
  documents. `dfs_out = dfs_in + upg - 1` and the test
  `dfs_in[p] <= dfs_in[q] <= dfs_out[p]` are unchanged.
* `dfs_interval(seq_dfs, upg, outlets)` and
  `dfs_interval_geotiff(..., outlets, rows=512)` take the outlets of the
  region's basins: index arrays on the array form, and on the GeoTIFF form the
  path of the region's basin table of FullBasin v1.1 (`local_outlet_row`,
  `local_outlet_col`, `basin_cell_count`) or `(rows, cols)`. The basins must
  take positions 0 to `n - 1` one after another, no position may be held by
  two cells, and `basin_cell_count` must equal `upg` at each outlet; otherwise
  the inputs are refused. These checks cannot prove that `seq_dfs` is a
  depth-first order of the flow network: `seq_dfs` and `upg` must come from
  the same flow network. The table is read as FullBasin writes it, fields
  split at every comma and each a plain integer; the line of a field that is
  missing or not an integer is named. On the GeoTIFF form `seq_dfs` must hold
  signed integers and `upg` integers, and one bit is kept for each cell of the
  sequence. The `n` argument of 1.1.2 is gone: the array form reads a whole
  region.
* `tests/test_dfs_interval.py` checks, on grids of several basins, that each
  basin's `dfs_in` is its block of `seq_dfs` read backward, that it differs
  from `n - 1 - seq_dfs` by a constant within the basin, and that missing,
  extra and repeated outlets, a table that disagrees with `upg`, a table with
  a field that is not a plain integer, a position held by two cells and
  rasters that are not integers are refused; a table with CRLF line ends gives
  the same files.
* The C tool `dfs_interval/` (MERIT-FlowTopo code 1.1.1) takes the basin table
  and gives the same files.
* Other results are unchanged.

## 1.1.2 (2026-10-28)

* `dfs_interval(seq_dfs, upg)` and `dfs_interval_geotiff(...)` give `dfs_in` and
  `dfs_out`, the depth-first interval of every cell, from a `seq_dfs` layer of
  FullTopo v1.1 and MERIT Hydro's `upg`: `dfs_in = n - 1 - seq_dfs`,
  `dfs_out = dfs_in + upg - 1`, and cell `q` drains through cell `p` exactly when
  `dfs_in[p] <= dfs_in[q] <= dfs_out[p]`. The GeoTIFF form reads and writes one
  region, from the region's `seq_dfs` and `upg`, a strip of rows at a time; a
  larger `upg` is read as well, across the antimeridian.
  `tests/test_dfs_interval.py` checks the intervals against the upstream cells of
  every cell.
* The C code archived with this release (MERIT-FlowTopo code 1.1.0,
  [10.5281/zenodo.22227620](https://doi.org/10.5281/zenodo.22227620)) has the same
  tool, `dfs_interval/`, which gives the same files, and the two partitions,
  `partition/`, which give the same subregions as `FlowTopo.partition` on the
  same METIS.
* Other results are unchanged.

## 1.1.1 (2026-10-28)

* The bundled MERIT Hydro v1.0.1 excerpt, in `data/` and in `tests/reference/expected.npz`,
  is distributed under CC BY-NC 4.0 or ODbL 1.0, at the user's choice, as stated in
  `DATA_NOTICE.md` and the README.
* Documentation and comments describe the methods themselves.
* The README and guide describe the 1.1 products only; the 1.0 records are
  cited from the paper.
* The FullTopo and region-map links are fixed.
* Six tests are renamed for the reference values they compare with.
* Results are unchanged.

## 1.1.0 (2026-10-23)

Released with [FullTopo v1.1](https://fullhydro.org/downloads/merit-hydro-v1.0.1-90m/fullbasin/fulltopo/v1.1/),
the structures on the 90 regions of FullDivide v1.1.

* The bundled example's header holds MERIT Hydro's grid exactly: origin
  (117.40291666666664, 24.445416666666667), cell size 1/1200 degree.
  The outlet and basin outline were moved with it; `tests/test_example_data.py`
  checks all three against the grid.
* Descriptions of the structures, kernels and locality metrics are clarified.

The changes before 1.1 are in [`CHANGELOG.md` at `v1.0.7`](https://github.com/hellolulujiang/flowtopo/blob/v1.0.7/CHANGELOG.md).

# Changelog

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

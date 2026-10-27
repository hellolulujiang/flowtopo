# Notice on the bundled data

The MIT terms in [`LICENSE`](LICENSE) cover the code and the documentation in this
repository: `flowtopo/`, `tests/` (but not `tests/reference/expected.npz`), `examples/`,
`docs/`, and the scripts at the top level. They do not cover the files in `data/` and
that file.

`data/dir_example.tif` is an excerpt of MERIT Hydro v1.0.1, 292 rows by 614 columns at 3
arc-seconds, rows 72,665 to 72,956 and columns 356,884 to 357,497 of its global grid
(Yamazaki et al., 2019, <https://doi.org/10.1029/2019WR024873>), which its authors
distribute under CC BY-NC 4.0 or ODbL 1.0, at the user's choice, terms that also hold
for data derived from it (<https://global-hydrodynamics.github.io/MERIT_Hydro/>). It is
redistributed here under the same two licences, either of them at your choice, for
reproducibility, with attribution to its authors. `data/bsn_example.geojson` and
`data/outlet_example.geojson` are the basin boundary and the outlet derived from it, and
carry the same terms. `tests/reference/expected.npz` holds the same flow directions and
the outputs computed from them, and carries the same terms too.

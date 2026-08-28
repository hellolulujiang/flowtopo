# FlowTopo

[![licence: MIT](https://img.shields.io/badge/licence-MIT-blue.svg)](LICENSE)

Orderings, layerings and kernels for D8 flow networks, in Python.

FlowTopo builds reusable traversal structures for a D8 flow network: three
serial orderings and three parallel layerings, computed once from the
flow-direction grid. Any flow-network computation can then run on them.

Four kernels are included to test the structures — upstream drainage area,
distance to outlet, longest upstream path, Strahler stream order. Each runs on
every structure and the results are cross-checked.

Video overview of the six structures and the three propagation manners:
<https://youtu.be/tE5K2wM3TTY>. The animations below are small previews: the
**▶ HD video** link under each one opens the full-resolution video, which can
be paused, scrubbed, restarted and viewed fullscreen. The video opens in this
tab; use the browser's Back button to return, or Ctrl/Cmd-click the link to
open it in a new tab. The same files are archived in
[`docs/media`](docs/media).

## Serial orderings

An ordering is a topological sort of the valid cells: every cell comes after
the cells it depends on. One pass over the sequence computes any kernel.

| topological sort from the sources | breadth-first from the pit | depth-first from the pit |
| :---: | :---: | :---: |
| [![](docs/media/seq_topo.gif)](https://github.com/user-attachments/assets/802b448c-46a1-4d7f-a1e3-c2cd1f1c9385) | [![](docs/media/seq_bfs.gif)](https://github.com/user-attachments/assets/5680a958-2d7e-4003-80af-0a1ac4ab769e) | [![](docs/media/seq_dfs.gif)](https://github.com/user-attachments/assets/0f48bc7a-98f2-4948-ae02-8769db772250) |
| a cell is appended once all its donors are done | cells in order of hop count from the pit | one tributary subtree at a time |
| [▶ HD video](https://github.com/user-attachments/assets/802b448c-46a1-4d7f-a1e3-c2cd1f1c9385) | [▶ HD video](https://github.com/user-attachments/assets/5680a958-2d7e-4003-80af-0a1ac4ab769e) | [▶ HD video](https://github.com/user-attachments/assets/0f48bc7a-98f2-4948-ae02-8769db772250) |

The three differ in memory access pattern. Depth-first has the lowest simulated
L1 miss rate on the example basin: 10.6%, against 21.8% (breadth-first) and
37.0% (topological sort).

## Parallel layerings

A layering assigns every cell a layer index such that cells in the same layer
are independent of each other. Layers run in order; cells within a layer run
in parallel. Layer 0 holds the headwaters.

| as soon as possible | conflict-free downstream | as late as possible |
| :---: | :---: | :---: |
| [![](docs/media/lyr_asap.gif)](https://github.com/user-attachments/assets/b204458c-1dbf-4cac-ac2b-a5bee193eb79) | [![](docs/media/lyr_cfds.gif)](https://github.com/user-attachments/assets/c7df9a2f-8ec4-475d-b9ed-c963ef762d54) | [![](docs/media/lyr_alap.gif)](https://github.com/user-attachments/assets/4407d9af-47e6-4fcc-b409-ea4bf850d3ff) |
| every cell in the earliest layer its donors allow | as soon as possible, plus one rule: no two cells in a layer share a receiver | every cell in the latest layer possible |
| [▶ HD video](https://github.com/user-attachments/assets/b204458c-1dbf-4cac-ac2b-a5bee193eb79) | [▶ HD video](https://github.com/user-attachments/assets/c7df9a2f-8ec4-475d-b9ed-c963ef762d54) | [▶ HD video](https://github.com/user-attachments/assets/4407d9af-47e6-4fcc-b409-ea4bf850d3ff) |

The minimum layer count is set by the longest flow path. The conflict-free rule
may add a few layers; on the example basin it adds none (949 layers for all
three).

## Write conflicts

Three ways to propagate values through the network:

| pull | atomic push | push |
| :---: | :---: | :---: |
| [![](docs/media/manner_pull.gif)](https://github.com/user-attachments/assets/c897fd03-b2dd-4371-aba8-cd2bd0692857) | [![](docs/media/manner_atomic_push.gif)](https://github.com/user-attachments/assets/f988af11-ce25-4fb4-9edd-1bfa5b025dde) | [![](docs/media/manner_push.gif)](https://github.com/user-attachments/assets/35cad382-2d3c-4378-aab4-7df1c9fc5384) |
| each receiver reads its donors; needs the upstream table | donors write through atomics; correct, but float sums are not reproducible | donors write directly; deterministic, no locks; requires the conflict-free layering |
| [▶ HD video](https://github.com/user-attachments/assets/c897fd03-b2dd-4371-aba8-cd2bd0692857) | [▶ HD video](https://github.com/user-attachments/assets/f988af11-ce25-4fb4-9edd-1bfa5b025dde) | [▶ HD video](https://github.com/user-attachments/assets/35cad382-2d3c-4378-aab4-7df1c9fc5384) |

A push is only safe if no two cells in a layer write to the same receiver.
Conflict counts on the example basin (93,432 cells):

| layering | conflicting writes inside a layer |
| --- | --- |
| as soon as possible | 12,122 |
| **conflict-free downstream** | **0** |
| as late as possible | 39,130 |

The count is a property of the layering and can be checked before running. A
test run is not a reliable check: a race does not always trigger. Strahler
order has no atomic form, because its confluence rule is a comparison rather
than an addition, so its only parallel push is under the conflict-free
layering.

If `manner` is not given, FlowTopo picks a safe one for the layering.

## Which structure to use

From the paper's benchmark on the full 90 m network (22.2 billion cells,
65 regions):

* **One sweep, one core** — the depth-first ordering. Up to 5.1× faster than
  the slowest serial ordering; subtree contiguity drops the L3 miss rate from
  37% to about 6%. It is also the baseline to measure parallel speedup
  against; a slower baseline overstates the speedup.
* **Repeated traversal** (calibration, ensembles) — the as-late-as-possible
  layering with pull. Fastest in parallel: each layer's working set stays in
  cache. Pull must store the donor table.
* **Non-linear kernels, or when RAM is tight** — the conflict-free downstream
  layering with push. Lock-free and deterministic, stores only the receiver
  pointer, and the only parallel option for Strahler order.
* **The largest basins** — the subbasin partition, released with
  MERIT-FlowTopo.

The structures are computed once from the static D8 field and reused without
limit.

## Install

```sh
pip install -e .            # numpy + rasterio
pip install -e ".[speed]"   # + numba, for threaded kernels
```

Requires Python ≥ 3.9. Without numba everything still runs, in pure Python.

## Quick start

```python
import flowtopo

topo = flowtopo.FlowTopo.from_raster("data/dir_example.tif")

upa = topo.upstream_area(ordering="dfs")                      # serial
upa = topo.upstream_area(layering="cfds", manner="push")      # a layer at a time
ldn = topo.distance_to_outlet(ordering="dfs")
lup = topo.longest_upstream_path(ldn, ordering="dfs")
strord = topo.strahler_order(ordering="dfs",
                             channel_mask=topo.channel_mask(upa, 10.0))

topo.to_2d(upa)                                               # back on the grid
```

With numba, the same kernels with threads:

```python
from flowtopo import parallel
upa = parallel.upstream_area(topo, layering="cfds", manner="push")
```

Run everything on the example basin:

```sh
python example.py       # all structures, kernels and manners; ends PASS or FAIL
python benchmark.py     # serial vs threaded at several grid sizes
```

## Documentation

* [`docs/user-guide.md`](docs/user-guide.md) — choosing an ordering, a layering
  and a manner; threads; raster I/O.
* [`docs/methods.md`](docs/methods.md) — each method with its origin and its
  complexity.
* [`examples/quickstart.ipynb`](examples/quickstart.ipynb) — a notebook on the
  bundled data, stored with its output so it reads without running anything.

## Example data

`data/dir_example.tif` is the example basin of the paper: 614 × 292 cells at
3 arc-seconds, 93,432 valid, 731 km², cut from
[MERIT Hydro](https://doi.org/10.1029/2019WR024873) (Yamazaki et al., 2019).
The GeoJSON files are the basin boundary and the outlet. Any D8 GeoTIFF in the
same convention works:

```sh
python example.py --data my_dir.tif
```

## Global products

This repository holds the method and one example basin. The global 90 m
products are on Zenodo, as per-region GeoTIFFs for 65 regions:

* **MERIT-FlowTopo** ([10.5281/zenodo.20653059](https://doi.org/10.5281/zenodo.20653059)) —
  the precomputed structures: depth-first sequence, conflict-free downstream
  and as-late-as-possible layerings, subbasin partition; all eight structures
  for Region 43 (South China).
* **MERIT-DrainAttr** ([10.5281/zenodo.20686665](https://doi.org/10.5281/zenodo.20686665)) —
  flow length downstream, flow length upstream and Strahler order for every
  region.
* **MERIT-FullBasin** ([10.5281/zenodo.20344113](https://doi.org/10.5281/zenodo.20344113)) —
  the partition into the 65 regions.

## Acknowledgements

**MERIT Hydro** (Yamazaki et al., 2019;
[10.1029/2019WR024873](https://doi.org/10.1029/2019WR024873)) provides the
flow-direction field everything here traverses: the example basin is cut from
it, and the global products are built on its 90 m network. The bundled excerpt
keeps its CC BY-NC 4.0 terms.

The representation this package works on comes from
[pyflwdir](https://github.com/Deltares/pyflwdir) (D. Eilander, Deltares and the
Institute for Environmental Studies, Vrije Universiteit Amsterdam; MIT licence;
[10.5281/zenodo.4287337](https://doi.org/10.5281/zenodo.4287337)). What FlowTopo
takes from it:

* **the flat downstream-pointer array** — for every cell, the linear index of
  the cell it drains into, with a pit pointing at itself. Every structure and
  every kernel here is derived from that one array;
* **the D8 decoding conventions** — codes as powers of two clockwise from east,
  0 and 255 terminal, 247 nodata, and a cell draining off the grid or into
  nodata treated as a pit;
* **the donor-count array** and its sentinel convention;
* **the chain-tracing rank and the breadth-first sequence builder**, which
  appear here in the variants set out in [`docs/methods.md`](docs/methods.md).

No pyflwdir source is included. The code here was written against those
conventions rather than copied from them, and `FlowTopo.from_d8` takes the same
array `pyflwdir.from_array(d8, ftype="d8")` takes, so the two read the same
rasters.

The three layerings, the conflict-free downstream rule, the propagation
manners, the locality metrics and the benchmark drivers are this project's own.

## Verification

Expected outputs for the example basin are stored with the tests, so a fresh
clone verifies itself:

```sh
pytest
```

27 tests: every ordering, layering, kernel and manner is cross-checked on the
example basin, including the write-conflict counts.

## Citing

See [`CITATION.cff`](CITATION.cff). The companion manuscript is
*MERIT-FlowTopo v1.0: a reusable computational foundation for hyperresolution
hydrology on the global 90 m drainage network* (Jiang et al., in preparation).

## Licence

MIT for the code. The bundled MERIT Hydro excerpt keeps its own CC BY-NC 4.0
terms; see [`LICENSE`](LICENSE).

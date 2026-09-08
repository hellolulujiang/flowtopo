# FlowTopo

[![tests](https://github.com/hellolulujiang/flowtopo/actions/workflows/tests.yml/badge.svg)](https://github.com/hellolulujiang/flowtopo/actions/workflows/tests.yml)
[![licence: MIT](https://img.shields.io/badge/licence-MIT-blue.svg)](LICENSE)

Orderings, layerings and partitions for D8 flow networks, in Python.

A D8 flow-direction grid constrains the order in which cells can be visited:
a cell can be processed only after its upstream neighbours, or, for some
kernels, only after its downstream one. FlowTopo builds such an order once,
from the flow-direction grid alone, and stores it as a reusable structure.
Any routine that walks the network in drainage order (drainage area, flow
length, stream order, or the upstream-to-downstream step of a routing scheme)
reads the structure instead of rebuilding the traversal each time.

This repository is the Python reference implementation with one example
basin. The structures for the 90 m MERIT Hydro network are on Zenodo
([Global products](#global-products)). The companion paper (Jiang et al.) is
in preparation. A ten-minute narrated video shows the structures and the three
ways of moving a value to its receiver: <https://youtu.be/tE5K2wM3TTY>.

## Install

```sh
git clone https://github.com/hellolulujiang/flowtopo.git
cd flowtopo
pip install -e .            # numpy + rasterio: structures and serial kernels
pip install -e ".[speed]"   # adds numba: the threaded kernels need it
```

Python 3.10 or newer. Without numba the structures and the serial kernels run
in pure Python; `flowtopo.parallel` raises rather than pretending to thread.

## Quick start

```python
import flowtopo

topo = flowtopo.FlowTopo.from_raster("data/dir_example.tif")

upa = topo.upstream_area(ordering="dfs")           # km², one serial pass
ldn = topo.distance_to_outlet(ordering="dfs")      # flow length downstream, m
strord = topo.strahler_order(                      # on cells draining ≥ 10 km²
    ordering="dfs", channel_mask=topo.channel_mask(upa, 10.0))

upa_grid = topo.to_2d(upa)                         # back on the raster grid
```

The same kernels on threads, a layer at a time (needs numba):

```python
from flowtopo import parallel
upa = parallel.upstream_area(topo, layering="cfds", manner="push")
```

## Example data

`data/dir_example.tif` is the example basin of the paper: 292 rows by 614
columns at 3 arc-seconds, 93,432 valid cells, 731 km², cut from
[MERIT Hydro](https://doi.org/10.1029/2019WR024873) (Yamazaki et al., 2019)
and kept under its CC BY-NC 4.0 terms ([`DATA_NOTICE.md`](DATA_NOTICE.md)).
The GeoJSON files are the basin boundary and the outlet.

Any D8 GeoTIFF in the same convention works: codes are powers of two clockwise
from east. By default 0 and 255 are terminals and 247 is nodata; a nodata
value declared in the file overrides that default.

Work one region or basin at a time. Indices are int32, so a raster must stay
under 2.1 billion cells, nodata included; the 38° × 38° regions of
MERIT-FullBasin do.

## What you get

Eight structures and three ways to move a value, all built from the D8 grid.
Each answers one question; each animation runs on the same two-basin example
and links to its full-size MP4 (all nine also play on the
[animation page](https://hellolulujiang.github.io/flowtopo/)).

<table>
<tr><th align="left">structure</th><th align="left">keyword</th><th align="left">rule</th><th align="left">animation</th></tr>
<tr><td colspan="4"><b>One core: in what order are the cells visited?</b> &nbsp;·&nbsp; serial orderings, one pass each</td></tr>
<tr><td>topological sort from the sources</td><td><code>ordering="topo"</code></td><td>a cell is appended once all its donors are done</td><td><a href="https://hellolulujiang.github.io/flowtopo/media/seq_topo.mp4"><img src="docs/media/seq_topo.gif" width="260"></a></td></tr>
<tr><td>breadth-first from the pit</td><td><code>ordering="bfs"</code></td><td>cells in order of hop count from the pit</td><td><a href="https://hellolulujiang.github.io/flowtopo/media/seq_bfs.mp4"><img src="docs/media/seq_bfs.gif" width="260"></a></td></tr>
<tr><td>depth-first from the pit</td><td><code>ordering="dfs"</code></td><td>one tributary subtree at a time; best cache locality</td><td><a href="https://hellolulujiang.github.io/flowtopo/media/seq_dfs.mp4"><img src="docs/media/seq_dfs.gif" width="260"></a></td></tr>
<tr><td colspan="4"><b>Many threads on one processor: which cells run together?</b> &nbsp;·&nbsp; parallel layerings; layers run in order, cells within a layer in parallel</td></tr>
<tr><td>as soon as possible</td><td><code>layering="asap"</code></td><td>every cell in the earliest layer its donors allow</td><td><a href="https://hellolulujiang.github.io/flowtopo/media/lyr_asap.mp4"><img src="docs/media/lyr_asap.gif" width="260"></a></td></tr>
<tr><td>conflict-free downstream</td><td><code>layering="cfds"</code></td><td>as soon as possible, plus: no two cells in a layer share a receiver</td><td><a href="https://hellolulujiang.github.io/flowtopo/media/lyr_cfds.mp4"><img src="docs/media/lyr_cfds.gif" width="260"></a></td></tr>
<tr><td>as late as possible</td><td><code>layering="alap"</code></td><td>every cell as late as the longest flow path allows; the most evenly filled layers in practice</td><td><a href="https://hellolulujiang.github.io/flowtopo/media/lyr_alap.mp4"><img src="docs/media/lyr_alap.gif" width="260"></a></td></tr>
<tr><td colspan="4"><b>Several processors: which cells go where?</b> &nbsp;·&nbsp; spatial partitions, one subregion per processor, no value crosses a boundary</td></tr>
<tr><td>basin-level</td><td><code>level="basin"</code></td><td>whole basins dealt to subregions; one dominant basin cannot be balanced</td><td rowspan="2"><a href="docs/media/partition_schematic.png"><img src="docs/media/partition_schematic.png" width="320"></a></td></tr>
<tr><td>subbasin-level</td><td><code>level="subbasin"</code></td><td>the dominant basin split along its mainstem; tributary subtrees move to the lighter subregion, the mainstem runs in a separate stage</td></tr>
<tr><td colspan="4"><b>At a confluence: how does a value reach the receiver?</b> &nbsp;·&nbsp; propagation manners</td></tr>
<tr><td>pull</td><td><code>manner="pull"</code></td><td>each receiver reads its donors and writes only itself; needs the donor table</td><td><a href="https://hellolulujiang.github.io/flowtopo/media/manner_pull.mp4"><img src="docs/media/manner_pull.gif" width="150"></a></td></tr>
<tr><td>atomic push</td><td><code>manner="atomic_push"</code></td><td>donors write through atomics; correct, but float sums are not reproducible</td><td><a href="https://hellolulujiang.github.io/flowtopo/media/manner_atomic_push.mp4"><img src="docs/media/manner_atomic_push.gif" width="150"></a></td></tr>
<tr><td>push</td><td><code>manner="push"</code></td><td>donors write directly; deterministic, no locks; safe only under <code>cfds</code></td><td><a href="https://hellolulujiang.github.io/flowtopo/media/manner_push.mp4"><img src="docs/media/manner_push.gif" width="150"></a></td></tr>
</table>

Three things to know:

* An ordering is built in one direction and reversed on demand. Kernels that
  accumulate into the receiver (drainage area, flow length upstream, Strahler
  order) walk upstream to downstream; flow length downstream walks the other
  way. The kernels flip the sequence for you; `topo.ordering("dfs", "u2d")`
  asks for a direction explicitly.
* The layer count is set by the longest flow path, and the conflict-free rule
  may add layers (on the example basin it adds none: 949 for all three). In
  the two headwater-anchored schemes layer 0 holds every headwater; in
  as-late-as-possible it holds only the farthest ones.
* `topo.partition(n_parts, level)` returns a label per cell and the load per
  subregion. Subbasin-level marks the mainstem `flowtopo.MAINSTEM` and leaves
  it to a separate stage: after the tributary subregions for kernels that
  accumulate downstream, before them for flow length downstream.

Four kernels are bundled to exercise the structures: upstream drainage area,
flow length downstream (`distance_to_outlet`), flow length upstream
(`longest_upstream_path`) and Strahler stream order. Each runs on every
supported combination of structure and manner, and the results are
cross-checked.

<details>
<summary>Write conflicts and cache locality on the example basin</summary>

A push is only safe if no two cells in a layer write to the same receiver.
Conflicting writes inside a layer, example basin (93,432 cells):

| layering | conflicting writes |
| --- | --- |
| as soon as possible | 12,122 |
| **conflict-free downstream** | **0** |
| as late as possible | 39,130 |

The count is a property of the layering and can be checked before running; a
test run is not a reliable check, because a race does not always trigger.
Because no two cells in a layer share a receiver, the conflict-free push sums
in the same order every time and returns bit-identical results at any thread
count. Strahler order has no single-atomic form, because its confluence rule
is a comparison and a count rather than one addition, so its only parallel
push is under the conflict-free layering. If `manner` is not given, FlowTopo
picks a safe one for the layering.

The three serial orderings differ in memory access. Simulated L1 miss rate on
the example basin: depth-first 10.5%, breadth-first 21.8%, topological sort
37.1% (`flowtopo.locality.miss_rates`). The figures shift by a few points with
how the two arrays are aligned in memory; their order does not.

</details>

## Which structure to use

From the paper's benchmark of the C implementation on the 90 m network
(22.2 billion cells, 65 regions):

* **One sweep, one core** — the depth-first ordering. Up to 5.1× faster than
  the slowest serial ordering; subtree contiguity drops the L3 miss rate from
  37% to about 6%. It is also the baseline to measure parallel speedup
  against; a slower baseline overstates the speedup.
* **Repeated traversal** (calibration, ensembles) — the as-late-as-possible
  layering, fastest in parallel for all four kernels because each layer's
  working set stays in cache. Under it, atomic push beat pull for the sum and
  maximum kernels (71.1 s against 85.1 s for flow length upstream); pull is
  the choice where the kernel has no atomic form, or where the sum must be
  reproducible, and it must store the donor table.
* **Non-linear kernels under push, or when RAM is tight** — the conflict-free
  downstream layering with push. Lock-free and deterministic, stores only the
  receiver pointer, and the only push that runs Strahler order at all. Pull
  runs it under any layering and was faster under as-late-as-possible (4.8 s
  against 9.1 s); push wins when the donor table does not fit.
* **Across processors** — the subbasin partition, one subregion per processor,
  each threaded to the point at which its processor's memory bandwidth
  saturates. Where that point lies depends on the processor; the paper's
  Fig. 15 shows it for the server tested, and it has to be measured again on
  other hardware.

This package is numba over numpy at a much smaller size and does not
reproduce those rankings term for term: on a 16-million-cell grid, push under
`cfds` is the fastest threaded form and pull loses even to the serial
sequence, because building and reading the donor table costs more than ten
threads save. Which one wins depends on your machine and your grid, so
measure with `benchmark.py` before choosing.

## Global products

Applied to the 90 m MERIT Hydro network with the C implementation, the same
methods produce two Zenodo records of per-region GeoTIFFs. Both cover the 65
continental regions of MERIT-FullBasin; the 29 island groups and the two
regions straddling the antimeridian are not included.

**MERIT-FlowTopo** ([10.5281/zenodo.20653059](https://doi.org/10.5281/zenodo.20653059))
holds the structures: for every region the depth-first sequence, the
conflict-free downstream and as-late-as-possible layerings and the subbasin
partition; for Region 43 (southern China) all eight, so the alternatives can be
compared somewhere. 978 GB uncompressed, 50 GB compressed.

**MERIT-DrainAttr** ([10.5281/zenodo.20686665](https://doi.org/10.5281/zenodo.20686665))
holds the kernels run over those structures: flow length downstream, flow
length upstream and Strahler stream order for every region, and upstream
drainage area for Region 43 only, since MERIT Hydro already distributes it
globally. 622 GB uncompressed, 60 GB compressed.

**MERIT-FullBasin** ([10.5281/zenodo.20344113](https://doi.org/10.5281/zenodo.20344113))
is the companion dataset that divides the network into 96 hydrologically
independent regions, the 65 continental ones being those above.

**MERIT-FlowTopo code** ([10.5281/zenodo.22227621](https://doi.org/10.5281/zenodo.22227621))
is the record the paper cites for code: the C implementation that produced
the two records above, the scripts and data behind every figure, and an
archived copy of this package at v1.0.

<details>
<summary>The structures and the kernel products over the 65 regions</summary>

[![](docs/media/global_orderings.png)](docs/media/global_orderings.png)

[![](docs/media/global_layerings.png)](docs/media/global_layerings.png)

[![](docs/media/global_partitions.png)](docs/media/global_partitions.png)

[![](docs/media/kernel_products.png)](docs/media/kernel_products.png)

</details>

### Using the released structures

The structures index into the MERIT Hydro flow-direction grid, which is not
redistributed here: get it from
<https://global-hydrodynamics.github.io/MERIT_Hydro/> under its own terms.
Which MERIT Hydro tiles a region needs, and the row and column offset of each,
is listed at <https://fullhydro.org/fullbasin/regions/>. Region boxes sit on
whole degrees and one degree is 1,200 cells, so a region is cut from the
global rasters by integer arithmetic, with no resampling.

You need not work with a whole region. A released sequence filtered to the
cells you keep is still a topological sort of them, and a filtered layering
keeps its layers independent, the conflict-free rule included, because
dropping cells can neither move a cell ahead of something it depends on nor
make two survivors depend on each other. A receiver that falls outside the
clip becomes an outlet when the clip is loaded. What a clip changes is the
kernel's answer, not the structure: drainage area computed
on a clip counts only the area inside it. Clip whole basins when the values
must match the global ones, or read them from MERIT-DrainAttr.

This package is a port of the C code that produced the release. The two were
written independently and agree on the example basin to floating-point
rounding in the accumulated area, so a region you build here with
`FlowTopo.from_raster` carries the same structures as the region you
download.

## Documentation

* [`docs/user-guide.md`](docs/user-guide.md) — choosing an ordering, a layering
  and a manner; threads; raster I/O.
* [`docs/methods.md`](docs/methods.md) — each method with its origin and its
  complexity, and what is borrowed from pyflwdir.
* [`examples/quickstart.ipynb`](examples/quickstart.ipynb) — a notebook on the
  bundled data, stored with its output.
* [`docs/review-checklist.md`](docs/review-checklist.md) — what has been
  checked and how, the bugs those checks found, and the angles still
  unattacked.

## Verification

```sh
pytest                  # needs pytest; numba for the threaded tests
python example.py       # every structure, kernel and manner on the example basin; ends PASS or FAIL
python benchmark.py     # serial vs threaded on synthetic grids
```

Every ordering, layering, partition, kernel and manner is cross-checked on the
example basin, write-conflict counts included. The structures are checked
against their definitions: an ordering is a topological sort, a layer is an
antichain, and in an upstream-to-downstream sequence a receiver comes after
its donors; all of that survives clipping a basin out of a region. The tests
and `example.py` run on Python 3.10, 3.11 and 3.12 on every push; the badge
at the top links to the runs.

## Citing

The companion manuscript is *MERIT-FlowTopo v1.0: a reusable computational
foundation for hyperresolution hydrology on the global 90 m drainage network*
(Jiang et al., in preparation); a citation file will be added once it
appears. Until then cite the code record
[10.5281/zenodo.22227621](https://doi.org/10.5281/zenodo.22227621), or this
repository with the commit you used; for a downloaded product cite its version
DOI above.

## Acknowledgements

**MERIT Hydro** (Yamazaki et al., 2019;
[10.1029/2019WR024873](https://doi.org/10.1029/2019WR024873)) provides the
flow-direction field everything here traverses.

The flat downstream-pointer representation, the D8 decoding conventions, the
donor-count array, the chain-tracing rank and the breadth-first sequence
builder follow [pyflwdir](https://github.com/Deltares/pyflwdir) (D. Eilander,
Deltares; MIT licence;
[10.5281/zenodo.4287337](https://doi.org/10.5281/zenodo.4287337)). No pyflwdir
source is included; the code was written against those conventions, and
`FlowTopo.from_d8` takes the same array `pyflwdir.from_array(d8, ftype="d8")`
takes. The borrowed and the original parts are set out in
[`docs/methods.md`](docs/methods.md).

**Claude** (Anthropic) assisted with the Python port, the tests, the
documentation and the packaging; the authors reviewed the work, and the
commit history records where. The methods and the results are the authors'
own, described in the companion manuscript.

## Contact

<lulu_jiang@pku.edu.cn>, or a
[GitHub issue](https://github.com/hellolulujiang/flowtopo/issues). Email is the
surer way to reach us.

## Licence

MIT for the code; see [`LICENSE`](LICENSE). The bundled MERIT Hydro excerpt keeps
its own CC BY-NC 4.0 terms; see [`DATA_NOTICE.md`](DATA_NOTICE.md).

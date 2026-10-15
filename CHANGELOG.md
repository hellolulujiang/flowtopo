# Changelog

## 1.0.7 (2026-10-15)

The results do not change.

* **Documentation: the three drainage attributes are named as in the FlowTopo paper**: flow length downstream
  (`distance_to_outlet`), flow length upstream (`longest_upstream_path`) and Strahler stream order
  (`strahler_order`), in place of "distance to (the) outlet", "longest upstream path", "upstream flow length" and
  "Strahler order". The function names stay.
* **The example notebook run again.** Its stored outputs dated from 0.1.0 and were never refreshed after the
  partition was rewritten in 1.0.4: the subbasin partition of the bundled basin is now loads 23,487, 23,107,
  23,107 and 23,108 cells (imbalance 1.01) with a mainstem of 623 cells, where the stored output showed four
  equal loads and a mainstem of 949. Every other output is unchanged; the plot titles use the names above. The
  notebook now says that the conflict count is of extra writes to a shared receiver (not of receivers), that the
  flow length upstream agrees bit for bit under every manner, and that `example.py` runs from the top folder; the
  README names what the notebook needs (Jupyter, matplotlib). The user guide points to the basin and region map of
  FullBasin version 1.0 for the MERIT Hydro tiles of a region (the section it named is no longer on the site).
* **`FLOWTOPO_METIS_LIBRARY` refuses a METIS 5.1 library.** METIS 5.2 moved the seed, contiguity and imbalance
  options (9, 13 and 18; 8, 11 and 16 in 5.1), so a 5.1 library took them as other options without an error. A 5.2
  library is told apart by `METIS_CacheFriendlyReordering`, which 5.1 does not have.

## 1.0.6 (2026-10-14)

The package brought in line with the C code as it now stands. The orderings, the layerings and the partition of the
bundled basin are unchanged; its flow lengths change by at most 0.16 m.

Results that change:

* **Flow lengths: a step is measured at the latitude between its two cell centres** (`geodist.pixel_length`), as
  the C code does. Until 1.0.5 it was the midpoint of the two row edges, half a pixel towards the first
  row. On the bundled basin `distance_to_outlet` changes by at most 0.16 m in 82.7 km and `longest_upstream_path`
  by 0.15 m in 98 km; on a one-degree grid at 60° N a step along a row was 1.5 % short. `tests/reference/expected.npz`
  holds the new values; the step of every link of the bundled basin equals the C code's.
* **Locality metrics follow the C code's definitions**, whose numbers the paper reports: a median is the upper of
  the two middle values and a percentile the value at the rounded rank (`numpy`'s interpolate); the receiver
  stride counts the cells that read a receiver (a pit reads none; until 1.0.5 it counted with a stride of 0); the
  reuse interval is found through a table of 2²⁰ cache lines, exact up to 16.7 million four-byte cells, and is the
  length of the sequence when no line is touched twice. On the bundled basin: `ds_stride_mean` 533.629 → 533.635,
  `gap_median_mean` of the CFDS layering 13,572.8 → 13,828.3; the miss rates are unchanged.
* **A cell that drains into a cell off the network is a pit, with or without a mask.** Built from `idxs_ds`
  without a mask, such a cell kept its link, and its basin was left out of the partition's loads.
  (`from_d8` never makes such a network.)
* **Ties in the partition are broken the same way as in the C code.** Of two tributaries with as many cells, the
  mainstem follows the one with the larger upstream area when `partition(..., upa=)` gives MERIT Hydro's `upa` (the
  C code's depth-first order takes the donors by it), then the one with the smaller cell index; until 1.0.5 it was
  the first in the donor table. Two one-cell tributaries on different rows of a geographic grid are such a tie, and
  their areas differ with the latitude, so without `upa` a mainstem can end in another headwater cell than the C's.
  Equally distant nodes are taken in the order of their indices when island components are linked to their
  neighbours (it was the order of SciPy's tree); that matters on regular grids of nodes. The METIS imbalance is
  rounded half up (`imbalance_target=1.0625` gave 62, now 63, as the C code). The `upa` given to the partition must
  be finite and positive on every cell of the network, as for `ordering("dfs", upa=)`.
* **Partition: a basin of a single cell takes no part, as in v1.0 and in the C code.** Such a basin (a pit that
  nothing flows into, mostly on a coast) was a node of the graph here, and the C code leaves it out, so a region
  with such basins could come out divided otherwise. Its cell now keeps `-1` in `part` and is in no load. The test
  of an archipelago, whose islands were single cells, now has islands of two cells.
* **Partition: the fragments of a subregion join however deep they lie.** A piece left apart from its subregion joins
  the subregion around it; a piece that touches only such pieces waits for the next round, and there were at most
  eight rounds. Pieces more than eight deep stayed apart, and the partition was returned as if they had joined (a
  path of 42 nodes with two heavy ends, labelled 0, 1, 0, 1, ..., kept 8 pieces apart). The rounds now go on until
  one moves nothing. Where eight rounds were enough the result is what it was. The same change is made in the C code
  and in pyflwdir's partition module, which this one follows.
* **Partition: an empty part takes the same node as in the C code.** METIS can leave a part empty; it then takes the
  lightest leaf of a breadth-first tree of the heaviest part. The tree was left to SciPy's order of visiting, which
  follows the edges stored from a node before those stored to it; the C code visits a node's neighbours by rising
  index, and another leaf could come out (five nodes with the edges 0-2, 1-2, 1-4, 2-3, 3-4: node 1 here, node 3 in
  C). The neighbours are now visited by rising index. A partition changes only where METIS left a part empty.
* **Partition: a node without a hint does not count when labels are aligned.** A hint of -1 was read as the last
  part. It is now left out, as in the C code.

New:

* **Partition: `basins=` numbers the basins.** METIS divides a graph according to the order of its nodes, and the C
  code numbers the basins as FlowDivide does; `partition(..., basins=bsn)` takes those labels (one positive label for
  each basin and one basin for each label, checked).
* **Partition: METIS 5.2.1 through `FLOWTOPO_METIS_LIBRARY`.** pymetis carries METIS 5.1.0, which divides the same
  graph with the same seed and options otherwise than METIS 5.2.1, the version the C code links. The environment
  variable names a METIS shared library built with 32-bit integers and reals; the partition then calls it through
  `ctypes` with the C code's arguments (pymetis stays the default). With FlowDivide's labels, that library, MERIT
  Hydro's `upa` and the default options, the subregions of MERIT Hydro regions 5699 and 27 are the current C code's
  to the pixel, at both levels.

Other changes:

* **`write_geotiff` does not write over a file.** A file already at the name is refused (`FileExistsError`), so that a
  run does not write over the result of an earlier one; `example.py` therefore needs an output directory without
  its files in it. (Two runs that write the same name at the same moment are not kept apart.)

Fixes:

* `parallel.upstream_area` and `parallel.longest_upstream_path` take raster-shaped inputs, row by row, as the serial
  kernels do (a 2-D `cell_area` or `ldn` raised IndexError), and refuse an array of the wrong size.
* The partition's nearest-neighbour search returns no neighbour for an empty set of points, as the C code, instead
  of failing in SciPy.
* `example.py` times the builders of the orderings and layerings themselves, after one run to compile them; it
  timed the object's cached copies, which the check of the network had built already.
* The tests compare the flow lengths with the C's to an absolute millimetre (the default relative tolerance let
  0.8 m through on a path of 83 km).
* `FlowTopo.upstream(mask)` no longer keeps the masked table in the cache: two threads asking with different masks
  could be handed each other's.
* `example.py` stops on a grid without a valid cell and on a network with a cycle, where every method agreed on
  the cells that were left and the check passed.
* `benchmark.py` says that the threaded forms need numba instead of announcing a fallback that does not exist.

Documentation:

* Descriptions brought in line with the code: the four manners of the kernels; what the base installation depends
  on; which METIS result is kept (within `METIS_SLACK` of its share the shortest boundary, otherwise the most
  balanced); when an opened basin is kept; how a file without a reference system is read (as degrees).

## 1.0.5 (2026-10-08)

Results that change:

* **The subbasin level opens as few basins as the balance needs.** Before (1.0.4), the largest basins
  were opened one after another, up to four, and the most balanced result was kept, however small its
  gain over fewer basins: a region could end with four opened mainstems where one did nearly as well.
  Now each round tries only the whole basins that can hold the balance up, each opened together with
  those kept so far: in every subregion over the target, the heaviest (which it cannot shed) and the
  heaviest bordering it; in the lightest subregion, the heaviest (which it cannot grow around) and the
  heaviest bordering it, such as a basin that walls off a piece of land. A basin heavier than an equal
  share in a subregion over the target can never be balanced whole and is then the only one tried. They
  are tried the heaviest first, at most `OPEN_TRIALS` (4), and a round stops at the first that brings
  the plan within the target. The best has the lowest max/mean, then the least overload (how much the subregions over the mean exceed
  it, in means); it is kept only when it lowers max/mean by `OPEN_MIN_GAIN` (0.01) or more, or leaves
  the heaviest subregion no heavier and lowers the overload by that much, so that a basin that lightens
  one of two equally heavy subregions counts. The rounds stop at the target, at a round without such a gain, or at
  four basins.
* The bundled example is one basin and gives the same loads as 1.0.4.

New: `flowtopo.partition.OPEN_MIN_GAIN` and `OPEN_TRIALS`.

## 1.0.4 (2026-10-07)

Results that change:

* **The partition keeps every subregion in one piece, then balances it.** Both levels now cut a graph
  whose nodes are basins or tributary subtrees and whose edges are their shared boundaries: the land
  masses get the subregions so that the heaviest is as light as can be, each mass is cut by contiguous
  weighted METIS, and boundary units then move from heavier to lighter neighbours. Within a land mass a
  subregion is one piece (the two banks of a held-back mainstem count as touching); land is the largest
  components, at most one per subregion (those of at least half an equal share, and the next largest
  when these are too few to be cut into every subregion), and every other component goes as an island
  with a subregion near it. Before, basins and tributary
  subtrees were dealt largest first to the lightest subregion, wherever they lay.
* **The subbasin level opens the largest basins while the subregions are unequal**, up to four, one
  layer only (a tributary is never opened further), and keeps the most balanced result. The mainstem
  stays with the subregion of its most upstream tributaries down to `P_min`, the first cell where a
  tributary of another subregion enters; only the mainstem below it is held back as
  `flowtopo.MAINSTEM` (before: every cell of the mainstem).
* On the example basin the subbasin loads are `[23487, 23107, 23107, 23108]`, 1.2 % above the mean,
  each subregion one piece with the held-back mainstem; before, `[23121, 23121, 23121, 23120]` with
  each subregion in about 650 pieces.
* The partitions released in MERIT-FlowTopo were made by the C code with the earlier rule; this
  package now partitions differently from it.

New: `partition()` and `FlowTopo.partition()` take `seed`, `refine`, `min_subtree_size` and
`imbalance_target`. The package now depends on `scipy` and `pymetis`.

Input checks, from internal versions after 1.0.3 (each was accepted before and gave a wrong answer or an
unclear error):

* A constructor mask may have the raster's two-dimensional shape; it is flattened before the network is
  cut.
* An incomplete transform raises a `ValueError` before any of its terms is read.
* Grid dimensions must be two whole, nonnegative numbers.
* A raster-shaped `upa` is flattened for the depth-first order, and the channel and Strahler inputs must
  hold one value per cell.

## 1.0.3 (2026-09-28)

1.0.1 and 1.0.2 were internal versions and were not released.

Every fix below was checked against the C code and on a small grid; 182 tests
pass.

Results that change:

* **A mask given to the constructor cuts the network.** A cell outside it leaves the network (`-1`) and a
  cell inside it whose receiver is outside becomes a pit. Before, the kernels honoured the mask but the
  orderings, basin labels, partition and layerings were built on the whole `idxs_ds`.
* **Strahler order honours the object's mask.** Without a `channel_mask` the order was computed over every
  cell with a receiver: on `idxs_ds=[3,3,-1,3]`, `mask=[F,T,F,T]` the outlet came out 2 where the network
  holds one stream.
* **The subbasin partition** counts cells in float64 (float32 is exact only up to 2²⁴ cells, and the pit
  could be mistaken for the cell above it), takes the pit from the flow directions, and follows only donors
  inside the mask.
* **A cycle no longer becomes a layer.** `reverse_layers` flipped the `-1` of a cell in a cycle into the
  largest layer, where the two cells of a two-cell cycle wrote into each other.
* **A projected grid is read as metres.** Every grid used to be taken for degrees (a 100 m cell came out as
  100 degrees). The CRS now decides; a result written back carries the grid's own CRS. A rotated grid and
  a projection in another unit than the metre are refused.
* **A step across the antimeridian is the short way round** on a geographic grid that spans the globe,
  as in the C code.
* **The kernels stay inside the mask.** The serial distance to outlet, the longest upstream path and the
  serial Strahler order each wrote into a cell or receiver outside the mask.
* **The threaded drainage area keeps the type of the areas it is given**; it forced float32 and differed
  from the serial answer at the outlet of a large basin.
* **A GeoTIFF's own mask is applied when it is read.** Cells an internal mask or alpha band marks invalid
  become nodata (a stored 0 there became an outlet). A file whose only mask is its nodata value reads as
  before. A type that cannot hold the fill (Int8 with no nodata) is widened.
* **The flow-direction nodata is found on the grid's own type**, before the cast to uint8: a UInt16 nodata
  of 65535 turned into 255, a valid terminal. A NaN nodata is found with `isnan`, and another value
  outside 0..255 is refused.
* **A layer's cache-line hit fraction compares line numbers**, not index differences, as the C code does.

Input checks (each of these was accepted before and gave a wrong answer without a word):

* downstream indices, orderings, masks and basin labels are checked on the type they come in, before they
  are cast (an int64 2**32 wrapped to cell 0); every module takes them through one check in `core`;
* the kernels check the length of `cell_area`, `plen`, `ldn` and `mask`, and that their nodata is negative
  in the type they hold it in;
* the constructor refuses a transform that is not six finite numbers, a pixel side of 0, and a rotation;
* a geographic CRS must be in degrees;
* `ordering("dfs", upa=...)` refuses an area that is not finite and positive on the network;
* `decomposition()` checks its direction as `ordering()` does, and `reverse_layers` refuses a layer number
  past int32;
* the cached upstream table and decomposition are read-only, like every other cached array.

New:

* **`FlowTopo(..., earth="wgs84")`** computes the cell areas on the WGS84 ellipsoid, the C code's default.
  Against the sphere the ellipsoid's cell is 0.45 per cent smaller at the equator, 0.22 per cent larger at
  45 degrees and 0.56 per cent larger at 60. The default stays the sphere of radius 6 371 000 m, which is
  the earth the released products are on.
* **`ordering("dfs", upa=...)`** takes the donors of a confluence by upstream drainage area, largest first,
  which is the order of the released MERIT-FlowTopo `seq_dfs` layer; the default takes them by cell index.
  Both are topological orders and every kernel gives the same answer under either, to the order the
  floating-point sums are added in.

Documentation: the as-soon-as-possible layering schedules a cell that drains into a cycle; the
conflict-free layering can hold a headwater back from layer 0; only the headwater farthest from the pit
holds the basin's largest rank; the depth-first docstring says which way the returned sequence runs.

Two places where this package deliberately differs from the C code, both documented in
[`docs/methods.md`](docs/methods.md):

* a cell in a cycle keeps `-1` in every layering and has no place in the sequences, where the C code's
  conflict-free layering returns an error instead. The value such a cell ends with is not meaningful and
  depends on the traversal. A grid with a cycle has to be refused, not read: test `(layers < 0) & mask`,
  or compare the length of a sequence with the valid-cell count;
* the cell areas are the sphere's unless `earth="wgs84"` is asked for, while the C code's default is the
  ellipsoid. The sphere is what the released products carry.

## 1.0.0 (2026-09-22)

The same code as 0.1.0, renumbered: the public version of this package now follows the product and the paper
(MERIT-FlowTopo v1.0) and the Zenodo code record 10.5281/zenodo.22227621 (version 1.0.0).


## 0.1.0

First release.

* Three serial orderings: depth-first from the pits, breadth-first from the
  pits, topological sort from the sources.
* Three parallel layerings: as soon as possible, conflict-free downstream, as
  late as possible.
* Two spatial partitions: basin-level and subbasin-level, both cutting along
  the drainage hierarchy so subregions need no communication.
* Four kernels to exercise the structures: upstream drainage area, distance to
  outlet, longest upstream path, Strahler stream order, each under the
  propagation manners it supports.
* Memory-locality metrics with a simulated N-way set-associative LRU cache.
* Threaded kernels through numba `prange` in `flowtopo.parallel`.
* Expected outputs for the bundled example basin ship with the tests.

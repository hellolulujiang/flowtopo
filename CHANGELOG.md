# Changelog

## 1.0.4 (2026-10-07)

Results that change:

* **The partition keeps every subregion in one piece, then balances it.** Both levels now cut a graph
  whose nodes are basins or tributary subtrees and whose edges are their shared boundaries: the land
  masses get the subregions so that the heaviest is as light as can be, each mass is cut by contiguous
  weighted METIS, and boundary units then move from heavier to lighter neighbours. Within a land mass a
  subregion is one piece (the two banks of a held-back mainstem count as touching); land is one of the
  largest components, each at least half an equal share and at most one per subregion, and every other
  component goes as an island with a subregion near it. Before, basins and tributary
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
(MERIT-FlowTopo v1.0) and the Zenodo code record 10.5281/zenodo.22227621 (version 1.0.0). From here on, packages
that are not yet public stay at 0.x locally and take 1.0.0 at their first release.


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

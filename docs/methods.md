# Methods

Every construction in the package, with where it comes from and what it costs.
Notation:

| | |
| --- | --- |
| `N` | cells in the grid, nodata included |
| `E` | edges in the flow graph; for D8, about `N` |
| `L` | longest downstream path, in hops |
| `P` | threads |
| `K` | largest donor count of any cell; at most 8 for D8 |
| `B` | number of basins |

## Rank

`rank[i]` is the number of downstream hops from cell `i` to its pit.
`rank[pit] = 0`, `rank[headwater]` is the largest value in its basin.

### `core.rank_to_pit`

Serial memoized chain tracing. A cell being traced is marked in progress, so a
cycle is detected in constant time and every cell is resolved once.

*Complexity:* O(N) time, O(N) space.

## Serial orderings

Each returns a topologically sorted list of cell indices, ready for one
sequential pass. `d2u` means position 0 is a pit; `u2d` means position 0 is a
headwater. The two directions are reverses of each other.

### `core.seq_dfs_from_pit` — depth-first, `d2u`

Walks each basin depth-first from its pit, finishing one tributary subtree
before starting the next. A cell's receiver therefore sits only a few positions
back in the sequence, which is why this ordering has the lowest simulated cache
miss rate of the three.

*Origin:* Braun and Willett (2013), Geomorphology 180–181:170–179.
*Complexity:* O(N) time, O(N + E) space.

### `core.seq_bfs_from_pit` — breadth-first, `d2u`

Starts from every pit and expands upstream one ring of donors at a time, so the
output is grouped by downstream hop count.

*Origin:* [pyflwdir](https://github.com/Deltares/pyflwdir) (Eilander et al.,
2021, Deltares).
*Complexity:* O(N + E) time, O(N + E) space.

### `core.seq_topo_from_source` — topological sort, `u2d`

Counts each cell's donors, starts from the headwaters, and releases a receiver
when its last donor has been output. Needs only the donor counts, never a donor
list, so it is the lightest of the three in memory.

*Origin:* Kahn (1962), "Topological sorting of large networks."
*Complexity:* O(N) time, O(N) space.

### `parallel.seq_bfs_from_pit` — breadth-first with threads, `d2u`

The same sequence as the serial breadth-first form, built by counting,
prefix-summing and writing each frontier across threads.

*Complexity:* O(N/P + L) time over L frontiers, plus the prefix sum each
frontier costs, which is a scan over the P thread totals. O(N + E) space.
Needs numba.

## Layerings

Each returns one layer index per cell, `-1` outside the network. The members of
a layer are mutually independent, so a layer can be processed in any order or
all at once. Layer 0 holds the headwaters and the index grows toward the pit.

### `core.layering_asap` — as soon as possible

Every headwater in layer 0, every other cell one layer beyond its
last-scheduled donor. Each cell lands in the earliest layer its dependencies
allow, so most cells pile into the first layers. Reaches the minimum layer
count, which the longest flow path sets.

*Origin:* Kahn (1962).
*Complexity:* O(N) time, O(N) space.

### `core.layering_cfds` — conflict-free downstream

As soon as possible, with one added rule: a ready cell whose receiver has
already been claimed by another cell in the current layer waits for the next
layer. No two cells of a layer then write into the same receiver.

That is what lets a threaded push run with no atomic and stay deterministic.
It is also the only one of the three under which Strahler stream order can run
under push at all. The confluence rule there is a comparison and a count, not
an addition, so no atomic applies. Costs at most a few layers over the plain
form.

*Complexity:* O(N + E) time, O(N) space.

### `core.layering_alap` — as late as possible

Per basin, `layer = max_rank_in_basin - rank`, so short tributaries are held
back to later layers rather than crowding into the first ones. Reaches the same
minimum layer count as the as-soon-as-possible form but spreads the work
differently, which changes the memory-access pattern.

*Complexity:* O(N) time, O(N + B) space.

## Spatial partitions

Each returns a subregion index per cell, `-1` outside the network. Both cut
along the drainage hierarchy, so no value crosses a subregion boundary while a
kernel runs. Assignment is longest-processing-time-first: items are placed
largest first, each into the lightest subregion so far.

### `partition(topo, n_parts, level="basin")`

Whole basins assigned to subregions, weighted by cell count. A basin is never
split, so a dominant basin leaves the other subregions idle.

*Complexity:* O(N + B log B) time, O(N) space.

### `partition(topo, n_parts, level="subbasin")`

Any basin larger than one subregion's share is decomposed along its mainstem,
found by walking upstream from the outlet and taking the larger tributary at
each confluence. The tributary subtrees become items in the assignment; the
mainstem depends on them and is held back to a second stage, marked
`flowtopo.MAINSTEM`.

*Complexity:* O(N + E + M log M) time, M being the number of items the
assignment sorts: one per undecomposed basin plus one per tributary subtree
split off a decomposed one. O(N) space.

## Kernels

| kernel | shape | manners |
| --- | --- | --- |
| `upstream_area` | many-to-one accumulation, linear | serial, pull, push, atomic_push |
| `distance_to_outlet` | one-to-one propagation | serial, and any layering |
| `longest_upstream_path` | many-to-one maximum, linear | serial, pull, push, atomic_push |
| `strahler_order` | many-to-one, non-linear | serial, pull, push |

All four are one pass over a structure: O(N) time serial, O(N) space beyond
the structure itself. Three of them also have a threaded form in
`flowtopo.parallel`, O(N/P + L) over L layers. Strahler order has none, since
its confluence rule is a comparison rather than an addition.

`push` writes into the receiver and is correct only where no two cells of a
layer share one, that is, under the conflict-free downstream layering.
`atomic_push` handles repeated receivers through `np.add.at` or
`np.maximum.at`. `pull` gathers from the donors, so a cell writes only to
itself; it needs the upstream adjacency table.

### `upstream_area`

Accumulates a per-cell quantity downstream. Accumulates in float32 when
`cell_area` is float32 and in float64 otherwise, so counting cells with an
integer array stays exact:
a float32 total past about 2e5 km² no longer moves when one 90 m cell, about
0.007 km², is added to it, so pass float64 on a continental network.

### `distance_to_outlet`

Cumulative along-stream distance to the pit. One-to-one, so each cell is
written once and any layering is safe. Needs a `d2u` traversal, since a
receiver must be finished before its donors.

### `longest_upstream_path`

Propagates the maximum distance-to-outlet seen upstream, then subtracts the
cell's own. Takes the output of `distance_to_outlet`.

### `strahler_order`

Two branches of equal order meeting raise the order by one; otherwise the
largest incoming order carries through. A comparison and a count rather than
an addition, so no single atomic expresses it: under a layering the
conflict-free push is the one that runs it in a single pass with no atomic.

*Origin:* Strahler (1957), Trans. Am. Geophys. Union 38:913–920.

## Supporting structures

### `core.d8_to_downstream`

The flattening of the D8 raster into one downstream pointer per cell. 247 is
nodata; a second nodata code can be given for a grid that uses another. A cell
draining off the grid, into nodata, or nowhere becomes a pit.

*Origin:* [pyflwdir](https://github.com/Deltares/pyflwdir) (Eilander et al.,
2021, Deltares).
*Complexity:* O(N) time, O(N) space.

### `core.unknown_codes`

Counts codes the convention does not define. A non-empty result usually means
the grid follows TauDEM or GRASS, which number their directions 1 to 8; read
as powers of two, every undefined code becomes a pit.

### `core.upstream_count`, `core.upstream_table`

Donors per cell, and the adjacency table the pull manner reads. The table
costs `N × K` entries, K being the largest fan-in, at most 8 for D8.

*Origin:* [pyflwdir](https://github.com/Deltares/pyflwdir) (Eilander et al.,
2021, Deltares), for the count and its sentinel convention.
*Complexity:* O(N + E) time, O(N·K) space.

### `core.basin_labels`

One-based basin id per cell, from a `d2u` pass: a pit opens a basin and every
other cell copies its receiver's label. Required by the as-late-as-possible
layering and by both partitions.

*Complexity:* O(N) time, O(N) space.

### `geodist`

Metres per degree from the WGS-84 series expansion, and the spherical area of
a lat-lon pixel. Checked against pyproj: metres per degree agree to 1e-5, the
series' own accuracy. Cell area matches the closed-form spherical value on
square and non-square grids alike, and summing a whole-globe grid reproduces
4 pi R squared.

Two approximations are worth knowing about, both small on a 3 arc-second grid
and larger on a coarse one.

Cell-to-cell distance is planar, not geodesic: metres per degree are evaluated
once and combined with Pythagoras, rather than solving the inverse geodesic
problem. Over one 90 m cell the two agree far below the resolution of the data.

The latitude at which they are evaluated is the midpoint of the two row edges
rather than of the two cell centres, so it sits half a pixel north of the
true midpoint. This matches the C implementation that produced the released
products, and the two are kept identical deliberately. On the bundled basin it
costs 0.16 m over a 98 km path, 1.7e-6 relative; on a one-degree grid at 60 N
it approaches a percent.

### `raster`

`read_geotiff` and `write_geotiff`, through rasterio. `GridHeader` carries the
shape, dtype, nodata and geotransform.

### `layering.Decomposition`, `layering.reverse_layers`

A layering is one layer index per cell; the kernels need the opposite view,
the members of each layer. `Decomposition` holds that as one flat array of
cell indices plus the offsets that cut it into layers, so a layer is a
zero-copy slice, and cells inside a layer stay in ascending index order.
`reverse_layers` flips a `u2d` layering into a `d2u` one for the
downstream-propagating kernel.

*Complexity:* O(N log N) to build, one stable sort; O(N) space.

### `api.FlowTopo`

The front end. Wraps a D8 network, builds each structure on first use and
caches it. Cached arrays are handed out read-only, so a caller cannot damage
the cache by writing to a result.

## Threaded forms

`flowtopo.parallel` compiles the layer traversal with numba `prange`:
`upstream_area`, `distance_to_outlet` and `longest_upstream_path`, plus
`seq_bfs_from_pit`, which expands each frontier across threads. Requires
numba; without it the module raises rather than pretending to use threads.

## Locality metrics

Feeds the interleaved (cell, receiver) access stream through a simulated N-way
set-associative LRU cache, so the two arrays a traversal reads compete for one
cache. Default geometry: 32 KB 8-way L1, 1 MB 16-way L2, 35.75 MB 11-way L3.

`serial_locality` reports, for one ordering: stride statistics, the row-jump
fraction, cache-line reuse interval, receiver distance, and the simulated miss
rate per level. `parallel_locality` reports, for one layering: intra-layer
span, gap statistics, cache-line footprint, receiver distance, and the miss
rate on the flattened layer order, so the two are directly comparable.

The reuse interval is the number of accesses between two touches of the same
cache line. It is not the reuse distance of the cache literature, which counts
the distinct lines touched in between and is the LRU stack distance.

The simulator is checked against an LRU written independently: the two agree
exactly on the example basin and on random access sequences. Six further tests
pin it to cases with a known answer, including the associativity boundary —
eight lines per set fit, the ninth thrashes.

*Complexity:* O(A · W) time for A accesses and W ways, O(capacity) space.

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

*Complexity:* O(N/P + L) time, O(N + E) space. Needs numba.

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

That is what lets a threaded push run with no atomic and stay deterministic,
and it is the only one of the three under which Strahler stream order can run
under push at all: its confluence rule is a comparison and a count, not an
addition, so no atomic applies. Costs at most a few layers over the plain form.

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

*Complexity:* O(N + E) time, O(N) space.

## Kernels

| kernel | shape | manners |
| --- | --- | --- |
| upstream drainage area | many-to-one accumulation, linear | serial, pull, push, atomic_push |
| distance to outlet | one-to-one propagation | serial, and any layering |
| longest upstream path | many-to-one maximum, linear | serial, pull, push, atomic_push |
| Strahler stream order | many-to-one, non-linear | serial, pull, push |

`push` writes into the receiver and is correct only where no two cells of a
layer share one, that is, under the conflict-free downstream layering.
`atomic_push` handles repeated receivers through `np.add.at` or
`np.maximum.at`. `pull` gathers from the donors, so a cell writes only to
itself; it needs the upstream adjacency table.

## Locality metrics

Feeds the interleaved (cell, receiver) access stream through a simulated N-way
set-associative LRU cache, so the two arrays a traversal reads compete for one
cache. Default geometry: 32 KB 8-way L1, 1 MB 16-way L2, 35.75 MB 11-way L3.

Reported per ordering: stride statistics, row-jump fraction, cache-line reuse
distance, receiver distance, and the simulated miss rate per level. Per
layering: intra-layer span, gap statistics, cache-line footprint, receiver
distance, and the miss rate on the flattened layer order, so the two are
directly comparable.

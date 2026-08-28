# User guide

## What this package is for

You have a D8 flow-direction raster and you want to compute something that
travels along the network: drainage area, distance to the outlet, stream order.
Every such computation needs the cells visited in an order that respects the
flow, and there is more than one such order. This package builds six of them
and runs four kernels over any of them, so you can pick the one that suits your
grid instead of taking whatever a library happened to build.

If you only want one drainage-area map on one grid, use the three lines under
[Five minutes](#five-minutes) and stop there. The rest of this guide is for
when the choice starts to matter: large grids, repeated kernels, or threads.

A narrated video overview animates the orderings, the layerings and the three propagation
manners — worth ten minutes before reading further:
<https://youtu.be/tE5K2wM3TTY>.

## Install

```sh
pip install -e .            # numpy + rasterio
pip install -e ".[speed]"   # + numba, for the threaded kernels
```

The structures and the serial kernels work without numba. The scalar builders
fall back to pure Python loops, correct but slow on a large grid. The threaded
kernels do not: `flowtopo.parallel` raises rather than pretending to use
threads, so install the `speed` extra before running anything below that uses
it.

## Five minutes

```python
import flowtopo

topo = flowtopo.FlowTopo.from_d8(d8, transform=transform)

upa = topo.upstream_area(ordering="dfs")
ldn = topo.distance_to_outlet(ordering="dfs")
lup = topo.longest_upstream_path(ldn, ordering="dfs")
ord = topo.strahler_order(ordering="dfs",
                          channel_mask=topo.channel_mask(upa, 10.0))
```

`d8` is a 2-D uint8 array in the MERIT Hydro convention: powers of two clockwise
from east, 0 and 255 terminal, 247 nodata. `transform` is a six-element GDAL
GeoTransform, needed only by the two distance kernels.

To read the bundled example instead:

```python
topo = flowtopo.FlowTopo.from_raster("data/dir_example.tif")
```

## The two ideas

### An ordering is a list; a layering is a schedule

An **ordering** is every valid cell in a single topologically sorted list. Walk
it start to end and every cell is reached after everything it depends on. One
loop, one thread.

A **layering** is one layer index per cell. Cells in a layer do not depend on
each other, so the whole layer can be done at once. Layer 0 is the headwaters
and the index grows toward the pit. The number of layers cannot be smaller than
the longest flow path.

Orderings and layerings are two views of the same constraint. Use an ordering
when you have one core, and a layering when you have several.

### Direction

`d2u` runs downstream to upstream, position 0 at a pit. `u2d` runs the other
way, position 0 at a headwater. Which one a kernel needs follows from which way
its information travels:

* drainage area, longest upstream path and stream order accumulate **upward
  into** the receiver, so they need `u2d`;
* distance to outlet reads **from** the receiver, so it needs `d2u`.

The package picks the right direction for you. You only meet it if you call
`topo.ordering(name, direction)` yourself.

## Choosing an ordering

```python
topo.ordering("dfs")    # depth-first from the pits
topo.ordering("bfs")    # breadth-first from the pits
topo.ordering("topo")   # topological sort from the sources
```

All three give a correct answer. They differ in how the memory is walked:

* **`dfs`** finishes one tributary subtree before starting the next, so a
  cell's receiver is only a few positions back. Lowest simulated cache miss
  rate of the three; on the bundled basin, 10.55% L1 against 21.83% and 37.02%.
  This is the one to reach for when the kernel will run repeatedly.
* **`bfs`** groups cells by hop count from the pit.
* **`topo`** needs only donor counts, never a donor list, so it allocates least.

Run `python example.py` to see the miss rates on your own grid.

## Choosing a layering

```python
topo.layering("asap")   # as soon as possible
topo.layering("cfds")   # conflict-free downstream
topo.layering("alap")   # as late as possible
```

Start with `cfds`. The difference between the three decides whether a parallel
push is correct.

### Write conflicts

A kernel that pushes writes into its receiver. Inside one layer that is safe
only if no two cells share a receiver. `asap` and `alap` do not promise that;
`cfds` does, by holding a cell back one layer when its receiver is taken.

The promise is a property of the layering, so you can count it rather than hope:

```python
import numpy as np

def conflicts(topo, layering):
    total = 0
    for members in topo.decomposition(layering, "u2d"):
        ds = topo.idxs_ds[members]
        ds = ds[(ds >= 0) & (ds != members)]
        total += ds.size - np.unique(ds).size
    return total
```

On the bundled 731 km² basin, 93,432 cells, 949 layers under all three:
`asap` 12,122, `cfds` 0, `alap` 39,130.

A test run is not a reliable check. On a 16,000,000-cell grid with 152,258
conflicts under `asap`, a threaded push still returned correct results on one
run, because the colliding cells landed on the same thread. Count the
conflicts instead.

### Cells a structure cannot cover

A cycle is a set of cells that drain into each other in a loop. No traversal
can order them, because each depends on the next. MERIT Hydro is cycle-free,
so this stays at zero on released data, but a grid you built or repaired
yourself may not be, and nothing raises. Check it:

```python
layers, _ = topo.layering("cfds")
stranded = np.count_nonzero((layers < 0) & topo.mask)   # 0 on a clean grid
```

A sequence shorter than `topo.ncells` says the same thing.

When a cycle is present the six structures do not agree on what is left, and
each is right by its own definition. A twelve-cell grid, an eight-cell chain
draining into a four-cell ring, covers:

| structure | cells covered |
| --- | --- |
| `dfs`, `bfs` | 0, they start at a pit and there is none |
| `topo` | 8, it starts at the headwaters and stops at the ring |
| `asap`, `cfds` | 8, same reason |
| `alap` | 0, it needs a rank measured from a pit |

So the number of cells you get back depends on which structure you asked for.
Rule out cycles first and the question does not arise: on a clean grid all six
cover every valid cell.

## Choosing a manner

For a layering you also choose how a cell and its receiver exchange the value:

| manner | what it does | correct when |
| --- | --- | --- |
| `pull` | the receiver gathers from its donors | always; needs the adjacency table |
| `push` | each cell writes into its receiver | only under `cfds` |
| `atomic_push` | the same scatter through `np.add.at` | always, except Strahler order, which has none |

```python
upa = topo.upstream_area(layering="cfds", manner="push")
```

If `manner` is left unset, FlowTopo picks a safe one for the layering: `push`
under `cfds`, `atomic_push` (or `pull` for Strahler) under the other two.

Ask for an unsafe combination and it runs anyway, quietly returning the wrong
number. That is deliberate: seeing the conflict is the point, and there is no
cheap way to tell a genuine request apart from a mistake. Three cells, two
headwaters meeting at a pit, is enough to show it:

```python
topo.strahler_order(layering="cfds", manner="push")   # [1 2 1], correct
topo.strahler_order(layering="asap", manner="push")   # [1 1 1], wrong
```

Both donors sit in the same `asap` layer, so one write lands on top of the
other and the confluence never happens. Drainage area goes the same way, one
donor short. If you set `manner` yourself, count the conflicts first.

Strahler order has no `atomic_push`, and this is not an oversight. Its
confluence rule is *two branches of equal order raise the order by one*, which
is a comparison and a count, not an addition. No atomic implements that, so
under a layering the only safe push is the conflict-free one.

### Precision on a large network

Drainage area accumulates in the precision of the array you give it, float32 by
default. One 90 m cell is about 0.007 km², and a float32 total large enough
stops registering it: past roughly 2e5 km² a single cell no longer moves the
total, at 1e6 km² it takes about five cells, at 5e6 km² about thirty-six. So on
a continental river the headwaters accumulate exactly and the cells near the
outlet stop counting. Hand it a float64 array when that matters:

```python
import numpy as np
upa = topo.upstream_area(ordering="dfs",
                         cell_area=np.asarray(topo.cell_area, dtype=np.float64))
```

## Splitting across processors

A layering spreads a layer across threads that share memory. Running on several
processors needs a second cut, and it has to follow the drainage hierarchy so
that no value crosses a subregion boundary mid-kernel:

```python
part, load = topo.partition(n_parts=4, level="subbasin")
```

Set `n_parts` to the number of processors, one subregion each. The paper's
benchmark uses four, one per NUMA node of a four-socket Xeon Platinum 8270
server, with about 13 threads working inside each subregion.

`part` holds a subregion index per cell, `-1` outside the network and
`flowtopo.MAINSTEM` (-2) on a mainstem cell held back to the second stage. With
`level="basin"` whole basins are assigned to subregions by cell count; a basin
is never split, so one dominant basin leaves the other processors idle. The
bundled example is a single basin, so it shows this directly: basin-level gives
`[93432, 0, 0, 0]`, subbasin-level `[23121, 23121, 23121, 23120]`.

`level="subbasin"` cuts an oversized basin along its mainstem, found by walking
upstream from the outlet and taking the larger tributary at each confluence.
The tributary subtrees are dealt to the lighter subregions; the mainstem
depends on them, so it is marked `flowtopo.MAINSTEM` and runs in a second stage
after they finish.

## Threads

The kernels in `flowtopo.kernels` run one numpy operation per layer. That makes
the write conflict visible deterministically, because `a[idx] += v` in numpy
keeps only the last write when `idx` repeats, exactly as a non-atomic threaded
push does. But numpy has no threads, so for speed use `flowtopo.parallel`,
which compiles the same layer traversal with numba:

```python
from flowtopo import parallel

upa = parallel.upstream_area(topo, layering="cfds", manner="push")
```

What to expect, from three runs of `benchmark.py` on a synthetic
16,000,000-cell network with 10 threads on an Apple M-series laptop:

| form | median s, three runs | speedup |
| --- | --- | --- |
| serial ordering, one thread | 0.0854 – 0.0880 | 1.00× |
| `cfds` layering, push, threads | 0.0505 – 0.0539 | 1.58 – 1.74× |
| `cfds` layering, pull, threads | 0.0898 – 0.0971 | 0.91 – 0.95× |

Drainage-area accumulation is memory-bound, so under 2× on ten threads is
close to the ceiling for this kernel. Among the always-correct forms, push
under `cfds` runs about 1.8× faster than pull and needs no adjacency table.
Pull is slower than the single-threaded sequence here: building and reading
the upstream table costs more than the threads save.

Timings move a few percent between runs and a lot between machines. Treat the
ratios as the message and the seconds as one laptop's answer.

Run `python benchmark.py` on your own machine.

## Working with your own rasters

`FlowTopo.from_raster` reads any single-band D8 GeoTIFF in the MERIT Hydro
convention: codes are powers of two clockwise from east, 0 and 255 are
terminals, 247 is nodata. The file's own nodata value is honoured as well,
which matters because 255 means an endorheic terminal here, not nodata. Write a result back next to it:

```python
import flowtopo
from flowtopo import write_geotiff

topo = flowtopo.FlowTopo.from_raster("my_dir.tif")
upa = topo.upstream_area(ordering="dfs")
write_geotiff("upa.tif", upa, topo.header(dtype="float32", nodata=-9999.0))
```

## Checking a result

`example.py` runs every structure, kernel and manner on one grid, compares all
results against the serial depth-first reference, and prints PASS or FAIL:

```sh
python example.py --data my_dir.tif --out out_mine
```

Drainage area is the one result not expected to agree bit for bit across
manners. Floating-point addition is not associative and each manner sums a
confluence's donors in a different order; on the bundled basin the spread is
0.0027 km², about 1e-6 of the largest value. Everything else agrees exactly,
longest upstream path included, because its confluence rule takes the larger
of two donors rather than adding them. The checks treat them that way.

## Where to look next

* `docs/methods.md` — every construction, its origin and its cost.
* `example.py` — the whole package exercised on one basin.
* `benchmark.py` — serial against threaded, at several grid sizes.

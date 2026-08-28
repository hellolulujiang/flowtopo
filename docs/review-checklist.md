# Review checklist

Several passes of review have been run on this package. Each attacked one
class of defect, and the real bugs that came out are listed at the end. A
reviewer picking this up should not repeat the rounds that found nothing;
attack the angles that are still untested, and re-run the regression tests that
pin the bugs already found.

Every claim below can be checked by running something. Nothing here asks
anyone to take a statement on trust.

## What has been checked, and how

### Correctness against something independent

1. **A brute-force reference.** A naive implementation written from the D8
   codes alone — no package code — agrees on the downstream pointers, the
   mask, upstream cell counts and distance to outlet.
2. **pyflwdir on the same grid.** Downstream pointers, pit set, upstream
   counts and rank agree exactly; `accuflux` agrees to 0; `stream_order`
   agrees on 100% of cells.
3. **WGS84 through pyproj.** Metres per degree agree to 1e-5, the series
   expansion's own accuracy. Cell area matches the spherical integral exactly,
   and integrating the globe reproduces 4πR² to 7e-14.
4. **An independent LRU.** The cache simulator agrees exactly with an
   `OrderedDict` LRU written from scratch, on the example basin and on random
   sequences.

### Internal consistency

5. **Every combination.** Each kernel over each ordering, each layering and
   each manner, on six synthetic networks: 504 comparisons, all within
   floating-point tolerance, and Strahler exact.
6. **numba on and off.** Thirteen outputs are bit-identical with and without
   numba.
7. **`kernels` against `parallel`.** The numpy and the threaded
   implementations agree, and the conflict-free push is bit-identical at 1, 2,
   4 and 8 threads.

### Contracts

8. **Determinism.** Twelve products are bit-identical between two runs.
9. **Immutability.** No call writes to its arguments, and cached arrays are
   read-only.
10. **Thread safety.** Eight threads sharing one `FlowTopo` produce identical
    results with no exception.
11. **Bad arguments.** Ten kinds of invalid call are rejected with a message
    that names the valid options.
12. **Pickling.** A `FlowTopo` round-trips and still computes the same answers.

### Degenerate and awkward inputs

13. **Empty grids, lone cells, cycles.** Nothing raises; a sequence shorter
    than the valid-cell count reports the cycle.
14. **Raster variants.** Southern hemisphere, the antimeridian, non-square
    pixels, coarse resolution, 80°N, and a D8 grid stored as float32. Cell
    area is now checked against the closed-form spherical value on four grid
    shapes, square and not; the earlier pass looked at these rasters but never
    compared an area, which is how the bug below survived it.
15. **Foreign nodata codes.** A grid that marks nodata as 255, 0 or anything
    else is read correctly.

### Numerics and resources

16. **Dtypes and limits.** int32 indices reach 2.1e9 cells, comfortably above
    the 3.4e8 of the largest MERIT region. Strahler saturates at 255 rather
    than wrapping.
17. **Accumulation precision.** A float32 total past about 2e5 km² no longer
    moves when one 90 m cell is added to it; at 1e6 km² it takes five cells and
    at 5e6 about thirty-six. float64 `cell_area` accumulates in float64, an
    integer one does too, and all manners then agree to 1e-9.
18. **Memory.** About 400 bytes per cell with every structure cached. The
    upstream table is the largest single item; skip it by not using `pull`.
19. **Stability.** A hundred repeated calls produce no drift and no growth.

### Packaging and documentation

20. **The wheel is 37 KB** and holds only the modules; the 20 MB of media stays
    out of it. Every python block in the documentation executes. Every relative
    link resolves. Every number quoted in the documentation is reproduced by
    running the code.

## Bugs these rounds found

| Round | Bug | Why it mattered |
| --- | --- | --- |
| API defaults | `upstream_area(layering="asap")` ran the unsafe push | Silently wrong on 27,029 of 360,000 cells |
| Aliasing | Cached arrays were handed out by reference | Writing to one corrupted every later call, with no sign |
| Degenerate input | `layering_alap` raised on a grid with no pit | An all-nodata tile is normal when tiling a global dataset |
| Complexity | `partition` rescanned the whole grid per basin | 1,166 basins cost 0.41 s where 26 cost 0.21 s |
| Precision | `upstream_area` forced float32 | A continental basin lost the cells nearest its outlet, with no way to ask for more |
| Cache model | LRU ages started at 0, like an untouched way | Every set lost one way; the simulator disagreed with a true LRU by 5e-6 |
| Decoding | The grid's own nodata code was ignored | 255 is a terminal here, so a grid using it for nodata silently gained cells |
| Cycles | `layering_cfds` swept unschedulable cells into one final layer | A ring then sat wholly inside one layer, where a push reads a value another cell in the same layer is writing. Conflicts stayed at zero, so the count could not see it |
| Precision | An integer `cell_area` fell to float32 | Counting upstream cells, the obvious use for an integer array, stops being exact above 2**24 |
| Balancing | `partition` counted cycle cells as work | With 36% of cells in a ring, load read 167/51/51/51 where the real work was 51 everywhere |
| Links | Nine README video links pointed at attachments from a discarded issue draft | GitHub does not keep those, so every reader got 404 where the page promised HD playback |
| Documentation | The rectangle-clip figure was given as a flat 98% | It is not a constant: measured across thirteen rectangles it runs 92.7% to 99.6% |
| Documentation | The parallel timing table was one stale run | Its pull row sat outside the spread of three fresh runs on the same machine |
| Geodesy | `pixel_area_km2` read the pixel width and used it for the height too | Cell area was out by the aspect ratio on any grid without square pixels: half on a 0.5 x 1 degree grid, double on 1 x 0.5. MERIT Hydro is square, so nothing released moves |
| Licence | The MIT grant named src/ and tools/, neither of which exists here | Carried over from the C package, so the grant named nothing in this repository |
| Documentation | Longest upstream path was listed as not agreeing bit for bit across manners | It agrees exactly: its confluence rule takes the larger of two donors, so there is no summation order to differ. Only drainage area moves |
| Documentation | `flowtopo.parallel` was documented as falling back to a serial loop without numba | It raises. Two other places already said so |
| Cache model | The offset between the two simulated arrays is a whole number of L1 and L2 sets | Cell i of each array always lands in the same set, the worst case for conflict misses. Depth-first reads 10.5% there and 8.8% at other alignments, breadth-first 21.8% against 14.2%. Their order is stable at every alignment tried, so the conclusion holds and the digits were being read too closely |

## A third pass: prose, layout and coverage

The rounds above attacked behaviour. A later pass attacked the writing and the
shape of the repository, and found its own defects.

* **`docs/methods.md` described nine of twenty-five public methods** while its
  own opening promised every one, with its origin and its complexity.
* **Three sentences in the README ran past forty words**, one to fifty-two.
* **Global products had grown into the longest section on the page**, saying
  less per line than the sections about the structures themselves.
* **Two figures were left unreferenced** after a section was rewritten, the
  same fault an earlier round had cleaned up once already.
* **A foreign D8 convention decoded silently.** TauDEM and GRASS number their
  directions 1 to 8; read as powers of two every undefined code becomes a pit,
  and the network falls into fragments with nothing to say so.
* **A missing raster raised a rasterio traceback** rather than naming the file.

What these have in common: none of them break a test. A reviewer looking only
at behaviour will not find them, and a reviewer reading only the README will
not find the ones in the code.

### Checks worth repeating

```sh
# every public name appears in the catalogue
python -c "
import pathlib, flowtopo, flowtopo.parallel as P
m = pathlib.Path('docs/methods.md').read_text()
pub = {n for n in flowtopo.__all__ if callable(getattr(flowtopo, n))}
pub |= {n for n in dir(P) if not n.startswith('_')
        and getattr(getattr(P, n), '__module__', '') == 'flowtopo.parallel'}
print(sorted(n for n in pub if n.split('.')[-1] not in m) or 'all covered')"

# every figure in docs/media is referenced by something
grep -o 'docs/media/[^)"]*' README.md docs/*.md | sed 's/.*://' | sort -u > /tmp/used
ls docs/media | sed 's|^|docs/media/|' | sort > /tmp/have
comm -13 /tmp/used /tmp/have    # anything printed is an orphan

# sentences over 28 words
python -c "
import re, pathlib
s = re.sub(r'\`\`\`.*?\`\`\`', '', pathlib.Path('README.md').read_text(), flags=re.S)
for p in s.split(chr(10)*2):
    f = ' '.join(p.split())
    if f[:1] in '#*|>' or not f: continue
    for x in re.split(r'(?<=[.:])\\s+', f):
        if len(x.split()) > 28: print(len(x.split()), x[:80])"
```

## Angles not yet attacked

A reviewer looking for something new should start here.

* **A second real dataset.** Everything is checked on one 731 km² basin and on
  synthetic networks. A HydroSHEDS tile, or a MERIT region with many basins and
  a dominant one, would exercise the partition code where it matters.
* **Very large grids.** `benchmark.py` runs 16 million cells and a single
  chain of 4 million orders without overflowing any stack, but the int32 limit
  at 2.1e9 is still argued from arithmetic rather than tested.
* **Non-MERIT D8 conventions.** Only the powers-of-two-clockwise-from-east
  encoding is handled. ArcGIS uses the same codes; other tools do not.
* **Windows and Linux.** Everything so far has run on macOS, arm64. CI covers
  Linux for the tests but nothing checks the numbers there.
* **numba version drift.** One version has been used throughout.
* **The locality metrics beyond the miss rate.** Stride, reuse interval and the
  per-layer statistics are computed but only the miss rate has been validated
  against an independent implementation.
* **Exhaustive enumeration beyond six cells.** Every network of three to six
  cells has been enumerated: the conflict-free layering never shares a receiver
  within a layer and never holds a cell together with its own receiver, and the
  three orderings are topological in their stated direction. Seven cells and up
  is untested, as is any property that needs a wider fan-in to show itself.

## Running the checks

```sh
pip install -e ".[speed,test]"
pytest                 # 177 tests
python example.py      # every structure, kernel and manner; ends PASS or FAIL
python benchmark.py    # serial against threaded
```

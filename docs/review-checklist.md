# Review checklist

Twenty rounds of review have been run on this package. Each round attacked one
class of defect, and seven real bugs came out of them, listed at the end. A
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
    pixels, coarse resolution, 80°N, and a D8 grid stored as float32.
15. **Foreign nodata codes.** A grid that marks nodata as 255, 0 or anything
    else is read correctly.

### Numerics and resources

16. **Dtypes and limits.** int32 indices reach 2.1e9 cells, comfortably above
    the 3.4e8 of the largest MERIT region. Strahler saturates at 255 rather
    than wrapping.
17. **Accumulation precision.** float32 stops resolving one 90 m cell once the
    running total passes about 1e5 km²; `cell_area` in float64 accumulates in
    float64, and all manners then agree to 1e-9.
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

## Angles not yet attacked

A reviewer looking for something new should start here.

* **A second real dataset.** Everything is checked on one 731 km² basin and on
  synthetic networks. A HydroSHEDS tile, or a MERIT region with many basins and
  a dominant one, would exercise the partition code where it matters.
* **Very large grids.** Nothing above 4 million cells has been run. The int32
  limit is argued, not tested.
* **Non-MERIT D8 conventions.** Only the powers-of-two-clockwise-from-east
  encoding is handled. ArcGIS uses the same codes; other tools do not.
* **Windows and Linux.** Everything so far has run on macOS, arm64. CI covers
  Linux for the tests but nothing checks the numbers there.
* **numba version drift.** One version has been used throughout.
* **The locality metrics beyond the miss rate.** Stride, reuse distance and the
  per-layer statistics are computed but only the miss rate has been validated
  against an independent implementation.

## Running the checks

```sh
pip install -e ".[speed,test]"
pytest                 # 86 tests
python example.py      # every structure, kernel and manner; ends PASS or FAIL
python benchmark.py    # serial against threaded
```

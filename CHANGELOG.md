# Changelog

## 0.1.0

First release.

* Three serial orderings: depth-first from the pits, breadth-first from the
  pits, topological sort from the sources.
* Three parallel layerings: as soon as possible, conflict-free downstream, as
  late as possible.
* Four kernels: upstream drainage area, distance to outlet, longest upstream
  path, Strahler stream order, each under the propagation manners it supports.
* Memory-locality metrics with a simulated N-way set-associative LRU cache.
* Threaded kernels through numba `prange` in `flowtopo.parallel`.
* Expected outputs for the bundled example basin ship with the tests.

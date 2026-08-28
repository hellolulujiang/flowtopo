"""Does the layering pay in Python, and what does the conflict-free rule buy?

Four ways to accumulate drainage area over the same network:

* a serial scalar loop over a topological ordering, compiled by numba;
* the layers threaded with ``prange``, pushing into the receiver with no
  atomic, under the conflict-free downstream layering;
* the same push under the as-soon-as-possible layering, which does not
  guarantee exclusive receivers;
* the threaded pull form, which gathers from the donors so every thread writes
  only to its own cell.

Two columns matter and they say different things.

``conflicts`` counts, deterministically, how many times a layer writes into a
receiver that another cell in the same layer also writes into.  It is a
property of the layering, not of a run: zero means a threaded push can never
race, non-zero means it can.

``wrong`` is what a single run actually got wrong.  A race is probabilistic,
so a layering with thousands of conflicts can still come out clean on a run
where the colliding cells happened to land on one thread.  That is exactly why
the conflict count is the thing to look at.

    python benchmark.py [sizes...]
"""

import sys
import time

import numpy as np

import flowtopo
from flowtopo import parallel
from flowtopo.synthetic import synthetic_d8


def count_conflicts(decomp, idxs_ds):
    """Receivers written more than once inside one layer."""
    total = 0
    for members in decomp:
        ds = idxs_ds[members]
        ds = ds[(ds >= 0) & (ds != members)]
        if ds.size:
            total += ds.size - np.unique(ds).size
    return total


def run(size, repeats=5):
    topo = flowtopo.FlowTopo.from_d8(
        synthetic_d8(size), transform=(0.0, 1 / 1200, 0.0, 40.0, 0.0, -1 / 1200)
    )
    reference = topo.upstream_area(ordering="dfs")
    decomp = topo.decomposition("cfds", "u2d")
    layer_sizes = decomp.layer_sizes

    print(
        f"\n{size} x {size} = {topo.ncells:,} cells, {topo.nbasins:,} basins, "
        f"{decomp.nlayers} layers "
        f"(largest {layer_sizes.max():,}, median {int(np.median(layer_sizes))})"
    )

    forms = (
        ("serial ordering, 1 thread", None,
         lambda: topo.upstream_area(ordering="dfs")),
        ("cfds layering, push, threads", "cfds",
         lambda: parallel.upstream_area(topo, "cfds", "push")),
        ("asap layering, push, threads", "asap",
         lambda: parallel.upstream_area(topo, "asap", "push")),
        ("cfds layering, pull, threads", "cfds",
         lambda: parallel.upstream_area(topo, "cfds", "pull")),
    )

    rows = []
    for label, layering, call in forms:
        call()  # compile
        times = []
        for _ in range(repeats):
            start = time.perf_counter()
            result = call()
            times.append(time.perf_counter() - start)
        wrong = int(
            np.count_nonzero(
                ~np.isclose(result[topo.mask], reference[topo.mask],
                            rtol=1e-4, atol=1e-2)
            )
        )
        conflicts = (
            0 if layering is None
            else count_conflicts(topo.decomposition(layering, "u2d"), topo.idxs_ds)
        )
        rows.append((label, float(np.median(times)), conflicts, wrong, layering))

    base = rows[0][1]
    if topo.ncells < 500_000:
        print("    (this grid is small: starting threads costs more than the")
        print("     work saved, so the threaded rows will look slow. Try 2000+.)")
    print(f"    {'form':30s} {'median s':>9s} {'speedup':>8s} "
          f"{'conflicts':>10s} {'wrong':>10s}")
    for label, seconds, conflicts, wrong, layering in rows:
        conflict_text = "-" if layering is None else f"{conflicts:,}"
        wrong_text = "-" if wrong == 0 and layering is None else f"{wrong:,}"
        print(f"    {label:30s} {seconds:9.4f} {base / seconds:7.2f}x "
              f"{conflict_text:>10s} {wrong_text:>10s}")

    push = next(r for r in rows if r[0].startswith("cfds layering, push"))
    pull = next(r for r in rows if r[0].startswith("cfds layering, pull"))
    print(f"    -> the two threaded forms that are always correct: "
          f"push is {pull[1] / push[1]:.2f}x faster than pull")


if __name__ == "__main__":
    try:
        from numba import config

        print(f"numba threads available: {config.NUMBA_NUM_THREADS}")
    except ImportError:
        print("numba not installed: the threaded forms fall back to one thread")

    for argument in [int(a) for a in sys.argv[1:]] or [1000, 2000, 4000]:
        run(argument)

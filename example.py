"""Run every structure, kernel and manner on the bundled example basin.

The Python counterpart of ``flowtopo_example`` in the C package.  It

1. reads the D8 grid;
2. builds the three serial orderings and the three parallel layerings;
3. counts, per layering, how many times a layer writes twice into one receiver;
4. runs the four kernels under every manner and checks they agree;
5. reports the memory-locality metrics;
6. writes the arrays to ``out/``.

    python example.py [--data data/dir_example.tif] [--out out]
"""

import argparse
import os
import time

import numpy as np

import flowtopo
from flowtopo.raster import write_geotiff

ORDERINGS = ("dfs", "bfs", "topo")
LAYERINGS = ("asap", "cfds", "alap")
ORDER_LABEL = {"dfs": "seq_dfs", "bfs": "seq_bfs", "topo": "seq_topo"}
LAYER_LABEL = {"asap": "lyr_asap", "cfds": "lyr_cfds", "alap": "lyr_alap"}


def banner(title):
    print("\n" + "=" * 62)
    print(f" {title}")
    print("=" * 62)


def timed(call, repeats=3):
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = call()
        times.append(time.perf_counter() - start)
    return float(np.median(times)), result


def count_conflicts(decomp, idxs_ds):
    """Receivers written more than once inside one layer."""
    total = 0
    worst = 0
    for members in decomp:
        ds = idxs_ds[members]
        ds = ds[(ds >= 0) & (ds != members)]
        if ds.size:
            repeats = ds.size - np.unique(ds).size
            total += repeats
            worst = max(worst, repeats)
    return total, worst


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=os.path.join("data", "dir_example.tif"))
    parser.add_argument("--out", default="out")
    parser.add_argument("--upa-threshold", type=float, default=10.0,
                        help="drainage area in km2 that defines the channel network")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    banner("Input")
    topo = flowtopo.FlowTopo.from_raster(args.data)
    total = topo.idxs_ds.size
    print(f"  grid            : {topo.nrow} x {topo.ncol} = {total:,} cells")
    print(f"  valid cells     : {topo.ncells:,} "
          f"({100.0 * topo.ncells / total:.1f}% of the grid)")
    print(f"  basins          : {topo.nbasins}")
    print(f"  numba           : {'yes' if flowtopo.HAS_NUMBA else 'no (pure python, slow)'}")

    # ---- Stage 1: the six structures ------------------------------------
    banner("Stage 1  Building the six topological representations")
    for name in ORDERINGS:
        seconds, seq = timed(
            lambda n=name: topo.ordering(n, "u2d"), args.repeats
        )
        print(f"  {ORDER_LABEL[name]:9s} {seq.size:10,} cells   {seconds:8.4f} s")
    for name in LAYERINGS:
        seconds, (layers, nlayers) = timed(
            lambda n=name: topo.layering(n), args.repeats
        )
        print(f"  {LAYER_LABEL[name]:9s} {nlayers:10,} layers  {seconds:8.4f} s")

    # ---- Stage 2: write conflicts ---------------------------------------
    banner("Stage 2  Write conflicts inside a layer")
    print("  A push writes into the receiver.  That is only safe if no two")
    print("  cells of a layer share one.  This counts how often they do.\n")
    print(f"  {'layering':10s} {'layers':>8s} {'conflicts':>12s} {'worst layer':>12s}")
    for name in LAYERINGS:
        decomp = topo.decomposition(name, "u2d")
        conflicts, worst = count_conflicts(decomp, topo.idxs_ds)
        print(f"  {LAYER_LABEL[name]:10s} {decomp.nlayers:8,} "
              f"{conflicts:12,} {worst:12,}")
    print("\n  Only the conflict-free downstream layering reaches zero, and it")
    print("  is the only one under which a plain push is safe.")

    # ---- Stage 3: the four kernels --------------------------------------
    banner("Stage 3  The four kernels under every manner")
    ldn_ref = topo.distance_to_outlet(ordering="dfs")
    upa_ref = topo.upstream_area(ordering="dfs")
    lup_ref = topo.longest_upstream_path(ldn_ref, ordering="dfs")
    channel = topo.channel_mask(upa_ref, args.upa_threshold)
    ord_ref = topo.strahler_order(ordering="dfs", channel_mask=channel)
    print(f"  channel network : {int(channel.sum()):,} cells above "
          f"{args.upa_threshold:g} km2")

    valid = topo.mask
    worst = {"ldn": 0.0, "upa": 0.0, "lup": 0.0, "ord": 0}
    unsafe_rows = []

    print(f"\n  {'structure':10s} {'kernel':24s} {'manner':12s} "
          f"{'median s':>9s} {'check':>12s}")

    def report(structure, kernel, manner, seconds, diff, unit, safe=True):
        flag = "" if safe else "   <- unsafe"
        print(f"  {structure:10s} {kernel:24s} {manner:12s} {seconds:9.4f} "
              f"{diff:9.3g} {unit}{flag}")

    for name in ORDERINGS:
        label = ORDER_LABEL[name]
        seconds, value = timed(
            lambda n=name: topo.distance_to_outlet(ordering=n), args.repeats)
        diff = float(np.abs(value[valid] - ldn_ref[valid]).max())
        worst["ldn"] = max(worst["ldn"], diff)
        report(label, "flow length downstream", "serial", seconds, diff, "m")

        seconds, value = timed(
            lambda n=name: topo.upstream_area(ordering=n), args.repeats)
        diff = float(np.abs(value[valid] - upa_ref[valid]).max())
        worst["upa"] = max(worst["upa"], diff)
        report(label, "upstream drainage area", "serial", seconds, diff, "km2")

        seconds, value = timed(
            lambda n=name: topo.longest_upstream_path(ldn_ref, ordering=n),
            args.repeats)
        diff = float(np.abs(value[valid] - lup_ref[valid]).max())
        worst["lup"] = max(worst["lup"], diff)
        report(label, "flow length upstream", "serial", seconds, diff, "m")

        seconds, value = timed(
            lambda n=name: topo.strahler_order(ordering=n, channel_mask=channel),
            args.repeats)
        diff = int(np.count_nonzero(value[channel] != ord_ref[channel]))
        worst["ord"] = max(worst["ord"], diff)
        report(label, "Strahler stream order", "serial", seconds, diff, "cells")

    for name in LAYERINGS:
        label = LAYER_LABEL[name]
        conflicts, _ = count_conflicts(topo.decomposition(name, "u2d"),
                                       topo.idxs_ds)

        seconds, value = timed(
            lambda n=name: topo.distance_to_outlet(layering=n, manner="push"),
            args.repeats)
        diff = float(np.abs(value[valid] - ldn_ref[valid]).max())
        worst["ldn"] = max(worst["ldn"], diff)
        report(label, "flow length downstream", "push", seconds, diff, "m")

        for manner in ("pull", "push", "atomic_push"):
            safe = manner != "push" or conflicts == 0
            seconds, value = timed(
                lambda n=name, m=manner: topo.upstream_area(layering=n, manner=m),
                args.repeats)
            diff = float(np.abs(value[valid] - upa_ref[valid]).max())
            if safe:
                worst["upa"] = max(worst["upa"], diff)
            else:
                unsafe_rows.append((label, "upstream drainage area", diff))
            report(label, "upstream drainage area", manner, seconds, diff,
                   "km2", safe)

            seconds, value = timed(
                lambda n=name, m=manner: topo.longest_upstream_path(
                    ldn_ref, layering=n, manner=m), args.repeats)
            diff = float(np.abs(value[valid] - lup_ref[valid]).max())
            if safe:
                worst["lup"] = max(worst["lup"], diff)
            else:
                unsafe_rows.append((label, "flow length upstream", diff))
            report(label, "flow length upstream", manner, seconds, diff, "m", safe)

        for manner in ("pull", "push"):
            safe = manner != "push" or conflicts == 0
            seconds, value = timed(
                lambda n=name, m=manner: topo.strahler_order(
                    layering=n, manner=m, channel_mask=channel), args.repeats)
            diff = int(np.count_nonzero(value[channel] != ord_ref[channel]))
            if safe:
                worst["ord"] = max(worst["ord"], diff)
            else:
                unsafe_rows.append((label, "Strahler stream order", diff))
            report(label, "Strahler stream order", manner, seconds, diff,
                   "cells", safe)

    # ---- Stage 4: locality ----------------------------------------------
    banner("Stage 4  Memory-locality metrics")
    print(f"  {'structure':10s} {'L1 miss':>10s} {'L2 miss':>10s} "
          f"{'L3 miss':>10s} {'row jump':>10s}")
    for name in ORDERINGS:
        metrics = topo.serial_locality(name)
        print(f"  {ORDER_LABEL[name]:10s} "
              f"{100 * metrics['miss_rate_L1']:9.4f}% "
              f"{100 * metrics['miss_rate_L2']:9.4f}% "
              f"{100 * metrics['miss_rate_L3']:9.4f}% "
              f"{metrics['row_jump_frac']:10.4f}")
    for name in LAYERINGS:
        metrics = topo.parallel_locality(name)
        print(f"  {LAYER_LABEL[name]:10s} "
              f"{100 * metrics['miss_rate_L1']:9.4f}% "
              f"{100 * metrics['miss_rate_L2']:9.4f}% "
              f"{100 * metrics['miss_rate_L3']:9.4f}% "
              f"{'-':>10s}  ({metrics['nlayers']} layers)")
    print("\n  The simulated cache is the machine of the companion paper:")
    print("  32 KB 8-way L1, 1 MB 16-way L2, 35.75 MB 11-way L3.  This basin")
    print("  fits inside L3, so the miss rates stay low.")

    # ---- Stage 5: write --------------------------------------------------
    banner("Stage 5  Writing outputs")
    os.makedirs(args.out, exist_ok=True)

    def dump(name, array, dtype, nodata):
        path = os.path.join(args.out, f"{name}.tif")
        write_geotiff(path, np.ascontiguousarray(array, dtype=dtype),
                      topo.header(dtype=np.dtype(dtype).name, nodata=nodata))
        print(f"  {path}")

    for name in ORDERINGS:
        seq = topo.ordering(name, "u2d")
        position = np.full(total, -1, dtype=np.int32)
        position[seq] = np.arange(seq.size, dtype=np.int32)
        dump(ORDER_LABEL[name], position, np.int32, -1)
    for name in LAYERINGS:
        dump(LAYER_LABEL[name], topo.layering(name)[0], np.int32, -1)
    dump("ldn", ldn_ref, np.float32, -9999.0)
    dump("lup", lup_ref, np.float32, -9999.0)
    dump("upa", upa_ref, np.float32, -9999.0)
    dump("ord", ord_ref, np.uint8, 0)

    # ---- verdict ---------------------------------------------------------
    tolerance_upa = float(upa_ref[valid].max()) * 1e-5
    checks = [
        ("flow length downstream", worst["ldn"], 1e-3, "m"),
        ("flow length upstream", worst["lup"], 1e-3, "m"),
        ("upstream drainage area", worst["upa"], tolerance_upa, "km2"),
        ("Strahler stream order", worst["ord"], 0, "cells"),
    ]
    passed = all(value <= limit for _, value, limit, _ in checks)

    banner("PASS" if passed else "FAIL")
    print("  Worst disagreement against the serial depth-first reference,")
    print("  over every safe combination of structure and manner:\n")
    for kernel, value, limit, unit in checks:
        verdict = "ok" if value <= limit else "FAILED"
        print(f"    {kernel:24s} {value:12.6g} {unit:5s} "
              f"tolerance {limit:.4g} {unit:5s} {verdict}")
    print("\n  Drainage area and longest path are not expected to match bit for")
    print("  bit: floating-point addition is not associative and each manner")
    print("  sums a confluence's donors in a different order.")

    if unsafe_rows:
        print("\n  Rows marked unsafe above ran a plain push under a layering")
        print("  that puts two donors of one receiver in the same layer.  They")
        print("  are excluded from the verdict; losing a write there is the")
        print("  expected behaviour, and it is what the layering is for:")
        for label, kernel, diff in unsafe_rows:
            print(f"    {label:10s} {kernel:24s} off by {diff:g}")

    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

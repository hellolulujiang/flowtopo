"""The promises the API makes to a caller.

Cached arrays are handed out on every call, so writing to one would corrupt
every later call. Results must not depend on how many times something is asked
for, and an accumulator must keep the precision it was given.
"""

import numpy as np
import pytest

import flowtopo
from flowtopo.synthetic import synthetic_d8

TRANSFORM = (0.0, 1 / 1200, 0.0, 35.0, 0.0, -1 / 1200)


@pytest.fixture(scope="module")
def topo():
    return flowtopo.FlowTopo.from_d8(synthetic_d8(150, seed=3),
                                     transform=TRANSFORM)


# ---------------------------------------------------------------------------
# Cached arrays cannot be written through
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("attribute", ["idxs_ds", "mask", "basins",
                                       "cell_area", "pixel_length"])
def test_cached_attributes_are_read_only(topo, attribute):
    with pytest.raises(ValueError):
        getattr(topo, attribute)[0] = 0


@pytest.mark.parametrize("name", ["dfs", "bfs", "topo"])
@pytest.mark.parametrize("direction", ["d2u", "u2d"])
def test_orderings_are_read_only(topo, name, direction):
    with pytest.raises(ValueError):
        topo.ordering(name, direction)[0] = 0


@pytest.mark.parametrize("name", ["asap", "cfds", "alap"])
def test_layerings_are_read_only(topo, name):
    with pytest.raises(ValueError):
        topo.layering(name)[0][0] = 0


# ---------------------------------------------------------------------------
# Results do not drift between calls
# ---------------------------------------------------------------------------


def test_repeated_calls_agree_bit_for_bit(topo):
    ldn = topo.distance_to_outlet(ordering="dfs")
    pairs = [
        (topo.upstream_area(ordering="dfs"), topo.upstream_area(ordering="dfs")),
        (topo.upstream_area(layering="cfds", manner="push"),
         topo.upstream_area(layering="cfds", manner="push")),
        (topo.longest_upstream_path(ldn, ordering="dfs"),
         topo.longest_upstream_path(ldn, ordering="dfs")),
        (topo.partition(4, "subbasin")[0], topo.partition(4, "subbasin")[0]),
    ]
    for first, second in pairs:
        assert np.array_equal(first, second)


def test_a_kernel_does_not_touch_what_it_was_given(topo):
    ldn = topo.distance_to_outlet(ordering="dfs")
    before = ldn.copy()
    topo.longest_upstream_path(ldn, ordering="dfs")
    topo.longest_upstream_path(ldn, layering="cfds", manner="push")
    assert np.array_equal(ldn, before)


# ---------------------------------------------------------------------------
# Accumulation keeps the precision it was handed
# ---------------------------------------------------------------------------


def test_upstream_area_follows_the_dtype_of_cell_area(topo):
    assert topo.upstream_area(ordering="dfs").dtype == np.float32
    wide = np.asarray(topo.cell_area, dtype=np.float64)
    assert topo.upstream_area(ordering="dfs", cell_area=wide).dtype == np.float64


@pytest.mark.parametrize("kwargs", [
    {"layering": "cfds", "manner": "push"},
    {"layering": "alap", "manner": "pull"},
    {"layering": "asap", "manner": "atomic_push"},
    {"ordering": "topo"},
])
def test_float64_accumulation_agrees_far_more_closely(topo, kwargs):
    wide = np.asarray(topo.cell_area, dtype=np.float64)
    reference = topo.upstream_area(ordering="dfs", cell_area=wide)
    got = topo.upstream_area(cell_area=wide, **kwargs)
    assert np.allclose(got[topo.mask], reference[topo.mask], rtol=1e-9, atol=1e-9)


# ---------------------------------------------------------------------------
# The threaded push is reproducible, which is the reason to prefer it
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not flowtopo.HAS_NUMBA, reason="threading needs numba")
def test_conflict_free_push_gives_the_same_answer_at_any_thread_count(topo):
    """No two cells in a layer share a receiver, so the sum order is fixed."""
    import numba

    from flowtopo import parallel

    before = numba.get_num_threads()
    try:
        results = []
        for threads in (1, 2, 4):
            numba.set_num_threads(threads)
            results.append(parallel.upstream_area(topo, "cfds", "push").copy())
    finally:
        numba.set_num_threads(before)

    for other in results[1:]:
        assert np.array_equal(results[0], other)

    serial = topo.upstream_area(ordering="dfs")
    assert np.allclose(results[0][topo.mask], serial[topo.mask],
                       rtol=1e-4, atol=1e-2)


# ---------------------------------------------------------------------------
# The cache simulator is a measuring instrument, so it gets checked against
# cases whose answer is known in advance
# ---------------------------------------------------------------------------


def _miss(sequence, receivers, **kwargs):
    from flowtopo.locality import miss_rates

    return miss_rates(np.asarray(sequence, dtype=np.int32),
                      np.asarray(receivers, dtype=np.int32),
                      elem_bytes=4, levels=("L1",), **kwargs)["L1"]


def test_an_empty_way_is_used_before_anything_is_evicted():
    """Two lines in one eight-way set: revisiting the first must hit."""
    a, b = 0, 64 * 16          # both land in set 0
    misses = _miss([a, b, a], np.full(b + 1, -1))
    assert misses == pytest.approx(2 / 3)


def test_walking_straight_through_memory_misses_once_per_cache_line():
    cells = np.arange(200_000)
    assert _miss(cells, np.full(cells.size, -1)) == pytest.approx(1 / 16, abs=2e-3)


def test_returning_to_one_cell_misses_only_the_first_time():
    cells = np.zeros(50_000)
    assert _miss(cells, np.full(1, -1)) == pytest.approx(1 / 50_000, abs=1e-6)


def test_a_working_set_larger_than_the_cache_misses_every_time():
    lines = np.arange(512 * 8) * 16          # eight times the L1 capacity
    cells = np.tile(lines, 4)
    assert _miss(cells, np.full(int(lines.max()) + 1, -1)) == pytest.approx(1.0)


@pytest.mark.parametrize("lines,thrashes", [(8, False), (9, True)])
def test_associativity_boundary_sits_exactly_at_the_way_count(lines, thrashes):
    """Eight lines per set fit; the ninth makes every revisit miss."""
    cells = np.tile(np.arange(lines) * 64 * 16, 20)
    misses = _miss(cells, np.full(int(cells.max()) + 1, -1))
    if thrashes:
        assert misses == pytest.approx(1.0)
    else:
        assert misses == pytest.approx(lines / cells.size, abs=1e-6)

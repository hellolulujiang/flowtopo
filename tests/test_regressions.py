"""Regression tests of the input contracts and the numerical conventions."""
import importlib

import numpy as np
import pytest

import flowtopo
from flowtopo import FlowTopo


def test_raster_mask_cuts_receivers_like_flat_mask():
    downstream = np.array([1, 3, 3, 3])
    raster_mask = np.array([[True, False], [True, True]])
    grid = FlowTopo(downstream, (2, 2), mask=raster_mask)
    flat = FlowTopo(downstream, (2, 2), mask=raster_mask.ravel())
    assert np.array_equal(grid.idxs_ds, [0, -1, 3, 3])
    assert np.array_equal(grid.idxs_ds, flat.idxs_ds)
    assert np.array_equal(grid.basins, flat.basins)


@pytest.mark.parametrize('transform', [(), (0, 1), (0, 1, 0, 2), (0, 1, 0, 2, 0)])
def test_incomplete_transform_is_rejected_before_indexing(transform):
    with pytest.raises(ValueError, match='six finite numbers'):
        FlowTopo(np.array([0]), (1, 1), transform=transform)


# ---- 1.1: checked against the C code ----

def test_a_step_is_measured_at_the_latitude_between_the_two_centres():
    from flowtopo import geodist
    # two cells of one row of a one-degree grid whose top is at 60 N: their centres are at 59.5 N
    length = geodist.pixel_length(np.array([1, 1], dtype=np.int32), 2, (0.0, 1.0, 0.0, 60.0, 0.0, -1.0), True)
    assert length[0] == np.float32(geodist.degree_metres_x(59.5))
    assert abs(float(length[0]) - 56639.81) < 0.5          # 55,799.98 m at 60 N, the row's edge (until 1.0.5)


def test_locality_statistics_are_the_c_codes_order_statistics():
    from flowtopo import locality
    idxs_ds = np.array([2, 2, 2], dtype=np.int32)          # two cells drain into cell 2, a pit
    out = locality.serial_locality(np.array([2, 0, 1], dtype=np.int32), idxs_ds, ncol=3, levels=())
    assert out["ds_stride_mean"] == 1.5                     # the pit reads no receiver: strides 2 and 1
    assert out["ds_stride_median"] == 2.0                   # the upper of the two middle values
    assert out["stride_median"] == 2.0 and out["stride_p95"] == 2.0
    # no line is touched twice: the interval is the length of the sequence
    far = locality.serial_locality(np.array([0, 16], dtype=np.int32), np.full(17, -1, dtype=np.int32), ncol=17,
                                   levels=())
    assert far["reuse_interval_mean"] == 2.0 and far["reuse_interval_median"] == 2.0
    layers = np.array([1, 1, 0], dtype=np.int64)
    par = locality.parallel_locality(layers, idxs_ds, np.ones(3, dtype=bool), ncol=3, levels=())
    assert par["ds_stride_mean"] == 1.5


def test_a_cell_draining_off_the_network_is_a_pit_with_or_without_a_mask():
    implicit = flowtopo.FlowTopo(np.array([1, -1], dtype=np.int32), (1, 2))
    explicit = flowtopo.FlowTopo(np.array([1, -1], dtype=np.int32), (1, 2), mask=np.array([True, False]))
    assert np.array_equal(implicit.idxs_ds, explicit.idxs_ds) and list(implicit.idxs_ds) == [0, -1]
    # the one cell is a basin of a single cell, which takes no part in the partition (as the C code)
    assert list(implicit.partition(1)[1]) == list(explicit.partition(1)[1]) == [0]
    assert list(implicit.partition(1)[0]) == list(explicit.partition(1)[0]) == [-1, -1]


def test_the_mainstem_takes_the_smaller_index_of_two_equal_tributaries():
    p = importlib.import_module("flowtopo.partition")   # flowtopo.partition itself is the function
    topo = flowtopo.FlowTopo(np.array([3, 3, -1, 3], dtype=np.int32), (2, 2))
    table, n_up = topo.upstream()
    upstream = np.array([1.0, 1.0, 0.0, 3.0])
    stem = p._mainstem(topo.idxs_ds, 3, upstream, table, n_up, np.asarray(topo.mask))
    assert list(stem) == [3, 0]


def test_equally_distant_neighbours_come_in_the_order_of_their_indices():
    p = importlib.import_module("flowtopo.partition")   # flowtopo.partition itself is the function
    points = np.array([(3, 4), (3, -4), (-3, 4), (-3, -4), (4, 3), (4, -3), (-4, 3), (-4, -3), (0, 0)], dtype=float)
    squared, near = p._nearest(points, points[8:9], 5)
    assert list(near[0]) == [8, 0, 1, 2, 3] and list(squared[0]) == [0.0, 25.0, 25.0, 25.0, 25.0]
    # more points at the cut-off distance than the tree was asked for
    ring = np.array([(np.cos(a), np.sin(a)) for a in np.linspace(0.0, 2.0 * np.pi, 40, endpoint=False)])
    ring = np.round(ring * 0.0)                             # forty points at the origin
    _, near = p._nearest(np.vstack([ring, [(5.0, 5.0)]]), np.array([[0.0, 0.0]]), 3)
    assert list(near[0]) == [0, 1, 2]


def test_masked_donor_tables_are_not_shared_between_calls():
    topo = flowtopo.FlowTopo(np.array([3, 3, -1, 3], dtype=np.int32), (2, 2))
    first = topo.upstream(np.array([1, 0, 0, 1], dtype=np.uint8))
    second = topo.upstream(np.array([0, 1, 0, 1], dtype=np.uint8))
    assert first[0][3][0] == 0 and second[0][3][0] == 1
    assert topo.upstream()[1][3] == 2


# ---- 1.1, continued ----

def test_the_mainstem_takes_the_larger_upa_of_two_equal_tributaries_when_given():
    p = importlib.import_module("flowtopo.partition")   # flowtopo.partition itself is the function
    topo = flowtopo.FlowTopo(np.array([3, -1, 3, 3], dtype=np.int32), (2, 2))
    table, n_up = topo.upstream()
    upstream = np.array([1.0, 0.0, 1.0, 3.0])
    mask = np.asarray(topo.mask)
    # two one-cell donors, 0 and 2: the C code takes the larger MERIT upa first (cell 2, a row further south)
    upa = np.array([0.00431728, 0.0, 0.00431739, 0.0129], dtype=np.float32)
    assert list(p._mainstem(topo.idxs_ds, 3, upstream, table, n_up, mask, upa)) == [3, 2]
    assert list(p._mainstem(topo.idxs_ds, 3, upstream, table, n_up, mask)) == [3, 0]
    # equal areas: the smaller index
    upa[2] = upa[0]
    assert list(p._mainstem(topo.idxs_ds, 3, upstream, table, n_up, mask, upa)) == [3, 0]


def test_the_partition_takes_upa_of_one_value_a_cell():
    topo = flowtopo.FlowTopo(np.array([3, 3, 3, 3], dtype=np.int32), (2, 2))
    with pytest.raises(ValueError, match="upa holds"):
        topo.partition(1, upa=np.ones(3, dtype=np.float32))


def test_the_partition_refuses_an_area_that_is_not_finite_and_positive_on_the_network():
    # 3 is the pit; 0 and 2 drain into it with one donor each (4 and 5): two tributaries of two cells
    topo = flowtopo.FlowTopo(np.array([3, -1, 3, 3, 0, 2], dtype=np.int64), (2, 3))
    upa = np.array([2.0, 0.0, 2.0, 5.0, 1.0, 1.0], dtype=np.float32)
    part, _ = topo.partition(n_parts=2, level="subbasin", upa=upa, min_subtree_size=1)      # off the network: any value
    assert np.asarray(part).size == 6
    for bad in (np.nan, np.inf, 0.0, -9999.0):
        broken = upa.copy()
        broken[2] = bad
        with pytest.raises(ValueError, match="finite, positive area"):
            topo.partition(n_parts=2, level="subbasin", upa=broken, min_subtree_size=1)


def test_nearest_of_no_point_finds_none():
    p = importlib.import_module("flowtopo.partition")
    squared, near = p._nearest(np.empty((0, 2)), np.array([[0.0, 0.0]]), 3)
    assert squared.shape == (1, 0) and near.shape == (1, 0)


@pytest.mark.skipif(not flowtopo.HAS_NUMBA, reason="the threaded kernels need numba")
def test_threaded_kernels_take_raster_shaped_inputs():
    from flowtopo import parallel
    topo = flowtopo.FlowTopo(np.array([3, 3, 3, 3], dtype=np.int32), (2, 2))
    flat = parallel.upstream_area(topo, cell_area=np.ones(4))
    grid = parallel.upstream_area(topo, cell_area=np.ones((2, 2)))
    assert np.array_equal(flat, grid) and grid[3] == 4.0
    ldn = np.zeros((2, 2), dtype=np.float32)
    assert parallel.longest_upstream_path(topo, ldn).shape == (4,)
    with pytest.raises(ValueError, match="holds 3 values"):
        parallel.upstream_area(topo, cell_area=np.ones(3))

import numpy as np
import pytest

from flowtopo import FlowTopo
from flowtopo import core


@pytest.mark.parametrize("shape", [(1.9, 1), (-1, -1), (1, np.nan), (1, np.inf),
                                 (True, 1), (1,), (1, 1, 1), ("1", 1)])
def test_constructor_rejects_noninteger_or_invalid_shape(shape):
    with pytest.raises(ValueError, match="shape"):
        FlowTopo(np.array([0]), shape)


def test_flat_d8_rejects_fractional_shape_before_building_pointers():
    with pytest.raises(ValueError, match="shape"):
        FlowTopo.from_d8(np.array([0], np.uint8), shape=(1.9, 1))


def test_whole_dimensions_and_empty_grid_remain_supported():
    assert FlowTopo(np.array([0]), (1., np.int64(1))).shape == (1, 1)
    assert FlowTopo(np.array([], np.int32), (0, 2)).ncells == 0


def test_dfs_accepts_raster_and_flat_upstream_area_with_the_same_order():
    downstream = np.array([2, 2, 2], np.int32)
    area = np.array([[2., 1., 3.]], np.float64)
    topo = FlowTopo(downstream, (1, 3))
    np.testing.assert_array_equal(topo.ordering(upa=area), [2, 0, 1])
    np.testing.assert_array_equal(topo.ordering(upa=area), topo.ordering(upa=area.ravel()))
    np.testing.assert_array_equal(core.seq_dfs_from_pit(downstream, area), [2, 0, 1])


def test_channel_and_strahler_masks_accept_grids_and_return_one_value_per_cell():
    topo = FlowTopo(np.array([3, 3, 3, 3]), (2, 2))
    area = np.array([[1., 2.], [3., 4.]], np.float64)
    channel = topo.channel_mask(area, 2.)
    assert channel.shape == (4,)
    np.testing.assert_array_equal(channel, [False, True, True, True])
    np.testing.assert_array_equal(channel, topo.channel_mask(area.ravel(), 2.))
    flat = topo.strahler_order(ordering="dfs", channel_mask=channel)
    grid = topo.strahler_order(ordering="dfs", channel_mask=channel.reshape((2, 2)))
    np.testing.assert_array_equal(flat, grid)


@pytest.mark.parametrize("values", [np.array([True]), np.ones(3), np.ones(5)])
def test_masks_and_areas_cannot_broadcast_wrong_cell_counts(values):
    topo = FlowTopo(np.array([3, 3, 3, 3]), (2, 2))
    with pytest.raises(ValueError, match="one value each"):
        topo.channel_mask(values)
    with pytest.raises(ValueError, match="one value each"):
        topo.strahler_order(ordering="dfs", channel_mask=values)

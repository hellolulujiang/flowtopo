"""Input contract regressions found in the October 2026 source review."""
import numpy as np
import pytest

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

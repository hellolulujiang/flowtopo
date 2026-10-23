"""The bundled example sits where it is cut from MERIT Hydro.

``data/dir_example.tif`` is an excerpt of MERIT Hydro, whose cells are 1/1200 degree with edges at
-180 - 1/2400 + j/1200 and 85 - 1/2400 - i/1200.  Its header, the outlet point and the basin outline
must lie on that grid, so that the example overlays MERIT Hydro cell for cell.
"""

import json
import os

import numpy as np

from flowtopo.raster import read_geotiff

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, os.pardir, "data")
MERIT_WEST = -180.0 - 1.0 / 2400.0
MERIT_NORTH = 85.0 - 1.0 / 2400.0


def _header():
    _, header = read_geotiff(os.path.join(DATA, "dir_example.tif"))
    return header


def _off_by(value):
    return abs(value - round(value))


def test_the_example_header_is_on_the_merit_grid():
    t = _header().transform
    assert t[1] == 1.0 / 1200.0 and t[5] == -1.0 / 1200.0, t
    assert t[2] == 0.0 and t[4] == 0.0
    assert _off_by((t[0] - MERIT_WEST) * 1200.0) < 1e-6, t[0]
    assert _off_by((MERIT_NORTH - t[3]) * 1200.0) < 1e-6, t[3]


def test_the_outlet_point_is_the_centre_of_the_outlet_cell():
    data, header = read_geotiff(os.path.join(DATA, "dir_example.tif"))
    grid = data.reshape(header.shape)
    t = header.transform
    with open(os.path.join(DATA, "outlet_example.geojson")) as f:
        x, y = json.load(f)["features"][0]["geometry"]["coordinates"]
    column = (x - t[0]) / t[1] - 0.5
    row = (t[3] - y) / -t[5] - 0.5
    assert _off_by(column) < 1e-6 and _off_by(row) < 1e-6, (row, column)
    assert grid[int(round(row)), int(round(column))] == 0          # the cell drains out of the basin


def test_the_basin_outline_runs_along_cell_edges():
    t = _header().transform
    with open(os.path.join(DATA, "bsn_example.geojson")) as f:
        rings = json.load(f)["features"][0]["geometry"]["coordinates"]
    points = np.array([point for ring in rings for point in ring], dtype=np.float64)
    columns = (points[:, 0] - t[0]) / t[1]
    rows = (t[3] - points[:, 1]) / -t[5]
    assert np.max(np.abs(columns - np.round(columns))) < 1e-6
    assert np.max(np.abs(rows - np.round(rows))) < 1e-6

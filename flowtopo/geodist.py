"""Geodesic helpers for a regular lat-lon raster.

Computes per-pixel arc lengths and areas without depending on GDAL.  ``transform`` is a six-element
GDAL GeoTransform: ``(x_origin, pixel_w, rot_x, y_origin, rot_y, pixel_h)``.
"""

import numpy as np

EARTH_RADIUS_M = 6371000.0
"""The sphere the released MERIT-FlowTopo and MERIT-DrainAttr areas are on."""

WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)

EARTH_MODELS = ("sphere", "wgs84")

GT_X_ORIGIN, GT_PIXEL_W, GT_ROT_X, GT_Y_ORIGIN, GT_ROT_Y, GT_PIXEL_H = range(6)


def degree_metres_y(lat):
    """Metres per degree of latitude (WGS-84 series expansion)."""
    rad = np.radians(lat)
    return (
        111132.92
        - 559.82 * np.cos(2.0 * rad)
        + 1.175 * np.cos(4.0 * rad)
        - 0.0023 * np.cos(6.0 * rad)
    )


def degree_metres_x(lat):
    """Metres per degree of longitude (WGS-84 series expansion)."""
    rad = np.radians(lat)
    return (
        111412.84 * np.cos(rad)
        - 93.5 * np.cos(3.0 * rad)
        + 0.118 * np.cos(5.0 * rad)
    )


def _wgs84_zone(lat_rad):
    """F(l) of the WGS84 zone integral, the C's wgs84_F."""
    e = np.sqrt(WGS84_E2)
    sl = np.sin(lat_rad)
    return sl / (1.0 - WGS84_E2 * sl * sl) + np.log((1.0 + e * sl) / (1.0 - e * sl)) / (2.0 * e)


def cell_area_m2(lat, xres, yres, model="sphere"):
    """Area of one lat-lon pixel, in square metres.

    ``model="sphere"`` is the sphere of radius 6 371 000 m the released products are on, and
    ``"wgs84"`` the exact zone of the WGS84 ellipsoid, which is what the C code computes by
    default.  The ellipsoid's cell is 0.45 per cent smaller at the equator and 0.56 per cent larger
    at 60 degrees: the difference changes sign with latitude.
    """
    if model not in EARTH_MODELS:
        raise ValueError(f"unknown earth model {model!r}; use one of {EARTH_MODELS}")
    l1 = np.radians(lat - abs(yres) / 2.0)
    l2 = np.radians(lat + abs(yres) / 2.0)
    dx = np.radians(abs(xres))
    if model == "sphere":
        return EARTH_RADIUS_M**2 * dx * (np.sin(l2) - np.sin(l1))
    return WGS84_A * WGS84_A * (1.0 - WGS84_E2) * dx * 0.5 * (_wgs84_zone(l2) - _wgs84_zone(l1))


def pixel_length(idxs_ds, ncol, transform, latlon=True):
    """Centre-to-centre distance from each cell to its receiver.

    Returns an array of float32 in metres (``latlon=True``) or in projected
    units.  A pit gets length 0.
    """
    # the downstream indices checked as every other entry checks them (a receiver 2 on
    # a grid of one cell gave a length of 2)
    from .core import _checked_downstream
    idxs_ds = _checked_downstream(idxs_ds).astype(np.int64)
    size = idxs_ds.size
    out = np.zeros(size, dtype=np.float32)

    idx = np.nonzero(idxs_ds >= 0)[0]
    ds = idxs_ds[idx]
    moving = ds != idx
    idx, ds = idx[moving], ds[moving]
    if idx.size == 0:
        return out

    xres = transform[GT_PIXEL_W]
    yres = transform[GT_PIXEL_H]
    north = transform[GT_Y_ORIGIN]

    r0, r1 = idx // ncol, ds // ncol
    dr = np.abs(r1 - r0)
    dc = np.abs((ds % ncol) - (idx % ncol))
    # on a geographic grid that spans the whole globe the step across
    # the antimeridian is the short way round, as the C code's distance() takes it
    # (1 column, not 359)
    if latlon and abs(ncol * xres) > 359.9:
        dc = np.where(dc > ncol // 2, ncol - dc, dc)

    if latlon:
        # Midpoint of the two row edges, not of the two cell centres: a cell
        # centre sits at north + (r + 0.5) * yres, so this latitude is half a
        # pixel north of the true midpoint. It is what the C implementation
        # that produced the released products does, and the two are kept
        # identical on purpose. The cost is 0.16 m over a 98 km path on the
        # bundled basin, 1.7e-6 relative, and it grows with the pixel: on a
        # one-degree grid at 60 N it is near a percent. See docs/methods.md.
        lat = north + (r0 + r1) / 2.0 * yres
        dy = np.where(dr == 0, 0.0, degree_metres_y(lat) * abs(yres))
        dx = np.where(dc == 0, 0.0, degree_metres_x(lat) * abs(xres))
    else:
        dy = np.full(idx.size, abs(yres))
        dx = np.full(idx.size, abs(xres))

    out[idx] = np.hypot(dy * dr, dx * dc).astype(np.float32)
    return out


def pixel_area_km2(nrow, ncol, transform, latlon=True, model="sphere"):
    """Per-cell area in square kilometres, broadcast over the grid.

    With ``latlon=False`` the transform is in metres and every cell has the same area; the spherical
    formula below would read those metres as degrees."""
    if not latlon:
        area_km2 = abs(transform[GT_PIXEL_W] * transform[GT_PIXEL_H]) * 1e-6
        return np.full(nrow * ncol, area_km2, dtype=np.float32)
    lats = transform[GT_Y_ORIGIN] + transform[GT_PIXEL_H] * (np.arange(nrow) + 0.5)
    # The two sides are read separately. They are equal on MERIT Hydro, three
    # arc-seconds each way, but a grid with taller pixels than wide is a valid
    # geographic grid and using one side for both halves or doubles the area.
    per_row = cell_area_m2(lats, abs(transform[GT_PIXEL_W]),
                           abs(transform[GT_PIXEL_H]), model) * 1e-6
    return np.repeat(per_row.astype(np.float32), ncol)

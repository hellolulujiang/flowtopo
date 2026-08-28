"""Geodesic helpers for a regular lat-lon raster.

Computes per-pixel arc lengths and areas without depending on GDAL.  ``transform`` is a six-element
GDAL GeoTransform: ``(x_origin, pixel_w, rot_x, y_origin, rot_y, pixel_h)``.
"""

import numpy as np

EARTH_RADIUS_M = 6371000.0

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


def cell_area_m2(lat, xres, yres):
    """Spherical area of one lat-lon pixel, in square metres."""
    l1 = np.radians(lat - abs(yres) / 2.0)
    l2 = np.radians(lat + abs(yres) / 2.0)
    dx = np.radians(abs(xres))
    return EARTH_RADIUS_M**2 * dx * (np.sin(l2) - np.sin(l1))


def pixel_length(idxs_ds, ncol, transform, latlon=True):
    """Centre-to-centre distance from each cell to its receiver.

    Returns an array of float32 in metres (``latlon=True``) or in projected
    units.  A pit gets length 0.
    """
    idxs_ds = np.ascontiguousarray(idxs_ds, dtype=np.int64)
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


def pixel_area_km2(nrow, ncol, transform):
    """Per-cell area in square kilometres, broadcast over the grid."""
    lats = transform[GT_Y_ORIGIN] + transform[GT_PIXEL_H] * (np.arange(nrow) + 0.5)
    # The two sides are read separately. They are equal on MERIT Hydro, three
    # arc-seconds each way, but a grid with taller pixels than wide is a valid
    # geographic grid and using one side for both halves or doubles the area.
    per_row = cell_area_m2(lats, abs(transform[GT_PIXEL_W]),
                           abs(transform[GT_PIXEL_H])) * 1e-6
    return np.repeat(per_row.astype(np.float32), ncol)

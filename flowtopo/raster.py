"""GeoTIFF I/O for D8 grids and kernel results, through rasterio."""

import os

import numpy as np


class GridHeader:
    """Shape, dtype, nodata, geotransform and CRS of a grid.

    ``latlon`` says whether the coordinates are degrees: the distance and area of a cell are worked
    out from the latitude when they are, and from the transform alone when they are metres."""

    __slots__ = ("ncol", "nrow", "dtype", "nodata", "transform", "crs", "latlon")

    def __init__(self, ncol, nrow, dtype="float32", nodata=0.0,
                 transform=(0.0, 1.0, 0.0, 0.0, 0.0, -1.0), crs="EPSG:4326", latlon=True):
        self.ncol = int(ncol)
        self.nrow = int(nrow)
        self.dtype = str(dtype)
        self.nodata = float(nodata)
        self.transform = tuple(float(v) for v in transform)
        self.crs = str(crs)
        self.latlon = bool(latlon)

    @property
    def shape(self):
        return (self.nrow, self.ncol)

    def __repr__(self):
        return (f"GridHeader(nrow={self.nrow}, ncol={self.ncol}, "
                f"dtype={self.dtype!r}, nodata={self.nodata})")


def _rasterio():
    try:
        import rasterio
    except ImportError as error:
        raise ImportError("GeoTIFF I/O needs rasterio: pip install rasterio") from error
    return rasterio


def read_geotiff(path):
    """Read a single-band GeoTIFF into a flat array and a :class:`GridHeader`."""
    rasterio = _rasterio()
    if not os.path.exists(path):
        raise FileNotFoundError(f"no such raster: {path}")
    try:
        src = rasterio.open(path)
    except Exception as error:
        raise ValueError(
            f"{path} could not be read as a raster: {error}"
        ) from error
    with src:
        data = src.read(1)
        # the cells the file's own mask marks invalid (an internal mask or alpha band, not only the
        # nodata value) become nodata; read(1) returns their stored values, and a 0 there was taken as a real outlet.
        # With no mask but the nodata value this changes nothing
        fill = src.nodata if src.nodata is not None else 247.0
        valid = src.read_masks(1)
        if not valid.all():
            # a type that cannot hold the fill (247 in Int8, with no nodata declared) is widened to one
            # that can, rather than the file refused
            if data.dtype.kind in "iu" and float(fill).is_integer():
                data = data.astype(np.promote_types(data.dtype, np.min_scalar_type(int(fill))))
            elif data.dtype.kind in "iu":
                data = data.astype(np.float64)
            else:
                data = data.copy()
            data[valid == 0] = fill
        t = src.transform
        # what this package can measure on: a north-up grid, in degrees or in metres.  A rotated grid
        # and a projection in another unit are refused rather than read as one of the two, since every
        # length and area below would come out in the wrong unit
        if t.b != 0.0 or t.d != 0.0:
            raise ValueError(f"{path} is rotated; this package reads north-up grids")
        # a geographic CRS is read as degrees only when its angular unit is the degree; one in grads or
        # radians was taken for degrees and every length and area came out wrong
        if src.crs is not None and src.crs.is_geographic:
            angular_unit_radians = src.crs.units_factor[1] if src.crs.units_factor else 0.0
            if abs(angular_unit_radians - np.pi / 180.0) > 1e-12:
                raise ValueError(
                    f"{path} is geographic in units of {src.crs.units_factor!r}; this package reads degrees, so the "
                    f"grid has to be given in degrees first"
                )
        if src.crs is not None and not src.crs.is_geographic:
            factor = src.crs.linear_units_factor[1] if src.crs.linear_units_factor else 0.0
            if abs(factor - 1.0) > 1e-9:
                raise ValueError(
                    f"{path} is projected in units of {factor:g} m ({src.crs.linear_units!r}); the "
                    f"lengths and areas here are metres, so the grid has to be reprojected first"
                )
        header = GridHeader(
            ncol=src.width, nrow=src.height, dtype=data.dtype.name,
            nodata=src.nodata if src.nodata is not None else 247.0,
            transform=(t.c, t.a, t.b, t.f, t.d, t.e),
            crs=str(src.crs) if src.crs else "",       # no CRS in, no CRS out
            # a projected grid is not degrees, and reading it as degrees turns a 100 m cell into
            # 100 degrees; with no CRS at all the file is read as degrees, which
            # is what every D8 product this package was written for carries
            latlon=bool(src.crs.is_geographic) if src.crs else True,
        )
    return data.ravel(), header


def write_geotiff(path, data, header):
    """Write a flat array as a single-band compressed GeoTIFF."""
    rasterio = _rasterio()
    from rasterio.transform import Affine

    x0, xw, xr, y0, yr, yh = header.transform
    grid = np.ascontiguousarray(data).reshape(header.shape)
    with rasterio.open(
        path, "w", driver="GTiff", width=header.ncol, height=header.nrow,
        count=1, dtype=grid.dtype.name, nodata=header.nodata,
        transform=Affine(xw, xr, x0, yr, yh, y0),
        crs=header.crs if header.crs else None,          # a grid with no CRS is written without one
        compress="deflate",
    ) as dst:
        dst.write(grid, 1)

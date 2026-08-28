"""GeoTIFF I/O for D8 grids and kernel results, through rasterio."""

import numpy as np


class GridHeader:
    """Shape, dtype, nodata and geotransform of a grid."""

    __slots__ = ("ncol", "nrow", "dtype", "nodata", "transform", "crs")

    def __init__(self, ncol, nrow, dtype="float32", nodata=0.0,
                 transform=(0.0, 1.0, 0.0, 0.0, 0.0, -1.0), crs="EPSG:4326"):
        self.ncol = int(ncol)
        self.nrow = int(nrow)
        self.dtype = str(dtype)
        self.nodata = float(nodata)
        self.transform = tuple(float(v) for v in transform)
        self.crs = str(crs)

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
    with rasterio.open(path) as src:
        data = src.read(1)
        t = src.transform
        header = GridHeader(
            ncol=src.width, nrow=src.height, dtype=data.dtype.name,
            nodata=src.nodata if src.nodata is not None else 247.0,
            transform=(t.c, t.a, t.b, t.f, t.d, t.e),
            crs=str(src.crs) if src.crs else "EPSG:4326",
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
        transform=Affine(xw, xr, x0, yr, yh, y0), crs=header.crs,
        compress="deflate",
    ) as dst:
        dst.write(grid, 1)

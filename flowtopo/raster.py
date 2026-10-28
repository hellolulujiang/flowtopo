"""GeoTIFF I/O for D8 grids and kernel results, through rasterio."""

import operator
import os

import numpy as np

from .core import dfs_interval


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
        # honour the file's mask (an internal mask or alpha band as well as the nodata value):
        # read(1) returns stored values, and a masked 0 must not become an outlet
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
        # a geographic CRS must use degrees; reading grads or radians as degrees would give wrong lengths and areas
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
                    f"lengths here are in metres, so the grid has to be reprojected first"
                )
        header = GridHeader(
            ncol=src.width, nrow=src.height, dtype=data.dtype.name,
            nodata=src.nodata if src.nodata is not None else 247.0,
            transform=(t.c, t.a, t.b, t.f, t.d, t.e),
            crs=str(src.crs) if src.crs else "",       # no CRS in, no CRS out
            # a projected grid is not degrees, and reading it as degrees turns a 100 m cell into
            # 100 degrees; a file with no CRS is read as degrees
            latlon=bool(src.crs.is_geographic) if src.crs else True,
        )
    return data.ravel(), header


def write_geotiff(path, data, header):
    """Write a flat array as a single-band compressed GeoTIFF.

    A file already at ``path`` is not written over: the call is refused (move the file
    away, or give another name).  The check is made once, before the write: two runs that
    write the same name at the same moment are not kept apart."""
    if os.path.lexists(path):
        raise FileExistsError(f"{path} exists and is not written over: move it away or give another name")
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


def _read_columns(src, row, nrow, col0, ncol):
    """Rows ``row`` to ``row + nrow - 1`` and ``ncol`` columns from ``col0`` on, the columns taken
    modulo the raster's width (the raster spans 360 degrees of longitude)."""
    from rasterio.windows import Window

    first = min(ncol, src.width - col0)
    data = src.read(1, window=Window(col0, row, first, nrow))
    if first < ncol:
        rest = src.read(1, window=Window(0, row, ncol - first, nrow))
        data = np.concatenate([data, rest], axis=1)
    return data


def _remove(paths):
    for path in paths:
        try:
            os.remove(path)
        except OSError:
            pass


def _place(parts, paths):
    """Rename the two finished files ``parts`` to ``paths``, both or neither."""
    # the target is looked at before each rename, which would write over it: an output
    # spelled otherwise may be the other output or the other's temporary file
    for path in paths:
        if os.path.lexists(path):
            raise FileExistsError(f"{path} appeared while the outputs were written")
    os.replace(parts[0], paths[0])
    try:
        if os.path.lexists(paths[1]):
            raise FileExistsError(f"{paths[0]} and {paths[1]} are the same file")
        os.replace(parts[1], paths[1])
    except BaseException:
        os.replace(paths[0], parts[0])
        raise


def dfs_interval_geotiff(seq_dfs_path, upg_path, dfs_in_path, dfs_out_path, rows=512):
    """Write the ``dfs_in`` and ``dfs_out`` rasters of a ``seq_dfs`` raster, the
    upstream cell counts read from ``upg_path`` (see :func:`flowtopo.core.dfs_interval`).

    One region is read and written: ``seq_dfs_path`` is a region of FullTopo v1.1,
    which must hold a whole sequence, and ``upg_path`` is MERIT Hydro's ``upg``
    cut to the same window.  A larger ``upg``, such as MERIT Hydro's ``upg`` tiles
    joined in a VRT (``gdalbuildvrt upg.vrt *_upg.tif``), is read as well: the
    window under ``seq_dfs`` is taken from it, across the antimeridian when it
    spans 360 degrees of longitude.  The two need the same cell size and cell
    edges that coincide.

    The rasters are read and written ``rows`` rows at a time, so a region is never
    held whole in memory; most of the memory used is GDAL's block cache, which
    ``GDAL_CACHEMAX`` limits.  Each output is a tiled GeoTIFF
    (512 x 512, ZSTD) on the grid of ``seq_dfs``, in its type, nodata ``-1``.
    The two are written as ``dfs_in_path + ".part"`` and ``dfs_out_path + ".part"``
    and renamed when both are complete; a run that fails removes them and leaves
    neither output, and a run killed before it ends can leave some of these files,
    to be removed before the next run.  A file or a link already at any of these
    four paths is not written over.  Two runs writing the same outputs at the same
    time are not kept apart.  Returns ``n``, the number of cells in the sequence.
    """
    rows = operator.index(rows)
    if rows < 1:
        raise ValueError(f"rows must be at least 1, not {rows}")
    paths = (dfs_in_path, dfs_out_path)
    parts = (f"{dfs_in_path}.part", f"{dfs_out_path}.part")
    if os.path.abspath(dfs_in_path) == os.path.abspath(dfs_out_path):
        raise ValueError(f"dfs_in and dfs_out are both {dfs_in_path}")
    for path in paths + parts:
        if os.path.lexists(path):
            raise FileExistsError(f"{path} exists and is not written over: move it away or give another name")
    rasterio = _rasterio()

    with rasterio.open(seq_dfs_path) as seq_src, rasterio.open(upg_path) as upg_src:
        st, ut = seq_src.transform, upg_src.transform
        if st.b or st.d or ut.b or ut.d:
            raise ValueError("seq_dfs and upg must be north-up grids")
        if not (np.isclose(st.a, ut.a, rtol=1e-9, atol=0) and np.isclose(st.e, ut.e, rtol=1e-9, atol=0)):
            raise ValueError(f"cell sizes differ: seq_dfs {st.a!r} x {st.e!r}, upg {ut.a!r} x {ut.e!r}")
        if seq_src.crs and upg_src.crs and seq_src.crs != upg_src.crs:
            raise ValueError(f"seq_dfs is in {seq_src.crs} and upg in {upg_src.crs}")
        col_at = (st.c - ut.c) / ut.a
        row_at = (st.f - ut.f) / ut.e
        col0, row0 = round(col_at), round(row_at)
        if abs(col_at - col0) > 1e-6 or abs(row_at - row0) > 1e-6:
            raise ValueError(
                f"the cell edges of seq_dfs fall {col_at:.6f} columns and {row_at:.6f} rows into upg, "
                f"not on whole cells"
            )
        ncol, nrow = seq_src.width, seq_src.height
        # a raster as wide as the globe continues past its east edge at its west edge
        wraps = abs(upg_src.width * abs(ut.a) - 360.0) < 0.5 * abs(ut.a)
        if wraps:
            col0 %= upg_src.width
        if row0 < 0 or row0 + nrow > upg_src.height or (
                not wraps and (col0 < 0 or col0 + ncol > upg_src.width)):
            raise ValueError("upg does not cover seq_dfs")

        # first pass: the number of cells in the sequence
        top, count = -1, 0
        for row in range(0, nrow, rows):
            seq = seq_src.read(1, window=((row, min(row + rows, nrow)), (0, ncol)))
            on = seq >= 0
            if on.any():
                top = max(top, int(seq.max()))
                count += int(np.count_nonzero(on))
        if count != top + 1:
            raise ValueError(
                f"{seq_dfs_path} holds {count} cells of the network and positions up to {top}: "
                f"it is not a whole sequence"
            )
        n = top + 1

        profile = dict(
            driver="GTiff", width=ncol, height=nrow, count=1, dtype=seq_src.dtypes[0],
            nodata=-1, transform=st, crs=seq_src.crs, tiled=True, blockxsize=512,
            blockysize=512, compress="zstd", predictor=2, BIGTIFF="IF_SAFER",
        )
        try:
            with rasterio.open(parts[0], "w", **profile) as in_dst, \
                    rasterio.open(parts[1], "w", **profile) as out_dst:
                for row in range(0, nrow, rows):
                    height = min(rows, nrow - row)
                    window = ((row, row + height), (0, ncol))
                    seq = seq_src.read(1, window=window)
                    upg = _read_columns(upg_src, row0 + row, height, col0, ncol)
                    first, last = dfs_interval(seq, upg, n)
                    in_dst.write(first, 1, window=window)
                    out_dst.write(last, 1, window=window)
            _place(parts, paths)
        except BaseException:
            _remove(parts)          # the temporary files, which were not there before
            raise
    return n

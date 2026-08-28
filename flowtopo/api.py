"""The :class:`FlowTopo` front end.

Wraps a D8 flow network and builds its reusable structures on demand,
caching each one.  Every kernel then takes an ordering or a layering
and a propagation manner, so the same network can be traversed six ways and
the answers compared.
"""

import warnings

import numpy as np

from . import core, geodist, kernels, locality, partition as _partition
from .layering import Decomposition, reverse_layers
from .raster import GridHeader, read_geotiff

def _frozen(array):
    """Mark a cached array read-only.

    These arrays are handed out on every call and kept inside the object. If a
    caller wrote to one, every later call would return the damaged array with
    no sign that anything had happened. Writing now raises instead; take a
    ``.copy()`` if you need to modify one.
    """
    array.flags.writeable = False
    return array


ORDERINGS = ("dfs", "bfs", "topo")
LAYERINGS = ("asap", "cfds", "alap")


class FlowTopo:
    """A D8 flow network and its topological representations.

    Parameters
    ----------
    idxs_ds : ndarray of int32
        Downstream linear index per cell; ``-1`` outside the network, and a
        pit points at itself.
    shape : tuple of int
        ``(nrow, ncol)``.
    transform : tuple of float, optional
        Six-element GDAL GeoTransform.  Needed for the distance kernels.
    latlon : bool
        Treat the transform as geographic degrees.
    mask : ndarray of bool, optional
        Valid cells.  Defaults to ``idxs_ds >= 0``.

    Examples
    --------
    >>> topo = FlowTopo.from_d8(d8, transform=transform)     # doctest: +SKIP
    >>> upa = topo.upstream_area(layering="cfds", manner="push")   # doctest: +SKIP
    """

    def __init__(self, idxs_ds, shape, transform=None, latlon=True, mask=None):
        self.idxs_ds = _frozen(np.ascontiguousarray(idxs_ds, dtype=np.int32))
        self.shape = tuple(int(v) for v in shape)
        self.nrow, self.ncol = self.shape
        if self.idxs_ds.size != self.nrow * self.ncol:
            raise ValueError("idxs_ds does not match shape")
        self.transform = tuple(transform) if transform is not None else None
        self.latlon = bool(latlon)
        self.mask = _frozen(
            np.ascontiguousarray(mask, dtype=bool)
            if mask is not None
            else self.idxs_ds >= 0
        )
        self._cache = {}

    # -- constructors ------------------------------------------------------

    @classmethod
    def from_d8(cls, d8, shape=None, transform=None, latlon=True, nodata=None):
        """Build from a D8 flow-direction raster.

        Takes the same array ``pyflwdir.from_array(d8, ftype="d8")`` takes:
        powers of two clockwise from east, 0 and 255 terminal, 247 nodata.
        A 2-D array is flattened; a flat one needs ``shape``.

        ``nodata`` names a second nodata code, for a grid that does not use
        247. :meth:`from_raster` fills it in from the file.
        """
        d8 = np.asarray(d8)
        if d8.ndim == 2:
            shape = d8.shape
            d8 = d8.ravel()
        elif shape is None:
            raise ValueError("a flat d8 array needs shape=(nrow, ncol)")
        nrow, ncol = shape

        strange = core.unknown_codes(d8, nodata=nodata)
        if strange:
            total = sum(strange.values())
            worst = sorted(strange.items(), key=lambda kv: -kv[1])[:5]
            listed = ", ".join(f"{code} ({count:,} cells)" for code, count in worst)
            warnings.warn(
                f"{total:,} cells carry a code this D8 convention does not "
                f"define: {listed}. Codes here are powers of two clockwise "
                f"from east, with 0 and 255 terminal and 247 nodata. TauDEM "
                f"and GRASS number their directions 1 to 8 instead, and read "
                f"that way every undefined cell becomes a pit. Reclassify the "
                f"grid, or pass nodata= if this is the grid's nodata value.",
                RuntimeWarning,
                stacklevel=2,
            )

        idxs_ds = core.d8_to_downstream(d8, nrow, ncol, nodata=nodata)
        obj = cls(idxs_ds, shape, transform=transform, latlon=latlon,
                  mask=idxs_ds >= 0)
        obj._cache["d8"] = np.ascontiguousarray(d8, dtype=np.uint8)
        return obj

    @classmethod
    def from_raster(cls, path):
        """Build from a single-band D8 GeoTIFF."""
        data, header = read_geotiff(path)
        return cls.from_d8(data, shape=header.shape,
                           transform=header.transform, latlon=True,
                           nodata=header.nodata)

    # -- geometry ----------------------------------------------------------

    def _require_transform(self):
        if self.transform is None:
            raise ValueError("this operation needs a geotransform")
        return self.transform

    @property
    def pixel_length(self):
        """Centre-to-centre distance from each cell to its receiver, in metres."""
        if "plen" not in self._cache:
            self._cache["plen"] = _frozen(geodist.pixel_length(
                self.idxs_ds, self.ncol, self._require_transform(), self.latlon
            ))
        return self._cache["plen"]

    @property
    def cell_area(self):
        """Per-cell area in square kilometres, zero outside the network."""
        if "area" not in self._cache:
            area = geodist.pixel_area_km2(self.nrow, self.ncol,
                                          self._require_transform())
            area[~self.mask] = 0.0
            self._cache["area"] = _frozen(area)
        return self._cache["area"]

    @property
    def basins(self):
        """One-based basin id per cell, ``0`` outside the network."""
        if "basins" not in self._cache:
            labels, count = core.basin_labels(self.idxs_ds, self.ordering("dfs"))
            self._cache["basins"] = _frozen(labels)
            self._cache["nbasins"] = count
        return self._cache["basins"]

    @property
    def nbasins(self):
        """Number of basins, that is, of cells that drain to themselves."""
        self.basins
        return self._cache["nbasins"]

    @property
    def ncells(self):
        """Number of valid cells, those inside the network."""
        return int(np.count_nonzero(self.mask))

    # -- orderings ---------------------------------------------------------

    def ordering(self, name="dfs", direction="d2u"):
        """One of the three serial orderings.

        ``name`` is ``"dfs"``, ``"bfs"`` or ``"topo"``.  ``direction`` is
        ``"d2u"`` (position 0 is a pit) or ``"u2d"`` (position 0 is a
        headwater); the two are reverses of each other.
        """
        if name not in ORDERINGS:
            raise ValueError(f"unknown ordering {name!r}; use one of {ORDERINGS}")
        if direction not in ("d2u", "u2d"):
            raise ValueError("direction must be 'd2u' or 'u2d'")

        key = f"seq_{name}"
        if key not in self._cache:
            if name == "dfs":
                seq = core.seq_dfs_from_pit(self.idxs_ds)
                native = "d2u"
            elif name == "bfs":
                seq = core.seq_bfs_from_pit(self.idxs_ds)
                native = "d2u"
            else:
                seq = core.seq_topo_from_source(self.idxs_ds)
                native = "u2d"
            self._cache[key] = (_frozen(seq), native)

        seq, native = self._cache[key]
        return seq if direction == native else seq[::-1]

    # -- layerings ---------------------------------------------------------

    def layering(self, name="cfds"):
        """One of the three layerings, as ``(layers, nlayers)``.

        ``name`` is ``"asap"``, ``"cfds"`` or ``"alap"``.  ``layers[i]`` is the
        layer index of cell ``i``, ``-1`` outside the network.
        """
        if name not in LAYERINGS:
            raise ValueError(f"unknown layering {name!r}; use one of {LAYERINGS}")

        key = f"lyr_{name}"
        if key not in self._cache:
            if name == "asap":
                layers, nlayers = core.layering_asap(self.idxs_ds)
            elif name == "cfds":
                layers, nlayers = core.layering_cfds(self.idxs_ds)
            else:
                layers, nlayers = core.layering_alap(self.idxs_ds, self.basins)
            layers = layers.copy()
            layers[~self.mask] = core.LAYER_NODATA
            self._cache[key] = (_frozen(layers), nlayers)
        return self._cache[key]

    def decomposition(self, name="cfds", direction="u2d"):
        """Per-layer member lists of a layering.

        ``direction="d2u"`` flips the layering so receivers come before their
        donors, which the downstream-propagation kernel needs.
        """
        key = f"dd_{name}_{direction}"
        if key not in self._cache:
            layers, nlayers = self.layering(name)
            if direction == "d2u":
                layers = reverse_layers(layers, self.mask)
                nlayers = None
            self._cache[key] = Decomposition.from_layers(layers, nlayers)
        return self._cache[key]

    # -- upstream adjacency ------------------------------------------------

    def upstream(self, mask=None):
        """``(us_table, n_up)``: the upstream adjacency the pull manner reads."""
        key = "us" if mask is None else "us_masked"
        if key not in self._cache or mask is not None:
            msk = None if mask is None else np.asarray(mask, dtype=np.uint8)
            self._cache[key] = core.upstream_table(self.idxs_ds, msk)
        return self._cache[key]

    # -- kernels -----------------------------------------------------------

    def _traversal(self, ordering, layering, direction):
        if (ordering is None) == (layering is None):
            raise ValueError("give exactly one of ordering= or layering=")
        if ordering is not None:
            return self.ordering(ordering, direction), None
        return None, self.decomposition(layering, direction)

    def upstream_area(self, ordering=None, layering=None, manner=None,
                      cell_area=None):
        """Upstream drainage area in square kilometres.

        Give either ``ordering=`` (serial) or ``layering=`` (parallel).  For a
        layering, ``manner`` is ``"pull"``, ``"push"`` or ``"atomic_push"``;
        plain ``"push"`` is only correct under ``layering="cfds"``.
        """
        seq, decomp = self._traversal(ordering, layering, "u2d")
        if manner is None:
            # push is only safe under the conflict-free layering
            manner = "serial" if seq is not None else (
                "push" if layering == "cfds" else "atomic_push")
        us_table, n_up = self.upstream() if manner == "pull" else (None, None)
        area = self.cell_area if cell_area is None else cell_area
        return kernels.upstream_area(
            self.idxs_ds, area, seq_u2d=seq, decomp=decomp,
            us_table=us_table, n_up=n_up, manner=manner, mask=self.mask,
        )

    def distance_to_outlet(self, ordering=None, layering=None, manner=None):
        """Along-stream distance from each cell down to its pit, in metres."""
        seq, decomp = self._traversal(ordering, layering, "d2u")
        manner = manner or ("serial" if seq is not None else "push")
        return kernels.distance_to_outlet(
            self.idxs_ds, self.pixel_length, seq_d2u=seq, decomp=decomp,
            manner=manner, mask=self.mask,
        )

    def longest_upstream_path(self, ldn, ordering=None, layering=None,
                              manner=None):
        """Distance up to the furthest headwater above each cell, in metres."""
        seq, decomp = self._traversal(ordering, layering, "u2d")
        if manner is None:
            manner = "serial" if seq is not None else (
                "push" if layering == "cfds" else "atomic_push")
        us_table, n_up = self.upstream() if manner == "pull" else (None, None)
        return kernels.longest_upstream_path(
            self.idxs_ds, ldn, seq_u2d=seq, decomp=decomp, us_table=us_table,
            n_up=n_up, manner=manner, mask=self.mask,
        )

    def strahler_order(self, ordering=None, layering=None, manner=None,
                       channel_mask=None):
        """Strahler stream order over the channel network.

        ``manner="push"`` is correct only under ``layering="cfds"``: the
        confluence rule is a comparison and a count, so no atomic can make the
        other layerings safe.
        """
        seq, decomp = self._traversal(ordering, layering, "u2d")
        if manner is None:
            manner = "serial" if seq is not None else (
                "push" if layering == "cfds" else "pull")
        us_table = n_up = None
        if manner == "pull":
            us_table, n_up = core.upstream_table(
                self.idxs_ds,
                None if channel_mask is None
                else np.asarray(channel_mask, dtype=np.uint8),
            )
        return kernels.strahler_order(
            self.idxs_ds, seq_u2d=seq, decomp=decomp, us_table=us_table,
            n_up=n_up, manner=manner, mask=channel_mask,
        )

    def channel_mask(self, upa, threshold_km2=10.0):
        """Cells whose drainage area reaches ``threshold_km2``."""
        return self.mask & (np.asarray(upa) >= np.float32(threshold_km2))

    # -- partitions --------------------------------------------------------

    def partition(self, n_parts=4, level="subbasin"):
        """Split the network into ``n_parts`` independent subregions.

        ``level="basin"`` keeps every basin whole; ``level="subbasin"``
        decomposes an oversized basin along its mainstem. Returns
        ``(part, load)``; see :func:`flowtopo.partition.partition`.
        """
        return _partition.partition(self, n_parts=n_parts, level=level)

    # -- locality ----------------------------------------------------------

    def serial_locality(self, ordering="dfs", levels=("L1", "L2", "L3")):
        """Memory-locality metrics for one serial ordering."""
        return locality.serial_locality(
            self.ordering(ordering, "d2u"), self.idxs_ds, self.ncol,
            levels=levels,
        )

    def parallel_locality(self, layering="cfds", levels=("L1", "L2", "L3")):
        """Memory-locality metrics for one layering."""
        layers, _ = self.layering(layering)
        return locality.parallel_locality(
            layers, self.idxs_ds, self.mask, self.ncol, levels=levels,
        )

    # -- output ------------------------------------------------------------

    def header(self, dtype="float32", nodata=-9999.0):
        """A :class:`~flowtopo.raster.GridHeader` matching this grid."""
        return GridHeader(
            ncol=self.ncol, nrow=self.nrow, dtype=dtype, nodata=nodata,
            transform=self.transform or (0.0, 1.0, 0.0, 0.0, 0.0, -1.0),
        )

    def to_2d(self, flat):
        """Reshape a per-cell array back to the grid."""
        return np.asarray(flat).reshape(self.shape)

    def __repr__(self):
        return (
            f"FlowTopo(shape={self.shape}, ncells={self.ncells}, "
            f"nbasins={self.nbasins})"
        )

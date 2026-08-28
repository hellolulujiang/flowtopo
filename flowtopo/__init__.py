"""FlowTopo -- orderings, layerings and kernels for D8 flow networks.

A D8 flow network can be traversed in more than one way, and the choice
changes both how fast the traversal runs and whether a parallel form is
correct at all.  This package builds six representations of the same network

* three serial orderings: depth-first from the pits, breadth-first from the
  pits, and a topological sort from the sources;
* three parallel layerings: as-soon-as-possible, conflict-free downstream, and
  as-late-as-possible,

then runs four kernels over them -- upstream drainage area, distance to outlet,
longest upstream path and Strahler stream order -- under every propagation
manner each kernel supports, so the six can be compared on the same input.

The conflict-free downstream layering is the piece worth a second look: it
guarantees that no two cells in one layer share a receiver.  A scatter over a
layer therefore touches each target exactly once, which makes a plain
``a[idx] += v`` correct where it would otherwise silently drop writes, and
makes Strahler order runnable under push at all.

Quick start
-----------
>>> import flowtopo                                          # doctest: +SKIP
>>> topo = flowtopo.FlowTopo.from_d8(d8, transform=transform)  # doctest: +SKIP
>>> upa = topo.upstream_area(layering="cfds", manner="push")   # doctest: +SKIP

``from_d8`` takes the same array ``pyflwdir.from_array(d8, ftype="d8")`` takes.
"""

from ._compat import HAS_NUMBA
from .api import LAYERINGS, ORDERINGS, FlowTopo
from .core import (
    D8_NODATA,
    LAYER_NODATA,
    basin_labels,
    d8_to_downstream,
    layering_alap,
    layering_asap,
    layering_cfds,
    rank_to_pit,
    seq_bfs_from_pit,
    seq_dfs_from_pit,
    seq_topo_from_source,
    upstream_count,
    upstream_table,
)
from .kernels import (
    MANNERS,
    distance_to_outlet,
    longest_upstream_path,
    strahler_order,
    upstream_area,
)
from .layering import Decomposition, reverse_layers
from .locality import parallel_locality, serial_locality
from .raster import GridHeader, read_geotiff, write_geotiff

__version__ = "0.1.0"

__all__ = [
    "FlowTopo",
    "ORDERINGS",
    "LAYERINGS",
    "MANNERS",
    "Decomposition",
    "GridHeader",
    "HAS_NUMBA",
    "D8_NODATA",
    "LAYER_NODATA",
    "__version__",
    "basin_labels",
    "d8_to_downstream",
    "distance_to_outlet",
    "layering_alap",
    "layering_asap",
    "layering_cfds",
    "longest_upstream_path",
    "parallel_locality",
    "rank_to_pit",
    "read_geotiff",
    "read_geotiff",
    "reverse_layers",
    "seq_bfs_from_pit",
    "seq_dfs_from_pit",
    "seq_topo_from_source",
    "serial_locality",
    "strahler_order",
    "upstream_area",
    "upstream_count",
    "upstream_table",
    "write_geotiff",
]

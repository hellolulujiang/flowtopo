"""Turn a per-cell layer array into a traversable decomposition.

A layering as produced by :mod:`flowtopo.core` is one layer index per cell.
The kernels need the opposite view: for each layer, the list of its cells.
:class:`Decomposition` holds that view as one flat array of cell indices plus
the offsets that cut it into layers, so a layer is a zero-copy slice.

Cells inside one layer are kept in ascending index order.  That is the order a
static loop schedule would walk them in, and the locality metrics depend on it.
"""

import numpy as np


class Decomposition:
    """Per-layer member lists of a layering.

    Attributes
    ----------
    cells : ndarray of int32
        Every valid cell, grouped by layer, ascending index within a layer.
    offsets : ndarray of int64, shape (nlayers + 1,)
        ``cells[offsets[L]:offsets[L + 1]]`` are the members of layer ``L``.
    nlayers : int
    ncells : int
    """

    __slots__ = ("cells", "offsets", "nlayers", "ncells")

    def __init__(self, cells, offsets):
        self.cells = cells
        self.offsets = offsets
        self.nlayers = int(offsets.size - 1)
        self.ncells = int(cells.size)

    @classmethod
    def from_layers(cls, layers, nlayers=None):
        """Build from a per-cell layer array; negative layers are skipped."""
        layers = np.ascontiguousarray(layers, dtype=np.int64)
        cells = np.nonzero(layers >= 0)[0].astype(np.int32)
        lyr = layers[cells]
        if nlayers is None:
            nlayers = int(lyr.max()) + 1 if lyr.size else 0

        # Stable sort by layer; cells were already in ascending index order,
        # so each layer comes out ascending too.
        order = np.argsort(lyr, kind="stable")
        cells = np.ascontiguousarray(cells[order])

        sizes = np.bincount(lyr, minlength=nlayers)
        offsets = np.zeros(nlayers + 1, dtype=np.int64)
        np.cumsum(sizes, out=offsets[1:])
        return cls(cells, offsets)

    def layer(self, index):
        """Cell indices of one layer, as a view."""
        return self.cells[self.offsets[index] : self.offsets[index + 1]]

    def __iter__(self):
        for layer_index in range(self.nlayers):
            yield self.layer(layer_index)

    def __len__(self):
        return self.nlayers

    @property
    def layer_sizes(self):
        return np.diff(self.offsets)

    def flattened(self):
        """The layers concatenated into one visitation order."""
        return self.cells

    def __repr__(self):
        return (
            f"Decomposition(nlayers={self.nlayers}, ncells={self.ncells}, "
            f"largest_layer={int(self.layer_sizes.max()) if self.nlayers else 0})"
        )


def reverse_layers(layers, msk):
    """Flip a u2d layering into a d2u one: ``layer = max_layer - layer``.

    Used by the downstream-propagation kernel, which has to visit a receiver
    before its donors instead of after.
    """
    layers = np.ascontiguousarray(layers, dtype=np.int64)
    msk = np.ascontiguousarray(msk, dtype=bool)
    lyr_max = int(layers[msk].max()) if np.any(msk) else -1
    out = np.full(layers.size, -1, dtype=np.int32)
    out[msk] = (lyr_max - layers[msk]).astype(np.int32)
    return out

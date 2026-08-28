"""Optional numba acceleration.

The graph builders in :mod:`flowtopo.core` are scalar loops over the flow
network.  They are correct in pure Python and fast under numba.  Import numba
if it is there, otherwise install no-op decorators so the same source runs
either way.  The kernels in :mod:`flowtopo.kernels` do not go through here:
they are vectorised over a whole layer with numpy and need no jit.
"""

try:  # pragma: no cover - trivial
    from numba import njit, prange

    HAS_NUMBA = True
except ImportError:  # pragma: no cover - trivial
    HAS_NUMBA = False

    def njit(*args, **kwargs):
        """Stand-in for numba.njit that returns the function unchanged."""
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]

        def decorator(func):
            return func

        return decorator

    prange = range


__all__ = ["njit", "prange", "HAS_NUMBA"]

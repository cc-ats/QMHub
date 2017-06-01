try:
    import numba
    from .ewald_numba import EwaldNumba as EwaldSum
except ImportError:
    from .ewald_numpy import EwaldNumPy as EwaldSum

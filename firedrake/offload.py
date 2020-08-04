from contextlib import contextmanager


def set_offloading_backend(backend):
    """
    Must be called before performing any data allocation firedrake operations.
    To mark any operations for offloading the operations should be wrapped
    within :func:`~firedrake.offload.offloading` context.

    :arg backend: An instance of :class:`pyop2.backend.AbstractComputeBackend`.
    """
    from pyop2 import op2
    from pyop2.backend import AbstractComputeBackend
    assert isinstance(backend, AbstractComputeBackend)
    op2.compute_backend = backend


@contextmanager
def offloading():
    """
    Operations (for ex. assemble, interpolation, etc) within the offloading
    region will be executed on backend as previously set by
    :func:`~firedrake.offload.set_offloading_backend`.
    """
    from pyop2 import op2
    op2.compute_backend.turn_on_offloading()
    yield
    op2.compute_backend.turn_off_offloading()
    return

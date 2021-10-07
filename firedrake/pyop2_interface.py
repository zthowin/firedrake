import abc
import dataclasses as dc

from tsfc.kernel_interface import firedrake_loopy as tsfc_utils
from pyop2 import op2

from firedrake import tsfc_interface


@dc.dataclass(frozen=True)
class LocalKernel:

    tsfc_kernel: tsfc_interface.SplitKernel


class LocalKernelBuilder:

    def __init__(self, expr):
        self.expr = expr

    @abc.abstractmethod
    def build(self):
        ...


def as_pyop2_local_kernel(ast, name, arguments, access=op2.INC, **kwargs):
    """TODO"""
    access_map = {tsfc_utils.Intent.IN: op2.READ, tsfc_utils.Intent.OUT: access}
    kernel_args = [
        op2.LocalKernelArg(access_map[arg.intent], arg.dtype)
        for arg in arguments
    ]
    return op2.Kernel(
        ast,
        name,
        kernel_args,
        requires_zeroed_output_arguments=True,
        **kwargs
    )

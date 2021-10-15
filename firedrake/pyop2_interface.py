import abc
from dataclasses import dataclass

from tsfc import kernel_args
from pyop2 import op2

from firedrake import tsfc_interface


@dataclass(frozen=True)
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
    access_map = {kernel_args.Intent.IN: op2.READ, kernel_args.Intent.OUT: access}
    knl_args = [op2.LocalKernelArg(access_map[arg.intent], arg.dtype)
                for arg in arguments]
    return op2.Kernel(ast, name, knl_args, requires_zeroed_output_arguments=True, **kwargs)

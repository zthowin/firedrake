from tsfc import kernel_args
from pyop2 import op2


def as_pyop2_local_kernel(ast, name, arguments, access=op2.INC, **kwargs):
    """TODO"""
    access_map = {kernel_args.Intent.IN: op2.READ, kernel_args.Intent.OUT: access}
    knl_args = [op2.LocalKernelArg(access_map[arg.intent], arg.dtype)
                for arg in arguments if arg.intent is not None]
    return op2.Kernel(ast, name, knl_args, requires_zeroed_output_arguments=True, **kwargs)

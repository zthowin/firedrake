"""
Provides the interface to TSFC for compiling a form, and
transforms the TSFC-generated code to make it suitable for
passing to the backends.

"""
import pickle

from hashlib import md5
from os import path, environ, getuid, makedirs
import functools
import os
import tempfile
import collections

import ufl
from ufl import Form, conj
from firedrake.constant import Constant
from firedrake.function import Function
from .ufl_expr import TestFunction

from tsfc import compile_form as tsfc_compile_form
from tsfc.parameters import PARAMETERS as tsfc_default_parameters
import tsfc.kernel_interface.firedrake_loopy as tsfc_utils

from pyop2.caching import Cached
from pyop2 import op2
from pyop2.mpi import COMM_WORLD, MPI

from firedrake.formmanipulation import split_form
from firedrake.parameters import parameters as default_parameters
from firedrake.petsc import PETSc
from firedrake import utils

# Set TSFC default scalar type at load time
tsfc_default_parameters["scalar_type"] = utils.ScalarType
tsfc_default_parameters["scalar_type_c"] = utils.ScalarType_c


KernelInfo = collections.namedtuple("KernelInfo",
                                    ["kernel",
                                     "integral_type",
                                     "oriented",
                                     "subdomain_id",
                                     "domain_number",
                                     "coefficient_map",
                                     "needs_cell_facets",
                                     "pass_layer_arg",
                                     "needs_cell_sizes",
                                     "tsfc_kernel_args"])


class TSFCKernel(Cached):

    _cache = {}

    _cachedir = environ.get('FIREDRAKE_TSFC_KERNEL_CACHE_DIR',
                            path.join(tempfile.gettempdir(),
                                      'firedrake-tsfc-kernel-cache-uid%d' % getuid()))

    @classmethod
    def _cache_lookup(cls, key):
        key, comm = key
        # comm has to be part of the in memory key so that when
        # compiling the same code on different subcommunicators we
        # don't get deadlocks. But MPI_Comm objects are not hashable,
        # so use comm.py2f() since this is an internal communicator and
        # hence the C handle is stable.
        commkey = comm.py2f()
        assert commkey != MPI.COMM_NULL.py2f()
        return cls._cache.get((key, commkey)) or cls._read_from_disk(key, comm)

    @classmethod
    def _read_from_disk(cls, key, comm):
        if comm.rank == 0:
            filepath = os.path.join(cls._cachedir, key[:2], key[2:])
            try:
                with open(filepath, "rb") as f:
                    val = pickle.load(f)
            except FileNotFoundError:
                raise KeyError(f"Object with key {key} not found")
            comm.bcast(val, root=0)
        else:
            val = comm.bcast(None, root=0)

        return cls._cache.setdefault((key, comm.py2f()), val)

    @classmethod
    def _cache_store(cls, key, val):
        key, comm = key
        cls._cache[(key, comm.py2f())] = val
        _ensure_cachedir(comm=comm)
        if comm.rank == 0:
            val._key = key
            shard, disk_key = key[:2], key[2:]
            filepath = os.path.join(cls._cachedir, shard, disk_key)
            tempfile = os.path.join(cls._cachedir, shard, "%s_p%d.tmp" % (disk_key, os.getpid()))
            # No need for a barrier after this, since non root
            # processes will never race on this file.
            os.makedirs(os.path.join(cls._cachedir, shard), exist_ok=True)
            with open(tempfile, "wb") as f:
                pickle.dump(val, f)
            os.rename(tempfile, filepath)
        comm.barrier()

    @classmethod
    def _cache_key(cls, form, name, parameters, number_map, interface, coffee=False, diagonal=False):
        # FIXME Making the COFFEE parameters part of the cache key causes
        # unnecessary repeated calls to TSFC when actually only the kernel code
        # needs to be regenerated
        return md5((form.signature() + name
                    + str(sorted(default_parameters["coffee"].items()))
                    + str(sorted(parameters.items()))
                    + str(number_map)
                    + str(type(interface))
                    + str(coffee)
                    + str(diagonal)).encode()).hexdigest(), form.ufl_domains()[0].comm

    def __init__(self, form, name, parameters, number_map, interface, coffee=False, diagonal=False):
        """A wrapper object for one or more TSFC kernels compiled from a given :class:`~ufl.classes.Form`.

        :arg form: the :class:`~ufl.classes.Form` from which to compile the kernels.
        :arg name: a prefix to be applied to the compiled kernel names. This is primarily useful for debugging.
        :arg parameters: a dict of parameters to pass to the form compiler.
        :arg number_map: a map from local coefficient numbers to global ones (useful for split forms).
        :arg interface: the KernelBuilder interface for TSFC (may be None)
        """
        if self._initialized:
            return
        tree = tsfc_compile_form(form, prefix=name, parameters=parameters, interface=interface, coffee=coffee, diagonal=diagonal)
        kernels = []
        for kernel in tree:
            # Set optimization options
            opts = default_parameters["coffee"]
            ast = kernel.ast
            # Unwind coefficient numbering
            numbers = tuple(number_map[c] for c in kernel.coefficient_numbers)
            kernels.append(KernelInfo(kernel=as_pyop2_local_kernel(kernel, opts),
                                      integral_type=kernel.integral_type,
                                      oriented=kernel.oriented,
                                      subdomain_id=kernel.subdomain_id,
                                      domain_number=kernel.domain_number,
                                      coefficient_map=numbers,
                                      needs_cell_facets=False,
                                      pass_layer_arg=False,
                                      needs_cell_sizes=kernel.needs_cell_sizes,
                                      tsfc_kernel_args=kernel.arguments))
        self.kernels = tuple(kernels)
        self._initialized = True


SplitKernel = collections.namedtuple("SplitKernel", ["indices",
                                                     "kinfo"])


@PETSc.Log.EventDecorator()
def compile_form(form, name, parameters=None, split=True, interface=None, coffee=False, diagonal=False):
    """Compile a form using TSFC.

    :arg form: the :class:`~ufl.classes.Form` to compile.
    :arg name: a prefix for the generated kernel functions.
    :arg parameters: optional dict of parameters to pass to the form
         compiler. If not provided, parameters are read from the
         ``form_compiler`` slot of the Firedrake
         :data:`~.parameters` dictionary (which see).
    :arg split: If ``False``, then don't split mixed forms.
    :arg coffee: compile coffee kernel instead of loopy kernel

    Returns a tuple of tuples of
    (index, integral type, subdomain id, coordinates, coefficients, needs_orientations, :class:`Kernels <pyop2.op2.Kernel>`).

    ``needs_orientations`` indicates whether the form requires cell
    orientation information (for correctly pulling back to reference
    elements on embedded manifolds).

    The coordinates are extracted from the domain of the integral (a
    :func:`~.Mesh`)

    """

    # Check that we get a Form
    if not isinstance(form, Form):
        raise RuntimeError("Unable to convert object to a UFL form: %s" % repr(form))

    if parameters is None:
        parameters = default_parameters["form_compiler"].copy()
    else:
        # Override defaults with user-specified values
        _ = parameters
        parameters = default_parameters["form_compiler"].copy()
        parameters.update(_)

    # We stash the compiled kernels on the form so we don't have to recompile
    # if we assemble the same form again with the same optimisations
    cache = form._cache.setdefault("firedrake_kernels", {})

    def tuplify(params):
        return tuple((k, params[k]) for k in sorted(params))

    key = (tuplify(default_parameters["coffee"]), name, tuplify(parameters), split, diagonal)
    try:
        return cache[key]
    except KeyError:
        pass

    kernels = []
    # A map from all form coefficients to their number.
    coefficient_numbers = dict((c, n)
                               for (n, c) in enumerate(form.coefficients()))
    if split:
        iterable = split_form(form, diagonal=diagonal)
    else:
        nargs = len(form.arguments())
        if diagonal:
            assert nargs == 2
            nargs = 1
        iterable = ([(None, )*nargs, form], )
    for idx, f in iterable:
        f = _real_mangle(f)
        # Map local coefficient numbers (as seen inside the
        # compiler) to the split global coefficient numbers
        number_map = dict((n, (coefficient_numbers[c], tuple(range(len(c.split())))))
                          if isinstance(c, Function) or isinstance(c, Constant)
                          else (n, (coefficient_numbers[c], (0,)))
                          for (n, c) in enumerate(f.coefficients()))

        prefix = name + "".join(map(str, (i for i in idx if i is not None)))
        kinfos = TSFCKernel(f, prefix, parameters,
                            number_map, interface, coffee, diagonal).kernels
        for kinfo in kinfos:
            kernels.append(SplitKernel(idx, kinfo))

    kernels = tuple(kernels)
    return cache.setdefault(key, kernels)


def _real_mangle(form):
    """If the form contains arguments in the Real function space, replace these with literal 1 before passing to tsfc."""

    a = form.arguments()
    reals = [x.ufl_element().family() == "Real" for x in a]
    if not any(reals):
        return form
    replacements = {}
    for arg, r in zip(a, reals):
        if r:
            replacements[arg] = 1
    # If only the test space is Real, we need to turn the trial function into a test function.
    if reals == [True, False]:
        replacements[a[1]] = conj(TestFunction(a[1].function_space()))
    return ufl.replace(form, replacements)


def clear_cache(comm=None):
    """Clear the Firedrake TSFC kernel cache."""
    comm = comm or COMM_WORLD
    if comm.rank == 0:
        import shutil
        shutil.rmtree(TSFCKernel._cachedir, ignore_errors=True)
        _ensure_cachedir(comm=comm)


def _ensure_cachedir(comm=None):
    """Ensure that the TSFC kernel cache directory exists."""
    comm = comm or COMM_WORLD
    if comm.rank == 0:
        makedirs(TSFCKernel._cachedir, exist_ok=True)


def as_pyop2_local_kernel(tsfc_kernel, opts):
    """TODO"""
    def get_access(intent):
        return {
            tsfc_utils.Intent.IN: op2.READ,
            tsfc_utils.Intent.OUT: op2.INC
        }.get(intent)

    kernel_args = [
        op2.LocalKernelArg(get_access(arg.intent), arg.dtype)
        for arg in tsfc_kernel.arguments
    ]
    return op2.Kernel(
        tsfc_kernel.ast,
        tsfc_kernel.name,
        kernel_args,
        opts=opts,
        requires_zeroed_output_arguments=True
    )


class WrapperKernelBuilder:
    """A helper class for building :class:`pyop2.WrapperKernel` objects."""

    def __init__(self, local_kernel, *, unroll=None):
        """Create a :class:`WrapperKernelBuilder` object.

        :arg local_kernel:
            A :class:`KernelInfo` object resulting from the compilation of a form.
        :kwarg unroll:
            :class:`bool` indicating whether or not the maps for accessing the output
            tensor address DoFs (``True``) or nodes (``False``). Value is ``None`` if
            kernel does not output a matrix.
        """
        self._local_kernel = local_kernel
        self._unroll = unroll

    def build(self):
        """Build the wrapper kernel.

        :returns:
            A :class:`pyop2.WrapperKernel`.
        """
        wrapper_kernel_args = [
            self._as_pyop2_arg(arg) for arg in self._local_kernel.tsfc_kernel_args
        ]

        iteration_region = {
            "exterior_facet_top": op2.ON_TOP,
            "exterior_facet_bottom": op2.ON_BOTTOM,
            "interior_facet_horiz": op2.ON_INTERIOR_FACETS
        }.get(self._local_kernel.integral_type, None)

        return op2.WrapperKernel(
            self._local_kernel.kernel,
            wrapper_kernel_args,
            iteration_region=iteration_region,
            pass_layer_arg=self._local_kernel.pass_layer_arg
        )

    def _as_pyop2_arg(self, local_kernel_arg):
        """Convert a local kernel argument type to a :class:`pyop2.WrapperKernelArg`.

        :arg local_kernel_arg:
            A :class:`tsfc.KernelArg` object from the local kernel.

        :returns:
            The corresponding :class:`pyop2.WrapperKernelArg`.
        """
        # TODO Once the minimum supported Python version is 3.8 this method should be
        # converted to a functools.singledispatchmethod.
        if type(local_kernel_arg) in (tsfc_utils.ConstantKernelArg,
                                      tsfc_utils.ExteriorFacetKernelArg,
                                      tsfc_utils.InteriorFacetKernelArg):
            return op2.GlobalWrapperKernelArg(local_kernel_arg.shape)

        elif type(local_kernel_arg) in (tsfc_utils.CoefficientKernelArg,
                                        tsfc_utils.CoordinatesKernelArg,
                                        tsfc_utils.CellOrientationsKernelArg,
                                        tsfc_utils.CellSizesKernelArg):
            arity, *dim = local_kernel_arg.shape
            dim = tuple(dim) if dim else (1,)
            return op2.DatWrapperKernelArg(dim, arity)

        elif type(local_kernel_arg) == tsfc_utils.LocalTensorKernelArg:
            if local_kernel_arg.rank == 0:
                assert self._unroll is None
                return op2.GlobalWrapperKernelArg(local_kernel_arg.shape)
            elif local_kernel_arg.rank == 1:
                assert self._unroll is None
                shape, = local_kernel_arg.shape
                arity, *dim = shape
                dim = tuple(dim) if dim else (1,)
                return op2.DatWrapperKernelArg(dim, arity)
            elif local_kernel_arg.rank == 2:
                rshape, cshape = local_kernel_arg.shape
                rarity, *rdim = rshape
                rdim = tuple(rdim) if rdim else (1,)
                carity, *cdim = cshape
                cdim = tuple(cdim) if cdim else (1,)
                dims = (rdim + cdim)
                return op2.MatWrapperKernelArg(((dims,),), (rarity, carity), unroll=self._unroll)
            else:
                raise AssertionError

        else:
            raise NotImplementedError(f"Argument type {type(local_kernel_arg)} is not "
                                       "recognised")


def make_wrapper_kernel(local_kernel, *, unroll=None):
    """Construct a :class:`pyop2.WrapperKernel` from a local kernel.

    See :class:`WrapperKernelBuilder` for a description of the available arguments.
    """
    return WrapperKernelBuilder(local_kernel, unroll=unroll).build()

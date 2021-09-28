import functools
import operator
from collections import OrderedDict, defaultdict, namedtuple
from dataclasses import dataclass
from enum import IntEnum, auto
from itertools import chain
import typing

import firedrake
import numpy
import tsfc.kernel_interface.firedrake_loopy as tsfc_utils  # TODO Stopgap
import ufl
from firedrake import (assemble_expressions, matrix, parameters, solving,
                       tsfc_interface, utils)
from firedrake.adjoint import annotate_assemble
from firedrake.bcs import DirichletBC, EquationBC, EquationBCSplit
from firedrake.formmanipulation import split_form
from firedrake.petsc import PETSc
from firedrake.slate import slac, slate
from firedrake.utils import ScalarType
from pyop2 import op2
from pyop2.exceptions import MapValueError, SparsityFormatError


__all__ = ("assemble",)


#############################################################

@dataclass(frozen=True)
class LocalKernel:

    tsfc_kernel: tsfc_interface.SplitKernel


class _LocalKernelBuilder:

    def __init__(self, expr, *, diagonal=False, form_compiler_parameters=None):
        self._expr = expr
        self._diagonal = diagonal
        self._form_compiler_parameters = form_compiler_parameters or {}

    def build(self):
        try:
            topology, = set(d.topology for d in self._expr.ufl_domains())
        except ValueError:
            raise NotImplementedError("All integration domains must share a mesh topology")

        # Ensure mesh is 'initialised' as we could have got here without building a
        # function space (e.g. if integrating a constant).
        for m in self._expr.ufl_domains():
            m.init()

        for o in chain(self._expr.arguments(), self._expr.coefficients()):
            domain = o.ufl_domain()
            if domain is not None and domain.topology != topology:
                raise NotImplementedError("Assembly with multiple meshes is not supported")

        if isinstance(self._expr, slate.TensorBase):
            if self._diagonal:
                raise NotImplementedError("Diagonal assembly with Slate is not supported")
            tsfc_kernels = slac.compile_expression(
                self._expr, tsfc_parameters=self._form_compiler_parameters
            )
        else:
            tsfc_kernels = tsfc_interface.compile_form(
                self._expr, "form", parameters=self._form_compiler_parameters, diagonal=self._diagonal
            )

        local_kernels = []
        for tsfc_kernel in tsfc_kernels:
            local_kernels.append(LocalKernel(tsfc_kernel))
        return tuple(local_kernels)


# TODO cache this
def _make_local_kernels(expr, **kwargs):
    return _LocalKernelBuilder(expr, **kwargs).build()

#############################################################


@dataclass(frozen=True)
class WrapperKernel:

    pyop2_kernel: op2.WrapperKernel
    local_kernel: LocalKernel

    @property
    def tsfc_args(self):
        return self.tsfc_kernel.kinfo.tsfc_kernel_args

    @property
    def tsfc_kernel(self):
        return self.local_kernel.tsfc_kernel


@dataclass(frozen=True)
class WrapperKernelData:
    """Class encapsulating any additional information required to make a wrapper kernel.

    This information is 'additional' in the sense that one could not obtain it from a
    pure-UFL form.
    """

    extruded: bool = False
    constant_layers: bool = False
    subset: bool = False
    unroll: bool = False

    def __post_init__(self):
        if self.constant_layers and not self.extruded:
            raise ValueError("constant_layers is only valid when extruded is True")


# TODO different class for Slate (local + wrapper)
class _WrapperKernelBuilder:

    def __init__(self, expr, kernel_data, **kwargs):
        """TODO

        .. note::

            expr should work even if it is 'pure UFL'.
        """
        self._expr = expr
        self._kernel_data = kernel_data
        self._local_kernel_kwargs = kwargs or {}

    def build(self):
        local_kernels = _make_local_kernels(self._expr, **self._local_kernel_kwargs)

        wrapper_kernels = []
        for local_kernel, kernel_data in zip(local_kernels, self._kernel_data):
            kinfo = local_kernel.tsfc_kernel.kinfo

            wrapper_kernel_args = [
                tsfc_interface.as_pyop2_wrapper_kernel_arg(arg, unroll=kernel_data.unroll)
                for arg in kinfo.tsfc_kernel_args
            ]

            iteration_region = {
                "exterior_facet_top": op2.ON_TOP,
                "exterior_facet_bottom": op2.ON_BOTTOM,
                "interior_facet_horiz": op2.ON_INTERIOR_FACETS
            }.get(local_kernel.tsfc_kernel.kinfo.integral_type, None)

            pyop2_kernel = op2.WrapperKernel(
                kinfo.kernel,
                wrapper_kernel_args,
                iteration_region=iteration_region,
                pass_layer_arg=kinfo.pass_layer_arg,
                extruded=kernel_data.extruded,
                constant_layers=kernel_data.constant_layers,
                subset=kernel_data.subset
            )

            wrapper_kernel = WrapperKernel(pyop2_kernel, local_kernel)
            wrapper_kernels.append(wrapper_kernel)

        return wrapper_kernels

    def _as_wrapper_kernel_argument(self, tsfc_arg):
        return _as_wrapper_kernel_argument(tsfc_arg, self)


# TODO cache this
def _make_wrapper_kernels(*args, **kwargs):
    return _WrapperKernelBuilder(*args, **kwargs).build()


#############################################################


@dataclass(frozen=True)
class ParloopData:
    """TODO"""

    lgmaps: typing.Optional = None
    unroll: bool = False


class ParloopExecutor:

    def __init__(self, expr, parloop_data, *, diagonal=False, tensor=None, **kwargs):
        """

        .. note::

            Here expr is a 'Firedrake-level' entity since we now recognise that data is
            attached. This means that we cannot safely cache the resulting object.
        """
        self._expr = expr
        self._parloop_data = parloop_data
        self._tensor = tensor
        self._diagonal = diagonal
        self._wrapper_kernel_kwargs = kwargs

    def run(self):
        # TODO find another way
        # These will be used to correctly interpret the "otherwise" subdomain
        all_integer_subdomain_ids = defaultdict(list)
        for _, subexpr in split_form(self._expr, diagonal=self._diagonal):
            for integral in subexpr.integrals():
                if integral.subdomain_id() != "otherwise":
                    all_integer_subdomain_ids[integral.integral_type()].append(integral.subdomain_id())

        for k, v in all_integer_subdomain_ids.items():
            all_integer_subdomain_ids[k] = tuple(sorted(v))

        wrapper_kernel_data = []
        integrals = chain(*[subexpr.integrals() for _, subexpr in split_form(self._expr, diagonal=self._diagonal)])
        for integral, parloop_data in zip(integrals, self._parloop_data):
            iterset = _get_iterset(self._expr, integral, all_integer_subdomain_ids)
            # since we are at 'Firedrake-level' we can inspect Firedrake objects
            extruded = isinstance(iterset, op2.ExtrudedSet)
            constant_layers = extruded and iterset.constant_layers
            subset = isinstance(iterset, op2.Subset)
            kernel_data_ = WrapperKernelData(unroll=parloop_data.unroll)
            wrapper_kernel_data.append(kernel_data_)

        wrapper_kernels = _make_wrapper_kernels(
            self._expr,
            wrapper_kernel_data,
            diagonal=self._diagonal,
            **self._wrapper_kernel_kwargs
        )


        # if kinfo.needs_cell_facets:
        #     raise NotImplementedError("Need to fix in Slate")
        #     assert integral_type == "cell"
        #     extra_args.append(m.cell_to_facets(op2.READ))

        # if kinfo.pass_layer_arg:
        #     raise NotImplementedError("Need to fix in Slate")
        #     c = op2.Global(1, itspace.layers-2, dtype=numpy.dtype(numpy.int32))
        #     o = c(op2.READ)
        #     extra_args.append(o)

        integrals = chain(*[subexpr.integrals() for _, subexpr in split_form(self._expr, diagonal=self._diagonal)])
        for integral, wrapper_kernel, parloop_data in zip(integrals, wrapper_kernels, self._parloop_data):
            # Icky generator so we can access the correct coefficients in order
            kinfo = wrapper_kernel.tsfc_kernel.kinfo
            def coeffs():
                for n, split_map in kinfo.coefficient_map:
                    c = self._expr.coefficients()[n]
                    split_c = c.split()
                    for c_ in (split_c[i] for i in split_map):
                        yield c_
            self.coeffs_iterator = iter(coeffs())

            iterset = _get_iterset(self._expr, integral, all_integer_subdomain_ids)
            parloop_args = [
                _as_parloop_arg(tsfc_arg, self, wrapper_kernel, parloop_data)
                for tsfc_arg in wrapper_kernel.tsfc_args
            ]
            try:
                op2.parloop(wrapper_kernel.pyop2_kernel, iterset, parloop_args)
            except MapValueError:
                raise RuntimeError("Integral measure does not match measure of all "
                                   "coefficients/arguments")


# TODO Make into a singledispatchmethod when we have Python 3.8
@functools.singledispatch
def _as_parloop_arg(tsfc_arg, self, wrapper_kernel, parloop_data):
    """Return a :class:`op2.ParloopArg` corresponding to the provided
    :class:`tsfc.KernelArg`.
    """
    raise NotImplementedError


@_as_parloop_arg.register(tsfc_utils.CoordinatesKernelArg)
def _(tsfc_arg, self, wrapper_kernel, parloop_data):
    kinfo = wrapper_kernel.tsfc_kernel.kinfo
    mesh = _get_mesh(self._expr, wrapper_kernel.tsfc_kernel)
    func = mesh.coordinates
    map_ = _get_map(func.function_space(), kinfo.integral_type)
    return op2.DatParloopArg(func.dat, map_)

@_as_parloop_arg.register(tsfc_utils.CellOrientationsKernelArg)
def _(tsfc_arg, self, wrapper_kernel, parloop_data):
    kinfo = wrapper_kernel.tsfc_kernel.kinfo
    mesh = _get_mesh(self._expr, wrapper_kernel.tsfc_kernel)
    func = mesh.cell_orientations()
    map_ = _get_map(func.function_space(), kinfo.integral_type)
    return op2.DatParloopArg(func.dat, map_)

@_as_parloop_arg.register(tsfc_utils.CellSizesKernelArg)
def _(tsfc_arg, self, wrapper_kernel, parloop_data):
    kinfo = wrapper_kernel.tsfc_kernel.kinfo
    mesh = _get_mesh(self._expr, wrapper_kernel.tsfc_kernel)
    func = mesh.cell_sizes
    map_ = _get_map(func.function_space(), kinfo.integral_type)
    return op2.DatParloopArg(func.dat, map_)

@_as_parloop_arg.register(tsfc_utils.ExteriorFacetKernelArg)
def _(tsfc_arg, self, wrapper_kernel, parloop_data):
    mesh = _get_mesh(self._expr, wrapper_kernel.tsfc_kernel)
    return op2.DatParloopArg(mesh.exterior_facets.local_facet_dat)

@_as_parloop_arg.register(tsfc_utils.InteriorFacetKernelArg)
def _(tsfc_arg, self, wrapper_kernel, parloop_data):
    mesh = _get_mesh(self._expr, wrapper_kernel.tsfc_kernel)
    return op2.DatParloopArg(mesh.interior_facets.local_facet_dat)

@_as_parloop_arg.register(tsfc_utils.ConstantKernelArg)
@_as_parloop_arg.register(tsfc_utils.CoefficientKernelArg)
def _(tsfc_arg, self, wrapper_kernel, parloop_data):
    kinfo = wrapper_kernel.tsfc_kernel.kinfo

    coeff = next(self.coeffs_iterator)
    if tsfc_arg.rank == 0:
        return op2.GlobalParloopArg(coeff.dat)
    elif tsfc_arg.rank == 1:
        return op2.DatParloopArg(coeff.dat, _get_map(coeff.function_space(), kinfo.integral_type))
    else:
        raise AssertionError("TODO")

@_as_parloop_arg.register(tsfc_utils.LocalTensorKernelArg)
def _(tsfc_arg, self, wrapper_kernel, parloop_data):
    indices = wrapper_kernel.tsfc_kernel.indices
    tensor =self._tensor
    kinfo = wrapper_kernel.tsfc_kernel.kinfo
    # TODO New types for Scalar/Vector/Matrix
    if tsfc_arg.rank == 0:
        return op2.GlobalParloopArg(tensor)
    elif tsfc_arg.rank == 1:
        i, = indices
        if i is None:
            return op2.DatParloopArg(
                self._tensor.dat, _get_map(self._tensor.function_space(), kinfo.integral_type)
            )
        else:
            return op2.DatParloopArg(
                self._tensor.dat[i], 
                _get_map(self._tensor.function_space()[i], kinfo.integral_type)
            )
    elif tsfc_arg.rank == 2:
        i, j = indices
        test, trial = tensor.a.arguments()
        if i is None and j is None:
            rmap = _get_map(test.function_space(), kinfo.integral_type)
            cmap = _get_map(trial.function_space(), kinfo.integral_type)
            return op2.MatParloopArg(tensor.M, (rmap, cmap), lgmaps=tuple(parloop_data.lgmaps))
        else:
            assert i is not None and j is not None
            rmap = _get_map(test.function_space()[i], kinfo.integral_type)
            cmap = _get_map(trial.function_space()[j], kinfo.integral_type)
            return op2.MatParloopArg(tensor.M[i, j], (rmap, cmap), lgmaps=(parloop_data.lgmaps,))
    else:
        raise AssertionError(f"Provided rank ({tsfc_arg.rank}) is not in {{0, 1, 2}}")




# Do NOT cache this
def _execute_parloops(*args, **kwargs):
    ParloopExecutor(*args, **kwargs).run()


class _FormAssembler(abc.ABC):

    def __init__(self, *, form_compiler_parameters=None):
        self._form_compiler_parameters = form_compiler_parameters

    @property
    def form_compiler_parameters(self):
        return self._form_compiler_parameters

    @abc.abstractproperty
    def result(self):
        ...

    @abc.abstractmethod
    def assemble(self, *args):
        ...



class _ZeroFormAssembler(_FormAssembler):

    def __init__(self, expr, **kwargs)
        super().__init__(**kwargs)

        if len(expr.arguments()) != 0:
            raise ValueError("Cannot assemble a 0-form with arguments")

        self._expr = expr
        self._tensor = op2.Global(1, [0.0], dtype=utils.ScalarType)

    @property
    def result(self):
        return self._tensor.data[0]

    def assemble(self):
        _execute_parloops(
            self._expr,
            [ParloopData() for _ in split_form(self._expr)],
            tensor=self._tensor,
            form_compiler_parameters=self.form_compiler_parameters
        )


class _OneFormAssembler(_FormAssembler):

    def __init__(
        self,
        expr,
        tensor=None,
        bcs=None,
        *,
        diagonal=False,
        assembly_type=AssemblyType.SOLUTION,
        **kwargs
    ):
        super().__init__(**kwargs)

        if diagonal:
            test, trial = expr.arguments()
            if test.function_space() != trial.function_space():
                raise ValueError("Can only assemble the diagonal of 2-form if the "
                                 "function spaces match")
        else:
            test, = expr.arguments()

        if tensor:
            if test.function_space() != tensor.function_space():
                raise ValueError("Form's argument does not match provided result tensor")
            tensor.dat.zero()
        else:
            tensor = firedrake.Function(test.function_space())
        self._tensor = tensor

        self._expr = expr
        self._diagonal = diagonal

    @property
    def result(self):
        return self.tensor

    def assemble(self, expr=None, bcs=None):
        expr = self._expr if not expr
        bcs = self._bcs if not bcs

        parloop_data = [ParloopData() for _ in split_form(expr)]
        _execute_parloops(expr, parloop_data, tensor=tensor)

        # Might have gotten here without EquationBC objects preprocessed
        if any(isinstance(bc, EquationBC) for bc in bcs):
            bcs = tuple(bc.extract_form("F") for bc in bcs)

        eq_bcs = tuple(bc for bc in bcs if isinstance(bc, EquationBCSplit))
        if eq_bcs and opts.diagonal:
            raise NotImplementedError("Diagonal assembly and EquationBC not supported")

        for bc in solving._extract_bcs(bcs):
            self._apply_bc(bc) 

    def _apply_bc(self, bc):
        # TODO Maybe this could be a singledispatchmethod?
        if isinstance(bc, DirichletBC):
            self._apply_dirichlet_bc(bc)
        elif isinstance(bc, EquationBCSplit):
            bc.zero(self._tensor)
            self.assemble(bc.f, bc.bcs)
        else:
            raise AssertionError

    def _apply_dirichlet_bc(self, bc):
        if self._assembly_type == AssemblyType.SOLUTION:
            if self._diagonal:
                bc.set(self._tensor, 1)
            else:
                bc.apply(self._tensor)
        elif self._assembly_type == AssemblyType.RESIDUAL:
            bc.zero(self._tensor)
        else:
            raise AssertionError


class _TwoFormAssembler(_FormAssembler):

    def __init__(
        self,
        expr,
        tensor=None,
        *,
        mat_type=None,
        sub_mat_type=None,
        appctx=None,
        form_compiler_parameters=None,
        options_prefix=None,
    ):
        super().__init__(form_compiler_parameters=form_compiler_parameters)

        if expr.arguments() != tensor.a.arguments():
            raise ValueError("Form's arguments do not match provided result tensor")

        mat_type, sub_mat_type = _get_mat_type(mat_type, sub_mat_type, expr.arguments())

        self._mat_type = mat_type
        self._sub_mat_type = sub_mat_type
        self._appctx = appctx or {}
        self._options_prefix = options_prefix

        if tensor:
            if mat_type != "matfree":
                tensor.M.zero()
            if matrix.a.arguments() != expr.arguments():
                raise ValueError("Form's arguments do not match provided result tensor")
        else:
            tensor = allocate_matrix(
                expr, bcs,
                form_compiler_parameters=form_compiler_parameters
            )
        self._tensor = tensor


    @property
    def result(self):
        if self._mat_type == "matfree":
            self._tensor.assemble()
        else:
            self._tensor.M.assemble()
        return self._tensor

    def assemble(self, expr=None, bcs=None):
        if self._mat_type == "matfree":
            return

        expr = self._expr if not expr
        bcs = self._bcs if not bcs

        parloop_data = []
        for indices, _ in split_form(expr):
            test, trial = self._expr.arguments()
            Vrow = test.function_space()
            Vcol = trial.function_space()
            row, col = indices
            if row is None and col is None:
                lgmaps, unroll = zip(*(_collect_lgmaps(self._tensor, self._bcs, Vrow, Vcol, i, j)
                                       for i, j in numpy.ndindex(self._tensor.block_shape)))
                unroll = any(unroll)
            else:
                assert row is not None and col is not None
                lgmaps, unroll = _collect_lgmaps(self._tensor, self._bcs, Vrow, Vcol, row, col)

            parloop_data.append(ParloopData(lgmaps, unroll))
        _execute_parloops(expr, parloop_data, tensor=self._tensor)

        for bc in solving._extract_bcs(bcs):
            self._apply_bc(bc)

    def _apply_bc(self, bc):
        if isinstance(bc, DirichletBC):
            self._apply_dirichlet_bc(bc)
        elif isinstance(bc, EquationBCSplit):
            self.assemble(bc.f, bc.bcs)
        else:
            raise AssertionError

    def _apply_dirichlet_bc(self, bc):
        op2tensor = self._tensor.M
        shape = tuple(len(a.function_space()) for a in self._tensor.a.arguments())

        V = bc.function_space()
        nodes = bc.nodes
        for i, j in numpy.ndindex(shape):
            # Set diagonal entries on bc nodes to 1 if the current
            # block is on the matrix diagonal and its index matches the
            # index of the function space the bc is defined on.
            if i != j:
                continue
            if V.component is None and V.index is not None:
                # Mixed, index (no ComponentFunctionSpace)
                if V.index == i:
                    op2tensor[i, j].set_local_diagonal_entries(nodes)
            elif V.component is not None:
                # ComponentFunctionSpace, check parent index
                if V.parent.index is not None:
                    # Mixed, index doesn't match
                    if V.parent.index != i:
                        continue
                    # Index matches
                op2tensor[i, j].set_local_diagonal_entries(nodes, idx=V.component)
            elif V.index is None:
                op2tensor[i, j].set_local_diagonal_entries(nodes)
            else:
                raise RuntimeError("Unhandled BC case")


#############################################################

class AssemblyType(IntEnum):
    """Enum enumerating possible assembly types.

    See ``"assembly_type"`` from :func:`assemble` for more information.
    """
    SOLUTION = auto()
    RESIDUAL = auto()


@PETSc.Log.EventDecorator()
@annotate_assemble
def assemble(expr, *args, **kwargs):
    r"""Evaluate expr.

    :arg expr: a :class:`~ufl.classes.Form`, :class:`~ufl.classes.Expr` or
        a :class:`~slate.TensorBase` expression.
    :arg tensor: Existing tensor object to place the result in.
    :arg bcs: Iterable of boundary conditions to apply.
    :kwarg diagonal: If assembling a matrix is it diagonal?
    TODO update here
    :kwarg assembly_type: String indicating how boundary conditions are applied
        (may be ``"solution"`` or ``"residual"``). If ``"solution"`` then the
        boundary conditions are applied as expected whereas ``"residual"`` zeros
        the selected components of the tensor.
    :kwarg form_compiler_parameters: Dictionary of parameters to pass to
        the form compiler. Ignored if not assembling a :class:`~ufl.classes.Form`.
        Any parameters provided here will be overridden by parameters set on the
        :class:`~ufl.classes.Measure` in the form. For example, if a
        ``quadrature_degree`` of 4 is specified in this argument, but a degree of
        3 is requested in the measure, the latter will be used.
    :kwarg mat_type: String indicating how a 2-form (matrix) should be
        assembled -- either as a monolithic matrix (``"aij"`` or ``"baij"``),
        a block matrix (``"nest"``), or left as a :class:`.ImplicitMatrix` giving
        matrix-free actions (``'matfree'``). If not supplied, the default value in
        ``parameters["default_matrix_type"]`` is used.  BAIJ differs
        from AIJ in that only the block sparsity rather than the dof
        sparsity is constructed.  This can result in some memory
        savings, but does not work with all PETSc preconditioners.
        BAIJ matrices only make sense for non-mixed matrices.
    :kwarg sub_mat_type: String indicating the matrix type to
        use *inside* a nested block matrix.  Only makes sense if
        ``mat_type`` is ``nest``.  May be one of ``"aij"`` or ``"baij"``.  If
        not supplied, defaults to ``parameters["default_sub_matrix_type"]``.
    :kwarg appctx: Additional information to hang on the assembled
        matrix if an implicit matrix is requested (mat_type ``"matfree"``).
    :kwarg options_prefix: PETSc options prefix to apply to matrices.

    :returns: See below.

    If expr is a :class:`~ufl.classes.Form` or Slate tensor expression then
    this evaluates the corresponding integral(s) and returns a :class:`float`
    for 0-forms, a :class:`.Function` for 1-forms and a :class:`.Matrix` or
    :class:`.ImplicitMatrix` for 2-forms. In the case of 2-forms the rows
    correspond to the test functions and the columns to the trial functions.

    If expr is an expression other than a form, it will be evaluated
    pointwise on the :class:`.Function`\s in the expression. This will
    only succeed if all the Functions are on the same
    :class:`.FunctionSpace`.

    If ``tensor`` is supplied, the assembled result will be placed
    there, otherwise a new object of the appropriate type will be
    returned.

    If ``bcs`` is supplied and ``expr`` is a 2-form, the rows and columns
    of the resulting :class:`.Matrix` corresponding to boundary nodes
    will be set to 0 and the diagonal entries to 1. If ``expr`` is a
    1-form, the vector entries at boundary nodes are set to the
    boundary condition values.

    .. note::
        For 1-form assembly, the resulting object should in fact be a *cofunction*
        instead of a :class:`.Function`. However, since cofunctions are not
        currently supported in UFL, functions are used instead.
    """
    # TODO It bothers me that we use the same code for two types of expression. Ideally
    # we would define a shared interface for the two to follow. Otherwise I feel like
    # having assemble_form and assemble_slate as separate functions would be desirable.
    if isinstance(expr, (ufl.form.Form, slate.TensorBase)):
        return _assemble_form(expr, *args, **kwargs)
    elif isinstance(expr, ufl.core.expr.Expr):
        return assemble_expressions.assemble_expression(expr)
    else:
        raise TypeError(f"Unable to assemble: {expr}")


def _assemble_form(expr, *args, *, diagonal=False, **kwargs):
    """Assemble an expression.

    :arg expr: a :class:`~ufl.classes.Form` or a :class:`~slate.TensorBase`
        expression.

    See :func:`assemble` for a description of the possible additional arguments
    and return values.
    """
    rank = len(expr.arguments())
    if rank == 0:
        assembler = _ScalarAssembler(expr, *args, **kwargs)
    elif rank == 1 or (rank == 2 and diagonal):
        assembler = _VectorAssembler(expr, *args, diagonal=diagonal, **kwargs)
    elif rank == 2:
        assembler = _MatrixAssembler(expr, *args, **kwargs)
    else:
        raise AssertionError

    assembler.assemble()
    return assembler.result  # needed because of final communication


@PETSc.Log.EventDecorator()
def allocate_matrix(
    expr,
    bcs=None,
    *,
    mat_type=None,
    sub_mat_type=None,
    appctx=None,
    form_compiler_parameters=None,
    options_prefix=None
):
    r"""Allocate a matrix given an expression.

    .. warning::

       Do not use this function unless you know what you're doing.
    """
    bcs = bcs or ()
    appctx = appctx or {}

    matfree = mat_type == "matfree"
    arguments = expr.arguments()
    if bcs is None:
        bcs = ()
    else:
        if any(isinstance(bc, EquationBC) for bc in bcs):
            raise TypeError("EquationBC objects not expected here. "
                            "Preprocess by extracting the appropriate form with bc.extract_form('Jp') or bc.extract_form('J')")
    if matfree:
        return matrix.ImplicitMatrix(expr, bcs,
                                     appctx=appctx,
                                     fc_params=form_compiler_parameters,
                                     options_prefix=options_prefix)

    integral_types = set(i.integral_type() for i in expr.integrals())
    for bc in bcs:
        integral_types.update(integral.integral_type()
                              for integral in bc.integrals())
    nest = opts.mat_type == "nest"
    if nest:
        baij = opts.sub_mat_type == "baij"
    else:
        baij = opts.mat_type == "baij"

    if any(len(a.function_space()) > 1 for a in arguments) and opts.mat_type == "baij":
        raise ValueError("BAIJ matrix type makes no sense for mixed spaces, use 'aij'")

    get_cell_map = operator.methodcaller("cell_node_map")
    get_extf_map = operator.methodcaller("exterior_facet_node_map")
    get_intf_map = operator.methodcaller("interior_facet_node_map")
    domains = OrderedDict((k, set()) for k in (get_cell_map,
                                               get_extf_map,
                                               get_intf_map))
    mapping = {"cell": (get_cell_map, op2.ALL),
               "exterior_facet_bottom": (get_cell_map, op2.ON_BOTTOM),
               "exterior_facet_top": (get_cell_map, op2.ON_TOP),
               "interior_facet_horiz": (get_cell_map, op2.ON_INTERIOR_FACETS),
               "exterior_facet": (get_extf_map, op2.ALL),
               "exterior_facet_vert": (get_extf_map, op2.ALL),
               "interior_facet": (get_intf_map, op2.ALL),
               "interior_facet_vert": (get_intf_map, op2.ALL)}
    for integral_type in integral_types:
        try:
            get_map, region = mapping[integral_type]
        except KeyError:
            raise ValueError(f"Unknown integral type '{integral_type}'")
        domains[get_map].add(region)

    test, trial = arguments
    map_pairs, iteration_regions = zip(*(((get_map(test), get_map(trial)),
                                          tuple(sorted(regions)))
                                         for get_map, regions in domains.items()
                                         if regions))
    try:
        sparsity = op2.Sparsity((test.function_space().dof_dset,
                                 trial.function_space().dof_dset),
                                tuple(map_pairs),
                                iteration_regions=tuple(iteration_regions),
                                nest=nest,
                                block_sparse=baij)
    except SparsityFormatError:
        raise ValueError("Monolithic matrix assembly not supported for systems "
                         "with R-space blocks")

    return matrix.Matrix(expr, bcs, mat_type, sparsity, ScalarType,
                         options_prefix=options_prefix)




@PETSc.Log.EventDecorator()
def create_assembly_callable(expr, tensor=None, bcs=None, form_compiler_parameters=None,
                             mat_type=None, sub_mat_type=None, diagonal=False):
    r"""Create a callable object than be used to assemble expr into a tensor.

    This is really only designed to be used inside residual and
    jacobian callbacks, since it always assembles back into the
    initially provided tensor.  See also :func:`allocate_matrix`.

    .. warning::

        This function is now deprecated.

    .. warning::

       Really do not use this function unless you know what you're doing.
    """
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("once", DeprecationWarning)
        warnings.warn("create_assembly_callable is now deprecated. Please use assemble instead.",
                      DeprecationWarning)

    if tensor is None:
        raise ValueError("Have to provide tensor to write to")
    return functools.partial(assemble, expr,
                             tensor=tensor,
                             bcs=bcs,
                             form_compiler_parameters=form_compiler_parameters,
                             mat_type=mat_type,
                             sub_mat_type=sub_mat_type,
                             diagonal=diagonal,
                             assembly_type="residual")


def _get_mat_type(mat_type, sub_mat_type, arguments):
    """Validate the matrix types provided by the user and set any that are
    undefined to default values.

    :arg mat_type: (:class:`str`) PETSc matrix type for the assembled matrix.
    :arg sub_mat_type: (:class:`str`) PETSc matrix type for blocks if
        ``mat_type`` is ``"nest"``.
    :arg arguments: The test and trial functions of the expression being assembled.
    :raises ValueError: On bad arguments.
    :returns: 2-:class:`tuple` of validated/default ``mat_type`` and ``sub_mat_type``.
    """
    if mat_type is None:
        mat_type = parameters.parameters["default_matrix_type"]
        if any(V.ufl_element().family() == "Real"
               for arg in arguments
               for V in arg.function_space()):
            mat_type = "nest"
    if mat_type not in {"matfree", "aij", "baij", "nest", "dense"}:
        raise ValueError(f"Unrecognised matrix type, '{mat_type}'")
    if sub_mat_type is None:
        sub_mat_type = parameters.parameters["default_sub_matrix_type"]
    if sub_mat_type not in {"aij", "baij"}:
        raise ValueError(f"Invalid submatrix type, '{sub_mat_type}' (not 'aij' or 'baij')")
    return mat_type, sub_mat_type


def _collect_lgmaps(matrix, all_bcs, Vrow, Vcol, row, col):
    """Obtain local to global maps for matrix insertion in the
    presence of boundary conditions.

    :arg matrix: the matrix.
    :arg all_bcs: all boundary conditions involved in the assembly of
        the matrix.
    :arg Vrow: function space for rows.
    :arg Vcol: function space for columns.
    :arg row: index into Vrow (by block).
    :arg col: index into Vcol (by block).
    :returns: 2-tuple ``(row_lgmap, col_lgmap), unroll``. unroll will
       indicate to the codegeneration if the lgmaps need to be
       unrolled from any blocking they contain.
    """
    if len(Vrow) > 1:
        bcrow = tuple(bc for bc in all_bcs
                      if bc.function_space_index() == row)
    else:
        bcrow = all_bcs
    if len(Vcol) > 1:
        bccol = tuple(bc for bc in all_bcs
                      if bc.function_space_index() == col
                      and isinstance(bc, DirichletBC))
    else:
        bccol = tuple(bc for bc in all_bcs
                      if isinstance(bc, DirichletBC))
    rlgmap, clgmap = matrix.M[row, col].local_to_global_maps
    rlgmap = Vrow[row].local_to_global_map(bcrow, lgmap=rlgmap)
    clgmap = Vcol[col].local_to_global_map(bccol, lgmap=clgmap)
    unroll = any(bc.function_space().component is not None
                 for bc in chain(bcrow, bccol))
    return (rlgmap, clgmap), unroll


def _get_map(func_space, integral_type):
    """TODO"""
    assert isinstance(func_space, ufl.FunctionSpace)

    if integral_type in (
        "cell",
        "exterior_facet_top",
        "exterior_facet_bottom",
        "interior_facet_horiz"
    ):
        return func_space.cell_node_map()
    elif integral_type in ("exterior_facet", "exterior_facet_vert"):
        return func_space.exterior_facet_node_map()
    elif integral_type in ("interior_facet", "interior_facet_vert"):
        return func_space.interior_facet_node_map()
    else:
        raise AssertionError(f"Unknown integral type '{integral_type}'")


def _get_mesh(expr, expr_kernel):
    return expr.ufl_domains()[expr_kernel.kinfo.domain_number]


def _get_iterset(expr, integral, all_integer_subdomain_ids):
    mesh = integral.ufl_domain()
    subdomain_data = expr.subdomain_data()[mesh].get(integral.integral_type(), None)
    if subdomain_data is not None:
        if integral.integral_type() != "cell":
            raise NotImplementedError("subdomain_data only supported with cell integrals")
        if integral.subdomain_id() not in ("otherwise", "everywhere"):
            raise ValueError("Cannot use subdomain data and subdomain_id")
        return subdomain_data
    else:
        return mesh.measure_set(integral.integral_type(), integral.subdomain_id(),
                                all_integer_subdomain_ids)

import abc
import collections
from collections import OrderedDict, defaultdict
import dataclasses
from dataclasses import dataclass
from enum import IntEnum, auto
import functools
import itertools
import operator

import cachetools
import finat
import firedrake
import numpy
from tsfc import kernel_args
import ufl
from firedrake import (assemble_expressions, matrix, parameters, solving,
                       tsfc_interface, utils)
from firedrake.adjoint import annotate_assemble
from firedrake.bcs import DirichletBC, EquationBC, EquationBCSplit
from firedrake.extrusion_utils import calc_offset
from firedrake.functionspacedata import (preprocess_finat_element, entity_dofs_key,
                                         entity_permutations_key)
from firedrake.petsc import PETSc
from firedrake.slate import slac, slate
from firedrake.slate.slac.kernel_builder import CellFacetKernelArg, LayerCountKernelArg, LayerKernelArg
from firedrake.utils import ScalarType
from pyop2 import op2
import pyop2.wrapper_kernel
from pyop2.exceptions import MapValueError, SparsityFormatError


__all__ = ("AssemblyType", "assemble")


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
    nest = mat_type == "nest"
    if nest:
        baij = sub_mat_type == "baij"
    else:
        baij = mat_type == "baij"

    if any(len(a.function_space()) > 1 for a in arguments) and mat_type == "baij":
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
                             assembly_type=AssemblyType.SOLUTION)


class _FormAssembler(abc.ABC):

    def __init__(self, *, form_compiler_parameters=None):
        self._form_compiler_params = form_compiler_parameters

    @property
    def form_compiler_params(self):
        return self._form_compiler_params

    @abc.abstractproperty
    def result(self):
        ...

    @abc.abstractmethod
    def assemble(self, *args):
        ...


class _ZeroFormAssembler(_FormAssembler):

    def __init__(self, expr, **kwargs):
        super().__init__(**kwargs)

        if len(expr.arguments()) != 0:
            raise ValueError("Cannot assemble a 0-form with arguments")

        self._expr = expr
        self._tensor = op2.Global(1, [0.0], dtype=utils.ScalarType)

    @property
    def result(self):
        return self._tensor.data[0]

    def assemble(self):
        knls = _make_wrapper_kernels(
            self._expr, form_compiler_params=self.form_compiler_params
        )

        tsfc_knls = [knl.tsfc_kernel for knl in knls]
        all_integer_subdomain_ids = _get_all_integer_subdomain_ids(tsfc_knls)

        for knl in knls:
            iterset = _get_iterset(self._expr, knl.tsfc_kernel.kinfo, all_integer_subdomain_ids)
            _execute_parloop(self._expr, knl, iterset, tensor=self._tensor)


class _OneFormAssembler(_FormAssembler):

    def __init__(
        self,
        form,
        tensor=None,
        bcs=None,
        *,
        assembly_type=AssemblyType.SOLUTION,
        diagonal=False,
        **kwargs
    ):
        super().__init__(**kwargs)

        if diagonal:
            test, trial = form.arguments()
            if test.function_space() != trial.function_space():
                raise ValueError("Can only assemble the diagonal of 2-form if the "
                                 "function spaces match")
        else:
            test, = form.arguments()

        if tensor:
            if test.function_space() != tensor.function_space():
                raise ValueError("Form's argument does not match provided result tensor")
            tensor.dat.zero()
        else:
            tensor = firedrake.Function(test.function_space())

        self._form = form
        self._bcs = bcs
        self._tensor = tensor
        self._assembly_type = assembly_type
        self._diagonal = diagonal

    @property
    def result(self):
        return self._tensor

    def assemble(self, form=None, bcs=None):
        # These arguments are optional in case we are calling this function recursively
        if form is None:
            assert bcs is None
            form = self._form
            bcs = self._bcs

        knls = _make_wrapper_kernels(form, diagonal=self._diagonal,
                                     form_compiler_params=self.form_compiler_params)

        tsfc_knls = [knl.tsfc_kernel for knl in knls]
        all_integer_subdomain_ids = _get_all_integer_subdomain_ids(tsfc_knls)

        for knl in knls:
            iterset = _get_iterset(form, knl.tsfc_kernel.kinfo, all_integer_subdomain_ids)
            _execute_parloop(form, knl, iterset, tensor=self._tensor, diagonal=self._diagonal)

        for bc in bcs:
            if isinstance(bc, EquationBC):
                bc = bc.extract_form("F")
            self._apply_bc(bc)

    def _apply_bc(self, bc):
        # TODO Maybe this could be a singledispatchmethod?
        if isinstance(bc, DirichletBC):
            self._apply_dirichlet_bc(bc)
        elif isinstance(bc, EquationBCSplit):
            if self._diagonal:
                raise NotImplementedError("Diagonal assembly and EquationBC not supported")
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
        bcs=None,
        *,
        mat_type=None,
        sub_mat_type=None,
        appctx=None,
        form_compiler_parameters=None,
        options_prefix=None,
    ):
        super().__init__(form_compiler_parameters=form_compiler_parameters)

        mat_type, sub_mat_type = self._get_mat_type(mat_type, sub_mat_type, expr.arguments())

        if tensor:
            if mat_type != "matfree":
                tensor.M.zero()
            if tensor.a.arguments() != expr.arguments():
                raise ValueError("Form's arguments do not match provided result tensor")
        else:
            tensor = allocate_matrix(
                expr, bcs, mat_type=mat_type, sub_mat_type=sub_mat_type, appctx=appctx,
                form_compiler_parameters=form_compiler_parameters,
                options_prefix=options_prefix
            )

        self._expr = expr
        self._tensor = tensor
        self._bcs = bcs
        self._is_matfree = mat_type == "matfree"

    @property
    def result(self):
        if not self._is_matfree:
            self._tensor.M.assemble()
        else:
            self._tensor.assemble()
        return self._tensor

    def assemble(self, expr=None, bcs=None):
        if self._is_matfree:
            return

        if expr is None:
            assert bcs is None
            expr = self._expr
            bcs = self._bcs

        knls = _make_wrapper_kernels(expr, form_compiler_params=self.form_compiler_params)

        tsfc_knls = [knl.tsfc_kernel for knl in knls]
        all_integer_subdomain_ids = _get_all_integer_subdomain_ids(tsfc_knls)

        for knl in knls:
            indices, kinfo = knl.tsfc_kernel

            iterset = _get_iterset(expr, kinfo, all_integer_subdomain_ids)

            test, trial = expr.arguments()
            Vrow = test.function_space()
            Vcol = trial.function_space()
            row, col = indices
            if row is None and col is None:
                lgmaps, unroll = zip(*(self._collect_lgmaps(self._tensor, bcs, Vrow, Vcol, i, j)
                                       for i, j in numpy.ndindex(self._tensor.block_shape)))
                unroll = any(unroll)
            else:
                assert row is not None and col is not None
                lgmaps, unroll = self._collect_lgmaps(self._tensor, bcs, Vrow, Vcol, row, col)

            # If we need to handle boundary conditions then replace the first argument to
            # the wrapper kernel.
            if unroll:
                old_arg = knl.pyop2_kernel.arguments[0]
                new_arg = dataclasses.replace(old_arg, unroll=True)
                pyop2_kernel = pyop2.wrapper_kernel.replace_argument(knl.pyop2_kernel, old_arg, new_arg)
                knl = dataclasses.replace(knl, pyop2_kernel=pyop2_kernel)

            _execute_parloop(expr, knl, iterset, tensor=self._tensor, lgmaps=lgmaps)

        for bc in bcs:
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

    @staticmethod
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

    @staticmethod
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
                     for bc in itertools.chain(bcrow, bccol))
        return (rlgmap, clgmap), unroll


def _assemble_form(form, tensor=None, bcs=None, *, assembly_type=AssemblyType.SOLUTION, diagonal=False, **kwargs):
    """Assemble a form.

    :arg form:
        The :class:`~ufl.classes.Form` or :class:`~slate.TensorBase` to be assembled.
    :args args:
        Extra positional arguments to pass to the underlying :class:`_Assembler` instance.
        See :func:`assemble` for more information.
    :kwarg diagonal:
        Flag indicating whether or not we are assembling the diagonal of a matrix.
    :kwargs kwargs:
        Extra keyword arguments to pass to the underlying :class:`_Assembler` instance.
        See :func:`assemble` for more information.
    """
    # Ensure mesh is 'initialised' as we could have got here without building a
    # function space (e.g. if integrating a constant).
    for mesh in form.ufl_domains():
        mesh.init()

    bcs = solving._extract_bcs(bcs)

    rank = len(form.arguments())
    if rank == 0:
        assert tensor is None and bcs == ()
        assembler = _ZeroFormAssembler(form, **kwargs)
    elif rank == 1 or (rank == 2 and diagonal):
        assembler = _OneFormAssembler(form, tensor, bcs, assembly_type=assembly_type, diagonal=diagonal, **kwargs)
    elif rank == 2:
        assembler = _TwoFormAssembler(form, tensor, bcs, **kwargs)
    else:
        raise AssertionError

    assembler.assemble()
    return assembler.result


@dataclass(frozen=True)
class _WrapperKernel:

    pyop2_kernel: op2.WrapperKernel
    local_kernel: tsfc_interface.SplitKernel

    @property
    def tsfc_args(self):
        return self.tsfc_kernel.kinfo.tsfc_kernel_args

    @property
    def tsfc_kernel(self):
        return self.local_kernel


class _AssembleWrapperKernelBuilder:

    def __init__(self, expr, *, diagonal=False, form_compiler_params=None):
        """TODO

        .. note::

            expr should work even if it is 'pure UFL'.
        """
        self._expr = expr
        self._diagonal = diagonal
        self._form_compiler_params = form_compiler_params or {}

    def build(self):
        try:
            topology, = set(d.topology for d in self._expr.ufl_domains())
        except ValueError:
            raise NotImplementedError("All integration domains must share a mesh topology")

        for o in itertools.chain(self._expr.arguments(), self._expr.coefficients()):
            domain = o.ufl_domain()
            if domain is not None and domain.topology != topology:
                raise NotImplementedError("Assembly with multiple meshes is not supported")

        if isinstance(self._expr, ufl.Form):
            local_knls = tsfc_interface.compile_form(
                self._expr, "form",
                diagonal=self._diagonal,
                parameters=self._form_compiler_params
            )
        elif isinstance(self._expr, slate.TensorBase):
            local_knls = slac.compile_expression(
                self._expr, compiler_parameters=self._form_compiler_params
            )
        else:
            raise AssertionError

        all_integer_subdomain_ids = _get_all_integer_subdomain_ids(local_knls)
        wrapper_knls = []
        for local_knl in local_knls:
            kinfo = local_knl.kinfo

            iterset = _get_iterset(self._expr, kinfo, all_integer_subdomain_ids)

            # TODO This information should be available from UFL
            extruded = isinstance(iterset, op2.ExtrudedSet)
            constant_layers = extruded and iterset.constant_layers
            subset = isinstance(iterset, op2.Subset)

            self.extruded = extruded  # hack

            wrapper_kernel_args = [
                self._as_wrapper_kernel_arg(arg, kinfo.integral_type)
                for arg in kinfo.tsfc_kernel_args
                if arg.intent is not None
            ]

            iteration_region = {
                "exterior_facet_top": op2.ON_TOP,
                "exterior_facet_bottom": op2.ON_BOTTOM,
                "interior_facet_horiz": op2.ON_INTERIOR_FACETS
            }.get(kinfo.integral_type, None)

            pyop2_kernel = op2.WrapperKernel(
                kinfo.kernel,
                wrapper_kernel_args,
                iteration_region=iteration_region,
                pass_layer_arg=kinfo.pass_layer_arg,
                extruded=extruded,
                constant_layers=constant_layers,
                subset=subset
            )

            wrapper_kernel = _WrapperKernel(pyop2_kernel, local_knl)
            wrapper_knls.append(wrapper_kernel)

        return wrapper_knls

    def _as_wrapper_kernel_arg(self, tsfc_arg, integral_type):
        # TODO Make singledispatchmethod with Python 3.8
        return _as_wrapper_kernel_arg(tsfc_arg, self, integral_type)


@functools.singledispatch
def _as_wrapper_kernel_arg(tsfc_arg, self, integral_type):
    raise NotImplementedError


@_as_wrapper_kernel_arg.register(kernel_args.RankZeroKernelArg)
def _(tsfc_arg, self, integral_type):
    return op2.GlobalWrapperKernelArg(tsfc_arg.shape)


@_as_wrapper_kernel_arg.register(kernel_args.RankOneKernelArg)
def _(tsfc_arg, self, integral_type):
    elem = tsfc_arg._elem
    if elem.is_mixed:
        subargs = []
        for el in elem.split():
            subargs.append(_make_dat_wrapper_kernel_arg(el, integral_type, self.extruded))
        return op2.MixedDatWrapperKernelArg(subargs)
    else:
        return _make_dat_wrapper_kernel_arg(elem, integral_type, self.extruded)

def _make_dat_wrapper_kernel_arg(elem, integral_type, extruded=False):
    map_id = _get_map_id(elem._elem, integral_type)

    finat_element = elem._elem
    if isinstance(finat_element, finat.TensorFiniteElement):
        finat_element = finat_element.base_element
    entity_dofs, real_tensorproduct = preprocess_finat_element(finat_element)
    # offset only valid for extruded
    if extruded:
        offset = tuple(calc_offset(finat_element.cell, entity_dofs, finat_element.space_dimension(), real_tensorproduct))
    else:
        offset = None

    map_arg = op2.MapWrapperKernelArg(map_id, elem.node_shape, offset)
    return op2.DatWrapperKernelArg(elem.tensor_shape, map_arg)


@_as_wrapper_kernel_arg.register(kernel_args.FacetKernelArg)
def _(tsfc_arg, self, integral_type):
    # These are directly addressed (no map)
    return op2.DatWrapperKernelArg(tsfc_arg.shape)


@_as_wrapper_kernel_arg.register(CellFacetKernelArg)
def _(tsfc_arg, self, integral_type):
    return op2.DatWrapperKernelArg(tsfc_arg.shape)


@_as_wrapper_kernel_arg.register(kernel_args.CellOrientationsKernelArg)
def _(tsfc_arg, self, integral_type):
    # this is taken largely from mesh.py where we observe that the function space is
    # DG0.
    from ufl import FiniteElement
    from tsfc.finatinterface import create_element
    assert not self.extruded
    ufl_element = FiniteElement("DG", cell=self._expr.ufl_domain().ufl_cell(), degree=0)
    finat_element = _as_scalar_element(create_element(ufl_element))
    map_id = _get_map_id(finat_element, integral_type)
    map_arg = op2.MapWrapperKernelArg(map_id, tsfc_arg.node_shape)
    return op2.DatWrapperKernelArg(tsfc_arg.shape, map_arg)


@_as_wrapper_kernel_arg.register(kernel_args.RankTwoKernelArg)
def _(tsfc_arg, self, integral_type):
    relem = tsfc_arg._relem
    celem = tsfc_arg._celem

    if relem.is_mixed:
        if celem.is_mixed:
            subargs = []
            shape = len(relem.split()), len(celem.split())
            for rel, cel in itertools.product(relem.split(), celem.split()):
                subargs.append(_make_mat_wrapper_kernel_arg(rel, cel, integral_type, self.extruded))
            return op2.MixedMatWrapperKernelArg(subargs, shape)
        else:
            subargs = []
            shape = len(relem.split()), 1
            for rel in relem.split():
                subargs.append(_make_mat_wrapper_kernel_arg(rel, celem, integral_type, self.extruded))
            return op2.MixedMatWrapperKernelArg(subargs, shape)
    else:
        if celem.is_mixed:
            shape = 1, len(celem.split())
            subargs = []
            for cel in celem.split():
                subargs.append(_make_mat_wrapper_kernel_arg(relem, cel, integral_type, self.extruded))
            return op2.MixedMatWrapperKernelArg(subargs, shape)
        else:
            return _make_mat_wrapper_kernel_arg(relem, celem, integral_type, self.extruded)


def _make_mat_wrapper_kernel_arg(relem, celem, integral_type, extruded=False):
    rmap_id = _get_map_id(relem._elem, integral_type)
    cmap_id = _get_map_id(celem._elem, integral_type)

    ###

    finat_element = _as_scalar_element(relem._elem)
    entity_dofs, real_tensorproduct = preprocess_finat_element(finat_element)
    if extruded:
        roffset = tuple(calc_offset(finat_element.cell, entity_dofs, finat_element.space_dimension(), real_tensorproduct))
    else:
        roffset = None

    ###

    finat_element = _as_scalar_element(celem._elem)
    entity_dofs, real_tensorproduct = preprocess_finat_element(finat_element)
    if extruded:
        coffset = tuple(calc_offset(finat_element.cell, entity_dofs, finat_element.space_dimension(), real_tensorproduct))
    else:
        coffset = None

    ###

    rmap_arg = op2.MapWrapperKernelArg(rmap_id, relem.node_shape, roffset)
    cmap_arg = op2.MapWrapperKernelArg(cmap_id, celem.node_shape, coffset)

    # PyOP2 matrix objects have scalar dims so we cope with that here...
    rdim = (numpy.prod(relem.tensor_shape, dtype=int),)
    cdim = (numpy.prod(celem.tensor_shape, dtype=int),)

    return op2.MatWrapperKernelArg(((rdim+cdim,),), (rmap_arg, cmap_arg))


def _as_scalar_element(elem):
    # This is done to mirror what happens in functionspacedata.py
    return elem.base_element if isinstance(elem, finat.TensorFiniteElement) else elem


def _get_map_id(finat_element, integral_type):
    """Return a key that is used to check if we reuse maps.

    functionspacedata.py does the same thing.
    """
    # TODO need to look at measure...
    # functionspacedata does some magic and replaces tensorelements with base
    if isinstance(finat_element, finat.TensorFiniteElement):
        finat_element = finat_element.base_element

    entity_dofs, real_tensorproduct = preprocess_finat_element(finat_element)
    try:
        eperm_key = entity_permutations_key(finat_element.entity_permutations)
    except NotImplementedError:
        eperm_key = None
    return entity_dofs_key(entity_dofs), real_tensorproduct, eperm_key


def _get_map_type(integral_type):
    if integral_type in (
        "cell",
        "exterior_facet_top",
        "exterior_facet_bottom",
        "interior_facet_horiz"
    ):
        return "cell"
    elif integral_type in ("exterior_facet", "exterior_facet_vert"):
        return "exterior_facet"
    elif integral_type in ("interior_facet", "interior_facet_vert"):
        return "interior_facet"
    else:
        raise AssertionError


def _wrapper_kernel_cache_key(form, **kwargs):
    if isinstance(form, ufl.Form):
        sig = form.signature()
    elif isinstance(form, slate.TensorBase):
        sig = form.expression_hash
    return (
        (sig,)
        + _tuplify(kwargs.pop("form_compiler_params", None) or {})
        + cachetools.keys.hashkey(**kwargs)
    )


@cachetools.cached(cachetools.LRUCache(maxsize=128), key=_wrapper_kernel_cache_key)
def _make_wrapper_kernels(*args, **kwargs):
    return _AssembleWrapperKernelBuilder(*args, **kwargs).build()


class ParloopExecutor:

    def __init__(self, form, knl, iterset, *, tensor=None, diagonal=False, lgmaps=None):
        """

        .. note::

            Here expr is a 'Firedrake-level' entity since we now recognise that data is
            attached. This means that we cannot safely cache the resulting object.
        """
        self._form = form
        self._knl = knl
        self._iterset = iterset
        self._tensor = tensor
        self._diagonal = diagonal
        self._lgmaps = lgmaps

    def run(self):
        kinfo = self._knl.tsfc_kernel.kinfo

        # Icky generator so we can access the correct coefficients in order
        def coeffs():
            for n, split_map in kinfo.coefficient_map:
                c = self._form.coefficients()[n]
                split_c = c.split()
                for c_ in (split_c[i] for i in split_map):
                    yield c_
        self.coeffs_iterator = iter(coeffs())

        parloop_args = [
            _as_parloop_arg(tsfc_arg, self, self._knl, lgmaps=self._lgmaps)
            for tsfc_arg in self._knl.tsfc_args if tsfc_arg.intent is not None
        ]
        try:
            op2.parloop(self._knl.pyop2_kernel, self._iterset, parloop_args)
        except MapValueError:
            raise RuntimeError("Integral measure does not match measure of all "
                               "coefficients/arguments")

    def _get_mesh(self, expr_kernel):
        return self._form.ufl_domains()[expr_kernel.kinfo.domain_number]

    @staticmethod
    def _get_map(func_space, integral_type):
        """TODO"""
        assert isinstance(func_space, ufl.FunctionSpace)

        map_type = _get_map_type(integral_type)

        if map_type == "cell":
            return func_space.cell_node_map()
        elif map_type == "exterior_facet":
            return func_space.exterior_facet_node_map()
        elif map_type == "interior_facet":
            return func_space.interior_facet_node_map()
        else:
            raise AssertionError


# TODO Make into a singledispatchmethod when we have Python 3.8
@functools.singledispatch
def _as_parloop_arg(tsfc_arg, self, wrapper_kernel, **kwargs):
    """Return a :class:`op2.ParloopArg` corresponding to the provided
    :class:`tsfc.KernelArg`.
    """
    raise NotImplementedError


@_as_parloop_arg.register(kernel_args.CoordinatesKernelArg)
def _(tsfc_arg, self, wrapper_kernel, **kwargs):
    kinfo = wrapper_kernel.tsfc_kernel.kinfo
    mesh = self._get_mesh(wrapper_kernel.tsfc_kernel)
    func = mesh.coordinates
    map_ = self._get_map(func.function_space(), kinfo.integral_type)
    return op2.DatParloopArg(func.dat, map_)


@_as_parloop_arg.register(kernel_args.CellOrientationsKernelArg)
def _(tsfc_arg, self, wrapper_kernel, **kwargs):
    kinfo = wrapper_kernel.tsfc_kernel.kinfo
    mesh = self._get_mesh(wrapper_kernel.tsfc_kernel)
    func = mesh.cell_orientations()
    map_ = self._get_map(func.function_space(), kinfo.integral_type)
    return op2.DatParloopArg(func.dat, map_)


@_as_parloop_arg.register(kernel_args.CellSizesKernelArg)
def _(tsfc_arg, self, wrapper_kernel, **kwargs):
    kinfo = wrapper_kernel.tsfc_kernel.kinfo
    mesh = self._get_mesh(wrapper_kernel.tsfc_kernel)
    func = mesh.cell_sizes
    map_ = self._get_map(func.function_space(), kinfo.integral_type)
    return op2.DatParloopArg(func.dat, map_)


@_as_parloop_arg.register(kernel_args.ExteriorFacetKernelArg)
def _(tsfc_arg, self, wrapper_kernel, **kwargs):
    mesh = self._get_mesh(wrapper_kernel.tsfc_kernel)
    return op2.DatParloopArg(mesh.exterior_facets.local_facet_dat)


@_as_parloop_arg.register(kernel_args.InteriorFacetKernelArg)
def _(tsfc_arg, self, wrapper_kernel, **kwargs):
    mesh = self._get_mesh(wrapper_kernel.tsfc_kernel)
    return op2.DatParloopArg(mesh.interior_facets.local_facet_dat)


@_as_parloop_arg.register(CellFacetKernelArg)
def _(tsfc_arg, self, wrapper_kernel, **kwargs):
    mesh = self._get_mesh(wrapper_kernel.tsfc_kernel)
    return op2.DatParloopArg(mesh.cell_to_facets)


@_as_parloop_arg.register(kernel_args.ConstantKernelArg)
def _(tsfc_arg, self, wrapper_kernel, **kwargs):
    coeff = next(self.coeffs_iterator)
    return op2.GlobalParloopArg(coeff.dat)


@_as_parloop_arg.register(kernel_args.CoefficientKernelArg)
def _(tsfc_arg, self, wrapper_kernel, **kwargs):
    kinfo = wrapper_kernel.tsfc_kernel.kinfo

    coeff = next(self.coeffs_iterator)
    return op2.DatParloopArg(coeff.dat, self._get_map(coeff.function_space(), kinfo.integral_type))


@_as_parloop_arg.register(LayerCountKernelArg)
def _(tsfc_arg, self, wrapper_kernel, **kwargs):
    glob = op2.Global(tsfc_arg.shape, self._iterset.layers-2, dtype=tsfc_arg.dtype)
    return op2.GlobalParloopArg(glob)


@_as_parloop_arg.register(kernel_args.ScalarOutputKernelArg)
def _(tsfc_arg, self, *args, **kwargs):
    return op2.GlobalParloopArg(self._tensor)


@_as_parloop_arg.register(kernel_args.VectorOutputKernelArg)
def _(tsfc_arg, self, wrapper_kernel, **kwargs):
    kinfo = wrapper_kernel.tsfc_kernel.kinfo

    i, = wrapper_kernel.tsfc_kernel.indices
    if i is None:
        return op2.DatParloopArg(
            self._tensor.dat, self._get_map(self._tensor.function_space(), kinfo.integral_type)
        )
    else:
        return op2.DatParloopArg(
            self._tensor.dat[i],
            self._get_map(self._tensor.function_space()[i], kinfo.integral_type)
        )


@_as_parloop_arg.register(kernel_args.MatrixOutputKernelArg)
def _(tsfc_arg, self, wrapper_kernel, *, lgmaps):
    tensor = self._tensor
    kinfo = wrapper_kernel.tsfc_kernel.kinfo
    i, j = wrapper_kernel.tsfc_kernel.indices
    test, trial = tensor.a.arguments()
    if i is None and j is None:
        rmap = self._get_map(test.function_space(), kinfo.integral_type)
        cmap = self._get_map(trial.function_space(), kinfo.integral_type)
        return op2.MatParloopArg(tensor.M, (rmap, cmap), lgmaps=tuple(lgmaps))
    else:
        assert i is not None and j is not None
        rmap = self._get_map(test.function_space()[i], kinfo.integral_type)
        cmap = self._get_map(trial.function_space()[j], kinfo.integral_type)
        return op2.MatParloopArg(tensor.M[i, j], (rmap, cmap), lgmaps=(lgmaps,))


def _execute_parloop(*args, **kwargs):
    ParloopExecutor(*args, **kwargs).run()


# TODO put in utils
def _tuplify(params):
    if isinstance(params, collections.Hashable):
        return (params,)

    assert isinstance(params, dict)
    return tuple((k, _tuplify(params[k])) for k in sorted(params))


def _get_iterset(expr, kinfo, all_integer_subdomain_ids):
    mesh = expr.ufl_domains()[kinfo.domain_number]
    subdomain_data = expr.subdomain_data()[mesh].get(kinfo.integral_type, None)
    if subdomain_data is not None:
        if kinfo.integral_type != "cell":
            raise NotImplementedError("subdomain_data only supported with cell integrals")
        if kinfo.subdomain_id not in ["everywhere", "otherwise"]:
            raise ValueError("Cannot use subdomain data and subdomain_id")
        return subdomain_data
    else:
        return mesh.measure_set(kinfo.integral_type, kinfo.subdomain_id,
                                all_integer_subdomain_ids)


def _get_all_integer_subdomain_ids(knls):
    # These will be used to correctly interpret the "otherwise" subdomain
    all_integer_subdomain_ids = defaultdict(list)
    for _, kinfo in knls:
        if kinfo.subdomain_id != "otherwise":
            all_integer_subdomain_ids[kinfo.integral_type].append(kinfo.subdomain_id)

    for k, v in all_integer_subdomain_ids.items():
        all_integer_subdomain_ids[k] = tuple(sorted(v))
    return all_integer_subdomain_ids

import pytest
from firedrake import *
from pyop2.gpu.cuda import cuda_backend
from pyop2.sequential import cpu_backend
# from pyop2.gpu.opencl import opencl_backend


@pytest.mark.parametrize("offloading_backend", [cuda_backend, cpu_backend])
def test_nonlinear_variational_solver(offloading_backend):
    set_offloading_backend(offloading_backend)
    mesh = UnitSquareMesh(32, 32)
    V = FunctionSpace(mesh, "CG", 1)
    u = TrialFunction(V)
    v = TestFunction(V)
    x, y = SpatialCoordinate(mesh)

    a = (dot(grad(v), grad(u)) + v * u) * dx
    f = Function(V)
    f.interpolate((1+8*pi*pi)*cos(x*pi*2)*cos(y*pi*2))
    L = f * v * dx
    fem_soln = Function(V)
    sp = {"mat_type": "matfree",
          "ksp_monitor_true_residual": None,
          "ksp_converged_reason": None}
    with offloading():
        solve(a == L, fem_soln, solver_parameters=sp)

    f.interpolate(cos(x*pi*2)*cos(y*pi*2))

    assert norm(fem_soln-f) < 1e-2

    with offloading():
        assert norm(fem_soln-f) < 1e-2


@pytest.mark.parametrize("offloading_backend", [cuda_backend, cpu_backend])
def test_linear_variational_solver(offloading_backend):
    set_offloading_backend(offloading_backend)
    mesh = UnitSquareMesh(32, 32)
    V = FunctionSpace(mesh, "CG", 1)
    u = TrialFunction(V)
    v = TestFunction(V)
    f = Function(V)
    x, y = SpatialCoordinate(mesh)
    f.interpolate((1+8*pi*pi)*cos(x*pi*2)*cos(y*pi*2))

    L = assemble(f * v * dx)
    fem_soln = Function(V)

    with offloading():
        a = assemble((dot(grad(v), grad(u)) + v * u) * dx, mat_type='matfree')
        solve(a, fem_soln, L)

    f.interpolate(cos(x*pi*2)*cos(y*pi*2))

    assert norm(fem_soln-f) < 1e-2

    with offloading():
        assert norm(fem_soln-f) < 1e-2


@pytest.mark.parametrize("offloading_backend", [cuda_backend, cpu_backend])
def test_data_manipulation_on_host(offloading_backend):
    set_offloading_backend(offloading_backend)

    mesh = UnitSquareMesh(32, 32)
    V = FunctionSpace(mesh, "CG", 1)
    u = TrialFunction(V)
    v = TestFunction(V)
    f = Function(V)
    x, y = SpatialCoordinate(mesh)
    f.interpolate((1+8*pi*pi)*cos(x*pi*2)*cos(y*pi*2))

    L = assemble(f * v * dx)
    fem_soln = Function(V)

    with offloading():
        a = assemble((dot(grad(v), grad(u)) + v * u) * dx, mat_type='matfree')
        solve(a, fem_soln, L)

    fem_soln_host = fem_soln.dat.data
    kappa = 2.0
    fem_soln_host *= kappa  # update data on host

    with offloading():
        error = assemble(((dot(grad(v), grad(fem_soln)) + v * fem_soln)-kappa*f*v)*dx,
                         mat_type='matfree')

        assert norm(error) < 1e-6

    assert norm(error) < 1e-6

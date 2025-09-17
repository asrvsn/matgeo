'''
Gradient flows of chemical free energies
'''

import numpy as np
import meshio
from typing import Callable, Tuple
from functools import partial

import gmsh
import basix
import basix.ufl
import dolfinx
import dolfinx.fem
import dolfinx.fem.petsc
import dolfinx.io
import dolfinx.plot
import ufl
from ufl.core.expr import Expr as UFLExpr
from petsc4py import PETSc
from mpi4py import MPI
import pyvista

from ..triangulation import Triangulation
from ..domains.utils import get_exterior_tags

def monomer_mobility(D_m: float, phi_m: UFLExpr) -> UFLExpr:
    ''' Aqueous monomer mobility '''
    assert D_m > 0, 'Monomer diffusion coefficient must be positive'
    return D_m * phi_m

def polymer_mobility(D_p: float, phi_p: UFLExpr) -> UFLExpr:
    ''' Aqueous polymer mobility '''
    assert D_p > 0, 'Polymer diffusion coefficient must be positive'
    return D_p * phi_p * (1 - phi_p)

def polymerization_reaction(k_r: float, phi_m: UFLExpr, phi_p: UFLExpr) -> UFLExpr:
    ''' Simple autocatalytic polymerization '''
    assert k_r > 0, 'Polymerization reaction rate must be positive'
    return k_r * phi_m * phi_p

def polymerization_free_energy(N_m: float, N_p: float, chi_pm: float, chi_pw: float, phi_m: UFLExpr, phi_p: UFLExpr) -> UFLExpr:
    ''' Polymerization free energy '''
    assert 0 < N_m < np.inf, 'Monomer number must be positive'
    assert 0 < N_p <= np.inf, 'Polymer number must be positive'
    assert chi_pm <= 0, 'Flory-Huggins attractive coefficient for monomer-polymer interaction must be non-positive'
    assert chi_pw > 0, 'Flory-Huggins repulsive coefficient for polymer-water (eff. polymer-non polymer) interaction must be positive'
    # Entropy
    expr = - phi_m * ufl.log(phi_m) / N_m 
    if not np.isclose(N_p, np.inf):
        expr += - phi_p * ufl.log(phi_p) / N_p # Don't include the subexpression to avoid activating autodiff
    # Flory-Huggins
    expr += chi_pw * phi_p * (1 - phi_p)
    if not np.isclose(chi_pm, 0):
        expr += chi_pm * phi_m * phi_p # Don't include the subexpression to avoid activating autodiff
    return expr

def monomer_steady_shape(
        mesh: dolfinx.mesh.Mesh,
        D_m: float,
        q_m: float,
    ) -> dolfinx.fem.Function:
    '''
    Model a steady-state monomer distribution shape given influx q_m.
    '''
    assert D_m > 0, 'Monomer diffusion coefficient must be positive'
    assert q_m > 0, 'Monomer production rate must be positive'

    # Constants
    ft = get_exterior_tags(mesh)
    ds = ufl.Measure('ds', domain=mesh, subdomain_data=ft)
    dx = ufl.dx(domain=mesh)
    HOLES = 1
    def compute_scalar(expr: UFLExpr) -> float:
        return dolfinx.fem.assemble_scalar(dolfinx.fem.form(expr))
    q_net = compute_scalar(-q_m * ds(HOLES))
    area = compute_scalar(1.0 * dx)

    # Solve problem
    V = dolfinx.fem.functionspace(mesh, ('Lagrange', 1))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    a = ufl.inner(u, v) * dx
    L = (
        - q_net / (area * D_m) * v * dx # interior
        + (q_m / D_m) * v * ds(HOLES) # boundary
    )

    problem = dolfinx.fem.petsc.LinearProblem(a, L, bcs=[], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    h_sol = problem.solve()
    return h_sol


def polymerization_induced_phase_separation(
        mesh: dolfinx.mesh.Mesh,
        E_fun: Callable[[UFLExpr, UFLExpr], UFLExpr], # Free energy phi_m, phi_p -> E
        R_fun: Callable[[UFLExpr, UFLExpr], UFLExpr], # Reaction term phi_m, phi_p -> R
        M_m_fun: Callable[[UFLExpr], UFLExpr], # Mobility phi_m -> M_m
        M_p_fun: Callable[[UFLExpr], UFLExpr], # Mobility phi_p -> M_p
        kappa: float, # Ginzburg-Landau capillarity parameter for polymer phase
        q_m: float, # Monomer production rate at the boundary
        dt: float=1e-2,
        theta: float=1/2, # Theta method for time integration (theta=0 explicit, theta=1/2 Crank-Nicolson, theta=1 implicit)
    ):
    '''
    Model a generic single-component aqueous polymerization reaction.
    Ginzburg-Landau term present in the free energy for the polymer phase.
    
    Args:
        mesh: The dolfinx mesh
        E: Function that takes (phi_m, phi_p) and returns the free energy density
        R: Function that takes (phi_m, phi_p) and returns the reaction term R
        M_m: Function that takes (phi_m) and returns the mobility M_m
        M_p: Function that takes (phi_p) and returns the mobility M_p
        kappa: Ginzburg-Landau capillarity parameter
        q_m: Monomer production rate at the boundary
        dt: Time step
        theta: Crank-Nicolson (theta=0 explicit, theta=1 implicit)
    '''
    # Validation
    assert kappa > 0, 'Interface penalty must be positive'
    
    # Boundary measure: use exterior facets since the box is periodic
    ft = get_exterior_tags(mesh)
    ds = ufl.Measure('ds', domain=mesh, subdomain_data=ft)
    HOLES = 1
    
    # Elements and function spaces
    P1 = basix.ufl.element('Lagrange', mesh.basix_cell(), 1, dtype=dolfinx.default_real_type)
    W = dolfinx.fem.functionspace(mesh, (P1, P1, P1)) # phi_m, phi_p, mu_p

    # Unknowns
    u = dolfinx.fem.Function(W) # current solution
    u0 = dolfinx.fem.Function(W) # solution from previous converged step
    
    phi_m, phi_p, mu_p = ufl.split(u)
    phi_m0, phi_p0, mu_p0 = ufl.split(u0)

    # Test functions
    v_m, v_p, v_mu_p = ufl.TestFunctions(W)

    # Weak form
    ## Theta-method for intermediate expressions
    R_mid = (1 - theta) * R_fun(phi_m0, phi_p0) + theta * R_fun(phi_m, phi_p)

    E = E_fun(phi_m, phi_p)
    E0 = E_fun(phi_m0, phi_p0)
    mu_m = ufl.diff(E, phi_m) # Chemical potential mu_m has no space derivatives
    mu_m0 = ufl.diff(E0, phi_m0)

    J_m_mid = (1 - theta) * M_m_fun(phi_m0) * ufl.grad(mu_m0) + theta * M_m_fun(phi_m) * ufl.grad(mu_m)
    J_p_mid = (1 - theta) * M_p_fun(phi_p0) * ufl.grad(mu_p0) + theta * M_p_fun(phi_p) * ufl.grad(mu_p)

    ## Functionals governing unknowns
    F_phi_m = (
        ufl.inner(phi_m - phi_m0, v_m) * ufl.dx + dt * (
            ufl.inner(R_mid, v_m) * ufl.dx + 
            ufl.inner(J_m_mid, ufl.grad(v_m)) * ufl.dx + 
            -(q_m * v_m * ds(HOLES)) # Implements J_m . n = -q_m on boundaries
        ) 
    )

    F_phi_p = (
        ufl.inner(phi_p - phi_p0, v_p) * ufl.dx + dt * (
            -ufl.inner(R_mid, v_p) * ufl.dx + 
            ufl.inner(J_p_mid, ufl.grad(v_p)) * ufl.dx
            # Zero-flux BCs for phi_p (implicitly satisfied)
        )
    )

    ### Ginzburg-Landau contribution for phi_p requires another equation
    F_mu_p = (
        ufl.inner(mu_p, v_mu_p) * ufl.dx - 
        ufl.inner(ufl.diff(E, phi_p), v_mu_p) * ufl.dx - 
        kappa * ufl.inner(ufl.grad(phi_p), ufl.grad(v_mu_p)) * ufl.dx
        # Zero-flux BCs for grad phi_p (implicitly satisfied)
    )

    F = F_phi_m + F_phi_p + F_mu_p

    # Construct problem
    use_superlu = PETSc.IntType == np.int64  # or PETSc.ScalarType == np.complex64
    sys = PETSc.Sys()  
    if sys.hasExternalPackage("mumps") and not use_superlu:
        linear_solver = "mumps"
    elif sys.hasExternalPackage("superlu_dist"):
        linear_solver = "superlu_dist"
    else:
        linear_solver = "petsc"
    petsc_options = {
        "snes_type": "newtonls",
        "snes_linesearch_type": "none",
        "snes_stol": np.sqrt(np.finfo(dolfinx.default_real_type).eps) * 1e-2,
        "snes_atol": 0,
        "snes_rtol": 0,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": linear_solver,
        "snes_monitor": None,
    }
    problem = NonlinearProblem(
        F, u, petsc_options_prefix="PIPS_", petsc_options=petsc_options
    )

    # Solve IVP

if __name__ == '__main__':
    from ..points.matern import matern_II_torus
    from ..points.cvt import gradient_flow_cvt_torus
    from ..domains.periodic import swiss_cheese
    from ..domains.utils import visualize_field, visualize_exterior_tags

    rng = np.random.default_rng(42)

    r = 0.1
    xs = matern_II_torus(1000, r, rng=rng)
    xs = gradient_flow_cvt_torus(xs, 10, 1.0)

    model = swiss_cheese(xs, np.full(xs.shape[0], r), boundary_elements=50)
    # model.visualizeMesh()
    
    # After swiss_cheese(), the gmsh model is automatically active in gmsh.model
    mesh, _, _ = dolfinx.io.gmshio.model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=2)
    print('Created mesh with', mesh.geometry.x.shape[0], 'vertices')

    D_m = 1.0
    q_m = 1.0

    w_m = monomer_steady_shape(mesh, D_m, q_m)
    print('Computed steady-state monomer shape.')
    visualize_field(mesh, w_m)

    # visualize_exterior_tags(mesh)

    # chi_pm: float = 1.0 # Flory-Huggins attractive coefficient for monomer-polymer interaction
    # chi_pw: float = 1.0 # Flory-Huggins repulsive coefficient for polymer-water (eff. polymer-non polymer) interaction

    # polymerization_induced_phase_separation(
    #     mesh,
    #     partial(polymerization_free_energy, chi_pm=chi_pm, chi_pw=chi_pw),
    #     partial(polymerization_reaction, k_r=k_r),
    #     partial(monomer_mobility, D_m=D_m),
    #     partial(polymer_mobility, D_p=D_p),
    #     kappa=kappa,
    #     q_m=q_m,
    #     dt=1e-2,
    #     theta=1/2,
    # )
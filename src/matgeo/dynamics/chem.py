'''
Gradient flows of chemical free energies
'''

import numpy as np
import meshio
from typing import Callable, Tuple

import gmsh
import basix
import basix.ufl.element
import dolfinx
import dolfinx.fem
import ufl
from petsc4py import PETSc

from ..triangulation import Triangulation

def get_inclusion_tags(mesh: dolfinx.mesh.Mesh) -> dolfinx.mesh.meshtags:
    '''
    Get the tags for the inclusions in the mesh
    '''
    tdim = mesh.topology.dim
    mesh.topology.create_connectivity(tdim-1, tdim)
    facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)
    vals = np.ones(len(facets), dtype=np.int32)
    return dolfinx.mesh.meshtags(mesh, tdim-1, facets, vals)

def polymerization_induced_phase_separation(
        mesh: dolfinx.mesh.Mesh,
        E_fun: Callable[[ufl.core.expr.Expr, ufl.core.expr.Expr], ufl.core.expr.Expr], # Free energy phi_m, phi_p -> E
        R_fun: Callable[[ufl.core.expr.Expr, ufl.core.expr.Expr], ufl.core.expr.Expr], # Reaction term phi_m, phi_p -> R
        M_m_fun: Callable[[ufl.core.expr.Expr], ufl.core.expr.Expr], # Mobility phi_m -> M_m
        M_p_fun: Callable[[ufl.core.expr.Expr], ufl.core.expr.Expr], # Mobility phi_p -> M_p
        kappa: float, # Ginzburg-Landau capillarity parameter
        q_m: float, # Monomer production rate at the boundary
        dt: float=1e-2,
        theta: float=1/2, # Crank-Nicolson (theta=0 explicit, theta=1 implicit)
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
    '''
    # Validation
    assert kappa > 0, 'Interface penalty must be positive'
    
    # Boundary measure: use exterior facets since the box is periodic
    ft = get_inclusion_tags(mesh)
    ds = ufl.Measure('ds', domain=mesh, subdomain_data=ft)
    HOLES = 1
    
    # Elements and function spaces
    P1 = basix.ufl.element('Lagrange', mesh.basix_cell(), 1, dtype=dolfinx.default_real_type)
    W = dolfinx.fem.FunctionSpace(mesh, (P1, P1, P1)) # phi_m, phi_p, mu_p

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
    F_phi_m = 
        ufl.inner(phi_m - phi_m0, v_m) * ufl.dx + dt * (
            ufl.inner(R_mid, v_m) * ufl.dx + 
            ufl.inner(J_m_mid, ufl.grad(v_m)) * ufl.dx + 
            -(q_m * v_m * ds(HOLES)) # Implements J_m . n = -q_m on boundaries
        ) 

    F_phi_p = 
        ufl.inner(phi_p - phi_p0, v_p) * ufl.dx + dt * (
            -ufl.inner(R_mid, v_p) * ufl.dx + 
            ufl.inner(J_p_mid, ufl.grad(v_p)) * ufl.dx
            # Zero-flux BCs for phi_p (implicitly satisfied)
        )

    ### Ginzburg-Landau contribution for phi_p requires another equation
    F_mu_p = 
        ufl.inner(mu_p, v_mu_p) * ufl.dx - 
        ufl.inner(ufl.diff(E, phi_p), v_mu_p) * ufl.dx - 
        kappa * ufl.inner(ufl.grad(phi_p), ufl.grad(v_mu_p)) * ufl.dx
        # Zero-flux BCs for grad phi_p (implicitly satisfied)

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
        "snes_stol": np.sqrt(np.finfo(default_real_type).eps) * 1e-2,
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

    rng = np.random.default_rng(42)

    r = 0.1
    xs = matern_II_torus(1000, r, rng=rng)
    xs = gradient_flow_cvt_torus(xs, 10, 1.0)

    model = swiss_cheese(xs, np.full(xs.shape[0], r), boundary_elements=50)
    # model.visualizeMesh()
    mesh = dolfinx.io.gmshio.model_to_mesh(model)
    ft = get_inclusion_tags(mesh)
    print('Created mesh.')

    fac_topo, fac_types, fac_geom = dolfinx.plot.vtk_mesh(mesh, mesh.topology.dim-1)
    # fac_topo is VTK connectivity: [n0, i0,i1,  n1, i2,i3, ...] for lines
    offset = 0
    xs, ys = [], []
    while offset < len(fac_topo):
        n = fac_topo[offset]; i0 = fac_topo[offset+1]; i1 = fac_topo[offset+2]
        offset += 1 + n
        pts = fac_geom[[i0, i1]]
        xs.append(pts[:,0]); ys.append(pts[:,1])

    plt.figure(figsize=(6,6))
    for k,(x,y) in enumerate(zip(xs,ys)):
        c = 'C1' if k in set(ft.indices) else 'k'
        plt.plot(x, y, c, lw=1.5)
    plt.gca().set_aspect('equal'); plt.axis('off')
    plt.show()

    # chi_pm: float = 1.0 # Flory-Huggins attractive coefficient for monomer-polymer interaction
    # chi_pw: float = 1.0 # Flory-Huggins repulsive coefficient for polymer-water (eff. polymer-non polymer) interaction

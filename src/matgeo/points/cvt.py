'''
Utilities relating to Centroidal Voronoi tessellations and Lloyd's algorithm
'''
import numpy as np
from typing import Tuple
from tqdm import tqdm

from asrvsn_math.torus import pdist_td

import matgeo.hmc_cpp as hmc_cpp
from ..plane import PlanarPolygon
from ..torus import voronoi_flat_torus

def U_grad_cvt_torus(x: np.ndarray, scale: float=1.0) -> Tuple[float, np.ndarray]:
    '''
    Potential energy and gradient in two dimensions
    Takes x in R^2, calculates Voronoi using wraparound to [0, 1)^2
    Returns energy and gradient
    '''
    assert x.ndim == 2
    assert x.shape[1] == 2, 'x must be 2D'
    vertices, regions = voronoi_flat_torus(x)
    polygons = [PlanarPolygon(vertices[region]) for region in regions]
    areas = np.array([p.area() for p in polygons])
    centroids = np.array([p.centroid() for p in polygons])
    assert x.shape == centroids.shape
    energy = np.sum([p.trace_M2(x_i) for p, x_i in zip(polygons, x)]) / 2
    grad = (x - centroids) * areas[:, None]
    energy *= scale
    grad *= scale
    return energy, grad

def gradient_flow_cvt_torus(
        x: np.ndarray,
        n_steps: int,
        dt: float,
        rho: float = 1.0,
    ):
    '''
    Gradient flow on the torus
    '''
    assert n_steps > 0
    assert x.ndim == 2 and x.shape[1] == 2
    assert dt > 0
    scale = (rho ** 2) / x.shape[0]

    for _ in tqdm(range(n_steps)):
        U, gradU = U_grad_cvt_torus(x, scale=scale)
        x -= dt * gradU
        x = x % 1.0
    return x

def sample_boltzmann_cvt_torus(
        x: np.ndarray,
        n_steps: int,
        dt: float,
        rho: float = 1.0,
        r: float = 0.0,
        beta: float = 1.0,
        L: int = 10,
        progress: bool = True,
        rng: np.random.Generator = np.random.default_rng(),
    ):
    """
    HMC on [0,1)^2 with periodic CVT energy and hard-core reflections (disk radius r).
    Target ∝ exp(-beta * (rho^2/n) * U_CVT(x)).
    """
    assert n_steps > 0
    assert x.ndim == 2 and x.shape[1] == 2
    assert dt > 0 and rho > 0 and r >= 0 and (0 < beta < np.inf)

    n = x.shape[0]
    scale = (rho ** 2) / n

    # --- Reflection kernel tolerances ---
    tol_len   = 1e-12   # length tolerance
    tol_time  = 1e-14   # time tolerance
    max_events = 64

    # Use an effective integrator step that scales for stiffness
    eps = dt / np.sqrt(beta)         # critical for large beta
    L_eff = max(1, int(L))           # keep your L; total length τ = L_eff * eps

    x = (x % 1.0).astype(np.float64, copy=False)

    acc = 0
    steps = tqdm(range(n_steps)) if progress else range(n_steps)
    for _ in steps:
        # Energy & gradient at current state (scaled)
        U, gradU = U_grad_cvt_torus(x, scale=scale)

        # Draw momentum (mass = I)
        p0 = rng.normal(size=x.shape).astype(np.float64, copy=False)

        # Half-kick
        x_prop = np.ascontiguousarray(x.copy(), dtype=np.float64)
        p_prop = np.ascontiguousarray(p0 - 0.5 * eps * beta * gradU, dtype=np.float64)

        # L leapfrog steps with reflective drift each substep
        for l in range(L_eff):
            # DRIFT with elastic/specular reflections (in-place on x_prop, p_prop)
            hmc_cpp.specular_reflect_torus(
                x_prop, p_prop, eps, r, max_events, tol_len, tol_time
            )

            # KICK (full except at the very end)
            U_new, gradU_new = U_grad_cvt_torus(x_prop, scale=scale)
            if l != L_eff - 1:
                p_prop -= eps * beta * gradU_new

        # Final half-kick
        p_prop -= 0.5 * eps * beta * gradU_new

        # Metropolis–Hastings accept/reject
        H_old = beta * U + 0.5 * np.sum(p0 ** 2)
        H_new = beta * U_new + 0.5 * np.sum(p_prop ** 2)
        log_alpha = H_old - H_new
        if np.log(rng.uniform()) < min(0.0, log_alpha):
            x = x_prop
            acc += 1

        if progress:
            steps.set_postfix_str(f"acc={(acc/(steps.n+1)):.2f}, U={U_new:.4e}, eps={eps:.2e}")

    return x
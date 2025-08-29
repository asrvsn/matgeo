'''
Utilities relating to Centroidal Voronoi tessellations and Lloyd's algorithm
'''
import numpy as np
from typing import Tuple
from tqdm import tqdm

from asrvsn_math.torus import pdist_td

from ..plane import PlanarPolygon
from ..torus import voronoi_flat_torus

def U_grad_torus(x: np.ndarray, scale: float=1.0) -> Tuple[float, np.ndarray]:
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

def sample_boltzmann_torus(
        x: np.ndarray,
        n_steps: int,
        dt: float,
        rho: float=1.0, 
        r: float=0.0,
        beta: float=1.0,  
        L: int=10, 
        progress: bool=False,
        rng: np.random.Generator=np.random.default_rng(),
    ):
    '''
    Gradient flow of the CVT energy on the torus at specified number density rho 
    and hard-core radius r
    '''
    assert n_steps > 0
    assert x.ndim == x.shape[1] == 2
    assert dt > 0
    assert rho > 0
    assert r >= 0
    assert 0 < beta < np.inf, 'Temperature must be finite'
    n = x.shape[0]
    scale = (rho ** 2) / n

    steps = tqdm(range(n_steps)) if progress else range(n_steps)
    for _ in steps:
        ''' 
        Proposal: Hamiltonian Monte Carlo with leapfrog integrator
        '''
        U, gradU = U_grad_torus(x, scale=scale)
        p = rng.normal(size=x.shape) # Mass matrix is identity
        x_ = x.copy()
        p_ = p - 0.5 * dt * beta * gradU # Take initial half-step in momentum
        for L_i in range(L):
            x_ = (x_ + dt * p_) % 1.0
            U_, gradU_ = U_grad_torus(x_, scale=scale)
            if L_i != L - 1:
                p_ -= dt * beta * gradU_
        p_ -= 0.5 * dt * beta * gradU_ # Take final half-step in momentum
        '''
        Metropolis-Hastings step
        '''
        H = beta * U + 0.5 * (p ** 2).sum()
        H_ = beta * U_ + 0.5 * (p_ ** 2).sum()
        log_alpha = H - H_
        if np.log(rng.uniform()) < min(0.0, log_alpha):
            x = x_
    return x
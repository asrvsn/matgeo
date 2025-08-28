'''
Utilities relating to Centroidal Voronoi tessellations and Lloyd's algorithm
'''
import numpy as np
from typing import Tuple
from tqdm import tqdm

from ..plane import PlanarPolygon
from ..torus import voronoi_flat_torus

def U_grad_torus(x: np.ndarray) -> Tuple[float, np.ndarray]:
    '''
    Potential energy and gradient in two dimensions
    Takes x in [0, 1]^2
    Returns energy and gradient
    '''
    assert x.ndim == 2
    assert x.shape[1] == 2, 'x must be 2D'
    x = x % 1
    vertices, regions = voronoi_flat_torus(x)
    polygons = [PlanarPolygon(vertices[region]) for region in regions]
    areas = np.array([p.area() for p in polygons])
    centroids = np.array([p.centroid() for p in polygons])
    assert x.shape == centroids.shape
    energy = np.sum([p.trace_M2(x_i) for p, x_i in zip(polygons, x)])
    grad = 2 * (x - centroids) * areas[:, None]
    return energy, grad

def gradient_flow_torus(
        x: np.ndarray,
        n_steps: int,
        dt: float,
        rho: float=1.0, 
        r: float=0.0,    
        progress: bool=False,
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
    d = 2 # Dimension
    n = x.shape[0]

    scale = (n / rho) ** (1 + 2 / d)
    y = x
    steps = tqdm(range(n_steps)) if progress else range(n_steps)
    for _ in steps:
        _, dydt = -scale * U_grad_torus(y)
        if r > 0:
            # Handle hard-core constraint
            pass
        y += dt * dydt
    return y
'''
Stuff on the torus [0, 1]^d
'''
import numpy as np
from scipy.spatial import Voronoi
from scipy.spatial.distance import cdist, pdist

from .surface import *
from .plane import *

class SquareTorus(Surface):
    '''
    Periodic unit square R^2 / Z^2
    '''
    @property
    def ndim(self) -> int:
        return 2

    def voronoi_tessellate(self, pts: np.ndarray) -> 'SquareTorusPartition':
        verts, regions = voronoi_flat_torus(pts)
        return [PlanarPolygon(verts[region]) for region in regions]
    
    def poisson_voronoi_tessellate(self, lam: float) -> 'SquareTorusPartition':
        n = np.random.poisson(lam)
        pts = np.random.uniform(0, 1, (n, 2))
        return self.voronoi_tessellate(pts)

class SquareTorusPolygon(PlanarPolygon):
    pass

class SquareTorusPartition(SurfacePartition):
    pass

'''
Helper functions
'''

def dists_td(xs: np.ndarray) -> np.ndarray:
    '''
    Pairwise Euclidean distance in [0, 1]^d respecting periodic boundary conditions
    Returns a n x n dense matrix
    '''
    assert xs.ndim in [1, 2], 'xs must be 1D or 2D'
    xs = xs % 1
    d = 1 if xs.ndim == 1 else xs.shape[1]
    d_dists = [] # Distance in dimension d
    for i in range(d):
        xs_d = (xs[:, i] if xs.ndim == 2 else xs)[:, None]
        d_mat = cdist(xs_d, xs_d)
        d_mat = np.minimum(d_mat, 1 - d_mat)
        d_dists.append(d_mat)
    dists = np.sqrt(np.array(d_dists).sum(axis=0))
    return dists

def pdists_td(xs: np.ndarray) -> np.ndarray:
    '''
    Same as above but only returns the upper triangular part (using scipy.spatial.distance.pdist)
    '''
    assert xs.ndim in [1, 2], 'xs must be 1D or 2D'
    xs = xs % 1
    d = 1 if xs.ndim == 1 else xs.shape[1]
    d_dists = [] # Distance in dimension d
    for i in range(d):
        xs_d = (xs[:, i] if xs.ndim == 2 else xs)[:, None]
        ds = pdist(xs_d)
        ds = np.minimum(ds, 1 - ds)
        d_dists.append(ds)
    dists = np.sqrt(np.array(d_dists).sum(axis=0))
    return dists

def voronoi_flat_torus(pts: np.ndarray) -> Tuple[np.ndarray, list]:
    '''
    #TODO: use CGAL instead
    Voronoi tessellation on flat torus [0, 1)^2. Returns:
    - Vertices to render
    - Regions to render (1-1 with input points)
    # - Translates points to [0, 1)^2
    '''
    assert pts.ndim == 2
    assert pts.shape[1] == 2
    N = pts.shape[0]
    # pts -= np.min(pts, axis=0)
    assert (pts.min() >= 0).all() and (pts.max() < 1).all(), 'Could not translate points to [0, 1)^2'
    pts_ext = np.concatenate([
        pts,
        pts + np.array([1, 0]),
        pts + np.array([0, 1]),
        pts + np.array([1, 1]),
        pts + np.array([-1, 0]),
        pts + np.array([0, -1]),
        pts + np.array([-1, -1]),
        pts + np.array([1, -1]),
        pts + np.array([-1, 1]),
    ], axis=0)
    
    vor = Voronoi(pts_ext)
    # Get regions corresponding to original points
    regions = [vor.regions[vor.point_region[i]] for i in range(N)]
    return vor.vertices, regions
'''
Stuff on the torus [0, 1]^d
'''
import numpy as np
from scipy.spatial import Voronoi
from scipy.spatial.distance import cdist, pdist

from .surface import *
from .plane import *

class Torus(Surface):
    '''
    Periodic unit square R^2 / Z^2
    '''
    @property
    def ndim(self) -> int:
        return 2

    def voronoi_tessellate(self, pts: np.ndarray) -> 'TorusPartition':
        verts, partitions = voronoi_flat_torus(pts)
        return TorusPartition(self, verts, partitions, pts)
    
    def poisson_voronoi_tessellate(self, lam: float, rng=np.random.default_rng()) -> 'TorusPartition':
        n = rng.poisson(lam)
        pts = rng.uniform(0, 1, (n, 2))
        return self.voronoi_tessellate(pts)

class TorusPolygon(PlanarPolygon):
    pass

class TorusPartition(SurfacePartition):
    def grad_second_moment(self) -> Tuple[np.ndarray, np.ndarray]:
        # TODO faster loop?
        grad_verts = np.zeros_like(self.vertices_nd)
        grad_seeds = np.zeros_like(self.seeds_nd)
        for i, (partition, seed) in enumerate(zip(self.partitions, self.seeds_nd)):
            vertices = self.vertices_nd[partition]
            grad_verts[partition] += grad_polygon_second_moment_vertices(vertices, seed)
            grad_seeds[i] = grad_polygon_second_moment_origin(vertices, seed)
        return grad_verts, grad_seeds
            
'''
Helper functions
'''

def voronoi_flat_torus(pts: np.ndarray) -> Tuple[np.ndarray, list]:
    '''
    #TODO: use CGAL instead
    Voronoi tessellation on flat torus [0, 1)^2. Returns:
    - Vertices to render
    - Regions to render (1-1 with input points)
    - Takes points in R^2, returns the corresponding Voronoi polygons (this _is_ the minimal-image unwrapping of T^2).
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
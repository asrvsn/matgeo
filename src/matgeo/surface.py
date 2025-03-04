'''
Surfaces and objects on surfaces
'''
from abc import ABC, abstractmethod
from typing import Optional, Tuple, List
import numpy as np

class Surface(ABC):
    '''
    Surface embedded in ambient Euclidean d-dimensional space
    '''
    @property
    @abstractmethod
    def ndim(self) -> int:
        ''' Dimension of ambient space '''
        pass

    @abstractmethod
    def voronoi_tessellate(self, pts: np.ndarray) -> 'SurfacePartition':
        ''' Compute the geodesic Voronoi tessellation '''
        pass

class SurfacePolygon(ABC):
    '''
    Polygon represented by vertices on a 2D surface, connected by geodesics on that surface.
    '''
    def __init__(self, vertices_nd: np.ndarray, surface: Optional[Surface]=None, check: bool=True):
        '''
        vertices: embedded coordinates of surface in d-dimensional space
        check: whether to check validity of the vertices
        '''
        if not (surface is None):
            assert surface.ndim >= 3 # If surface is provided, it must be at least ambient 3D
        self.vertices_nd = vertices_nd
        self.surface = surface

    @property
    def n(self) -> int:
        return self.vertices_nd.shape[0]
    
    @property
    def ndim(self) -> int:
        ''' Dimension of ambient space '''
        return self.vertices_nd.shape[1]
    
    def save(self, path: str) -> None:
        ''' Save polygon to file '''
        np.save(path, self.vertices_nd)

    @abstractmethod
    def load(self, path: str) -> 'SurfacePolygon':
        pass

    @abstractmethod
    def nth_moment(self, n: int, center=None, standardized: bool=False):
        '''
        Compute nth moment of the surface area about a given center.
        (About the centroid, normalized 1st moment, if None.)
        '''
        pass

    def area(self) -> float:
        return self.nth_moment(0)

    def centroid(self) -> np.ndarray:
        return self.nth_moment(1) / self.area()
    
class SurfacePartition(ABC):
    def __init__(self, surface: Surface, vertices_nd: np.ndarray, partitions: np.ndarray, seeds_nd: np.ndarray):
        '''
        vertices_nd: embedded coordinates of surface in d-dimensional space
        partitions: indices of vertices in each partition (representing a SurfacePolygon)
        seeds_nd: dual of the tessellation, in some sense

        It's assumed the vertices form valid polygons.
        '''
        assert len(partitions) == len(seeds_nd)
        self.surface = surface
        self.vertices_nd = vertices_nd
        self.partitions = partitions
        self.seeds_nd = seeds_nd

    @abstractmethod
    def grad_second_moment(self) -> Tuple[np.ndarray, np.ndarray]:
        '''
        Compute (covariant) gradient of the second-moment functional

        \sum_i \int_{D_i} d_g(x, x_i)^2 dx 
        
        where d_g(x, x_i) is the geodesic distance from x to x_i,
        D_i are surface polygons defined by self.vertices_nd, and
        x_i are self.seeds_nd
        
        Returns:
        (1) gradient with respect to self.vertices_nd
        (2) gradient with respect to self.seeds_nd
        '''
        pass

    @property
    @abstractmethod
    def polygons(self) -> List[SurfacePolygon]:
        '''
        Convert the partition indices into polygons
        '''
        pass

    @abstractmethod
    def refine(self, k: int) -> 'SurfacePartition':
        '''
        Refine edges in the partition using additional points.
        '''
        pass
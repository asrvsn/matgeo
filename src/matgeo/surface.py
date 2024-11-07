'''
Surfaces and objects on surfaces
'''
from abc import ABC, abstractmethod
import numpy as np

class Surface(ABC):
    pass

class SurfacePolygon(ABC):
    '''
    Polygon represented by vertices on a 2D surface, 
    connected by geodesics on that surface.
    The main purpose of this class is to provide methods to compute the 
    nth moments of the surface area of this polygon.
    '''
    def __init__(self, vertices_emb: np.ndarray):
        '''
        vertices: embedded coordinates of surface in d-dimensional space
        '''
        self.vertices_emb = vertices_emb

    @property
    def n(self) -> int:
        return self.vertices_emb.shape[0]
    
    @property
    def ndim(self) -> int:
        ''' Dimension of ambient space '''
        return self.vertices_emb.shape[1]
    
    def save(self, path: str) -> None:
        ''' Save polygon to file '''
        np.save(path, self.vertices_emb)

    @abstractmethod
    def nth_moment(self, n: int, center=None, standardized: bool=False):
        '''
        Compute nth moment of the surface area about a given center.
        (About the centroid, normalized 1st moment, if None.)
        '''
        pass

    def area(self) -> float:
        '''
        Calculate area of polygon.
        '''
        return self.nth_moment(0)
    
    def centroid(self) -> np.ndarray:
        '''
        Calculate first normalized moment (center of mass) of polygon.
        '''
        return self.nth_moment(1) / self.area()
'''
Delaunay triangulations
'''

import numpy as np
from typing import Tuple
from scipy.spatial import SphericalVoronoi, ConvexHull, Delaunay

def delaunay_chull(pts: np.ndarray) -> np.ndarray:
    '''
    Compute the outer faces of the Delaunay triangulation of a given set of points by taking their intersection with the convex hull.
    Returns the indices of the simplices (nfaces, 3)
    '''
    tri = Delaunay(pts)
    simplices = tri.convex_hull
    return simplices
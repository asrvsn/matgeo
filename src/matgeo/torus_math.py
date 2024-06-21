'''
Stuff on the torus [0, 1]^d
'''
import numpy as np
from scipy.spatial.distance import cdist, pdist

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
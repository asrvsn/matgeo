'''
Utilities for manipulating polygons and polyhedra
'''

import numpy as np
from typing import Generator, Tuple, List, Union
from shapely.geometry import Polygon, LineString, MultiPolygon
from shapely.geometry.polygon import orient
from sklearn.neighbors import KDTree
from scipy.ndimage import binary_erosion, binary_dilation

from .array import flatten_list

''' General helpers '''

def face_normals(vertices: np.ndarray, faces: list) -> np.ndarray:
    ''' Compute the face normals '''
    normals = []
    for f in faces:
        v1, v2, v3 = vertices[f[:3]]
        n = np.cross(v2-v1, v3-v2)
        n /= np.linalg.norm(n)
        normals.append(n)
    return np.array(normals)

def face_areas(vertices: np.ndarray, faces: list) -> np.ndarray:
    ''' Compute the face areas '''
    areas = np.zeros(len(faces))
    for f_i, f in enumerate(faces):
        A1AK = vertices[f[1:-1]] - vertices[f[0]]
        A1AK_ = vertices[f[2:]] - vertices[f[0]]
        cross = np.cross(A1AK, A1AK_, axis=1)
        areas[f_i] = np.linalg.norm(cross.sum(axis=0)) / 2
    return areas

def unique_edges(faces: list) -> np.ndarray:
    ''' Compute the unique edges '''
    edges = []
    seen = set()
    for f in faces:
        for i in range(len(f)):
            e = [f[i], f[(i+1)%len(f)]]
            if not (frozenset(e) in seen):
                seen.add(frozenset(e))
                edges.append(e)
    return np.array(edges)

def traverse_face_edges(faces: list) -> Generator:
    ''' Compute the incident face pairs '''
    seen = set()
    for i in range(len(faces)):
        for j in range(len(faces)):
            if i == j:
                continue
            e = [i, j]
            if not (frozenset(e) in seen):
                seen.add(frozenset(e))
                if len(set(faces[i]) & set(faces[j])) == 2: # Faces share an edge
                    yield e

def face_edges(faces: list) -> np.ndarray:
    return np.array([e for e in traverse_face_edges(faces)])

def face_adjacency(faces: list) -> dict:
    ''' Compute the adjacency list of faces '''
    adj = {}
    for e in traverse_face_edges(faces):
        i, j = e
        if not i in adj:
            adj[i] = [j]
        else:
            adj[i].append(j)
        if not j in adj:
            adj[j] = [i]
        else:
            adj[j].append(i)
    return adj

def face_neighbors(faces: list) -> np.ndarray:
    ''' Compute the number of faces incident to each face'''
    adj = face_adjacency(faces)
    return np.array([len(adj[f_i]) for f_i in range(len(faces))])

def edge_lengths(vertices: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)

def mean_curvature(face_normals: np.ndarray, face_adj: dict) -> np.ndarray:
    ''' Compute mean curvature at faces '''
    mc = np.zeros(len(face_normals))
    for f_i, f_n in enumerate(face_normals):
        mc[f_i] = sum([
            1 - np.dot(f_n, face_normals[f_j]) 
            for f_j in face_adj[f_i]
        ]) / len(face_adj[f_i])
    return mc

def dual_polyhedron(vertices: np.ndarray, faces: list, dual_vertices=None) -> Tuple[np.ndarray, list]:
    '''
    Compute the vertex-face dual of a given polyhedron
    '''
    if dual_vertices is None:
        dual_vertices = np.array([np.mean(vertices[f], axis=0) for f in faces]) # Dual vertices are centroids
    assert len(dual_vertices) == len(faces)
    dual_faces = []
    for v_i, v in enumerate(vertices):
        fs_ = [f_i for f_i, f in enumerate(faces) if v_i in f]
        fs = [fs_.pop()]
        # Order face
        while len(fs_) > 0:
            f1 = faces[fs[-1]]
            for f2_i, f2 in enumerate(fs_):
                f2 = faces[f2]
                if len(set(f1) & set(f2)) == 2: # Faces share an edge
                    fs.append(fs_.pop(f2_i))
                    break
        # Orient face CCW
        fv1, fv2, fv3 = dual_vertices[fs[0]], dual_vertices[fs[1]], dual_vertices[fs[2]]
        n = np.cross(fv2-fv1, fv3-fv2)
        if np.dot(n, v) < 0:
            fs = fs[::-1]
        # Add face
        dual_faces.append(np.array(fs, dtype=np.intp))
    assert len(dual_faces) == len(vertices)
    return dual_vertices, dual_faces

''' Shapely polygon helpers '''

def polygonize_mask(mask: np.ndarray, buffer: int=0) -> Polygon:
    """Takes a set of xy coordinates pp Numpy array(n,2) and reorders the array to make a polygon using a nearest neighbor approach.
    Adapted from https://stackoverflow.com/questions/34968838/python-finite-boundary-voronoi-cells
    Returned polygon is oriented.
    """
    mask = mask.astype(bool)
    assert buffer >= 0
    # find coordinates of contour
    if buffer > 0:
        mask = binary_dilation(mask, iterations=buffer)
    contour = mask ^ binary_erosion(mask)
    pp = np.array(np.where(contour==1)[::-1]).T

    # start with first index
    pp_new = np.zeros_like(pp)
    pp_new[0] = pp[0]
    p_current_idx = 0

    tree = KDTree(pp)

    for i in range(len(pp) - 1):

        nearest_dist, nearest_idx = tree.query([pp[p_current_idx]], k=4)  # k1 = identity
        nearest_idx = nearest_idx[0]

        # finds next nearest point along the contour and adds it
        for min_idx in nearest_idx[1:]:  # skip the first point (will be zero for same pixel)
            if not pp[min_idx].tolist() in pp_new.tolist():  # make sure it's not already in the list
                pp_new[i + 1] = pp[min_idx]
                p_current_idx = min_idx
                break

    pp_new[-1] = pp[0]
    return orient(Polygon(pp_new))

def to_simple_polygons(p: Union[Polygon, MultiPolygon]) -> List[Polygon]:
    '''
    Split complex polygons (possibly with holes) into simple polygons
    Inspired by https://github.com/fgassert/split_donuts
    '''
    if p.geom_type == 'Polygon':
        if len(p.interiors) > 0:
            pt = p.interiors[0].centroid
            # Split horizontally at pt
            nx, ny, xx, xy = p.bounds
            lEnv = LineString([(nx, ny), (pt.x, xy)]).envelope
            rEnv = LineString([(pt.x, ny), (xx, xy)]).envelope
            lp, rp = p.intersection(lEnv), p.intersection(rEnv)
            return to_simple_polygons(lp) + to_simple_polygons(rp)
        else:
            return [p]
    elif p.geom_type == 'MultiPolygon':
        return flatten_list([to_simple_polygons(q) for q in p.geoms])
    else:
        raise ValueError(f'Unsupported geometry type: {p.geom_type}')
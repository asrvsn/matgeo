'''
NO INTERNAL DEPENDENCIES except utility functions
'''

import numpy as np
import scipy as sp
from scipy.spatial import Voronoi, voronoi_plot_2d
import sys
from typing import Tuple, List
import shapely
from shapely.ops import polygonize, unary_union
from shapely.geometry import LineString, MultiPolygon, MultiPoint, Point, Polygon
from scipy.spatial import ConvexHull
import pdb
import matplotlib.pyplot as plt

import matgeo.voronoi_cpp as vor_cpp
from .utils.poly import polygonize_mask

eps = sys.float_info.epsilon

def in_box(coords, bounding_box):
    return np.logical_and(np.logical_and(bounding_box[0] <= coords[:, 0],
                                         coords[:, 0] <= bounding_box[1]),
                          np.logical_and(bounding_box[2] <= coords[:, 1],
                                         coords[:, 1] <= bounding_box[3]))


def bounded_voronoi(bounds: Tuple[float, float], coords: np.ndarray) -> Voronoi:
    ''' Construct a Voronoi diagram with guaranteed finite polygons within a bounding box '''
    assert bounds[0] > 0
    assert bounds[1] > 0
    bounding_box = np.array([0, bounds[0], 0, bounds[1]])
    # Select coords inside the bounding box
    i = in_box(coords, bounding_box)
    # Mirror points
    points_center = coords[i, :]
    points_left = np.copy(points_center)
    points_left[:, 0] = bounding_box[0] - (points_left[:, 0] - bounding_box[0])
    points_right = np.copy(points_center)
    points_right[:, 0] = bounding_box[1] + (bounding_box[1] - points_right[:, 0])
    points_down = np.copy(points_center)
    points_down[:, 1] = bounding_box[2] - (points_down[:, 1] - bounding_box[2])
    points_up = np.copy(points_center)
    points_up[:, 1] = bounding_box[3] + (bounding_box[3] - points_up[:, 1])
    points = np.append(points_center,
                       np.append(np.append(points_left,
                                           points_right,
                                           axis=0),
                                 np.append(points_down,
                                           points_up,
                                           axis=0),
                                 axis=0),
                       axis=0)
    # Compute Voronoi
    vor = Voronoi(points)
    # Filter regions
    regions = []
    for region in vor.regions:
        flag = True
        for index in region:
            if index == -1:
                flag = False
                break
            else:
                x = vor.vertices[index, 0]
                y = vor.vertices[index, 1]
                if not(bounding_box[0] - eps <= x and x <= bounding_box[1] + eps and
                       bounding_box[2] - eps <= y and y <= bounding_box[3] + eps):
                    flag = False
                    break
        if region != [] and flag:
            regions.append(region)
    vor.filtered_points = points_center
    vor.filtered_regions = regions
    return vor

def voronoi_finite_polygons_2d(vor: Voronoi, radius=None) -> Tuple[np.ndarray, list]:
    """
    Reconstruct infinite voronoi regions in a 2D diagram to finite regions.
    adapted from https://stackoverflow.com/questions/36063533/clipping-a-voronoi-diagram-python/43023639#43023639
    Parameters
    ----------
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.
    Returns
    -------
    vertices : 2d array 
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.
    regions : list of tuples
        Indices of vertices in each revised Voronoi regions.
    """

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()*2

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1] # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return np.asarray(new_vertices), new_regions

def mask_bounded_voronoi(pts: np.ndarray, mask: np.ndarray, buffer: int=0) -> Tuple[np.ndarray, list]:
    '''
    Compute the intersection of a 2D voronoi diagram with a mask.
    '''
    assert pts.ndim == 2
    assert pts.shape[1] == 2
    boundary = polygonize_mask(mask, buffer=buffer)
    return poly_bounded_voronoi(pts, boundary)

def chull_bounded_voronoi(pts: np.ndarray, buffer: int=0) -> Tuple[np.ndarray, list, np.ndarray]:
    '''
    Compute the intersection of a 2D voronoi diagram with the convex hull.
    '''
    assert pts.ndim == 2
    assert pts.shape[1] == 2
    boundary = ConvexHull(pts).vertices
    boundary = pts[boundary]
    boundary = Polygon(boundary).buffer(buffer)
    return poly_bounded_voronoi(pts, boundary)

def region_to_polygon(region: np.ndarray) -> Polygon:
    shape = list(region.shape)
    shape[0] += 1
    region = np.append(region, region[0]).reshape(*shape)
    return Polygon(region)

def poly_bounded_voronoi(pts: np.ndarray, boundary: Polygon, filter_pts: bool=True, strictly_contained: bool=False) -> Tuple[np.ndarray, np.ndarray, list, np.ndarray]:
    '''
    Compute the intersection of a 2D voronoi diagram with a polygon.
    adapted from https://stackoverflow.com/questions/34968838/python-finite-boundary-voronoi-cells
    Returns:
    - pts: seed points
    - vertices: (n,2) array of voronoi vertices
    - regions: (F,~) list of indices of vertices that make up each voronoi region in 1-1 correspondence with pts if strictly_contained=True
    - on_boundary: (F,) boolean array indicating whether each region is on the boundary (True) or not (False)
    '''
    # returns a list of the seeds that are contained within the polygon
    pts_ = []
    for i, pt in enumerate(pts):
        if boundary.contains(Point(pt)) or (not filter_pts):
            pts_.append(pt)
        else:
            assert not strictly_contained, 'Point not contained in boundary but strictly_contained=True'
    pts = np.array(pts_)
    assert pts.shape[0] > 3, 'Not enough points contained in mask'

    vor = Voronoi(pts)
    vertices, regions = voronoi_finite_polygons_2d(vor)
    assert len(regions) == pts.shape[0], 'Not enough regions'
    
    #clips tesselation to the mask
    new_vertices = []
    new_regions = []
    vertex_map = dict()
    on_boundary = []
    for region in regions:
        poly_reg = vertices[region]
        shape = list(poly_reg.shape)
        shape[0] += 1
        p = Polygon(np.append(poly_reg, poly_reg[0]).reshape(*shape))
        on_boundary.append(boundary.intersects(p) and not boundary.contains(p))
        p = p.intersection(boundary)
        p = shapely.geometry.polygon.orient(p, sign=1.0) # Orient CCW
        poly = [tuple(c) for c in p.exterior.coords]
        for coord in poly:
            if coord not in vertex_map:
                vertex_map[coord] = len(new_vertices)
                new_vertices.append(coord)
        region = [vertex_map[coord] for coord in poly[:-1]]
        new_regions.append(region)
    
    if filter_pts:
        assert len(pts) == len(new_regions)
    return pts, np.array(new_vertices), new_regions, np.array(on_boundary)
    
    # #plots the results
    # fig, ax = plt.subplots()
    # ax.imshow(mask,cmap='Greys_r')
    # for poly in new_vertices:
    #     ax.fill(*zip(*poly), alpha=0.7)
    # ax.plot(pts[:,0],pts[:,1],'ro',ms=2)
    # plt.show()
    
    # lines = [
    #     LineString(vor.vertices[line])
    #     for line in vor.ridge_vertices if -1 not in line
    # ]
    # result = MultiPolygon([poly.intersection(boundary) for poly in polygonize(lines)])
    # result = [p for p in result.geoms] + [p for p in boundary.difference(unary_union(result)).geoms]

    # vertices = []
    # faces = []
    # for poly in result:
    #     poly = shapely.geometry.polygon.orient(poly, sign=1.0)
    #     # pdb.set_trace()
    #     coords = list(zip(poly.boundary.coords.xy[0][:-1], poly.boundary.coords.xy[1][:-1]))
    #     for coord in coords:
    #         if coord not in vertex_map:
    #             vertex_map[coord] = len(vertices)
    #             vertices.append(coord)
    #     face = np.array([vertex_map[coord] for coord in coords], dtype=np.intp)
    #     faces.append(face)

    
    # return np.array(vertices), faces
            

    # plt.plot(pts[:,0], pts[:,1], 'ko')
    # for r in result.geoms:
    # 	plt.fill(*zip(*np.array(list(
    # 		zip(r.boundary.coords.xy[0][:-1], r.boundary.coords.xy[1][:-1])
    # 		))),
    # 		alpha=0.4)
    # plt.show()

## Slow / stupid version
# def voronoi_flat_torus(pts: np.ndarray) -> Tuple[np.ndarray, list, np.ndarray]:
#     '''
#     Voronoi tessellation on flat torus [0, 1)^2. Returns:
#     - Vertices to render
#     - Regions to render
#     - Region areas (1-1 with regions)
#     '''
#     assert pts.ndim == 2
#     assert pts.shape[1] == 2
#     assert (pts.min() >= 0).all() and (pts.max() <= 1).all()
#     pts_ext = np.concatenate([
#         pts,
#         pts + np.array([1, 0]),
#         pts + np.array([0, 1]),
#         pts + np.array([1, 1]),
#         pts + np.array([-1, 0]),
#         pts + np.array([0, -1]),
#         pts + np.array([-1, -1]),
#         pts + np.array([1, -1]),
#         pts + np.array([-1, 1]),
#     ], axis=0)
    
#     vor = Voronoi(pts_ext)
#     vertices, regions = voronoi_finite_polygons_2d(vor)
#     boundary = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])

#     # Retains only those regions and vertices that periodically tessellate the flat torus
#     vertices_ = []
#     regions_ = []
#     areas = []
#     vertex_map = dict()

#     def add_poly(poly):
#         poly = shapely.geometry.polygon.orient(poly, sign=1.0) # Orient CCW
#         coords = [tuple(c) for c in p.exterior.coords]
#         for coord in coords:
#             if coord not in vertex_map:
#                 vertex_map[coord] = len(vertices_)
#                 vertices_.append(coord)
#         face = np.array([vertex_map[coord] for coord in coords], dtype=np.intp)
#         regions_.append(face)
#         areas.append(poly.area)

#     lb = LineString([(0, 0), (0, 1)])
#     rb = LineString([(1, 0), (1, 1)])
#     tb = LineString([(0, 1), (1, 1)])
#     bb = LineString([(0, 0), (1, 0)])
    
#     # Take only those regions intersecting the left and lower boundaries, avoiding double-counting the one at the top-left corner
#     for region in regions:
#         poly_reg = vertices[region]
#         shape = list(poly_reg.shape)
#         shape[0] += 1
#         p = Polygon(np.append(poly_reg, poly_reg[0]).reshape(*shape))
#         if p.intersects(boundary):
#             if boundary.contains(p):
#                 add_poly(p)
#             else:
#                 if p.intersects(lb) and not p.intersects(tb):
#                     add_poly(p)
#                 elif p.intersects(bb) and not p.intersects(rb):
#                     add_poly(p)

#     assert len(regions_) == len(areas)
#     return np.array(vertices_), regions_, np.array(areas)
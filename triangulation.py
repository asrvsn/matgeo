'''
Triangulations of surfaces
'''

import os
import numpy as np
from typing import Tuple, List, Dict, Set
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist
import pdb
from tqdm import tqdm
import potpourri3d as pp3d
import openmesh as omesh
import joblib as jl
import multiprocessing as mp
import pickle
import alphashape
import CGAL
from compas_cgal.reconstruction import poisson_surface_reconstruction, advancing_front_surface_reconstruction

from plane import Plane, PlanarPolygon
from parallel import parallel_farm_array

'''
Classes
'''

Incidence = Dict[int, List[int]]
Edges = Set[frozenset]

class Triangulation:

    def __init__(self, pts: np.ndarray, simplices: np.ndarray):
        '''
        pts: coordinates in d-dimensional space
        simplices: t x 3 indices of simplex coordinates
        '''
        assert pts.ndim == 2
        self.pts = pts
        self.simplices = simplices

        # Geodesic stuff
        self.d_ij = None # Cache of pre-computed geodesic distances

        # Incidence structure
        self.incidence: Incidence = None
        self.edges: Edges = None

    @property
    def n(self):
        return self.pts.shape[0]

    @property
    def ndim(self):
        return self.pts.shape[1]

    def compute_simplex_areas(self) -> np.ndarray:
        '''
        Return a t x 1 array of simplex areas
        '''
        assert self.ndim in [2, 3], 'Cross-product area calculation only implemented for 2d and 3d'
        AB = self.pts[self.simplices[:, 1]] - self.pts[self.simplices[:, 0]]
        AC = self.pts[self.simplices[:, 2]] - self.pts[self.simplices[:, 0]]
        areas = np.linalg.norm(np.cross(AB, AC), axis=1) / 2
        return areas
    
    def area(self) -> float:
        '''
        Return the total area of the triangulation
        '''
        return self.compute_simplex_areas().sum()

    def translate(self, v):
        '''
        Translate the triangulation by v
        '''
        self.pts += v

    def compute_circumcenters(self) -> np.ndarray:
        '''
        Compute the circumcenters of the simplices
        '''
        assert self.ndim == 3, 'Circumcenters only implemented for 3d'
        vts = self.pts[self.simplices]
        cps = circumcenter_3d(vts[:, 0], vts[:, 1], vts[:, 2])
        return cps
    
    def vor_dualize(self) -> Tuple[List[np.ndarray], np.ndarray]:
        '''
        Dualize the triangulation by Voronoi rule applied to quasi-2D patches about each vertex
        '''
        polygons = []
        centers = []
        for i in tqdm(range(self.pts.shape[0]), desc='Dualizing'):
            # Get all simplices incident to vertex
            idx = np.unique(np.where(self.simplices == i)[0])
            simplices_i = self.simplices[idx]
            # Create adjacency structure of the ring of vertices {j} around i
            adj = dict()
            for s in simplices_i:
                u, v = s[s != i]
                if u in adj:
                    adj[u].append(v)
                else:
                    adj[u] = [v]
                if v in adj:
                    adj[v].append(u)
                else:
                    adj[v] = [u]
            # Traverse ring 
            js = []
            j0 = list(adj.keys())[0]
            j_ = None # Previous
            j = j0
            while j_ is None or j != j0:
                # Add j to the list of coordinates
                js.append(j)
                # Get next that isn't previous
                if j_ is not None:
                    adj[j].remove(j_)
                j_ = j
                j = adj[j].pop()
            # Take planar approximation of curved patch
            patch = self.pts[[i] + js]
            plane = Plane.fit_l2(patch)
            # Dualize patch in 2D using circumcenter rule (geodesics are straight lines)
            patch = plane.project_l2(patch)
            center, ring = patch[0], patch[1:]
            N = len(ring)
            poly = np.array([
                circumcenter_3d(center, ring[j], ring[(j+1)%N]) for j in range(N)
            ])
            # Construct valid polygons from quasi-2D voronoi tessellation. Vor polygons are convex, so use the convex hull if constructing invalid ones.
            polygons.append(PlanarPolygon(poly, use_chull_if_invalid=True, check=True))
            centers.append(center)
        return polygons, np.array(centers)
    
    def orient_origin(self):
        '''
        Orient the simplices so their cross product is outward-pointing
        '''
        normals = self.compute_normals()
        centroids = self.compute_centroids()
        # Compute dot product of normals with vectors from centroid to origin
        signs = np.sign((normals * centroids).sum(axis=1))
        # Flip simplices with negative dot product
        self.simplices[signs < 0] = self.simplices[signs < 0][:, ::-1]

    def compute_normals(self) -> np.ndarray:
        '''
        Compute the normals of the simplices
        '''
        assert self.ndim == 3, 'Normals only implemented for 3d'
        AB = self.pts[self.simplices[:, 1]] - self.pts[self.simplices[:, 0]]
        AC = self.pts[self.simplices[:, 2]] - self.pts[self.simplices[:, 1]]
        normals = np.cross(AB, AC)
        return normals
    
    def compute_centroids(self) -> np.ndarray:
        '''
        Compute the centroids of the simplices
        '''
        return self.pts[self.simplices].mean(axis=1)
    
    def geodesic_distance(self, i: int, j: int) -> float:
        '''
        Compute the geodesic distance between two points
        '''
        return self.all_geodesic_distances()[i, j]
    
    def all_geodesic_distances(self, parallel=True, cache: str=None) -> np.ndarray:
        if self.d_ij is None:
            if not cache is None:
                self.d_ij = np.load(cache)
                print(f'Loaded geodesic distances from cache: {cache}')
            else:
                n = len(self.pts)
                if parallel:
                    setup_fun = lambda: pp3d.MeshHeatMethodDistanceSolver(self.pts, self.simplices)
                    compute_fun = lambda solver, i: solver.compute_distance(i)
                    self.d_ij = parallel_farm_array(setup_fun, compute_fun, range(n))
                else:
                    solver = pp3d.MeshHeatMethodDistanceSolver(self.pts, self.simplices)
                    self.d_ij = np.array(list(
                        tqdm((solver.compute_distance(i) for i in range(n)), total=n, desc='Computing geodesic distances')
                    ))
                if cache is not None:
                    np.save(cache, self.d_ij)
        return self.d_ij
    
    def max_geodesic_distance(self) -> float:
        return self.all_geodesic_distances().max()
    
    def geodesic_disk(self, i: int, r: float, cond=np.all) -> 'Triangulation':
        '''
        Compute the sub-triangulation corresponding to the approximate geodesic disk around a point
        '''
        assert cond in [np.any, np.all], 'Condition must be np.any or np.all'
        # Get the indices of all points within the geodesic contour
        d_ij = self.all_geodesic_distances()
        j_keep = d_ij[i] <= r
        # Filter the triangles
        simplices = self.simplices
        simplices = simplices[cond(j_keep[simplices], axis=1)]
        # Construct the geodesic contour
        return Triangulation(self.pts, simplices)
    
    def get_graph(self) -> Tuple[Incidence, Edges]:
        '''
        Compute the incidence structure of the triangulation
        '''
        if self.incidence is None:
            edges = set()
            for simplex in self.simplices:
                for i in range(3):
                    edge = frozenset([simplex[i], simplex[(i + 1) % 3]])
                    edges.add(edge)
            incidence = dict()
            for (i, j) in edges:
                for (u, v) in [(i, j), (j, i)]:
                    if u in incidence:
                        incidence[u].append(v)
                    else:
                        incidence[u] = [v]
            self.incidence = incidence
            self.edges = edges
        return self.incidence, self.edges
    
    def get_degree(self) -> np.ndarray:
        '''
        Compute the degree of each vertex
        '''
        incidence, _ = self.get_graph()
        return np.array([len(incidence[i]) for i in range(self.n)])
    
    def subdivide(self, n: int, mode: str='modified_butterfly') -> 'Triangulation':
        '''
        Refine the triangulation by either interpolating or approximating subdivision, creating 4**n as many triangles as the original 
        Modes:
        - "modified_butterfly":
            Interpolating Subdivision for Meshes with Arbitrary Topology
            https://dl.acm.org/doi/pdf/10.1145/237170.237254
        - "catmull_clark":
            Recursively generated B-spline surfaces on arbitrary topological meshes
            https://www.sciencedirect.com/science/article/pii/0010448578901100
        - "loop":
            Smooth Subdivision Surfaces Based on Triangles
            https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/thesis-10.pdf

        These subdivision methods _seem_ to have the property that the first n points 
        correspond exactly to the original mesh points.
        '''
        assert n >= 0
        tri = self.copy()
        mesh = omesh.TriMesh()
        mesh.add_vertices(tri.pts)
        mesh.add_faces(tri.simplices)
        if mode == 'modified_butterfly':
            mesh.subdivide_modified_butterfly(n)
        elif mode == 'catmull_clark':
            mesh.subdivide_catmull_clark(n)
        elif mode == 'loop':
            mesh.subdivide_loop(n)
        else:
            raise ValueError(f'Unknown subdivision mode: {mode}')
        tri.pts = mesh.points()
        tri.simplices = mesh.face_vertex_indices()
        return tri

    def copy(self) -> 'Triangulation':
        '''
        Return a copy of the triangulation
        '''
        return Triangulation(self.pts.copy(), self.simplices.copy())
    
    def match_pts(self, pts: np.ndarray) -> np.ndarray:
        '''
        Get indices of closest points in pts to each point in self.pts
        '''
        return np.argmin(cdist(self.pts, pts), axis=0)
    
    def rdf(
            self, 
            from_verts: np.ndarray, 
            to_verts: np.ndarray,
            n_bins=50
        ) -> Tuple[np.ndarray, np.ndarray]:
        '''
        Compute the radial distribution function between given source-destination pairs
        '''
        # Get indices of all points to compare
        n_from, n_to = len(from_verts), len(to_verts)
        d_ij = self.all_geodesic_distances()
        d_ij_ng = d_ij[from_verts][:, to_verts]
        r_max = d_ij_ng.max(axis=0).min() # Cut off where we don't see any more pairs
        # Compute the histogram of distances from each point
        def setup_fun():
            return self, d_ij
        def hist_fun(ctx, from_idx):
            tri, d_ij = ctx
            i = from_verts[from_idx]
            j_no_i = to_verts[to_verts != i]
            d_ij_ng = d_ij[i][j_no_i]
            hist, bin_edges = np.histogram(d_ij_ng, bins=n_bins, range=(0, r_max), density=False)
            hist = hist.astype(float)
            # Approximate the geodesic contour area for each bin
            shell_areas = np.array([
                tri.geodesic_disk(i, r, cond=np.all).area() for r in bin_edges
            ])
            bin_areas = np.diff(shell_areas) # dA
            assert (bin_areas > 0).all(), 'dA !> 0 for all r, refine the mesh or decrease n_bins'
            # Compute the radial distribution function
            hist = hist / bin_areas
            return hist
        g = parallel_farm_array(setup_fun, hist_fun, range(n_from), name=f'RDF')
        assert g.shape == (n_from, n_bins)
        # Normalize by number density (assuming homogeneity)
        g /= n_to / self.area()
        _, bin_edges = np.histogram(np.zeros(1), bins=n_bins, range=(0, r_max), density=False)
        bin_midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2
        return g, bin_midpoints

    def export(self, folder: str, name: str):
        '''
        Export the triangulation as (vertices, triangles) 
        '''
        assert os.path.isdir(folder), f'Folder {folder} does not exist'
        np.savetxt(os.path.join(folder, f'{name}.vertices.txt'), self.pts)
        np.savetxt(os.path.join(folder, f'{name}.triangles.txt'), self.simplices, fmt='%d')
        print(f'Exported {self.pts.shape[0]} vertices and {self.simplices.shape[0]} triangles to {folder}')

    @staticmethod
    def surface_3d(pts: np.ndarray, method='advancing_front', **kwargs) -> 'Triangulation':
        '''
        Extract a surface triangulation from a Delaunay tetrahedralization
        using one of several possible methods to extract triangles correspoding to a "surface"
        '''
        if method == 'alpha_shape':
            assert 'alpha' in kwargs, 'Alpha parameter must be specified for alpha shape method'
            mesh = alphashape.alphashape(pts, kwargs['alpha'])
            return Triangulation(mesh.vertices, mesh.faces)
        elif method == 'advancing_front':
            V, F = advancing_front_surface_reconstruction(pts)
            return Triangulation(V, F)
        elif method == 'poisson':
            assert 'normals' in kwargs, 'Normals must be specified for Poisson surface reconstruction'
            assert pts.shape == kwargs['normals'].shape, 'Points and normals must have the same shape'
            V, F = poisson_surface_reconstruction(pts, kwargs['normals'])
            return Triangulation(V, F)
        else:
            raise ValueError(f'Unknown surface extraction method: {method}')

class FaceTriangulation(Triangulation):

    def __init__(self, pts: np.ndarray, simplices: np.ndarray, faces: np.ndarray):
        '''
        Triangulation with support for curved faces consisting of multiple simplices
        '''
        super().__init__(pts, simplices)
        self.faces = faces

    def label_simplices(self) -> np.ndarray:
        '''
        Return an t x 1 array of labels for each simplex, labeled by face (e.g. for coloring faces)
        '''
        labels = np.zeros(self.simplices.shape[0], dtype=np.intp)
        for i, face in enumerate(self.faces):
            labels[face] = i
        return labels

    def compute_face_areas(self) -> np.ndarray:
        '''
        Return a f x 1 array of face areas, computed by adding areas of the constituent simplices
        '''
        simplex_areas = self.compute_simplex_areas()
        areas = np.zeros(len(self.faces))
        for i, face in enumerate(self.faces):
            areas[i] = simplex_areas[face].sum()
        return areas

    def planar_approximation(self) -> Tuple[List[PlanarPolygon], np.ndarray]:
        '''
        Approximate the faces by best-fit planar polygons and corresponding centers
        '''
        polygons = []
        centers = []
        for face in self.faces:
            N = len(face)
            # Get all vertices of face
            indices, counts = np.unique(self.simplices[face], return_counts=True)
            # Remove center vertex (every face consists of pairs of incident simplices, thus each vertex appears twice, except for the center vertex)
            assert counts.max() == N
            center_index = indices[counts == N][0]
            indices, counts = indices[counts < N], counts[counts < N]
            assert (counts == 2).all()
            # Get coordinates of vertices
            coords = self.pts[indices]
            center = self.pts[center_index]
            # Compute best-fit plane
            plane = Plane.fit_l2(coords)
            # Project vertices onto plane
            coords = plane.embed(plane.project_l2(coords))
            center = plane.embed(plane.project_l2(center[None]))[0]
            # Construct polygon from un-oriented coordinates
            polygon = PlanarPolygon.from_pointcloud(coords)
            polygons.append(polygon)
            centers.append(center.copy())
        return polygons, np.array(centers)

    def planar_faces(self) -> Tuple[List[np.ndarray], np.ndarray]:
        '''
        Approximate the faces by best-fit planar polygons and corresponding centers (in 3d)
        '''
        polygons = []
        centers = []
        for face in self.faces:
            N = len(face)
            # Get all vertices of face
            indices, counts = np.unique(self.simplices[face], return_counts=True)
            # Remove center vertex (every face consists of pairs of incident simplices, thus each vertex appears twice, except for the center vertex)
            assert counts.max() == N
            center_index = indices[counts == N][0]
            indices, counts = indices[counts < N], counts[counts < N]
            assert (counts == 2).all()
            # Get coordinates of vertices
            coords = self.pts[indices]
            center = self.pts[center_index]
            # Compute best-fit plane
            plane = Plane.fit_l2(coords)
            # Project vertices onto plane
            coords = plane.project_l2(coords)
            center = plane.project_l2(center[None])[0]
            coords_embed = plane.embed(coords)
            # Orient vertices
            hull = ConvexHull(coords_embed)
            coords = coords[hull.vertices]
            # Construct polygon 
            polygons.append(coords.copy())
            centers.append(center.copy())
        return polygons, np.array(centers)

    def planar_moments(self, n: int) -> np.ndarray:
        '''
        Compute the nth area moments of planar approximations of the polygonal faces.
        '''
        assert n in [0, 1, 2], 'Only 0th, 1st, and 2nd moments are implemented'
        if n == 0:
            '''
            Return areas of the planar faces
            '''
            pass
        
    @staticmethod
    def from_polygons(polygons: List[np.ndarray], centers: np.ndarray) -> 'FaceTriangulation':
        '''
        Compute a face-triangulation from a set of polygons and their centers.
        Arguments:
        - polygons: n-length list of (var) x d arrays of polygon vertices (These must be oriented CCW or CW!)
        - centers: n x d array of polygon centers
        '''
        assert len(polygons) == len(centers)
        pts = []
        n_pts = 0
        simplices = []
        n_simplices = 0
        faces = []
        for poly, center in zip(polygons, centers):
            # Add points from polygon and center
            pts.extend(poly.tolist())
            pts.append(center)
            s_c = n_pts + len(poly) # Index of face center
            face = []
            # Construct simplices assuming polygons are consistently oriented
            for i in range(len(poly) - 1):
                j = (i + 1) % len(poly)
                s_i, s_j = n_pts + i, n_pts + j # Indices of simplex vertices
                simplex = [s_i, s_j, s_c]
                simplices.append(simplex)
                face.append(n_simplices)
                n_simplices += 1
            faces.append(np.array(face))
            n_pts += len(poly) + 1

        pts = np.array(pts)
        simplices = np.array(simplices)
        assert len(faces) == len(polygons)
        return FaceTriangulation(pts, simplices, faces)


'''
Classes & functions for constructing Voronoi tessellations of curved closed surfaces embedded in R3
'''

class VoronoiTriangulation(FaceTriangulation):

    def __init__(self, pts: np.ndarray, simplices: np.ndarray, faces: list, circumcenters: np.ndarray=None, midpoints: np.ndarray=None):
        '''
        pts: coordinates in 3-dimensional space
        simplices: t x 3 indices of simplex coordinates
        faces: f x (variable) list of simplex indices constituting curved faces
        circumcenters (optional): t x 1 indices of circumcenter coordinates
        midpoints (optional): t x 1 indices of midpoint coordinates
        '''
        FaceTriangulation.__init__(self, pts, simplices, faces)
        assert pts.shape[1] == 3
        self.circumcenters = circumcenters
        self.midpoints = midpoints

    def compute_neighbors(self) -> np.ndarray:
        '''
        Return a f x 1 array of the number of neighboring polygons.
        '''
        nbs = np.array([len(face) // 2 for face in self.faces]) # Each neighboring face contributes two simplices
        return nbs

    @staticmethod
    def from_delaunay(pts: np.ndarray, simplices: np.ndarray) -> 'VoronoiTriangulation':
        '''
        Construct a Voronoi triangulation from a Delaunay triangulation
        The Voronoi triangulation is a vertex-face dual. For every vertex of the Delaunay, there is a face in the Voronoi (represented here in curved form as a decomposition into simplices).
        ''' 
        assert pts.ndim == 2
        assert pts.shape[1] == 3
        assert simplices.ndim == 2
        assert simplices.shape[1] == 3

        # Form vertex-simplex incidence matrix (represented sparsely as a dict)
        simplex_inc = dict()
        for i, simplex in enumerate(simplices):
            for j in simplex:
                if j in simplex_inc:
                    simplex_inc[j].append(i)
                else:
                    simplex_inc[j] = [i]

        # Construct Voronoi sub-triangulation
        vor_circumcenters = [] # Circumcenters of simplices used as "vertices" of the curved Voronoi polygons
        vor_midpoints = [] # Midpoints of edges used as "vertices" of the curved Voronoi polygons
        vor_faces = [] # Voronoi faces consisting of lists of simplex indices
        vor_simplices = [] # Voronoi simplices used to construct the faces
        
        # Add circumcenters of simplices as new points
        for i in range(simplices.shape[0]):
            sp = simplices[i]
            pt_c = circumcenter_3d(*pts[sp])
            vor_circumcenters.append(pt_c)
        vor_circumcenters = np.array(vor_circumcenters)

        # Add midpoints of edges as new points (without double-counting for incident simplices)
        midpoint_indices = dict() # Map from edge (vertex-pair) to midpoint index
        n_edges = 0
        for i in range(simplices.shape[0]):
            sp = simplices[i]
            for j in range(3):
                j_p1, j_p2 = sp[j], sp[(j + 1) % 3]
                edge = frozenset([j_p1, j_p2])
                if not edge in midpoint_indices:
                    pt_m = (pts[j_p1] + pts[j_p2]) / 2
                    vor_midpoints.append(pt_m)
                    midpoint_indices[edge] = n_edges
                    n_edges += 1
        vor_midpoints = np.array(vor_midpoints)
        assert vor_midpoints.shape[0] == n_edges == len(midpoint_indices), 'Number of midpoints does not match number of edges'

        # Final set of Voronoi points
        vor_pts = np.concatenate([pts, vor_circumcenters, vor_midpoints], axis=0)
        n1, n2, n3 = pts.shape[0], vor_circumcenters.shape[0], vor_midpoints.shape[0]

        # Construct Voronoi simplices & faces
        n_simplices = 0
        for j in range(pts.shape[0]): # j indexes points in the Delaunay triangulation
            # Construct a face for each point j
            face = []
            # Obtain re-triangulated face from simplices incident to point
            for i in simplex_inc[j]:
                # For each incident simplex, add 2 new simplices
                sp = simplices[i]
                [j_l, j_r] = [j_ for j_ in sp if j_ != j] # Get the two other points in the simplex
                j_c = n1 + i # Index of circumcenter of simplex
                j_l = n1 + n2 + midpoint_indices[frozenset([j, j_l])] # Index of midpoint of left edge
                j_r = n1 + n2 + midpoint_indices[frozenset([j, j_r])] # Index of midpoint of right edge
                sp_l = [j, j_l, j_c] # Left simplex
                sp_r = [j, j_r, j_c] # Right simplex
                vor_simplices.append(sp_l)
                vor_simplices.append(sp_r)
                face.append(n_simplices)
                face.append(n_simplices + 1)
                n_simplices += 2
            vor_faces.append(face)
        vor_simplices = np.array(vor_simplices, dtype=np.intp)

        # Construct final triangulation
        vor_circumcenter_indices = np.arange(n1, n1 + n2)
        vor_midpoint_indices = np.arange(n1 + n2, n1 + n2 + n3)
        return VoronoiTriangulation(vor_pts, vor_simplices, vor_faces, circumcenters=vor_circumcenter_indices, midpoints=vor_midpoint_indices)

'''
Utility functions
'''

def circumcenter_3d(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> np.ndarray:
    '''
    Circumcenter of triangle in arbitrary position in R3 
    see https://en.wikipedia.org/wiki/Circumcircle#Cartesian_coordinates_from_cross-_and_dot-products
    Supports broadcasting.
    '''
    assert p1.shape == p2.shape == p3.shape
    assert p1.shape[-1] == 3
    div = 2 * np.linalg.norm(np.cross(p1 - p2, p2 - p3), axis=-1)**2
    alpha = np.linalg.norm(p2 - p3, axis=-1)**2 * np.dot(p1 - p2, p1 - p3) / div
    beta = np.linalg.norm(p1 - p3, axis=-1)**2 * np.dot(p2 - p1, p2 - p3) / div
    gamma = np.linalg.norm(p1 - p2, axis=-1)**2 * np.dot(p3 - p1, p3 - p2) / div
    cp = alpha * p1 + beta * p2 + gamma * p3
    return cp

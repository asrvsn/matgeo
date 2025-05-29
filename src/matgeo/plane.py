'''
Functions for fitting planes to data.
'''

from typing import Tuple, List, Optional, Union
import numpy as np
import cvxpy as cp
import numpy.linalg as la
import scipy.optimize as scopt
import shapely
from shapely import Polygon
from shapely.geometry.polygon import orient as shapely_orient
from shapely.validation import explain_validity
from scipy.spatial import ConvexHull, Voronoi
from scipy.spatial.distance import pdist, cdist
from scipy.ndimage import find_objects
from skimage.morphology import binary_erosion, binary_dilation
import skimage
import skimage.measure
import upolygon
import pdb
import shapely
import shapely.geometry
import cv2
import jax.numpy as jnp
import jax

from .surface import *
from .voronoi import poly_bounded_voronoi
from .utils.poly import to_simple_polygons
from .utils import mask as mutil

class Plane(Surface):
    '''
    Hyperplane in d dimensions represented in normal-offset form.
    If in ambient two dimensions, then the "plane" is a line.
    '''
    def __init__(self, n: np.ndarray, v: np.ndarray, basis: Optional[np.ndarray]=None):
        d = n.shape[0]
        assert d >= 2, 'Plane must be at least planar :P'
        nm = la.norm(n)
        assert nm > 0, 'n must be nonzero'
        # Convention is that normal points in direction of positive offset
        if n @ v < 0:
            n = -n
        self.n = n / nm
        self.v = v
        if d == 2:
            # In two dimensions, the plane basis is simply [-y, x] where n = [x, y]
            self.basis = np.array([[-self.n[1], self.n[0]]])
        else:
            # In 3 and higher dimensions, pick a random basis for the plane
            if basis is None:
                e_x = np.cross(self.n, np.random.randn(d))
                assert la.norm(e_x) > 0
                e_x /= la.norm(e_x)
                e_y = np.cross(self.n, e_x)
                assert np.allclose(1, la.norm(e_y))
                self.basis = np.stack([e_x, e_y], axis=-1)
            else:
                assert basis.shape == (d, 2)
                self.basis = basis
            assert np.allclose(self.basis.T @ self.basis, np.eye(2))

    def __add__(self, v: np.ndarray) -> 'Plane':
        ''' Translate the plane '''
        return Plane(self.n, self.v + v)
    
    def __sub__(self, v: np.ndarray) -> 'Plane':
        ''' Translate the plane '''
        return Plane(self.n, self.v - v)
    
    def copy(self) -> 'Plane':
        return Plane(self.n.copy(), self.v.copy(), basis=self.basis.copy())

    @staticmethod
    def fit_l2(X: np.ndarray, tol=1e-3) -> 'Plane':
        '''
        Find best least-squares fit to a set of points.
        '''
        assert X.ndim == 2, 'X must be 2d'
        assert X.shape[0] >= X.shape[1] >= 2, 'X must have at least as many points as dimensions'
        v = np.mean(X, axis=0) # Offset is center of mass
        Xbar = X - v[None, :]
        U, S, V = la.svd(Xbar, full_matrices=False)
        n = V[-1, :]
        return Plane(n, v)

    @staticmethod
    def random(d: int, sigma_dist=10) -> 'Plane':
        '''
        Return a random plane in d dimensions.
        '''
        n = np.random.randn(d)
        v = np.random.randn(d) * sigma_dist
        return Plane(n, v)

    @staticmethod
    def fit_project_l2(X: np.ndarray) -> np.ndarray:
        '''
        Fit and project onto a best plane.
        '''
        plane = Plane.fit_l2(X)
        return plane, plane.project_l2(X)
    
    @staticmethod
    def fit_project_embed_l2(X: np.ndarray) -> np.ndarray:
        '''
        Fit, project, embed points in 2D plane in random basis
        '''
        plane = Plane.fit_l2(X)
        return plane.embed(plane.project_l2(X))

    def project_l2(self, X: np.ndarray) -> np.ndarray:
        '''
        Orthogonal projection of points to plane
        '''
        if X.ndim == 1:
            return X - ((X - self.v) @ self.n) * self.n
        elif X.ndim == 2:
            return X - (X - self.v) @ self.n[:, None] * self.n[None, :]
        else:
            raise ValueError('X must be 1d or 2d')

    def embed(self, X: np.ndarray):
        '''
        Embed ambient points in plane
        '''
        return (X - self.v) @ self.basis

    def reverse_embed(self, X: np.ndarray):
        '''
        Embed planar points in ambient space
        '''
        return X @ self.basis.T + self.v

    def sample_mgrid(self, xlim=(-10,10), ylim=(-10,10), n=100) -> np.ndarray:
        '''
        Sample points on the plane on a grid.
        '''
        xx, yy = np.meshgrid(np.linspace(*xlim, n), np.linspace(*ylim, n))
        zz = (self.b - self.n[0] * xx - self.n[1] * yy) / self.n[2]
        return np.stack([xx, yy, zz], axis=-1)

    def on_side_rhr(self, X: np.ndarray) -> np.ndarray:
        '''
        Return whether points are on the side of the plane corresponding to the right-hand rule.
        '''
        return (X - self.v) @ self.n >= 0
    
    def get_altaz(self) -> Tuple[float, float]:
        ''' Get altitude and azimuth from plane normal (in degrees) ''' 
        az = np.arctan2(self.n[1], self.n[0]) * 180 / np.pi
        alt = np.arcsin(self.n[2]) * 180 / np.pi
        return alt, az
    
    def flip(self) -> 'Plane':
        ''' Flip the plane's orientation '''
        return Plane(-self.n, self.v.copy())
    
    def affine_transform(self, T: callable, v: np.ndarray) -> 'Plane':
        '''
        Apply affine transform T(x) = L(x - v) to plane
        '''
        n = T(self.n.copy() + v) # n is a basepointed vector
        w = T(self.v.copy())
        return Plane(n, w)

    def slope_intercept(self) -> Tuple[float, float]:
        '''
        Return 1D plane in y = mx + c form
        '''
        assert self.ndim == 2, 'Plane must be a line in 2D'
        m = -self.n[0] / self.n[1]
        c = self.v @ self.n / self.n[1]
        return m, c

    def voronoi_tessellate(self, seeds: np.ndarray) -> 'PlanePartition':
        pts = seeds
        if pts.shape[1] > 2:
            assert pts.shape[1] == self.ndim, f'Points must be in same dimension as plane'
            pts = self.embed(pts)
        vor = Voronoi(pts)
        partitions = [r for r in vor.regions if len(r) > 0 and not (-1 in r)]
        vertices = self.reverse_embed(vor.vertices)
        return PlanePartition(
            self.copy(), vertices, partitions, seeds
        )
    
    def __eq__(self, other: 'Plane') -> bool:
        ''' Check if two planes are equal '''
        return np.allclose(self.n, other.n) and np.allclose(self.v, other.v)

    @property
    def ndim(self) -> int:
        '''
        Dimension of ambient space
        '''
        return self.n.shape[0]

    @property
    def b(self) -> float:
        '''
        Offset of plane
        '''
        return self.n @ self.v
    
    @staticmethod
    def XY() -> 'Plane':
        ''' XY plane in 3D '''
        return Plane(np.array([0,0,1]), np.array([0,0,0]), basis=np.array([[1,0],[0,1],[0,0]]))
    
    @staticmethod
    def XZ() -> 'Plane':
        ''' XZ plane in 3D '''
        return Plane(np.array([0,1,0]), np.array([0,0,0]), basis=np.array([[1,0],[0,0],[0,1]]))
    
    @staticmethod
    def YZ() -> 'Plane':
        ''' YZ plane in 3D '''
        return Plane(np.array([1,0,0]), np.array([0,0,0]), basis=np.array([[0,0],[1,0],[0,1]]))
    
class PlanePartition(SurfacePartition):
    def grad_second_moment(self):
        raise NotImplementedError

    def polygons(self):
        return [PlanarPolygon(v, plane=self.surface, check=False) for v in self.vertices_nd]

class PlanarPolygon(SurfacePolygon, Surface):
    '''
    Planar representation of coplanar polygonal points in d >= 2 dimensions.
    A polygon can be both a surface (manifold with boundary) and a polygon on a plane.
    '''
    def __init__(self, 
            vertices: np.ndarray, 
            check: bool=True,
            use_chull_if_invalid: bool=False,
            plane: Optional[Plane]=None,
        ):
        assert vertices.ndim == 2, 'X must be 2d'
        if check:
            assert vertices.shape[0] >= 3, 'X must have at least 3 points'
        nd = vertices.shape[1]
        assert nd >= 2, 'X must be at least 2-dimensional'
        # Compute planar embedding as needed
        if nd > 2:
            if plane is None:
                # Fit the plane if not given
                plane = Plane.fit_l2(vertices)
            vertices = plane.embed(plane.project_l2(vertices)) # Vertices are now coplanar in this plane
        # Check validity and try to recover / complain as needed (uses Shapely)
        if check: 
            poly = Polygon(vertices)
            if not poly.is_valid:
                if use_chull_if_invalid:
                    vertices = vertices[ConvexHull(vertices).vertices]
                    poly = Polygon(vertices)
                else:
                    # Try buffer(0) trick
                    poly = poly.buffer(0)
                    if type(poly) == shapely.geometry.MultiPolygon:
                        # Extract polygon with largest area
                        poly = max(poly.geoms, key=lambda x: x.area)
                    assert type(poly) == shapely.geometry.Polygon
                assert poly.is_valid, 'Polygon is invalid, reason:\n' + explain_validity(poly)
                assert len(poly.exterior.coords) > 3, 'Polygon must have at least 3 vertices'
            # Orient the polygon vertices CCW in 2D
            poly = shapely_orient(poly, sign=1)
            vertices = np.array(poly.exterior.coords)[:-1] # Last point is same as first
        # Compute oriented vertices in ambient space
        if nd == 2:
            plane = None
            vertices_nd = vertices
        else:
            vertices_nd = plane.reverse_embed(vertices)
        # Instance vars
        self.vertices = vertices
        super().__init__(vertices_nd, surface=plane)

    @property
    def ndim(self) -> int:
        ''' Dimension of ambient space '''
        return self.vertices_nd.shape[1]
    
    @property
    def plane(self) -> Plane:
        ''' Plane on which the polygon lies '''
        return self.surface
    
    @property 
    def normal(self) -> np.ndarray:
        ''' Normal vector of the plane '''
        assert self.ndim > 2, 'Normal vector is only defined in dimension d > 2'
        return self.plane.n
    
    def __eq__(self, other: 'PlanarPolygon') -> bool:
        ''' Check if two polygons are equal '''
        return self.plane == other.plane and np.allclose(self.vertices_nd, other.vertices_nd)
    
    def flip(self):
        ''' Flip the normal vector of the plane '''
        self.plane.n *= -1

    def reverse_embed(self, x: np.ndarray) -> np.ndarray:
        ''' Reverse embed a point from the plane '''
        if self.ndim == 2:
            return x
        else:
            return self.plane.reverse_embed(x)
    
    def nth_moment(self, n: int, center=None, standardized: bool=False):
        '''
        Compute nth moment of area with respect to a center.
        If no center is provided, the first moment (center of mass) is used (only applicable for n >= 2).
        Standardization indicates centralized and scale-invariant.
        '''
        if standardized:
            assert n >= 1, 'Standardized moments are only defined for n >= 1'
        X = self.vertices
        if n >= 2:
            if center is None:
                center = self.centroid()
            X = X - center # Calculate nth moment about given center
        X_ = np.roll(X, -1, axis=0) # Next points in sequence (common subexpression) 
        X_cross = X[:, 0] * X_[:, 1] - X[:, 1] * X_[:, 0] # Cross product of adjacent points (common subexpression)
        if n == 0:
            return X_cross.sum() / 2
        elif n == 1:
            M1 = X_cross @ (X + X_) / 6
            if standardized:
                M1 /= self.area()
            return M1
        elif n == 2:
            x, y = X.T
            x_, y_ = X_.T
            I_xx = X_cross @ (x**2 + x*x_ + x_**2) / 12
            I_yy = X_cross @ (y**2 + y*y_ + y_**2) / 12
            I_xy = X_cross @ (2*x*y + x*y_ + x_*y + 2*x_*y_) / 24
            M2 = np.array([
                [I_xx, I_xy],
                [I_xy, I_yy]
            ])
            if standardized:
                M2 /= (self.area() ** 2)
            return M2
            # # Note: X[:,:,None] * Y[:,None,:] is equivalent to einsum('ij,ik->ijk', X, Y)
            # inner = 2 * X_[:,:,None] * X_[:,None,:] \
            #         + X_[:,:,None] * X[:,None,:] \
            #         + X[:,:,None] * X_[:,None,:] \
            #         + 2 * X[:,:,None] * X[:,None,:]
            # inner = X_cross[:,None,None] * inner
            # return inner.sum(axis=0) / 24
        else:
            raise ValueError('moment calculation not implemented for n > 2')

    def circular_radius(self) -> float:
        '''
        Radius of a circle with the same area as the polygon.
        '''
        return np.sqrt(self.area() / np.pi)

    def elliptic_radii(self) -> Tuple[float, float]:
        '''
        Return radii of an ellipse with the same area and aspect ratio as the polygon (in ascending order).
        '''
        ar = self.aspect_ratio()
        a = np.sqrt(self.area() / (np.pi * ar))
        b = ar * a
        return a, b
    
    def elliptic_radius(self, point: np.ndarray) -> float:
        '''
        Elliptic radius in the direction of a test point (given in the same basis as the polygon coordinates).
        '''
        pass

    def principal_axes(self) -> Tuple[np.ndarray, np.ndarray]:
        '''
        Return principal axes of polygon (in ascending order of elliptic radius).
        '''
        S = self.nth_moment(2)
        _, V = la.eigh(S)
        return V[:, 0], V[:, 1]
    
    def major_axis(self) -> np.ndarray:
        '''
        Return major axis of polygon.
        '''
        return self.principal_axes()[1]

    def minor_axis(self) -> np.ndarray:
        '''
        Return minor axis of polygon.
        '''
        return self.principal_axes()[0]

    def covariance_matrix(self) -> np.ndarray:
        '''
        Calculate covariance matrix of polygon.
        '''
        return self.nth_moment(2, standardized=False) / self.area()

    def mahalanobis_distance(self, x: np.ndarray) -> float:
        '''
        Mahalanobis distance to center of mass.
        '''
        mu = self.centroid() # Center of mass
        Sigma = self.covariance_matrix()
        x_ = x - mu
        Sinv = la.inv(Sigma)
        if x.ndim == 1:
            return np.sqrt(x_ @ Sinv @ x_)
        elif x.ndim == 2:
            return np.sqrt(np.sum((x_ @ Sinv) * x_, axis=1))
        else:
            raise ValueError('Invalid input dimension')

    def aspect_ratio(self) -> float:
        '''
        Calculate aspect ratio metric as in paper.
        '''
        S = self.nth_moment(2)
        L = la.eigvalsh(S)
        return np.sqrt(L.max() / L.min())

    def anisotropy(self) -> float:
        '''
        Calculate anisotropy metric as in paper.
        '''
        return self.aspect_ratio() - 1

    def orientation(self) -> float:
        '''
        Return angle of major axis with respect to x-axis in basis of vertex points, in the range [0, pi] with periodic boundary conditions (a line has no polarity)
        '''
        _, e2 = self.principal_axes()
        theta = np.arctan2(e2[1], e2[0]) % np.pi
        return theta

    def exterior_angles(self) -> np.ndarray:
        '''
        Get exterior angles of polygon in the range [-pi, pi].
        '''
        e_ij = self.vertices - np.roll(self.vertices, 1, axis=0)
        e_jk = np.roll(e_ij, -1, axis=0)
        return np.arctan2(e_ij[:, 0] * e_jk[:, 1] - e_ij[:, 1] * e_jk[:, 0], (e_ij * e_jk).sum(axis=1))

    def elastic_energy(self) -> float:
        '''
        Discrete elastic energy in the sense of discrete differential geometry.
        '''
        e_ij = self.vertices - np.roll(self.vertices, 1, axis=0) # Edge vectors from node n-1 to node n
        ell_ij = la.norm(e_ij, axis=1) # Edge lengths
        ell_jk = np.roll(ell_ij, -1) # Edge lengths of next edge
        ell_i = (ell_ij + ell_jk) / 2 # Length elements at each node as average of adjacent edge lengths
        varphi_i = self.exterior_angles() # Exterior angles at each node
        energy = (varphi_i ** 2 / ell_i).sum() # Energy is total squared curvature weighted by length elements
        energy *= ell_ij.sum() # Rescale energy to be independent of perimeter
        return energy

    def perimeter(self) -> float:
        '''
        Perimeter of the polygon
        '''
        return la.norm(self.vertices - np.roll(self.vertices, 1, axis=0), axis=1).sum()

    def isoperimetric_deficit(self) -> float:
        '''
        "Quantitative isoperimetric inequality" in the sense of 4 pi A + lambda(A) <= L^2 with equality iff A is a circle.
        https://annals.math.princeton.edu/wp-content/uploads/annals-v168-n3-p06.pdf
        '''
        return self.perimeter() / (2 * np.sqrt(np.pi * self.area())) - 1
    
    def isoperimetric_quotient(self) -> float:
        '''
        Circularity metric in [0,1] defined by the isoperimetric inequality: L^2 >= 4 pi A
        '''
        return (4 * np.pi * self.area()) / (self.perimeter() ** 2)

    def isoperimetric_ratio(self) -> float:
        '''
        Sqrt() of isoperimetric_quotient
        '''
        return np.sqrt(self.isoperimetric_quotient())

    def whiten(self, eps: float=1e-12, return_W: bool=False) -> 'PlanarPolygon':
        '''
        Apply whitening transform to polygon vertices.
        '''
        V = self.vertices.copy()
        V -= self.centroid()
        Sigma = self.covariance_matrix()
        W = sqrt_pd_inv(Sigma, eps) # W is symmetric
        V = V @ W
        V += self.centroid()
        poly = PlanarPolygon(V)
        return (poly, W) if return_W else poly

    def random_whiten(self, ar_k: float=1., eps: float=1e-12, norm: float=1) -> Tuple['PlanarPolygon', np.ndarray]:
        '''
        Apply a "random whitening" transform, that is, perturb the points to achieve a random covariance matrix.
        Also returns the transformation matrix (for transforming test points).
        '''
        mu = self.centroid()
        poly, W = self.whiten(return_W=True)
        poly = poly - mu
        Sigma = random_gamma_covariance(ar_k=ar_k, norm=norm)
        W_ = sqrt_pd(Sigma)
        poly.vertices = poly.vertices @ W_
        poly = poly + mu
        return poly, W_ @ W

    def stretches(self) -> Tuple[float, float]:
        ''' Major and minor stretches in descending order '''
        S = self.covariance_matrix()
        L = la.eigvalsh(S)
        return np.sqrt(L.max()), np.sqrt(L.min())
    
    def trace_M2(self, center: np.ndarray=None, standardized: bool=False) -> float:
        ''' 
        Quantizer energy of a point 
        '''
        return np.trace(self.nth_moment(2, center=center, standardized=standardized))
    
    def voronoi_tessellate(self, xs: np.ndarray, interior: bool=False, **kwargs) -> 'PlanarPolygonPartition':
        '''
        Voronoi tessellation of the region bounded by this polygon containing given points
        '''
        assert xs.ndim == 2
        assert xs.shape[1] == 2
        seeds, vertices, partitions, on_bd = poly_bounded_voronoi(xs, self.to_shapely(), **kwargs)
        if interior:
            seeds = seeds[~on_bd]
            partitions = partitions[~on_bd]
        return PlanarPolygonPartition(self, vertices, partitions, seeds)
    
    def voronoi_quantization_energy(self, xs: np.ndarray, standardized: bool=False) -> float:
        '''
        Quantizer energy of the Voronoi tessellation (contained within self) of a set of points
        Dimensionless energy computed as in eq. (45), https://journals.aps.org/pre/pdf/10.1103/PhysRevE.82.056109
        '''
        polygons = self.voronoi_tessellate(xs).polygons
        e = sum([p.trace_M2(x, standardized=False) for p, x in zip(polygons, xs)])
        if standardized:
            e /= xs.shape[0]
            mean_area = np.array([p.area() for p in polygons]).mean()
            e /= (2 * (mean_area ** 2))
        return e
    
    def E2_energy(self, center=None) -> float:
        '''
        E^2 energy as defined in https://journals.aps.org/prresearch/pdf/10.1103/PhysRevResearch.3.L042001
        '''
        M = self.nth_moment(2, center=center)
        return np.trace(M) / (2 * np.sqrt(la.det(M))) - 1

    def __mul__(self, s: float) -> 'PlanarPolygon':
        ''' Scale the polygon about its centroid '''
        mu = self.reverse_embed(self.centroid())
        return PlanarPolygon((self.vertices_nd - mu) * s + mu)

    def __sub__(self, mu: np.ndarray) -> 'PlanarPolygon':
        ''' Translate the polygon '''
        return PlanarPolygon(self.vertices_nd - mu)

    def __add__(self, mu: np.ndarray) -> 'PlanarPolygon':
        ''' Translate the polygon '''
        return PlanarPolygon(self.vertices_nd + mu)

    def match_area(self, area: float) -> 'PlanarPolygon':
        ''' Scale polygon about centroid to match area of other polygon '''
        assert area >= 0
        return self * np.sqrt(area / self.area())

    def rescale(self, xyres: np.ndarray) -> 'PlanarPolygon':
        ''' Rescale the polygon to a given resolution, with origin in the original basis of the coordinates '''
        assert self.ndim == 2
        assert xyres.ndim == 1 and xyres.shape[0] == 2
        poly = self.copy()
        poly.vertices *= xyres
        return poly
    
    def flipxy(self) -> 'PlanarPolygon':
        ''' Flip polygon about x-axis '''
        assert self.ndim == 2
        return PlanarPolygon(self.vertices[:, ::-1])
    
    def flipy(self, yval: float) -> 'PlanarPolygon':
        ''' Flip polygon about y-value '''
        assert self.ndim == 2
        vertices = self.vertices.copy()
        vertices[:, 1] = yval - vertices[:, 1]
        return PlanarPolygon(vertices)
    
    def flipx(self, xval: float) -> 'PlanarPolygon':
        ''' Flip polygon about x-value '''
        assert self.ndim == 2
        vertices = self.vertices.copy()
        vertices[:, 0] = xval - vertices[:, 0]
        return PlanarPolygon(vertices)

    def copy(self) -> 'PlanarPolygon':
        plane = None if self.plane is None else self.plane.copy()
        return PlanarPolygon(self.vertices_nd.copy(), plane=plane, check=False) # Assume I am already valid
    
    def transform(self, A: np.ndarray, center: np.ndarray=None) -> 'PlanarPolygon':
        '''
        Apply a linear transform about a given center
        '''
        if center is None:
            center = self.centroid()
        poly = self.copy()
        poly.vertices = (poly.vertices - center) @ A.T + center
        return poly
    
    def transpose(self) -> 'PlanarPolygon':
        return PlanarPolygon(self.vertices[:, ::-1])
    
    def to_shapely(self) -> Polygon:
        return Polygon(self.vertices)
    
    def center(self) -> 'PlanarPolygon':
        ''' Center polygon at origin '''
        return self - self.reverse_embed(self.centroid())

    def intersects(self, other: 'PlanarPolygon') -> bool:
        ''' Check if two polygons intersect '''
        return self.to_shapely().intersects(other.to_shapely())

    def intersection(self, other: 'PlanarPolygon') -> List['PlanarPolygon']:
        ''' Intersection of two polygons '''
        assert self.ndim == 2 and other.ndim == 2
        inter = self.to_shapely().intersection(other.to_shapely())
        if inter.is_empty:
            return []
        inter = to_simple_polygons(inter)
        return [PlanarPolygon.from_shapely(p) for p in inter]
    
    def single_intersection(self, other: 'PlanarPolygon') -> 'PlanarPolygon':
        ''' Intersection of two polygons '''
        ps = self.intersection(other)
        assert len(ps) == 1, f'Expected single intersection, got {len(ps)}'
        return ps[0]
    
    def intersection_area(self, other: 'PlanarPolygon') -> float:
        ''' Intersection area of two polygons '''
        assert self.ndim == 2 and other.ndim == 2
        return self.to_shapely().intersection(other.to_shapely()).area
        
    def match_ellipse(self, ellipse) -> 'PlanarPolygon':
        ''' Match polygon center, area, and aspect ratio to the ellipse '''
        assert self.ndim == 2
        poly = self - self.centroid() # Center to 0
        poly = poly.whiten() # Set second moment to multiple of identity
        S = la.inv(ellipse.M) # Desired second moment
        W = sqrt_pd(S) # Desired whitening transform
        poly.vertices = poly.vertices @ W # Achieve desired second moment
        poly = poly.match_area(ellipse.area()) # Match area
        poly = poly + ellipse.v # Translate to ellipse center
        # Check
        assert np.allclose(poly.centroid(), ellipse.v)
        assert np.allclose(poly.area(), ellipse.area())
        assert np.isclose(poly.aspect_ratio(), ellipse.aspect_ratio())
        return poly
    
    def symmetric_difference(self, other: 'PlanarPolygon') -> List['PlanarPolygon']:
        ''' Symmetric difference of two polygons '''
        assert self.ndim == 2 and other.ndim == 2
        diff = self.to_shapely().symmetric_difference(other.to_shapely())
        diff = to_simple_polygons(diff)
        return [PlanarPolygon.from_shapely(p) for p in diff]
    
    def symmetric_difference_area(self, other: 'PlanarPolygon') -> float:
        ''' Area of symmetric difference of two polygons '''
        assert self.ndim == 2 and other.ndim == 2
        return self.to_shapely().symmetric_difference(other.to_shapely()).area
    
    def iou(self, other: 'PlanarPolygon') -> float:
        ''' Intersection over union of two polygons '''
        assert self.ndim == 2 and other.ndim == 2
        ia = self.to_shapely().intersection(other.to_shapely()).area
        ua = self.area() + other.area() - ia
        return ia / ua

    def bounding_box(self) -> Tuple[float, float, float, float]:
        ''' Bounding box of polygon in format (xmin, ymin, xmax, ymax) '''
        assert self.ndim == 2
        return self.to_shapely().bounds

    def contains(self, x: np.ndarray) -> Union[bool, np.ndarray]:
        ''' Check if point is contained in polygon '''
        assert self.ndim == 2
        if x.ndim == 1:
            assert x.shape[0] == self.ndim, f'Expected point to have {self.ndim} dimensions, got {x.shape[0]} dimensions'
            return self.to_shapely().contains(shapely.geometry.Point(x))
        elif x.ndim == 2:
            assert x.shape[1] == self.ndim, f'Expected points to have {self.ndim} dimensions, got {x.shape[1]} dimensions'
            sp = self.to_shapely()
            return np.array([sp.contains(shapely.geometry.Point(x_i)) for x_i in x])
        else:
            raise ValueError(f'Expected 1D or 2D array, got {x.ndim}D array')
    
    def shape_index(self) -> float:
        return self.perimeter() / np.sqrt(self.area())
    
    def hullify(self) -> 'PlanarPolygon':
        assert self.ndim == 2
        return PlanarPolygon.from_pointcloud(self.vertices)
    
    def simplify(self, eps: float, use_arclen: bool=True) -> 'PlanarPolygon':
        '''
        Simplify polygon using RDP algorithm
        '''
        assert eps >= 0
        if eps == 0:
            return self.copy()
        else:
            if use_arclen:
                eps *= self.perimeter()
            vertices = cv2.approxPolyDP(self.vertices.astype(np.float32), eps, True).reshape(-1, 2)
            return PlanarPolygon(vertices, check=False) # Assume cv2 produces valid output
        
    def subdivide_bspline(self, degree: int=2) -> 'PlanarPolygon':
        '''
        Subdivide polygon using B-spline subdivision
        '''
        assert self.ndim == 2
        return PlanarPolygon(skimage.measure.subdivide_polygon(self.vertices, degree=degree, preserve_ends=False))

    def diameter(self) -> float:
        ''' Diameter of polygon '''
        return pdist(self.vertices).max()

    def max_noncentrality(self) -> float:
        ''' Maximum noncentrality of polygon '''
        return la.norm(self.vertices - self.centroid(), axis=1).max()
    
    def apply_affine(self, T: np.ndarray) -> 'PlanarPolygon':
        vertices = self.vertices @ T[:2, :2].T + T[:2, 2]
        return PlanarPolygon(vertices, check=False)

    def to_mask(self, shape: Tuple[int, int]) -> np.ndarray:
        mask = np.zeros(shape, dtype=np.uint8)
        mask = mutil.draw_poly(mask, self.vertices.flatten().tolist(), 1)
        return mask.astype(bool)

    def draw_outline(self, img: np.ndarray, label: int=1) -> np.ndarray:
        assert self.ndim == 2
        mask = self.to_mask(img.shape[:2])
        mask = (mask - binary_erosion(mask).astype(np.uint8)).astype(bool)
        img = img.copy()
        img[mask] = label
        return img
    
    def embed_3d(self, z: float=0.) -> 'PlanarPolygon':
        ''' Embed polygons in XY plane'''
        assert self.ndim == 2
        plane = Plane.XY() + np.array([0, 0, z])
        vertices = np.hstack((self.vertices, np.full((self.vertices.shape[0], 1), z)))
        return PlanarPolygon(vertices, plane=plane, check=False) # Assume I am already valid

    def embed_XY(self) -> 'PlanarPolygon':
        return self.embed_3d(z=0.)
    
    def migrate_OLD(self) -> 'PlanarPolygon':
        '''
        TODO: hacky fix to migrate old polygons to new format
        '''
        if not hasattr(self, 'vertices_nd'):
            self.vertices_nd = self.vertices
        if not hasattr(self, 'surface'):
            assert self.ndim == 2
            self.surface = None
        return self.copy()

    @staticmethod
    def from_chull_polygons(polys: List['PlanarPolygon']) -> 'PlanarPolygon':
        '''
        Compute convex hull of list of polygons
        '''
        assert all(p.ndim == 2 for p in polys), 'All polygons must be 2D'
        return PlanarPolygon.from_pointcloud(np.concatenate([p.vertices for p in polys], axis=0))

    @staticmethod
    def from_pointcloud(coords: np.ndarray) -> 'PlanarPolygon':
        '''
        Compute using 2d convex hull
        '''
        hull = ConvexHull(coords)
        return PlanarPolygon(coords[hull.vertices])
    
    @staticmethod
    def from_shapely(poly: Polygon, allow_interiors: bool=False) -> 'PlanarPolygon':
        '''
        Convert from Shapely polygon
        '''
        assert allow_interiors or len(poly.interiors) == 0, 'Polygon has holes'
        return PlanarPolygon(np.array(poly.exterior.coords)[:-1])
    
    @staticmethod
    def from_image(img: np.ndarray) -> 'PlanarPolygon':
        '''
        Compute using convex hull of image nonzero values
        Assumes image is in YX convention as usual
        '''
        return PlanarPolygon.from_pointcloud(np.array(np.nonzero(img.T)).T)
    
    @staticmethod
    def from_shape(shape: Tuple[int, int]) -> 'PlanarPolygon':
        '''
        Create bounding box polygon from shape
        '''
        return PlanarPolygon(np.array([
            [0, 0],
            [0, shape[0]],
            [shape[1], shape[0]],
            [shape[1], 0]
        ]))
        
    @staticmethod
    def random(n: int=40, ar_k: float=1) -> 'PlanarPolygon':
        # Sample random points from multivariate normal
        Sigma = random_gamma_covariance(ar_k)
        pts = np.random.multivariate_normal([0, 0], Sigma, n)
        return PlanarPolygon.from_pointcloud(pts)

    @staticmethod
    def regular_ngon(n: int, r: float=1.) -> 'PlanarPolygon':
        ''' Regular n-gon of radius r centered at origin'''
        assert n >= 3
        theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
        pts = np.array([np.cos(theta), np.sin(theta)]).T * r
        return PlanarPolygon(pts)

    @staticmethod
    def random_regular(lam: float = 2, sigma: float=10, r: float=None) -> 'PlanarPolygon':
        n = 3 + np.random.poisson(lam)
        theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
        theta += np.random.rand() * 2 * np.pi # Random rotation
        v = np.random.randn(2) * sigma # Random offset
        r = np.random.exponential() if r is None else r # Random radius
        pts = np.array([np.cos(theta), np.sin(theta)]).T * r + v
        return PlanarPolygon(pts)
    
    @staticmethod
    def random_closed_curve(n: int=100, k_modes: int=10, k_exp: float=-1.5) -> 'PlanarPolygon':
        '''
        Generate random star-convex simple closed curve
        '''
        theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
        # Generate random 2pi-periodic function as polar coordinate
        phases = np.random.uniform(0, 2 * np.pi, k_modes)
        amplitudes = np.array([np.random.exponential(k ** k_exp) for k in range(1, k_modes+1)])
        omega = np.arange(1, k_modes+1)
        r = np.ones(n) + amplitudes.sum()
        for k in range(k_modes):
            r += amplitudes[k] * np.sin(omega[k] * theta + phases[k])
        pts = (np.array([np.cos(theta), np.sin(theta)]) * r).T
        assert pts.shape == (n, 2)
        return PlanarPolygon(pts)

    @staticmethod
    def id_ngon(n: int) -> float:
        ''' Isoperimetric deficit of regular n-gon '''
        assert n >= 3
        return np.sqrt(n * np.tan(np.pi / n) / np.pi) - 1
    
    @staticmethod
    def iq_ngon(n: int) -> float:
        assert n >= 3
        return np.pi / (n * np.tan(np.pi / n))
    
    @staticmethod
    def ir_ngon(n: int) -> float:
        return np.sqrt(PlanarPolygon.iq_ngon(n))

    @staticmethod
    def area_ngon(n: int, R: float=1, mode='circumradius') -> float:
        ''' Area of regular n-gon '''
        assert n >= 3
        if mode == 'circumradius':
            return n * R ** 2 * np.sin(2 * np.pi / n) / 2
        elif mode in ['apothem', 'inradius']:
            return n * R ** 2 * np.tan(np.pi / n)
        else:
            raise ValueError(f'Invalid mode: {mode}')
    
    @staticmethod
    def radius_ngon(n: int, a: float) -> float:
        ''' Circum-radius of regular n-gon given area '''
        assert n >= 3
        return np.sqrt(2 * a / (n * np.sin(2 * np.pi / n)))
    
    @staticmethod
    def apothem_ngon(n: int, a: float=1) -> float:
        ''' Apothem of regular n-gon given area '''
        assert n >= 3
        return PlanarPolygon.radius_ngon(n, a) * np.cos(np.pi / n)
    
    @staticmethod
    def second_moment_ngon(n: int, r: float=1) -> float:
        ''' Second moment of regular n-gon '''
        assert n >= 3
        c = (r ** 4) * n * (4 * np.sin(2 * np.pi / n) + np.sin(4 * np.pi / n)) / 48
        return c * np.eye(2)
    
    @staticmethod
    def trace_M2_ngon(n: int, r: float=1, standardized: bool=True) -> float:
        ''' Quantization energy of a regular n-gon '''
        e = np.trace(PlanarPolygon.second_moment_ngon(n, r))
        if standardized:
            e /= (PlanarPolygon.area_ngon(n, r) ** 2)
        return e
    
    @staticmethod
    def load(path: str) -> 'PlanarPolygon':
        return PlanarPolygon(np.load(path))
    
    @staticmethod
    def from_center_wh(center: np.ndarray, wh: Tuple[float, float]) -> 'PlanarPolygon':
        x, y = center
        w, h = wh
        return PlanarPolygon(np.array([[x - w, y - h], [x + w, y - h], [x + w, y + h], [x - w, y + h]]))

    @staticmethod
    def bounding_polygon(polys: List['PlanarPolygon']) -> 'PlanarPolygon':
        '''
        Compute convex hull of list of polygons
        '''
        assert all(p.ndim == 2 for p in polys), 'All polygons must be 2D'
        # pdb.set_trace()
        return PlanarPolygon.from_pointcloud(np.concatenate([p.vertices for p in polys], axis=0))
    
    @staticmethod
    def rect(x0, y0, w, h) -> 'PlanarPolygon':
        return PlanarPolygon(np.array([[x0, y0], [x0 + w, y0], [x0 + w, y0 + h], [x0, y0 + h]]))

class PlanarPolygonPartition(SurfacePartition):
    def __init__(self, surface: PlanarPolygon, vertices_nd: np.ndarray, partitions: np.ndarray, seeds_nd: np.ndarray):
        super().__init__(surface, vertices_nd, partitions, seeds_nd)

    def grad_second_moment(self) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError
    
    @property 
    def polygons(self) -> List[PlanarPolygon]:
        return [PlanarPolygon(self.vertices_nd[p], plane=self.surface.surface, check=False) for p in self.partitions]

    def refine(self, n: int) -> 'PlanarPolygonPartition':
        raise NotImplementedError
    
class PlanarPolygonPacking(SurfacePacking):

    def __init__(self, surface: PlanarPolygon, polygons: List[PlanarPolygon]):
        super().__init__(surface, polygons)

    def covered_area(self) -> float:
        ''' Accounts for overlapping polygons '''
        ps = [p.to_shapely() for p in self.polygons]
        return shapely.unary_union(ps).area

    def packing_fraction(self) -> float:
        return self.covered_area() / self.surface.area()
    
    def rescale(self, xyres: np.ndarray) -> 'PlanarPolygonPacking':
        ''' Rescale the packing to a given resolution, with origin in the original basis of the coordinates '''
        assert xyres.ndim == 1 and xyres.shape[0] == 2
        return PlanarPolygonPacking(self.surface.rescale(xyres), [p.rescale(xyres) for p in self.polygons])
        
    def copy(self) -> 'PlanarPolygonPacking':
        return PlanarPolygonPacking(self.surface.copy(), [p.copy() for p in self.polygons])
    
    def is_collinear(self, rtol: float=1e-1) -> bool:
        ''' Check if all polygons are approximately collinear '''
        axes = np.array([p.major_axis() for p in self.polygons])
        dots = np.abs(axes @ axes.T)
        return np.allclose(dots, 1, rtol=rtol)
        
    @property
    def coms(self) -> np.ndarray:
        return np.array([p.centroid() for p in self.polygons])
    
    @property
    def areas(self) -> np.ndarray:
        return np.array([p.area() for p in self.polygons])

    @staticmethod
    def from_mask(
            mask: np.ndarray,
            erode: int=0,
            dilate: int=0,
            use_chull_if_invalid: bool=False,
            method = 'marching_squares',
            boundary = None,
            check: bool=True,
        ) -> 'PlanarPolygonPacking':
        '''
        Extract polygons from mask
        '''
        assert mask.ndim == 2, f'Mask must be 2D, got {mask.ndim}D'
        assert erode >= 0 and dilate >= 0, f'Erode and dilate must be non-negative, got {erode} and {dilate}'
        assert dilate <= erode, f'Dilate must be less than or equal to erode, got {dilate} and {erode}'

        def do_erosion_dilation(submask):
            erode_n, dilate_n = erode, dilate
            while erode_n > 0:
                submask = binary_erosion(submask)
                erode_n -= 1
                if dilate_n > 0:
                    submask = binary_dilation(submask)
                    dilate_n -= 1
            return submask
        
        polygons = []

        if method == 'standard':
            polygons = mask_to_polygons(mask, erode=erode, dilate=dilate, use_chull_if_invalid=use_chull_if_invalid)
            polygons = [PlanarPolygon(p, check=check) for p in polygons if len(p) >= 3]
            # labels = np.unique(mask)
            # for label in labels:
            #     submask = (mask == label)
            #     submask = do_erosion_dilation(submask)
            #     if not submask.any():
            #         continue
            #     _, contours, __ = upolygon.find_contours(submask.astype(np.uint8))
            #     for contour in contours:
            #         if len(contour) >= 6:
            #             contour = np.array(contour).reshape(-1, 2)
            #             try:
            #                 poly = PlanarPolygon(contour, check=True, use_chull_if_invalid=use_chull_if_invalid)
            #                 polygons.append(poly)
            #             except Exception as e:
            #                 print(f'Error creating polygon from contour: {e}')
            #                 continue

        elif method == 'marching_squares':
            props = skimage.measure.regionprops(mask)
            for prop in props:
                if np.isclose(prop.area, 0):
                    continue
                submask = np.pad(prop.image, 1, mode='constant', constant_values=0) # add padding to get interior contour always
                submask = do_erosion_dilation(submask)
                if not submask.any():
                    continue
                contour = max(skimage.measure.find_contours(submask, level=0.9), key=len)
                if len(contour) >= 3:
                    contour -= 1 # Remove padding
                    for d in range(mask.ndim):
                        contour[:, d] += prop.bbox[d] # Add back bounding box offset
                    contour = np.fliplr(contour) # find_contours returns (row, col) which is (y, x) in image coordinates
                    try:
                        poly = PlanarPolygon(contour, check=True, use_chull_if_invalid=use_chull_if_invalid)
                        polygons.append(poly)
                    except Exception as e:
                        print(f'Error creating polygon from contour: {e}')
                        continue

        else:
            raise ValueError(f'Invalid method: {method}')

        if boundary is None:
            boundary = PlanarPolygon.from_shape(mask.shape)
        assert boundary.ndim == 2
        return PlanarPolygonPacking(boundary, polygons)
                        
    @staticmethod
    def match_by_containment(
            container: 'PlanarPolygonPacking',  
            containee: 'PlanarPolygonPacking',
            min_frac: float=0.9,
        ) -> Tuple['PlanarPolygonPacking', 'PlanarPolygonPacking']:
        '''
        Match containee to container by containment
        '''
        assert container.surface.ndim == containee.surface.ndim == 2
        assert container.surface == containee.surface, 'container and containee must be on the same surface'
        seen = set() # Set of seen containee polygons
        container_mask = np.zeros(container.n_polygons, dtype=bool)
        containee_areas = containee.areas
        containee_indices = []
        for i, poly_i in enumerate(container.polygons):
            j_i = None
            max_frac = 0
            for j, poly_j in enumerate(containee.polygons):
                if j in seen:
                    continue
                frac = poly_i.intersection_area(poly_j) / containee_areas[j]
                if frac > max_frac:
                    max_frac = frac
                    j_i = j
            if max_frac > min_frac:
                container_mask[i] = True
                containee_indices.append(j_i)
                seen.add(j_i)
        containee_indices = np.array(containee_indices, dtype=np.intp)
        assert container_mask.sum() == containee_indices.shape[0], f'Number of containee indices != number of container polygons'
        container = container.select(container_mask)
        containee = containee.select(containee_indices)
        return container, containee
'''
Utility functions
'''

def mask_to_polygons(mask: np.ndarray, rdp_eps: float=0., erode: int=0, dilate: int=0, use_chull_if_invalid=False) -> List[np.ndarray]:
    '''
    Compute outlines of objects in mask as a list of polygon coordinates
    Arguments:
    - mask: integer mask of shape (H, W)
    - rdp_eps: epsilon parameter in the Ramer-Douglas-Peucker algorithm for polygon simplification
    - erode: number of pixels to erode the mask by before computing the polygon to get rid of single-pixel artifacts

    TODO: use marching squares to get better polygons?
    https://nils-olovsson.se/articles/marching_squares/
    '''
    assert mask.ndim == 2, 'Mask must be 2D'
    assert rdp_eps >= 0
    assert erode >= 0 and dilate >= 0
    polygons = []

    # slices = find_objects(mask)
    # for i,si in enumerate(slices):
    #         if si is not None:
    #             sr, sc = si
    #             submask = mask[sr, sc] == (i+1) # Select ROI
    #             if erode > 0:
    #                 submask = binary_erosion(submask, iterations=erode) # Get rid of single-pixel artifacts
    #             if dilate > 0:
    #                 submask = binary_dilation(submask, iterations=dilate) # Correct area lost in erosion
    #             contours = cv2.findContours(submask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2]
    #             # Take contour with largest coord count
    #             n_coords = [c.size for c in contours]
    #             pvc, pvr = contours[np.argmax(n_coords)].squeeze().T
    #             vr, vc = pvr + sr.start, pvc + sc.start
    #             coords = np.stack([vc, vr], axis=1)
    #             # Construct polygon
    #             poly = Polygon(coords)
    #             if rdp_eps > 0: # Simplify polygon using RDP algorithm
    #                 poly = poly.simplify(rdp_eps, preserve_topology=True)
    #             poly = orient(poly, sign=1.0)
    #             coords = np.array(poly.exterior.coords) # Exterior coordinates oriented counter-clockwise
    #             polygons.append(coords)

    elems = np.unique(mask)
    elems = elems[elems != 0] # Ignore background
    for elem in elems:
        submask = mask == elem
        if erode > 1:
            submask = binary_erosion(submask, iterations=erode) # Get rid of single-pixel artifacts
        if dilate > 1:
            submask = binary_dilation(submask, iterations=dilate) # Correct area lost in erosion
        _, external_paths, internal_paths = upolygon.find_contours(submask.astype(np.uint8))
        # Take contour with largest coord count
        n_coords = [len(c) for c in external_paths]
        contour = external_paths[np.argmax(n_coords)] # in X, Y, X, Y, ... format
        contour = np.array(contour).reshape(-1, 2) # (N, 2) in X, Y format
        p = Polygon(contour)
        # Ensure valid polygons are extracted
        if not p.is_valid:
            if use_chull_if_invalid:
                p = p.convex_hull
            else:
                # Try buffer(0) trick
                p = p.buffer(0)
                if type(p) == shapely.geometry.MultiPolygon:
                    # Extract polygon with largest area
                    p = max(p.geoms, key=lambda x: x.area)
            assert type(p) == shapely.geometry.Polygon
            assert p.is_valid
        contour = np.array(p.exterior.coords)  
        polygons.append(contour)

    return polygons

def random_gamma_covariance(ar_k: float=1, norm: float=1) -> np.ndarray:
    '''
    Random covariance matrix for 2-variate random variable with aspect ratio sampled from gamma distribution.
    '''
    # Sample random aspect ratio from gamma distribution
    ar = np.random.gamma(ar_k)
    # Make random covariance matrix with aspect ratio
    L = np.diag([1, ar ** 2])
    P, _ = la.qr(np.random.randn(2, 2))
    P *= norm
    Sigma = P @ L @ P.T
    return Sigma

def sqrt_pd_inv(Sigma: np.ndarray, eps: float=1e-12) -> np.ndarray:
    '''
    Compute ZCA whitening transform from covariance matrix.
    '''
    U, L, _ = la.svd(Sigma)
    W = U @ np.diag(1 / np.sqrt(L + eps)) @ U.T
    return W

def sqrt_pd(Sigma: np.ndarray) -> np.ndarray:
    '''
    Compute square root of positive definite matrix.
    '''
    U, L, _ = la.svd(Sigma)
    return U @ np.diag(np.sqrt(L)) @ U.T

def polygon_area_unjitted(X: np.ndarray) -> float:
    '''
    Area of polygon using JAX
    '''
    X_ = jnp.roll(X, -1, axis=0)
    X_cross = X[:, 0] * X_[:, 1] - X[:, 1] * X_[:, 0]
    return jnp.sum(X_cross) / 2

polygon_area = jax.jit(polygon_area_unjitted)
grad_polygon_area = jax.jit(jax.grad(polygon_area_unjitted))

def polygon_second_moment_unjitted(X: np.ndarray, o: np.ndarray) -> np.ndarray:
    '''
    Trace second moment of polygon with respect to origin
    '''
    X = X - o
    X_ = jnp.roll(X, -1, axis=0)
    X_cross = X[:, 0] * X_[:, 1] - X[:, 1] * X_[:, 0]
    x, y = X.T
    x_, y_ = X_.T
    I_xx = X_cross @ (x**2 + x*x_ + x_**2) / 12
    I_yy = X_cross @ (y**2 + y*y_ + y_**2) / 12
    return I_xx + I_yy

polygon_second_moment = jax.jit(polygon_second_moment_unjitted)

grad_polygon_second_moment_vertices = jax.jit(jax.grad(polygon_second_moment_unjitted, argnums=0))
grad_polygon_second_moment_origin = jax.jit(jax.grad(polygon_second_moment_unjitted, argnums=1))

if __name__ == '__main__':
    import random
    
    # random.seed(0)
    # np.random.seed(0)

    # Random plane
    d = 3
    plane = Plane.random(d)

    # Sample random points from plane
    N = 1000
    X = np.random.randn(N, d) * 10
    X = plane.project_l2(X)

    # Check that planar representation is 1-1
    X_embedded = plane.reverse_embed(plane.embed(X))
    assert np.allclose(X, X_embedded), 'X_embedded != X'

    # Add noise
    sigma_noise = 1.0
    dX = np.random.randn(N, d) * sigma_noise
    assert dX.shape == X.shape
    Y = X + dX

    # Fit plane to noisy points
    plane_fit = Plane.fit_l2(X)
    assert np.allclose(plane_fit.n, plane.n, atol=1e-3), 'plane_fit.n != plane.n'
    assert np.allclose(plane_fit.b, plane.b, atol=1e-3), 'plane_fit.v != plane.v'
    plane_fit = Plane.fit_l2(Y)

    # Check unary_union for overlapping polygons
    for k in range(1000):
        w = np.random.exponential() + 1
        h = np.random.exponential() + 1
        overlap = np.random.uniform(0, 1)
        poly1 = PlanarPolygon.rect(0, 0, w, h)
        poly2 = PlanarPolygon.rect(w - overlap, 0, w, h)
        packing = PlanarPolygonPacking(None, [poly1, poly2])
        covered_area = h * (2 * w - overlap)
        assert np.isclose(covered_area, packing.covered_area())
    print('Covering area of overlapping polygons is correct.')
        
    # Check area of regular n-gon
    for k in range(10000):
        # Random regular polygon
        r = np.random.exponential()
        poly = PlanarPolygon.random_regular(r=r)
        assert np.allclose(poly.area(), PlanarPolygon.area_ngon(poly.n, R=r)), f'area of regular {poly.n}-gon != analytic formula'
    print('Area of regular n-gon is correct.')

    # Check area against Shapely
    for k in range(1000):
        # Random polygon
        poly = PlanarPolygon.random()
        poly_ = Polygon(poly.vertices)
        assert np.allclose(poly.area(), poly_.area), f'area of random polygon != shapely'
        assert np.allclose(poly.centroid(), poly_.centroid.coords[0]), f'centroid of random polygon != shapely'
    print('Area and centroid of random polygon is correct.')

    # Check second moment against another library
    import polygon_math
    for k in range(1000):
        # Random polygon
        poly = PlanarPolygon.random()
        poly_ = polygon_math.polygon(poly.vertices)
        # Second moment
        I2 = poly.nth_moment(2, center=np.zeros(2))
        [Iyy_, Ixx_, Ixy_] = poly_.SecondMomentArea
        Ixy_ *= -1 # Polygon library flips the sign
        assert np.allclose(I2[0,0], Ixx_), f'second moment of random polygon != polygon_math'
        assert np.allclose(I2[1,1], Iyy_), f'second moment of random polygon != polygon_math'
        assert np.allclose(I2[0,1], Ixy_), f'second moment of random polygon != polygon_math'
    print('Second moment of random polygon is correct.')

    # Check change of second moment formula
    for k in range(1000):
        # Random polygon
        poly = PlanarPolygon.random()
        # A point within polygon
        x = np.random.randn(2) * 10
        # Second moment about centroid
        I2 = poly.nth_moment(2, center=np.zeros(2))
        I2_x = poly.nth_moment(2, center=x)
        I0, I1 = poly.nth_moment(0), poly.nth_moment(1)
        I2_x_ = I2 - x * I1[:,None] - I1 * x[:,None] + I0 * x[:,None] * x[None,:]
        assert np.allclose(I2_x, I2_x_), f'second moment change of axis failed'
    print('Second moment change of axis works.')

    # Check regular polygons
    for n in range(3, 1000):
        theta = np.linspace(0, 2*np.pi, n, endpoint=False) + np.random.rand() * 2*np.pi # Random rotation
        v = np.random.randn(2) * 100 # Random offset
        X = np.stack([np.cos(theta), np.sin(theta)], axis=-1) + v
        poly = PlanarPolygon(X)
        assert np.allclose(poly.anisotropy(), 0), f'anisotropy of regular {poly.n}-gon != 0'
        assert np.allclose(poly.exterior_angles(), 2*np.pi/poly.n), f'exterior angles of regular {poly.n}-gon != 2pi/n'
        # print(f'Elastic energy of regular {poly.n}-gon: {poly.elastic_energy()}')
        # assert np.allclose(poly.elastic_energy(), 4 * np.pi**2), f'elastic energy of regular {poly.n}-gon != 4pi^2'
    print('All regular polygons have zero anisotropy and correct exterior angles.')

    # Check regular polygons
    for n in range(3, 1000):
        theta = np.linspace(0, 2*np.pi, n, endpoint=False) + np.random.rand() * 2*np.pi # Random rotation
        v = np.random.randn(2) * 100 # Random offset
        r = np.random.exponential() * 10 # Random radius
        X = r * np.stack([np.cos(theta), np.sin(theta)], axis=-1) + v
        poly = PlanarPolygon(X)
        assert np.allclose(poly.nth_moment(2), PlanarPolygon.second_moment_ngon(n, r=r)), f'second moment of regular {poly.n}-gon != analytic formula'
    print('Second moment of regular n-gon is correct.')

    # Check random polygons
    for k in range(10):
        # Construct random polygon
        n = np.random.randint(3, 100)
        theta = np.random.uniform(0, 2*np.pi, n)
        theta.sort()
        energies = []
        for l in range(100):
            # Apply random affine transform
            theta_ = theta + np.random.uniform(0, 2*np.pi) # Random rotation
            r = np.random.uniform(0.1, 10) # Random scale
            v = np.random.randn(2) * 100 # Random offset
            X = np.stack([np.cos(theta_), np.sin(theta_)], axis=-1) * r + v
            poly = PlanarPolygon(X)
            energies.append(poly.elastic_energy())
            assert np.allclose(poly.exterior_angles().sum(), 2*np.pi), f'exterior angles of random {n}-gon != 2pi'
        energies = np.array(energies)
        assert np.allclose(energies, energies[0]), f'elastic energy of random {n}-gon not scale-invariant'
    print('Elastic energy is scale-invariant and random polygons have correct total curvature.')

    # Check that standardized second moment is scale-invariant
    for k in range(10000):
        # Random polygon
        poly = PlanarPolygon.random()
        # Random scale
        poly2 = poly * np.random.uniform(0.1, 100)
        assert np.allclose(poly.trace_M2(standardized=True), poly2.trace_M2(standardized=True)), f'second moment of scaled polygon != original'
    print('Standardized second moment is scale-invariant.')

    # Check isoperimetric inequality for standardized second moment
    for k in range(10000):
        # Random polygon
        poly = PlanarPolygon.random()
        assert poly.trace_M2(standardized=True) > 1 / (2 * np.pi), f'second moment of random polygon does not obey isoperimetric inequality'
        # Random closed curve
        poly = PlanarPolygon.random_closed_curve()
        assert poly.trace_M2(standardized=True) > 1 / (2 * np.pi), f'second moment of random closed curve does not obey isoperimetric inequality'
    print('Standardized second moment obeys isoperimetric inequality.')

    # Check isoperimetric quotient for regular n-gons
    for k in range(1000):
        # Random polygon
        poly = PlanarPolygon.random_regular()
        iq = np.pi / (poly.n * np.tan(np.pi / poly.n))
        assert np.isclose(poly.isoperimetric_quotient(), iq), f'isoperimetric quotient of random {poly.n}-gon != pi/(n*tan(pi/n))'
    print('Isoperimetric quotient of regular n-gon is correct.') 

    # Check aspect ratio of whitened random polygons
    for k in range(1000):
        # Random point cloud
        n = np.random.randint(3, 100)
        pts = np.random.randn(n, 2)
        poly = PlanarPolygon.from_pointcloud(pts).whiten()
        # print('Aspect ratio of random polygon:', poly.aspect_ratio())
        ar = poly.aspect_ratio()
        assert np.allclose(ar, 1, atol=1e-3), f'aspect ratio of whitened random {n}-gon != 1, got {ar}'
        S = poly.nth_moment(2)
        r = PlanarPolygon.radius_ngon(n, poly.area())
        S0 = PlanarPolygon.second_moment_ngon(n, r=r)
        # pdb.set_trace()
        assert np.allclose(S, S0), f'second moment of whitened random {n}-gon != regular n-gon'
    print('Aspect ratio of whitened random polygons is approximately 1 and second moment is correct.')

    # Check that isoperimetric deficit is scale-invariant
    for k in range(10000):
        # Random polygon
        poly = PlanarPolygon.random()
        # Random scale
        poly2 = poly * np.random.uniform(0.1, 100)
        assert np.allclose(poly.isoperimetric_deficit(), poly2.isoperimetric_deficit()), f'isoperimetric deficit of scaled polygon != original'
    print('Isoperimetric deficit is scale-invariant.')

    # Check that whitened isoperimetric deficit is invariant to second moment change
    for k in range(1000):
        # Random polygon
        poly1 = PlanarPolygon.random()
        id1 = poly1.whiten().isoperimetric_deficit()
        # Random whitening transform
        poly2, _ = poly1.random_whiten()
        id2 = poly2.whiten().isoperimetric_deficit()
        assert np.allclose(id1, id2), f'isoperimetric deficit of whitened polygon != original'
    print('Whitened isoperimetric deficit is invariant to second moment change.')

    # Check that Mahalanobis distance is scale-invariant
    for k in range(1000):
        # Random polygon
        poly = PlanarPolygon.random()
        poly = poly - poly.centroid() # Center polygon
        # Random point
        x1 = np.random.randn(2)
        d1 = poly.mahalanobis_distance(x1)
        # Random scale
        s = np.random.uniform(0.1, 100)
        poly2 = poly * s
        x2 = x1 * s
        d2 = poly2.mahalanobis_distance(x2)
        assert np.allclose(d1, d2), f'Mahalanobis distance of scaled polygon != original'
    print('Mahalanobis distance is scale-invariant.')

    # Check that Mahalanobis distance is invariant to affine transform
    for k in range(1000):
        # Random polygon
        poly = PlanarPolygon.random()
        poly = poly - poly.centroid()
        # Random point
        x1 = np.random.randn(2)
        d1 = poly.mahalanobis_distance(x1)
        # Random whitening transform
        poly2, T = poly.random_whiten()
        x2 = T @ x1
        d2 = poly2.mahalanobis_distance(x2)
        assert np.allclose(d1, d2), f'Mahalanobis distance of whitened polygon != original'
    print('Mahalanobis distance is invariant to affine transform.')

    # Check area matching
    for k in range(1000):
        # Random polygons
        poly1, poly2 = PlanarPolygon.random(), PlanarPolygon.random()
        # Match area
        poly2 = poly2.match_area(poly1.area())
        assert np.allclose(poly1.area(), poly2.area()), f'area of matched polygon != original'
    print('Area matching works.')

    # Check that covariance matrix of regular polygon is scalar multiple of identity
    for k in range(1000):
        # Random regular polygon
        poly = PlanarPolygon.random_regular()
        C = poly.covariance_matrix()
        assert np.allclose(C, C[0,0]*np.eye(2)), f'covariance matrix of regular {poly.n}-gon != scalar multiple of identity'
    print('Covariance matrix of regular polygon is scalar multiple of identity.')

    # # Check that covariance matrix is scale-invariant
    # for k in range(1000):
    #     # Random polygon
    #     poly = PlanarPolygon.random()
    #     C = poly.covariance_matrix()
    #     # Random scale
    #     poly2 = poly * np.random.uniform(0.1, 100)
    #     C2 = poly2.covariance_matrix()
    #     pdb.set_trace()
    #     assert np.allclose(C, C2), f'covariance matrix of scaled polygon != original'
    # print('Covariance matrix is scale-invariant.')

    # Quantizer energy
    for k in range(1000):
        # Random point
        x = np.random.randn(2) * 10
        # Random polygon
        poly = PlanarPolygon.random() - x
        e = poly.trace_M2(np.zeros(2), dimensionless=True)
        assert e >= 0, f'quantizer energy of random polygon is negative'
        s = np.random.uniform(0.1, 100)
        poly_ = PlanarPolygon(poly.vertices * s) # Scale vertices in this coordinate system
        e_ = poly_.trace_M2(np.zeros(2), dimensionless=True)
        assert np.allclose(e, e_), f'dimensionless quantizer energy of scaled polygon != original'
    print('Dimensionless Quantizer energy is scale-invariant.')

    # E2 energy
    for k in range(1000):
        # Random point
        x = np.random.randn(2) * 10
        # Random polygon
        poly = PlanarPolygon.random() - x
        e = poly.E2_energy(np.zeros(2))
        assert e >= 0, f'E2 energy of random polygon is negative'
        s = np.random.uniform(0.1, 100)
        poly_ = PlanarPolygon(poly.vertices * s) # Scale vertices in this coordinate system
        e_ = poly_.E2_energy(np.zeros(2))
        assert np.allclose(e, e_), f'E2 energy of scaled polygon != original'
    print('E2 energy is scale-invariant.')
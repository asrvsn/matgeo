'''
Functions for fitting ellipsoids to data.
'''

import numpy as np
import cvxpy as cp
import numpy.linalg as la
import scipy.optimize as scopt
from scipy.linalg import sqrtm
import scipy.special as special
from typing import Tuple, Union, List, Optional
import pdb

from .surface import Surface
from .plane import Plane, PlanarPolygon

class Ellipsoid(Surface):
    ''' d-dimensional ellipsoid defined by a center and Hermitian inner product '''
    
    def __init__(self, M: np.ndarray, v: np.ndarray):
        self.M = M # Inner product
        self.v = v # Center

    def __eq__(self, other: 'Ellipsoid') -> bool:
        ''' Check if two ellipsoids are equal '''
        return np.allclose(self.M, other.M) and np.allclose(self.v, other.v)

    @staticmethod
    def fit_outer(points: np.ndarray) -> 'Ellipsoid':
        ''' Fit minimum-volume ellipsoid containing the given points using DCP '''
        assert points.ndim == 2, 'points must be 2d array'
        d = points.shape[1]
        A = cp.Variable((d, d), PSD=True) # sqrt(M)
        b = cp.Variable(d)
        cost = -cp.log_det(A)
        constraints = [
            cp.norm(points @ A + b[None, :], axis=1) <= 1
        ]
        problem = cp.Problem(cp.Minimize(cost), constraints)
        problem.solve()
        M = A.value.T @ A.value
        v = -la.inv(A.value) @ b.value
        return Ellipsoid(M, v)

    @staticmethod
    def fit_outer_iterative(points: np.ndarray, tol=1e-4) -> 'Ellipsoid':
        ''' Fit minimum-volume ellipsoid using iterative method. Standard implementation e.g. https://github.com/rmsandu/Ellipsoid-Fit/blob/main/outer_ellipsoid.py'''
        points = np.asmatrix(points)
        N, d = points.shape
        Q = np.column_stack((points, np.ones(N))).T
        u = np.ones(N) / N
        err = 1 + tol
        while err > tol:
            X = Q * np.diag(u) * Q.T
            M = np.diag(Q.T * la.inv(X) * Q)
            jdx = np.argmax(M)
            step_size = (M[jdx] - d - 1.0) / ((d + 1) * (M[jdx] - 1.0))
            new_u = (1 - step_size) * u
            new_u[jdx] += step_size
            err = la.norm(new_u - u)
            u = new_u

        c = u * points  # center of ellipsoid
        A = la.inv(points.T * np.diag(u) * points - c.T * c) / d
        return Ellipsoid(np.asarray(A), np.squeeze(np.asarray(c)))

    @staticmethod
    def fit_l2(X: np.ndarray) -> 'Ellipsoid':
        # TODO: problem not DCP
        assert X.ndim == 2, 'X must be 2d'
        d = X.shape[1]
        A = cp.Variable((d, d), PSD=True)
        b = cp.Variable(d)
        LHS = cp.norm(X @ A + b[None, :], axis=1) - 1
        cost = cp.sum_squares(LHS)
        problem = cp.Problem(cp.Minimize(cost), [])
        problem.solve()
        v = -la.inv(A.value) @ b.value
        M = A.value.T @ A.value
        return Ellipsoid(M, v)

    @staticmethod
    def random(sigma=5, v=None, d=2, ar_k: float=2.5, ar: float=None) -> 'Ellipsoid':
        ''' Random d-dimensional ellipsoid '''
        # Random rotation
        R = np.random.randn(d, d) * sigma
        R, _ = np.linalg.qr(R) # GOE

        # Random stretches
        if ar is None:
            ar = np.random.gamma(ar_k)
        ar = max(ar, 1/ar)
        L = np.diag(np.linspace(1, ar, d) ** -2) 
        M = R @ L @ R.T

        # Random translation
        v = np.random.randn(d) * sigma if v is None else (v * np.ones(d))
        return Ellipsoid(M, v)

    def map_sphere(self, X: np.ndarray, P=None, rs=None, v=None) -> np.ndarray:
        ''' Map points on the unit sphere to the ellipsoid '''
        assert X.ndim == 2, 'X must be 2d'
        assert X.shape[1] == self.ndim, 'X must have same dimension as ellipsoid'
        if P is None or rs is None:
            P, rs = self.get_axes_stretches()
        if v is None:
            v = self.v
        return X @ np.diag(rs) @ P.T + v[None, :]

    def sample_mgrid(self, n: int=20, P=None, rs=None, v=None):
        ''' Sample the ellipsoid on a meshgrid '''
        assert self.ndim == 3, 'Ellipsoid must be 2d or 3d'
        assert n > 0
        us, vs = np.mgrid[0:2*np.pi:2*n*1j, 0:np.pi:n*1j]
        x = np.cos(us)*np.sin(vs)
        y = np.sin(us)*np.sin(vs)
        z = np.cos(vs)
        X = np.vstack((x.flatten(), y.flatten(), z.flatten())).T
        X = self.map_sphere(X, P=P, rs=rs, v=v)
        X = X.reshape((2*n, n, self.ndim))
        return X

    def project_l2(self, X: np.ndarray, tol=1e-3, batch=10) -> np.ndarray:
        ''' Project points onto the ellipsoid with minimum l2 distance using SLSQP 
        # TODO: try 
        # Orthogonal projection of a point onto an ellipse (2D) or ellipsoid (3D)
        # https://github.com/nedelec/Orthogonal-Projection-on-Ellipsoid
        '''
        assert X.ndim == 2, 'X must be 2d'
        assert X.shape[1] == self.ndim, 'X must have same dimension as ellipsoid'
        n = X.shape[0]
        if n > batch:
            # This problem is embarrassingly parallel. Decrease problem size since constraint Jacobian is n^2
            return np.vstack([self.project_l2(X[i:i+batch, :], tol=tol) for i in range(0, n, batch)])
        Xbar = X - self.v[None, :]
        Xbar_ = Xbar.flatten()
        L, P = la.eigh(self.M)
        A = P @ np.diag(np.sqrt(L)) @ P.T # Use sqrt(M) in problem
        def cost(Ybar_):
            return la.norm(Ybar_ - Xbar_) ** 2
        def cost_grad(Ybar_):
            return 2 * (Ybar_ - Xbar_)
        def constraint_func(Ybar_):
            return la.norm(Ybar_.reshape(Xbar.shape) @ A, axis=1) ** 2 - 1
        constraint = scopt.NonlinearConstraint(constraint_func, lb=0, ub=0)
        result = scopt.minimize(cost, Xbar.flatten(), jac=cost_grad, constraints=[constraint], method='SLSQP', tol=tol)
        assert result.success, 'project_l2: optimization failed'
        Ybar = result.x.reshape(Xbar.shape)
        Y = Ybar + self.v[None, :]
        return Y

    def project_z_OLD(self, X: np.ndarray, tol=1e-3, batch=10, initial_z=-1000) -> np.ndarray:
        ''' Project points in an X-Y plane onto the ellipsoid in the ellipsoid's own basis '''
        assert X.ndim == 2, 'X must be 2d'
        assert X.shape[1] == 2, 'X must be planar'
        assert self.ndim == 3, 'Ellipsoid must be 3d'
        n = X.shape[0]
        if n > batch:
            return np.vstack([self.project_z(X[i:i+batch, :], tol=tol) for i in range(0, n, batch)])
        xybar = X - self.v[None, :2]
        A = self.sqrt_M
        def cost(zbar):
            return zbar.sum() # Minimize z to get lower projection
        def cost_grad(zbar):
            return zbar
        def constraint_func(zbar):
            Xbar = np.hstack((xybar, zbar[:, None]))
            return la.norm(Xbar @ A, axis=1) ** 2 - 1
        constraint = scopt.NonlinearConstraint(constraint_func, lb=0, ub=0)
        result = scopt.minimize(cost, np.full(n,initial_z), jac=cost_grad, constraints=[constraint], method='SLSQP', tol=tol)
        assert result.success, 'project_z: optimization failed'
        zbar = result.x
        Xbar = np.hstack((xybar, zbar[:, None]))
        X = Xbar + self.v[None, :]
        return X

    def project_z(self, X: np.ndarray) -> np.ndarray:
        '''
        Project points in XY plane onto ellipsoid using simple quadratic equation.
        Essentially casts X as parallel light rays onto the ellipsoid and finds the intersection points.
        '''
        assert X.ndim == 2, 'X must be 2d'
        assert X.shape[1] == 2, 'X must be planar'
        assert self.ndim == 3, 'Ellipsoid must be 3d'
        n = X.shape[0]
        M = self.M

        # Let y = x - v, solving for y2 in y^T M y = 1 where y = [y0, y1, y2]
        y01 = X - self.v[:2]
        y0, y1 = y01[:,0], y01[:,1]

        # Coefficients of ax^2 + bx + c = 0 for x
        a = np.full(n, M[2, 2])
        b = y01 @ M[2, :2] + y0 * M[0, 2] + y1 * M[1, 2]
        c = y0 * (y01 @ M[0, :2]) + y1 * (y01 @ M[1, :2]) - 1

        # Using Citardauq formula for numerical stability, take root closest to zero
        y2 = 2 * c / (-b - np.sqrt(b**2 - 4 * a * c))
        x2 = y2 + self.v[2]
        X_ = np.hstack((X, x2[:, None]))
        return X_

    def project_poly_z(self, poly: PlanarPolygon, plane: Optional[Plane]=None) -> PlanarPolygon:
        ''' Project a polygon onto the ellipsoid in the XY plane '''
        assert poly.ndim == 2
        return PlanarPolygon(self.project_z(poly.vertices), plane=plane)

    def dist(self, X: np.ndarray, **kwargs) -> np.ndarray:
        assert X.ndim == 2, 'X must be 2d'
        assert X.shape[1] == self.ndim, 'X must have same dimension as ellipsoid'
        Y = self.project_l2(X, **kwargs)
        return la.norm(X - Y, axis=1)
        
    @property
    def ndim(self):
        return self.M.shape[0]

    @property
    def sqrt_M(self) -> np.ndarray:
        L, P = la.eigh(self.M)
        A = P @ np.diag(np.sqrt(L)) @ P.T # Use sqrt(M) in problem
        return A

    def get_axes_stretches(self):
        ''' Get the principal axes and stretches of the ellipsoid. Axes are columns of P '''
        L, P = la.eigh(self.M)
        rs = 1/np.sqrt(L) # Radii are in descending order
        return P, rs
    
    def get_major_axis(self) -> np.ndarray:
        ''' Get vector of major axis, with length touching the ellipse '''
        P, rs = self.get_axes_stretches() # radii are in descending order
        n = P[:, 0]
        n *= rs[0]
        return n
    
    def get_major_plane(self) -> Plane:
        ''' Get plane normal to major axis '''
        return Plane(self.get_major_axis(), self.v.copy())
    
    def get_radii(self) -> np.ndarray:
        ''' Get principal radii in descending order '''
        return 1 / np.sqrt(la.eigvalsh(self.M)) 
    
    def get_major_radius(self) -> float:
        ''' Get major radius '''
        _, rs = self.get_axes_stretches()
        return rs[0]
    
    def get_minor_radius(self) -> float:
        ''' Get minor radius '''
        _, rs = self.get_axes_stretches()
        return rs[1]
    
    def translate(self, x: np.ndarray):
        self.v += x

    def __add__(self, x: np.ndarray):
        return Ellipsoid(self.M.copy(), self.v + x)

    def __sub__(self, x: np.ndarray):
        return Ellipsoid(self.M.copy(), self.v - x)

    def __mul__(self, x: float):
        return Ellipsoid(self.M * 1/(x**2), self.v.copy())

    def __truediv__(self, x: float):
        return Ellipsoid(self.M * (x**2), self.v.copy())
    
    def transpose(self, *idxs: Tuple) -> 'Ellipsoid':
        assert len(idxs) == self.ndim, 'Must have same dimension as ellipsoid'
        idxs = list(idxs)
        return Ellipsoid(self.M.copy()[idxs, :][:, idxs], self.v.copy()[idxs])

    def contains(self, x: np.ndarray) -> np.ndarray:
        ''' Whether the ellipsoid contains said points '''
        if x.ndim == 1:
            x = x[None, :]
        assert x.ndim == 2, 'x must be 2d'
        assert x.shape[1] == self.ndim, 'x must have same dimension as ellipsoid'
        x_ = x - self.v[None, :]
        return np.sum((x_ @ self.M) * x_, axis=1) <= 1

    def d_surface_area(self, tol=1e-3) -> float:
        ''' Proportional surface area of ellipsoid (only proportionally accurate to those within the same dimension d) '''
        _, rs = self.get_axes_stretches()
        return np.prod(rs)

    def circular_radius(self) -> float:
        ''' Geometric mean of major and minor axes '''
        _, rs = self.get_axes_stretches()
        return np.prod(rs) ** (1/self.ndim)
    
    def area(self) -> float:
        assert self.ndim == 2, 'Ellipsoid must be 2d'
        return np.pi * la.det(self.M) ** (-0.5) # Slightly more accurate?
    
    def volume(self) -> float:
        return Ellipsoid.vol_n_sphere(self.ndim) * la.det(self.M) ** (-0.5)
    
    def eccentricity(self) -> float:
        assert self.ndim == 2, 'Ellipsoid must be 2d'
        _, rs = self.get_axes_stretches()
        b, a = rs
        return 1.0 - a**2/b**2 

    def perimeter(self) -> float:
        '''
        Approximation by elliptic integral
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.ellipe.html
        '''
        assert self.ndim == 2, 'Ellipsoid must be 2d'
        ecc = self.eccentricity()
        b = self.get_major_radius()
        return 4 * b * special.ellipe(ecc)

    def rescale(self, scale: np.ndarray) -> 'Ellipsoid':
        ''' Rescale the polygon to a given resolution, with origin in the original basis of the coordinates '''
        assert scale.ndim == 1, 'scale must be 1d'
        assert scale.shape[0] == self.ndim, 'scale must have same dimension as ellipsoid'
        assert np.all(scale > 0), 'scale must be positive'
        v = self.v * scale
        # Apply transformation so if x^T M x = 1, then (x*scale)^T M' (x*scale) = 1
        scaleinv = np.diag(1/scale)
        M = scaleinv @ self.M @ scaleinv
        return Ellipsoid(M, v)
    
    def center(self) -> 'Ellipsoid':
        ''' Center the ellipsoid at the origin '''
        return Ellipsoid(self.M, np.zeros(self.ndim))
    
    def aspect_ratio(self) -> float:
        ''' Ratio of major to minor axes '''
        _, rs = self.get_axes_stretches()
        return rs.max() / rs.min()
    
    def match_volume(self, volume: float) -> 'Ellipsoid':
        ''' Rescale the ellipsoid to have a given volume '''
        return self * (volume / self.volume()) ** (1/self.ndim)
    
    def match_area(self, area: float) -> 'Ellipsoid':
        ''' Rescale the ellipsoid to have a given area '''
        assert self.ndim == 2, 'Ellipsoid must be 2d'
        return self.match_volume(area)
    
    def discretize(self, n: int) -> PlanarPolygon:
        ''' Descretize the ellipse into a polygon '''
        assert self.ndim == 2, 'Ellipsoid must be 2d'
        us = np.linspace(0, 2*np.pi, n, endpoint=False)
        X = np.vstack((np.cos(us), np.sin(us))).T
        X = self.map_sphere(X)
        return PlanarPolygon(X)
    
    def align_axes(self, order='ascending') -> Tuple['Ellipsoid', callable, np.ndarray]:
        '''
        Align ellipsoid principal radii with axes [x, y, z, ...] in specified order
        Return affine transform T such that T(x) = y where y is the aligned coordinates
        '''
        L, P = la.eigh(self.M)
        v = self.v.copy()
        assert order in ['ascending', 'descending'], 'order must be ascending or descending'
        if order == 'ascending':
            L = L[::-1]
            P = P[:, ::-1]
        T = lambda x: (x - v) @ P
        return Ellipsoid(np.diag(L), np.zeros_like(self.v)), T, v.copy()
    
    def invert_align_axes(self, order='ascending') -> callable:
        '''
        Get the inverse transform of align_axes()
        Align ellipsoid principal radii with axes [x, y, z, ...] in specified order
        Return affine transform T_ such that T_(y) = x where y is the aligned coordinates
        '''
        L, P = la.eigh(self.M)
        v = self.v.copy()
        assert order in ['ascending', 'descending'], 'order must be ascending or descending'
        if order == 'ascending':
            L = L[::-1]
            P = P[:, ::-1]
        T = lambda y: y @ P.T + v
        return T
    
    def get_affine_matrix(self, order='descending') -> np.ndarray:
        '''
        Same as invert_align_axes(), in affine matrix form
        '''
        _, P = la.eigh(self.M)
        assert order in ['ascending', 'descending'], 'order must be ascending or descending'
        if order == 'ascending':
            P = P[:, ::-1]
        return np.vstack((
            np.hstack((P, self.v[:, None])),
            np.array([0, 0, 1])
        ))
    
    def apply_affine(self, T: np.ndarray) -> 'Ellipsoid':
        ''' Apply affine transform T to the ellipsoid '''
        d = self.ndim
        P = T[:d, :d]
        M = P @ self.M @ P.T
        v = P @ self.v + T[:d, d]
        return Ellipsoid(M, v)
    
    def copy(self) -> 'Ellipsoid':
        return Ellipsoid(self.M.copy(), self.v.copy())
    
    def plane_intersection(self, plane: Plane) -> np.ndarray:
        '''
        Compute intersection of plane with ellipsoid, either as points (2D) or as an ellipse (3D)
        '''
        assert self.ndim == plane.ndim, 'Ellipsoid and plane must have the same ambient dimension'
        if self.ndim == 2:
            # 1. Put ellipse in standard form (center at origin, axes aligned)
            ell, T, v = self.align_axes()
            # 2. Apply affine transform T to plane to put it in the same basis
            plane = plane.affine_transform(T, v)
            # 3. Solve resulting quadratic equation x^2 / a^2 + (mx + c)^2 / b^2 = 1 for real roots
            a, b = ell.get_stretches_diagonal()
            m, c = plane.slope_intercept()
            # (a^2m^2 + b^2)x^2 + 2a^2mcx + a^2(c^2-b^2) = 0
            roots = np.roots(np.array([
                a**2 * m**2 + b**2,
                2 * a**2 * m * c,
                a**2 * (c**2 - b**2),
            ]))
            roots = roots[np.isclose(roots.imag, 0)].real
            # 4. Get 2D points in original basis
            pts = np.vstack((roots, m * roots + c)).T
            T_ = self.invert_align_axes()
            return T_(pts)
        elif self.ndim == 3:
            raise NotImplementedError('returning 2D ellipses in arbitrary 3D position not yet implemented')
        else:
            raise NotImplementedError('Hyperplane/Hyperellipsoid intersection not implemented')

    def get_stretches_diagonal(self) -> np.ndarray:
        '''
        Get the stretches in descending order when aligned with the principal axes
        '''
        assert np.count_nonzero(self.M - np.diag(np.diagonal(self.M))) == 0, 'Ellipse must be aligned so M is diagonal'
        return 1 / np.sqrt(np.diagonal(self.M))
        
    def voronoi_tessellate(self, pts):
        raise NotImplementedError
    
    def mahalanobis_distance(self, x: np.ndarray) -> float:
        return np.sqrt((x - self.v) @ self.M @ (x - self.v))
        
    @staticmethod
    def from_poly(poly: PlanarPolygon, equiarea: bool=True) -> 'Ellipsoid':
        v = poly.centroid()
        Sigma = poly.covariance_matrix()
        M = la.inv(Sigma) 
        ell = Ellipsoid(M, v)
        if equiarea:
            # Re-scale ellipsoid to have same area as polygon
            ell = ell.match_area(poly.area())
        return ell
    
    @staticmethod
    def vol_n_sphere(n: int) -> float:
        ''' Volume of n-sphere '''
        return np.pi ** (n/2) / special.gamma(n/2 + 1)
    
    @staticmethod
    def n_sphere(n: int, v: np.ndarray=None, r: float=1) -> 'Ellipsoid':
        ''' n-sphere '''
        v = np.zeros(n+1) if v is None else v
        return Ellipsoid(np.eye(n+1) * r**(-2), v)
    
    @staticmethod
    def circle(v: np.ndarray=None, r: float=1) -> 'Ellipsoid':
        return Ellipsoid.n_sphere(1, v=v, r=r)
    
    @staticmethod
    def sphere(v: np.ndarray=None, r: float=1) -> 'Ellipsoid':
        return Ellipsoid.n_sphere(2, v=v, r=r)

class Ellipse(Ellipsoid):
    def __init__(self, M: np.ndarray, v: np.ndarray):
        assert v.shape == (2,), 'v must be 2d'
        super().__init__(M, v)

    def __add__(self, x: np.ndarray):
        return Ellipse(self.M.copy(), self.v + x)

    def __sub__(self, x: np.ndarray):
        return Ellipse(self.M.copy(), self.v - x)

    def __mul__(self, x: float):
        return Ellipse(self.M * 1/(x**2), self.v.copy())

    def __truediv__(self, x: float):
        return Ellipse(self.M * (x**2), self.v.copy())

    def flipy(self, yval: float) -> 'Ellipse':
        Iy = np.diag([1, -1])
        b = np.array([0, yval]) 
        v = Iy @ self.v + b
        M = Iy @ self.M @ Iy.T
        return Ellipse(M, v)
    
    def copy(self) -> 'Ellipse':
        return Ellipse(self.M.copy(), self.v.copy())
    
    def get_rotation(self) -> float:
        ''' Get angle of rotation with respect to COM '''
        P, _ = self.get_axes_stretches()
        theta = np.arctan2(P[0,1], P[0,0]) # Angle from orthogonal matrix P.T
        return theta

    def revolve_major(self, v_z: float=0.0) -> Ellipsoid:
        ''' Revolve about major axis '''
        return self.revolve(0, v_z)
    
    def revolve_minor(self, v_z: float=0.0) -> Ellipsoid:
        ''' Revolve about minor axis '''
        return self.revolve(1, v_z)
    
    def revolve(self, axis: int, v_z: float=0.0) -> Ellipsoid:
        '''
        Transform 2D ellipse to 3D ellipsoid using specified axis as axis of rotation
        (defaults to z-coordinate in the XY plane)
        '''
        assert axis in [0, 1], 'axis of revolution must be major or minor axis'
        L, P = la.eigh(self.M)
        L_ = np.array([L[0], L[1], L[1-axis]]) # Eigenvalues in ascending order
        P_ = np.zeros((3, 3))
        P_[:2, :2] = P # Original eigenvectors remain in plane
        P_[:, 2] = np.array([0, 0, 1]) # New eigenvector is z-axis
        M_ = P_ @ np.diag(L_) @ P_.T
        v_ = np.array([self.v[0], self.v[1], v_z])
        return Ellipsoid(M_, v_)
    
    @staticmethod
    def from_ellipsoid(ell: Ellipsoid) -> 'Ellipse':
        return Ellipse(ell.M.copy(), ell.v.copy())
    
    @staticmethod
    def from_poly(poly: PlanarPolygon) -> 'Ellipse':
        return Ellipse.from_ellipsoid(Ellipsoid.from_poly(poly))
    
    @staticmethod
    def random(*args, **kwargs) -> 'Ellipse':
        return Ellipse.from_ellipsoid(Ellipsoid.random(*args, **kwargs))
    
if __name__ == '__main__':

    ''' Tests '''

    # Test area matching
    for _ in range(1000):
        ell = Ellipsoid.random(sigma=100, d=2)
        assert np.isclose(ell.area(), ell.volume()), 'Area and volume of 2d ellipsoid must be equal'
        area = np.random.uniform(0.1, 100)
        ell_ = ell.match_area(area)
        assert np.isclose(area, ell_.area()), 'Area of rescaled ellipsoid must match input area'
    print('Passed area matching test.')

    # Test revolve
    for _ in range(1000):
        ell = Ellipse.random(sigma=100, d=2)
        ell1 = ell.revolve_major()
        ell2 = ell.revolve_minor()
        major, minor = ell.get_radii()
        assert major > minor
        c = np.pi * 4/3
        try:
            assert np.isclose(ell1.volume(), c * major * minor**2)
            assert np.isclose(ell2.volume(), c * major**2 * minor)
        except AssertionError:
            pdb.set_trace()
    print('Passed revolve test.')

    # import matplotlib.pyplot as plt
    # import mpl_tools as pt
    
    # # Sample randomly points on a sphere
    # N = 1000
    # X = np.random.randn(N, 3)
    # X /= np.linalg.norm(X, axis=1, keepdims=True)

    # # Apply random stretches
    # rs = np.random.uniform(0.5, 3, size=(3,))
    # X = X * rs

    # # Apply random rotation
    # R = np.random.randn(3, 3) * 5
    # R, _ = np.linalg.qr(R)
    # X = X @ R.T

    # # Apply random translation
    # v = np.random.randn(3) * 5
    # X = X + v

    # # Add noise
    # X = X + np.random.normal(0, 0.1, size=X.shape)

    # # Fit ellipsoid
    # # ell = Ellipsoid.fit_outer(X)
    # ell = Ellipsoid.fit_outer_iterative(X, tol=1e-3)
    # # ell = Ellipsoid.fit_l2(X)
    # plane = ell.get_major_plane()
    # print('Fit outer ellipsoid.')

    # # Project to ellipsoid
    # Y = ell.project_l2(X)
    # print('Projected points to ellipsoid.')

    # # Plot
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(X[:, 0], X[:, 1], X[:, 2])
    # ax.scatter(Y[:, 0], Y[:, 1], Y[:, 2])
    # # Plot lines from X to Y
    # for i in range(N):
    #     ax.plot([X[i, 0], Y[i, 0]], [X[i, 1], Y[i, 1]], [X[i, 2], Y[i, 2]], 'r')
    # X = ell.sample_mgrid()
    # ax.plot_surface(X[..., 0], X[..., 1], X[..., 2], color='y', alpha=0.3)
    # pt.ax_plane(ax, plane)
    # plt.show()

    
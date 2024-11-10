from .surface import *
from .ellipsoid import *

''' Sphere '''

class Sphere(Ellipsoid):
    '''
    Convenience class to distinguish by type
    '''
    def __init__(self, v: np.ndarray, r: float):
        n = v.shape[0]
        M = np.eye(n) * r**(-2)
        super().__init__(M, v)

    @property
    def r(self) -> float:
        return self.M[0, 0] ** (-1/2)
    
    # Define setter for r
    @r.setter
    def r(self, r: float):
        self.M = np.eye(self.ndim) * r**(-2)

    def __add__(self, x: np.ndarray):
        return Sphere(self.v + x, self.r)

    def __sub__(self, x: np.ndarray):
        return Sphere(self.v - x, self.r)

    def __mul__(self, x: float):
        return Sphere(self.v.copy(), self.r * x)

    def __truediv__(self, x: float):
        return Sphere(self.v.copy(), self.r / x)
    
    def copy(self) -> 'Sphere':
        return Sphere(self.v.copy(), self.r)

    @staticmethod
    def from_poly(poly: PlanarPolygon) -> 'Sphere':
        v = poly.centroid()
        r = np.sqrt(poly.area() / np.pi)
        return Sphere(v, r)
    
    @staticmethod
    def from_ellipsoid(ell: Ellipsoid) -> 'Sphere':
        d = ell.ndim
        M_ = np.eye(d) * np.trace(ell.M) / d
        assert np.allclose(M_, ell.M), 'Ellipse matrix must be isotropic to cast into sphere'
        r = ell.M[0,0] ** (-1/2)
        return Sphere(ell.v, r)
    
class SphericalPolygon(SurfacePolygon):
    '''
    Polygon on surface of 2-sphere.
    '''
    def __init__(self, vertices_nd: np.ndarray, sphere: Sphere):
        super().__init__(vertices_nd, surface=sphere)

class SphericalPartition(SurfacePartition):
    pass
    
''' Circle'''
    
class Circle(Sphere):
    def __init__(self, v: np.ndarray, r: float):
        assert v.shape == (2,), 'v must be 2d'
        super().__init__(v, r)

    def __add__(self, x: np.ndarray):
        return Circle(self.v + x, self.r)

    def __sub__(self, x: np.ndarray):
        return Circle(self.v - x, self.r)

    def __mul__(self, x: float):
        return Circle(self.v.copy(), self.r * x)

    def __truediv__(self, x: float):
        return Circle(self.v.copy(), self.r / x)

    def flipy(self, yval: float) -> 'Circle':
        ''' Flip the y-coordinate of the center of the circle '''
        return Circle(np.array([self.v[0], yval-self.v[1]]), self.r)
    
    def copy(self) -> 'Circle':
        return Circle(self.v.copy(), self.r)
    
    @staticmethod
    def from_ellipse(ell: Ellipse) -> 'Circle':
        sph = Sphere.from_ellipsoid(ell)
        return Circle(sph.v, sph.r)
    
    @staticmethod
    def from_sphere(sph: Sphere) -> 'Circle':
        return Circle(sph.v, sph.r)
    
    @staticmethod
    def from_poly(poly: PlanarPolygon) -> 'Circle':
        return Circle.from_sphere(Sphere.from_poly(poly))
# Import compatibility patches first
from . import compat

from .ellipsoid import Ellipsoid, Ellipse
from .sphere import Sphere, Circle
from .plane import *
from .triangulation import Triangulation
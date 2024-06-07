from __future__ import print_function
from sys import exit
from sys import stderr

from CGAL.CGAL_Kernel import Point_3
from CGAL.CGAL_Polyhedron_3 import Polyhedron_3
from CGAL.CGAL_Point_set_3 import Point_set_3
# from CGAL.CGAL_Tr
from CGAL.CGAL_Advancing_front_surface_reconstruction import *

import numpy as np
import pdb

points = Point_set_3("/Users/asrvsn/Downloads/oni.xyz")
if points.empty():
    stderr.write("Error: cannot read file ../data/oni.xyz\n")
    exit()

P = Polyhedron_3()

advancing_front_surface_reconstruction(points, P)

# pts = np.array([[p.x(), p.y(), p.z()] for p in points])
# pdb.set_trace()

P.write_to_file("oni.off")

# Read file as numpy array of facets

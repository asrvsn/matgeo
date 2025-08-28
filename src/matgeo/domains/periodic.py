'''
A collection of periodic domain construction helpers
'''
import numpy as np
import pygmsh

from ..triangulation import Triangulation

def swiss_cheese(centers: np.ndarray, radii: np.ndarray, lcar_min: float, lcar_max: float) -> Triangulation:
    """
    Creates a 2D triangulation for a periodic domain [0, 1)^2 with circular holes.
    Holes are periodized: if a disk overlaps an edge, its image appears on the opposite side.
    """
    centers = np.asarray(centers, dtype=float)
    radii = np.asarray(radii, dtype=float)
    assert centers.ndim == 2 and centers.shape[1] == 2
    assert radii.ndim == 1 and radii.shape[0] == centers.shape[0]

    def circle_intersects_unit_square(c: np.ndarray, r: float) -> bool:
        # AABB intersection test with [0,1]x[0,1]
        return (c[0] + r > 0.0) and (c[0] - r < 1.0) and (c[1] + r > 0.0) and (c[1] - r < 1.0)

    offsets = np.array(
        [[-1.0, -1.0], [-1.0, 0.0], [-1.0, 1.0],
         [ 0.0, -1.0], [ 0.0, 0.0], [ 0.0, 1.0],
         [ 1.0, -1.0], [ 1.0, 0.0], [ 1.0, 1.0]],
        dtype=float
    )

    with pygmsh.geo.Geometry() as geom:
        domain = geom.add_rectangle([0.0, 0.0, 0.0], 1.0, 1.0, mesh_size=lcar_max)
        holes = []
        for c, r in zip(centers, radii):
            for off in offsets:
                c_img = c + off
                if circle_intersects_unit_square(c_img, float(r)):
                    holes.append(geom.add_disk([float(c_img[0]), float(c_img[1]), 0.0], float(r), mesh_size=lcar_min))

        if holes:
            geom.boolean_difference(domain, geom.boolean_union(holes))

        mesh = geom.generate_mesh()

    points = mesh.points[:, :2]
    cells = mesh.get_cells_type("triangle")
    return Triangulation(points, cells)

if __name__ == "__main__":
    pass

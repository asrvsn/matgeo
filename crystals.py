import freud
import matplotlib.cm
import numpy as np
import rowan
import plato.draw.fresnel
backend = plato.draw.fresnel

def fcc_lattice(nx=1, ny=1, nz=1, scale=1.0, noise=0.0):
    """Make an FCC crystal for testing

    :param nx: Number of repeats in the x direction, default is 1
    :param ny: Number of repeats in the y direction, default is 1
    :param nz: Number of repeats in the z direction, default is 1
    :param scale: Amount to scale the unit cell by (in distance units),
        default is 1.0
    :param noise: Apply Gaussian noise with this width to particle positions
    :type nx: int
    :type ny: int
    :type nz: int
    :type scale: float
    :type noise: float
    :return: freud Box, particle positions, shape=(nx*ny*nz, 3)
    :rtype: (:class:`freud.box.Box`, :class:`np.ndarray`)
    """
    fractions = np.array(
        [[0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5], [0, 0, 0]], dtype=np.float32
    )

    fractions = np.tile(
        fractions[np.newaxis, np.newaxis, np.newaxis], (nx, ny, nz, 1, 1)
    )
    fractions[..., 0] += np.arange(nx)[:, np.newaxis, np.newaxis, np.newaxis]
    fractions[..., 1] += np.arange(ny)[np.newaxis, :, np.newaxis, np.newaxis]
    fractions[..., 2] += np.arange(nz)[np.newaxis, np.newaxis, :, np.newaxis]
    fractions /= [nx, ny, nz]

    box = 2 * scale * np.array([nx, ny, nz], dtype=np.float32)
    positions = ((fractions - 0.5) * box).reshape((-1, 3))

    if noise != 0:
        positions += np.random.normal(scale=noise, size=positions.shape)

    return freud.box.Box(*box), positions

def plot_crystal(box, positions, colors=None, radii=None, backend=None,
                 polytopes=[], polytope_colors=None):
    if backend is None:
        backend = plato.draw.fresnel
    if colors is None:
        colors = np.array([[0.5, 0.5, 0.5, 1]] * len(positions))
    if radii is None:
        radii = np.array([0.5] * len(positions))
    sphere_prim = backend.Spheres(positions=positions, colors=colors, radii=radii)
    box_prim = backend.Lines(
        start_points=box.makeCoordinates(
            [[0, 0, 0], [0, 0, 0], [0, 0, 0], [1, 1, 0],
             [1, 1, 0], [1, 1, 0], [0, 1, 1], [0, 1, 1],
             [0, 1, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1]]),
        end_points=box.makeCoordinates(
            [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0],
             [0, 1, 0], [1, 1, 1], [1, 1, 1], [0, 1, 0],
             [0, 0, 1], [0, 0, 1], [1, 1, 1], [1, 0, 0]]),
        widths=0.1,
        colors=[0, 0, 0, 1])
    if polytope_colors is None:
        polytope_colors = colors * np.array([1, 1, 1, 0.4])
    polytope_prims = []
    for p, c in zip(polytopes, polytope_colors):
        p_prim = backend.ConvexPolyhedra(
            positions=[[0, 0, 0]], colors=c, vertices=p, outline=0)
        polytope_prims.append(p_prim)
    rotation = rowan.multiply(
        rowan.from_axis_angle([1, 0, 0], np.pi/10),
        rowan.from_axis_angle([0, 1, 0], -np.pi/10))
    scene = backend.Scene([sphere_prim, box_prim, *polytope_prims],
                          zoom=3, rotation=rotation)
    if backend is not plato.draw.fresnel:
        scene.enable('directional_light')
    #else:
    #    scene.enable('antialiasing')
    scene.show()

def plot_bcc(ax, n):
    box, positions = fcc_lattice(nx=n, ny=n, nz=n, scale=1.5)
    positions = box.wrap(positions)
    # cmap = mpl.cm.get_cmap('tab20')
    # colors = cmap(np.random.rand(len(positions)))
    voro = freud.voronoi.Voronoi(box)
    voro.compute(positions, buff=np.max(box.L)/2)
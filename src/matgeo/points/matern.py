'''
Sampling Matern point processes
'''
import numpy as np

from asrvsn_math.torus import dists_td

def matern_II_torus(mu: float, r: float, rng=np.random.default_rng()) -> np.ndarray:
    '''
    Type-II Matern point process on the torus [0, 1)^2
    '''
    assert r > 0
    assert mu > 0
    n = rng.poisson(mu)
    if n == 0:
        return np.empty((0, 2), dtype=float)

    pts = rng.uniform(0.0, 1.0, (n, 2))
    marks = rng.uniform(0.0, 1.0, n)
    ds = dists_td(pts)
    np.fill_diagonal(ds, np.inf)
    overlapping = ds < (2 * r)
    overlapping_marks = np.where(overlapping, marks[None, :], np.inf)
    min_overlapping_mark = overlapping_marks.min(axis=1)
    keep = marks < min_overlapping_mark
    return pts[keep]

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import asrvsn_mpl as pt
    r = 0.1
    pts = matern_II_torus(100, r)
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    pt.ax_square(ax, np.array([0.5, 0.5]), 0.5)
    for center in pts:
        pt.ax_circle(ax, center, r, color='black', linewidth=0.5)
    plt.show()
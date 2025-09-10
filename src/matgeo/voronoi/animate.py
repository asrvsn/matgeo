'''
Utilities for animating Voronoi diagram formation 
'''
import os
import numpy as np
import matplotlib.pyplot as plt

from asrvsn_math.torus import cdist_td

def colliding_fronts_torus(
        folder: str,
        pts: np.ndarray,    
        r: float,
        tmax: float=np.inf,
        dt: float=0.1,
        ndiscret: int=1000,
    ):
    '''
    Animate colliding fronts at unit speedfrom seed points at given starting radii on the torus [0, 1)^2.
    '''
    assert pts.ndim == pts.shape[1] == 2
    assert r > 0
    assert tmax > 0
    assert dt > 0
    assert ndiscret > 0

    os.makedirs(folder, exist_ok=True)
    # assert len(os.listdir(folder)) == 0, 'Folder must be empty'
    os.system(f'rm -rf {folder}/*')

    n = pts.shape[0]
    theta = np.linspace(0.0, 2.0 * np.pi, ndiscret, endpoint=False)  # (ndiscret,)
    unit = np.stack([np.cos(theta), np.sin(theta)], axis=1)          # (ndiscret, 2)
    seeds = pts[:, None, :] + r * unit[None, :, :]                  # (n, ndiscret, 2)
    normals = unit[None, :, :].copy()                                # (n, ndiscret, 2)
    displacements = np.zeros((n, ndiscret), dtype=float)

    Delta = pts[None, :, :] - pts[:, None, :]    # shape (n, n, 2): j - i
    Delta -= np.rint(Delta)
    dist2 = np.sum(Delta**2, axis=2)             # (n, n)
    dot = np.tensordot(Delta, unit.T, axes=([2], [0]))  # (n, n, K)
    with np.errstate(divide='ignore', invalid='ignore'):
        s_ijk = dist2[:, :, None] / (2.0 * dot)         # (n, n, K)
    s_ijk = np.where(dot > 0.0, s_ijk, np.inf)
    idx = np.arange(n)
    s_ijk[idx, idx, :] = np.inf                         # ignore self
    s_dir = np.min(s_ijk, axis=1)                       # (n, K)
    dmax  = np.maximum(s_dir - r, 0.0)                  # (n, K)  extra growth beyond r

    # dmax = cdist_td(pts)
    # np.fill_diagonal(dmax, np.inf)
    # dmax = dmax.min(axis=1) / 2.0

    fig, ax = plt.subplots(figsize=(5,5))
    frame_id = 0
    t = 0.0

    while True:
        ax.clear()
        ax.set_aspect('equal')
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

        dt_ = dt if frame_id > 0 else 0.0
        displacements += dt_
        if np.all(displacements >= dmax):
            break
        displacements = np.minimum(displacements, dmax)
        fronts = seeds + displacements[:, :, None] * normals

        for ox in (-1.0, 0.0, 1.0):
            for oy in (-1.0, 0.0, 1.0):
                offset = np.array([ox, oy], dtype=float)[None, None, :]  # (1,1,2)

                seeds_ = seeds + offset
                xb = np.concatenate([seeds_[:, :, 0], seeds_[:, :1, 0]], axis=1)
                yb = np.concatenate([seeds_[:, :, 1], seeds_[:, :1, 1]], axis=1)
                for i in range(n):
                    ax.plot(xb[i], yb[i], '-', lw=0.7, color='red')

                fronts_ = fronts + offset
                xa = np.concatenate([fronts_[:, :, 0], fronts_[:, :1, 0]], axis=1)
                ya = np.concatenate([fronts_[:, :, 1], fronts_[:, :1, 1]], axis=1)
                for i in range(n):
                    ax.plot(xa[i], ya[i], '-', lw=0.7, color='green')

        out_path = os.path.join(folder, f"frame_{frame_id:05d}.png")
        fig.savefig(out_path, dpi=200, bbox_inches='tight', pad_inches=0.0)
        print(f"Saved frame {frame_id} to {out_path}")
        frame_id += 1

        t += dt_
        if t > tmax:
            break
    
    plt.close(fig)


if __name__ == '__main__':
    import argparse
    from matgeo.points.matern import matern_II_torus

    parser = argparse.ArgumentParser(description='Animate colliding fronts on the flat torus and write frames to a folder.')
    parser.add_argument('folder', type=str, help='Output folder (must be empty)')
    parser.add_argument('--mu', type=float, default=1000, help='Poisson mean for number of seeds')
    parser.add_argument('--r', type=float, default=0.1, help='Initial front radius and hard-core radius for sampling')
    parser.add_argument('--tmax', type=float, default=np.inf, help='Maximum time to simulate')
    parser.add_argument('--dt', type=float, default=0.025, help='Time step')
    parser.add_argument('--ndiscret', type=int, default=1000, help='Angular discretization of fronts')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    args = parser.parse_args()

    if args.seed is not None:
        rng = np.random.default_rng(args.seed)
    else:
        rng = np.random.default_rng()

    pts = matern_II_torus(args.mu, args.r, rng=rng)
    colliding_fronts_torus(
        folder=args.folder,
        pts=pts,
        r=args.r,
        tmax=args.tmax,
        dt=args.dt,
        ndiscret=args.ndiscret,
    )
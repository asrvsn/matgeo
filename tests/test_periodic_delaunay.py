'''
Testing periodic triangulations
'''

if __name__ == '__main__':
    import numpy as np
    import pandas as pd
    import timeit
    from tqdm import tqdm

    from matgeo.spv_math import triangulate_periodic
    from matgeo.triangulation import Triangulation

    lam = 500
    rng = np.random.default_rng(0)
    box1 = np.array([1,1])
    box2 = np.array([[0,0], [1,1]])

    ## Correctness

    n_test = 100
    for _ in tqdm(range(n_test)):
        n = rng.poisson(lam)
        xs = rng.uniform(0, 1, size=(n, 2))
        simp1 = triangulate_periodic(xs, box1)
        # print(f'SPV got {len(simp1)} simplices')
        simp2 = Triangulation.periodic_delaunay(xs, box2).simplices
        # print(f'CGAL got {len(simp2)} simplices')
        # assert np.all(simp1 == simp2)

        simp1_set = set(map(frozenset, simp1))
        simp2_set = set(map(frozenset, simp2))
        # print(f'CGAL and SPV have {len(simp1_set & simp2_set)} common simplices')
        print(f'SPV and CGAL agree: {simp1_set == simp2_set}')

        # fig, axs = pt.default_mosaic([['a', 'b']])
        # pt.ax_tri_2d(axs['a'], xs, simp1)
        # axs['a'].set_title('CGAL')

        # pt.ax_tri_2d(axs['b'], xs, simp2)
        # axs['b'].set_title('SPV')

        # plt.show()
    
    ## Performance

    n = rng.poisson(lam)
    xs = rng.uniform(0, 1, size=(n, 2))
    n_loops = 100
    n_repeats = 10

    times1 = timeit.repeat(lambda: triangulate_periodic(xs, box1), repeat=n_repeats, number=n_loops)
    print('SPV:')
    print(pd.Series(times1).describe())

    times2 = timeit.repeat(lambda: Triangulation.periodic_delaunay(xs, box2).simplices, repeat=n_repeats, number=n_loops)
    print('CGAL:')
    print(pd.Series(times2).describe())
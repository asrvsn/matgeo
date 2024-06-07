'''
Parallel utilities
'''

from typing import Callable, Any, List
import joblib as jl
import multiprocessing as mp
import numpy as np
from tqdm import tqdm
import pdb

def parallel_farm_array(
        setup_fun: Callable[[], Any],
        compute_fun: Callable[[Any], Any],
        items: List[Any],
        nj: int=max(1, mp.cpu_count() - 2),  
        name: str='',
    ) -> np.ndarray:
    '''
    Parallel map with setup & worker reuse
    '''
    items = list(items)
    n = len(items)
    nperj = np.ceil(n / nj).astype(int)
    def fun(j):
        i0, i1 = j * nperj, min((j + 1) * nperj, n)
        assert i0 < n, f'Invalid i0={i0}, n={n}'
        ctx = setup_fun()
        return np.array([
            compute_fun(ctx, item) for item in tqdm(items[i0:i1], desc=f'Computing {name} ({i0}-{i1})')
        ])
    results = np.concatenate(
        jl.Parallel(n_jobs=nj)(jl.delayed(fun)(j) for j in range(nj)),
        axis=0
    )
    return results
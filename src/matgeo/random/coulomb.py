import numpy as np

def sample_ginibre(n: int, sigma: float=1, rng=None) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()
    A = (rng.normal(0, sigma, (n, n)) + 1j * rng.normal(0, sigma, (n, n))) / np.sqrt(2)

    evs = np.linalg.eigvals(A)
    evs = np.concatenate((evs.real[:, None], evs.imag[:, None]), axis=1)
    evs /= np.sqrt(n) * sigma

    return evs
import numpy as np
import numba
from math import sqrt


@numba.jit(nopython=True, fastmath=True)
def theta_distance(a, b):
    # type: (np.ndarray, np.ndarray) -> float
    """
    Euclidean distance between point `a` and point `b`
    :param a: Point a
    :param b: Point b
    :return: Euclidean distance
    """

    assert a.shape[0] == b.shape[0]

    val = 0.
    for i in range(a.shape[0]):
        val += np.abs(a[i] - b[i])
    return val

@numba.jit(nopython=True, fastmath=True)
def euclidean_distance(a, b):
    # type: (np.ndarray, np.ndarray) -> float
    """
    Euclidean distance between point `a` and point `b`
    :param a: Point a
    :param b: Point b
    :return: Euclidean distance
    """

    assert a.shape[0] == b.shape[0]

    val = 0.
    for i in range(a.shape[0]):
        val += (a[i] - b[i]) ** 2
    return sqrt(val)

@numba.jit(nopython=True, fastmath=True)
def hu_distance(t1, t2, w=None):
    # type: (np.ndarray, np.ndarray, np.ndarray) -> float
    """
    Compute the HU distance as the average Euclidean distance between points
    on two trajectories `t1` and `t2`

    :param t1: Trajectory a (L x coord)
    :param t2: Trajectory b (L x coord)
    :return: Hu distance
    """

    d = float(0)
    assert t1.shape[0] == t2.shape[0]

    for i in range(t1.shape[0]):
        _d = euclidean_distance(t1[i], t2[i])
        d += _d * w[i] if w is not None else _d

    return (d / t1.size)


@numba.jit(nopython=True, fastmath=True)
def hausdorff_distance(t1, t2):
    # type: (np.ndarray, np.ndarray) -> float
    """
    Compute the Hausdorff distance between two trajectories `t1` and `t2`

    :param t1: Trajectory a
    :param t2: Trajectory b
    :return: hausdorff distance
    """

    nA = t1.shape[0]
    nB = t2.shape[0]
    cmax = 0.
    for i in range(nA):
        cmin = np.inf
        for j in range(nB):
            d = euclidean_distance(t1[i, :], t2[j, :])
            if d<cmin:
                cmin = d
            if cmin<cmax:
                break
        if cmin>cmax and np.inf>cmin:
            cmax = cmin
    for j in range(nB):
        cmin = np.inf
        for i in range(nA):
            d = euclidean_distance(t1[i, :], t2[j, :])
            if d<cmin:
                cmin = d
            if cmin<cmax:
                break
        if cmin>cmax and np.inf>cmin:
            cmax = cmin
    return cmax


@numba.jit(nopython=True, fastmath=True)
def lcss_distance(t1, t2, th=100, delta=10):
    # type: (np.ndarray, np.ndarray, int, int) -> float
    """
    Compute LCSS distance between `t1` and `t2` trajectories

    :param t1: Trajectory a
    :param t2: Trajectory b
    :param th: Threshold
    :param delta: Delta
    :return: LCSS distance
    """


    m = t1.shape[0]
    n = t2.shape[0]

    L = np.full((n+1, m+1), 0.)

    for i in range(m+1):
        for j in range(n+1):
            if i == 0 or j == 0:
                L[i, j] = 0
            elif euclidean_distance(t1[i - 1], t2[j - 1]) < th and abs(i-j) < delta:
                L[i, j] = L[i-1, j-1] + 1
            else:
                L[i, j] = max(L[i-1, j], L[i, j - 1])

    return 1 - (L[-1, -1] / min(m, n))


@numba.jit(nopython=True, fastmath=True)
def dist_to_sim_mx(dist_matrix, sigma=15):
    # type: (np.ndarray, float) -> np.ndarray
    """
    Distance matrix to similarity matrix using Gaussian Kernel

    :param dist_matrix:  Distance matrix
    :param sigma: Sigma
    :return: Similarity matrix
    """
    return np.exp(- dist_matrix ** 2 / (2. * sigma ** 2))

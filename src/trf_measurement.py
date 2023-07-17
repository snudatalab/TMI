import functools
import numpy as np

from scipy.spatial import cKDTree
from scipy.special import gamma, digamma

log = np.log


def convert_vectors_to_2d_arrays_if_any(func):
    """
    Convert vectors to 2d arrays if any of the arguments is a vector
    Args:
        func: function to be wrapped

    Returns: wrapped function

    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        args = list(args)
        for ii, arg in enumerate(args):
            if isinstance(arg, (list, tuple, np.ndarray)):
                if np.ndim(arg) == 1:
                    args[ii] = np.array(arg)[:, np.newaxis]
                elif np.ndim(arg) == 2:
                    pass
                else:
                    raise ValueError("Arrays should have one or two dimensions.")
        for k, v in kwargs.items():
            if isinstance(v, (list, tuple, np.ndarray)):
                if np.ndim(v) == 1:
                    kwargs[k] = v[:, np.newaxis]
                elif np.ndim(v) == 2:
                    pass
                else:
                    raise ValueError("Arrays should have one or two dimensions.")
        return func(*args, **kwargs)

    return wrapper


@convert_vectors_to_2d_arrays_if_any
def get_h(x, k=1, norm='max', min_dist=3., workers=40):
    """
    Estimates the entropy H of a random variable x (in nats)
    Args:
        x: input (n, d) ndarray
        k: number of nearest neighbors
        norm: p-norm used when computing k-nearest neighbour distances
        min_dist: minimum distance between data points
        workers: number of workers

    Returns: entropy H(X)

    """

    n, d = x.shape

    if norm == 'max':  # max norm:
        p = np.inf
        log_c_d = 0  # volume of the d-dimensional unit ball
    elif norm == 'euclidean':  # euclidean norm
        p = 2
        log_c_d = (d / 2.) * log(np.pi) - log(gamma(d / 2. + 1))
    else:
        raise NotImplementedError("Variable 'norm' either 'max' or 'euclidean'")

    kdtree = cKDTree(x)

    # query all points -- k+1 as query point also in initial set
    distances, _ = kdtree.query(x, k + 1, eps=0, p=p, workers=workers)
    distances = distances[:, -1]

    # enforce non-zero distances
    distances[distances < min_dist] = min_dist
    sum_log_dist = np.sum(log(2 * distances))

    h = -digamma(k) + digamma(n) + log_c_d + (d / float(n)) * sum_log_dist

    return h


def TMI(x, y, k=3, norm='max', min_dist=1.0, workers=1):
    """
    Estimates the Conditional Entropy H(X|Y) of two random variables x and y (in nats)
    Args:
        x: input (n, d) ndarray
        y: input (n, ) ndarray
        k: number of nearest neighbors
        norm: p-norm used when computing k-nearest neighbour distances
        min_dist: minimum distance between data points
        workers: number of workers

    Returns: conditional entropy H(X|Y)

    """
    num_classes = int(y.max() + 1)
    x = x - np.mean(x, axis=0, keepdims=True)
    hzy_list = np.array([get_h(x[y == i], k=k, norm=norm, min_dist=min_dist, workers=workers)
                         if x[y == i].shape[0] != 0 and x[y == i].shape[0] != 1 else 0 for i in range(num_classes)])
    y_list = np.array([len(y[y == i]) / len(y) for i in range(num_classes)])
    inf_list = np.isfinite(hzy_list)
    hzy = np.multiply(hzy_list[inf_list], y_list[inf_list]).mean()

    return hzy

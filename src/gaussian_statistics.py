import numpy as np
from scipy.stats import norm
from .numerical_characteristics import pearson


def pearson_statistics(x: np.ndarray, threshold: float) -> np.ndarray:
    """
    Calculates statistics for testing N(N-1)/2 hypotheses of the form:
    H_ij: The Pearson measure of similarity between the i and j component
    of the random vector <= threshold vs K_ij: The Pearson measure of similarity
    between the i and j component of the random vector > threshold.

    Parameters
    ----------
    x : (n,N) array_like
        Sample of the size n from distribution of the N-dimensional Gaussian random vector.

    threshold : float
        The threshold in the interval (-1, 1).

    Returns
    -------
    statistics : (N,N) ndarray
        Matrix of statistics.

    """
    n, N = x.shape

    def z_transform(y):
        return 0.5 * np.log((1 + y) / (1 - y))

    def statistics(y):
        if y == 1:
            return np.inf
        if y == -1:
            return -np.inf
        return np.sqrt(n) * (z_transform(y) - z_transform(threshold))

    transformer = np.vectorize(statistics)
    return transformer(pearson(x))

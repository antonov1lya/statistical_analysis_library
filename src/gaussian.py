import numpy as np
from scipy.stats import norm
from .numerical_characteristics import pearson


def pearson_test(x: np.ndarray, threshold: float) -> np.ndarray:
    """
    Calculate p-value for testing N(N-1)/2 hypotheses of the form
    H_ij: The Pearson measure of similarity between the i and j component 
    of the random vector <= threshold
    vs
    K_ij: The Pearson measure of similarity between the i and j component 
    of the random vector > threshold.

    Parameters
    ----------
    x : (n,N) array_like
        Sample of the size n from distribution of the N-dimensional Gaussian random vector.

    threshold : float
        The threshold in the interval (-1, 1).

    Returns
    -------
    p_value : (N,N) ndarray
        Matrix of p-values.
    """
    n, N = x.shape

    def z_transform(y):
        return 0.5 * np.log((1+y)/(1-y))

    def statistics(y):
        return np.sqrt(n) * (z_transform(y) - z_transform(threshold))

    def calc_p_value(y):
        return 0 if y == 1 else 1 if y == -1 else 1 - norm.cdf(statistics(y))

    transformer = np.vectorize(calc_p_value)
    p_value = transformer(pearson(x))
    return p_value

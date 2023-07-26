import numpy as np
from scipy.stats import norm
from .numerical_characteristics import pearson, kurtosis, sign_similarity


def pearson_test(x: np.ndarray, threshold: float) -> np.ndarray:
    """
    Calculates p-values for testing N(N-1)/2 hypotheses of the form:
    H_ij: The Pearson measure of similarity between the i and j component
    of the random vector <= threshold vs K_ij: The Pearson measure of similarity
    between the i and j component of the random vector > threshold.

    Parameters
    ----------
    x : (n,N) array_like
        Sample of the size n from distribution of the N-dimensional elliptical random vector.

    threshold : float
        The threshold in the interval (-1, 1).

    Returns
    -------
    p_value : (N,N) ndarray
        Matrix of p-values.

    """
    n, N = x.shape
    kurt = kurtosis(x)

    def z_transform(y):
        return 0.5 * np.log((1 + y) / (1 - y))

    def statistics(y):
        if y == 1:
            return np.inf
        if y == -1:
            return -np.inf
        return np.sqrt(n / (1 + kurt)) * (z_transform(y) - z_transform(threshold))

    def calc_p_values(y):
        return 1 - norm.cdf(statistics(y))

    transformer = np.vectorize(calc_p_values)
    return transformer(pearson(x))


def sign_similarity_test(x: np.ndarray, threshold: float) -> np.ndarray:
    """
    Calculates p-values for testing N(N-1)/2 hypotheses of the form:
    H_ij: The sign measure of similarity between the i and j component
    of the random vector <= threshold vs K_ij: The sign measure of similarity
    between the i and j component of the random vector > threshold.

    Parameters
    ----------
    x : (n,N) array_like
        Sample of the size n from distribution of the N-dimensional elliptical random vector.

    threshold : float
        The threshold in the interval (0, 1).

    Returns
    -------
    p_value : (N,N) ndarray
        Matrix of p-values.

    """
    n, N = x.shape

    def statistics(y):
        return np.sqrt(n) * (y - threshold) / np.sqrt(threshold * (1 - threshold))

    def calc_p_values(y):
        return 1 - norm.cdf(statistics(y))
    
    transformer = np.vectorize(calc_p_values)
    return transformer(sign_similarity(x))
    

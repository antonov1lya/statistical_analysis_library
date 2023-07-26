import numpy as np
from .numerical_characteristics import *


def _calc_pearson_stat(corr: np.ndarray, threshold: float, kurt: float, n: int) -> np.ndarray:
    def z_transform(y):
        return 0.5 * np.log((1 + y) / (1 - y))

    def statistics(y):
        if y == 1:
            return np.inf
        if y == -1:
            return -np.inf
        return np.sqrt(n / (1 + kurt)) * (z_transform(y) - z_transform(threshold))

    transformer = np.vectorize(statistics)
    return transformer(corr)


def pearson_statistics(x: np.ndarray, threshold: float) -> np.ndarray:
    """
    Calculates statistics for testing N(N-1)/2 hypotheses of the form:
    H_ij: The Pearson correlation between the i and j component
    of the random vector <= threshold vs K_ij: The Pearson correlation
    between the i and j component of the random vector > threshold.

    Parameters
    ----------
    x : (n,N) array_like
        Sample of the size n from distribution of the N-dimensional elliptical random vector.

    threshold : float
        The threshold in the interval (-1, 1).

    Returns
    -------
    statistics : (N,N) ndarray
        Matrix of statistics.

    """
    n, N = x.shape
    kurt = kurtosis(x)
    return _calc_pearson_stat(pearson(x), threshold, kurt, n)


def sign_similarity_statistics(x: np.ndarray, threshold: float) -> np.ndarray:
    """
    Calculates statistics for testing N(N-1)/2 hypotheses of the form:
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
    statistics : (N,N) ndarray
        Matrix of statistics.

    """
    n, N = x.shape

    def statistics(y):
        return np.sqrt(n) * (y - threshold) / np.sqrt(threshold * (1 - threshold))

    transformer = np.vectorize(statistics)
    return transformer(sign_similarity(x))


def fechner_statistics(x: np.ndarray, threshold: float) -> np.ndarray:
    """
    Calculates statistics for testing N(N-1)/2 hypotheses of the form:
    H_ij: The Fechner correlation between the i and j component
    of the random vector <= threshold vs K_ij: The Fechner correlation
    between the i and j component of the random vector > threshold.

    Parameters
    ----------
    x : (n,N) array_like
        Sample of the size n from distribution of the N-dimensional elliptical random vector.

    threshold : float
        The threshold in the interval (-1, 1).

    Returns
    -------
    statistics : (N,N) ndarray
        Matrix of statistics.

    """
    n, N = x.shape

    def statistics(y):
        return np.sqrt(n) * (y - threshold) / np.sqrt(1 - threshold ** 2)

    transformer = np.vectorize(statistics)
    return transformer(fechner(x))


def kruskal_statistics(x: np.ndarray, threshold: float) -> np.ndarray:
    """
    Calculates statistics for testing N(N-1)/2 hypotheses of the form:
    H_ij: The Kruskal correlation between the i and j component
    of the random vector <= threshold vs K_ij: The Kruskal correlation
    between the i and j component of the random vector > threshold.

    Parameters
    ----------
    x : (n,N) array_like
        Sample of the size n from distribution of the N-dimensional elliptical random vector.

    threshold : float
        The threshold in the interval (-1, 1).

    Returns
    -------
    statistics : (N,N) ndarray
        Matrix of statistics.

    """
    n, N = x.shape

    def statistics(y):
        return np.sqrt(n) * (y - threshold) / np.sqrt(1 - threshold ** 2)

    transformer = np.vectorize(statistics)
    return transformer(kruskal(x))


def spearman_statistics(x: np.ndarray, threshold: float) -> np.ndarray:
    """
    Calculates statistics for testing N(N-1)/2 hypotheses of the form:
    H_ij: The Spearman correlation between the i and j component
    of the random vector <= threshold vs K_ij: The Spearman correlation
    between the i and j component of the random vector > threshold.

    Parameters
    ----------
    x : (n,N) array_like
        Sample of the size n from distribution of the N-dimensional elliptical random vector.

    threshold : float
        The threshold in the interval (-1, 1).

    Returns
    -------
    statistics : (N,N) ndarray
        Matrix of statistics.

    """
    n, N = x.shape

    def statistics(y):
        return np.sqrt(n - 1) * (y - threshold)

    transformer = np.vectorize(statistics)
    return transformer(spearman(x))


def partial_statistics(x: np.ndarray, threshold: float) -> np.ndarray:
    """
    Calculates statistics for testing N(N-1)/2 hypotheses of the form:
    H_ij: The partial Pearson correlation between the i and j component
    of the random vector <= threshold vs K_ij: The partial Pearson correlation
    between the i and j component of the random vector > threshold.

    Parameters
    ----------
    x : (n,N) array_like
        Sample of the size n from distribution of the N-dimensional elliptical random vector.

    threshold : float
        The threshold in the interval (-1, 1).

    Returns
    -------
    statistics : (N,N) ndarray
        Matrix of statistics.

    """
    n, N = x.shape
    kurt = kurtosis(x)
    return _calc_pearson_stat(partial(x), threshold, kurt, n)
from typing import Callable
import numpy as np
from scipy.stats import norm

from .test_statistics import *


def _calc_p_value(x: np.ndarray, measure: str, threshold: float, model: str, p_value: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    if measure == 'pearson':
        if model == 'gaussian':
            return p_value(pearson_statistics(x, 'gaussian', threshold))
        else:
            return p_value(pearson_statistics(x, 'elliptical', threshold))

    if measure == 'sign_similarity':
        return p_value(sign_similarity_statistics(x, threshold))

    if measure == 'fechner':
        return p_value(fechner_statistics(x, threshold))

    if measure == 'kruskal':
        return p_value(kruskal_statistics(x, threshold))

    if measure == 'spearman':
        return p_value(spearman_statistics(x, threshold))

    if measure == 'partial':
        return p_value(partial_statistics(x, threshold))


def threshold_graph_p_value(x: np.ndarray, measure: str, threshold: float, model: str = 'elliptical') -> np.ndarray:
    """
    Calculates p-values for testing N(N-1)/2 hypotheses of the form:
    H_ij: measure of similarity between the i and j component
    of the random vector <= threshold vs K_ij: measure of similarity
    between the i and j component of the random vector > threshold.

    Parameters
    ----------
    x : (n,N) array_like
        Sample of the size n from distribution of the N-dimensional random vector.

    measure: {'pearson', 'sign_similarity', 'fechner', 'kruskal', 'spearman', 'partial'}
        The measure of similarity relative to which the test is performed.

    threshold : float
        The threshold in the interval (0, 1) for sign similarity
        and in the interval (-1, 1) for other measures.

    model : {'gaussian', 'elliptical'}
        The model according to which the random vector is distributed.

    Returns
    -------
    p_value : (N,N) ndarray
        Matrix of p-values.

    """
    p_value = np.vectorize(lambda y: 1 - norm.cdf(y))
    return _calc_p_value(x, measure, threshold, model, p_value)


def concentration_graph_p_value(x: np.ndarray, measure: str, model: str = 'elliptical') -> np.ndarray:
    """
    Calculates p-values for testing N(N-1)/2 hypotheses of the form:
    H_ij: measure of similarity between the i and j component
    of the random vector = 0 vs K_ij: measure of similarity
    between the i and j component of the random vector != 0.

    Parameters
    ----------
    x : (n,N) array_like
        Sample of the size n from distribution of the N-dimensional random vector.

    measure: {'pearson', 'sign_similarity', 'fechner', 'kruskal', 'spearman', 'partial'}
        The measure of similarity relative to which the test is performed.

    model : {'gaussian', 'elliptical'}
        The model according to which the random vector is distributed.

    Returns
    -------
    p_value : (N,N) ndarray
        Matrix of p-values.

    """
    p_value = np.vectorize(lambda y: 2 * (1 - norm.cdf(np.abs(y))))
    return _calc_p_value(x, measure, 0, model, p_value)
    
import numpy as np


def from_pearson_to_sign_similarity(pearson: float) -> float:
    """
    Calculates from the value of the Pearson correlation
    the equivalent value of the sign measure of similarity
    for a bi-variate elliptical vector.

    Parameters
    ----------
    pearson : float
        The value of the Pearson correlation.

    Returns
    -------
    sign_similarity : float
        The equivalent value of the sign measure of similarity.

    """
    return 0.5 + (1 / np.pi) * np.arcsin(pearson)


def from_pearson_to_fechner(pearson: float) -> float:
    """
    Calculates from the value of the Pearson correlation
    the equivalent value of the Fechner correlation
    for a bi-variate elliptical vector.

    Parameters
    ----------
    pearson : float
        The value of the Pearson correlation.

    Returns
    -------
    fechner : float
        The equivalent value of the Fechner correlation.

    """
    return (2 / np.pi) * np.arcsin(pearson)


def from_pearson_to_kruskal(pearson: float) -> float:
    """
    Calculates from the value of the Pearson correlation
    the equivalent value of the Kruskal correlation
    for a bi-variate elliptical vector.

    Parameters
    ----------
    pearson : float
        The value of the Pearson correlation.

    Returns
    -------
    kruskal : float
        The equivalent value of the Kruskal correlation.

    """
    return (2 / np.pi) * np.arcsin(pearson)


def from_pearson_to_kendall(pearson: float) -> float:
    """
    Calculates from the value of the Pearson correlation
    the equivalent value of the Kendall correlation
    for a bi-variate elliptical vector.

    Parameters
    ----------
    pearson : float
        The value of the Pearson correlation.

    Returns
    -------
    kendall : float
        The equivalent value of the Kendall correlation.

    """
    return (2 / np.pi) * np.arcsin(pearson)


def from_pearson_to_spearman(pearson: float) -> float:
    """
    Calculates from the value of the Pearson correlation
    the equivalent value of the Spearman correlation
    for a bi-variate Gaussian vector.

    Parameters
    ----------
    pearson : float
        The value of the Pearson correlation.

    Returns
    -------
    spearman : float
        The equivalent value of the Spearman correlation.

    """
    return (6 / np.pi) * np.arcsin(pearson / 2)

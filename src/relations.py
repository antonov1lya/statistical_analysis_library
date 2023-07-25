import numpy as np


def from_pearson_to_sign_similarity(pearson: float) -> float:
    """
    Calculates from the value of the Pearson measure of similarity 
    the equivalent value of the sign measure of similarity 
    for a bi-variate elliptical vector.

    Parameters
    ----------
    pearson : float
        The value of the Pearson measure of similarity.

    Returns
    -------
    sign_similarity : float
        The equivalent value of the sign measure of similarity.
    """
    return 0.5 + (1 / np.pi) * np.arcsin(pearson)


def from_pearson_to_fechner(pearson: float) -> float:
    """
    Calculates from the value of the Pearson measure of similarity 
    the equivalent value of the Fechner measure of similarity 
    for a bi-variate elliptical vector.

    Parameters
    ----------
    pearson : float
        The value of the Pearson measure of similarity.

    Returns
    -------
    fechner : float
        The equivalent value of the Fechner measure of similarity.
    """
    return (2 / np.pi) * np.arcsin(pearson)


def from_pearson_to_kruskal(pearson: float) -> float:
    """
    Calculates from the value of the Pearson measure of similarity 
    the equivalent value of the Kruskal measure of similarity 
    for a bi-variate elliptical vector.

    Parameters
    ----------
    pearson : float
        The value of the Pearson measure of similarity.

    Returns
    -------
    kruskal : float
        The equivalent value of the Kruskal measure of similarity.
    """
    return (2 / np.pi) * np.arcsin(pearson)


def from_pearson_to_kendall(pearson: float) -> float:
    """
    Calculates from the value of the Pearson measure of similarity 
    the equivalent value of the Kendall measure of similarity 
    for a bi-variate elliptical vector.

    Parameters
    ----------
    pearson : float
        The value of the Pearson measure of similarity.

    Returns
    -------
    kendall : float
        The equivalent value of the Kendall measure of similarity.
    """
    return (2 / np.pi) * np.arcsin(pearson)


def from_pearson_to_spearman(pearson: float) -> float:
    """
    Calculates from the value of the Pearson measure of similarity 
    the equivalent value of the Spearman measure of similarity 
    for a bi-variate Gaussian vector.

    Parameters
    ----------
    pearson : float
        The value of the Pearson measure of similarity.

    Returns
    -------
    spearman : float
        The equivalent value of the Spearman measure of similarity.
    """
    return (6 / np.pi) * np.arcsin(pearson / 2)

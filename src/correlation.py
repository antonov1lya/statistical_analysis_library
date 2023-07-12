import numpy as np
from numba import njit


@njit
def _calculate_pearson(x, corr, mean, n, N):
    for i in range(N):
        for j in range(i, N):
            for t in range(n):
                corr[i][j] += (x[t][i] - mean[i]) * (x[t][j] - mean[j])
    for i in range(N):
        for j in range(i+1, N):
            corr[i][j] /= np.sqrt(corr[i][i] * corr[j][j])
    for i in range(N):
        corr[i][i] = 1


def pearson(x):
    """
    Sample Pearson correlation matrix.

    Parameters
    ----------
    x : (n,N) array_like
        Sample of the size n from distribution of the N-dimensional random vector.

    Returns
    -------
    corr : (N,N) ndarray
        Upper triangular sample Pearson correlation matrix.
    """
    x = np.array(x)
    n, N = x.shape
    mean = np.mean(x, axis=0)
    corr = np.zeros((N, N))
    _calculate_pearson(x, corr, mean, n, N)
    return corr


@njit
def _calculate_sign_similarity(x, corr, mean, n, N):
    for i in range(N):
        for j in range(i, N):
            for t in range(n):
                if (x[t][i]-mean[i])*(x[t][j]-mean[j]) >= 0:
                    corr[i][j] += 1
            corr[i][j] /= n


def sign_similarity(x):
    """
    Sample sign similarity matrix.

    Parameters
    ----------
    x : (n,N) array_like
        Sample of the size n from distribution of the N-dimensional random vector.

    Returns
    -------
    corr : (N,N) ndarray
        Upper triangular sample sign similarity matrix.
    """
    x = np.array(x)
    n, N = x.shape
    mean = np.mean(x, axis=0)
    corr = np.zeros((N, N))
    _calculate_sign_similarity(x, corr, mean, n, N)
    return corr


@njit
def _calculate_fechner(x, corr, mean, n, N):
    for i in range(N):
        for j in range(i, N):
            for t in range(n):
                if (x[t][i]-mean[i])*(x[t][j]-mean[j]) >= 0:
                    corr[i][j] += 1
                else:
                    corr[i][j] -= 1
            corr[i][j] /= n


def fechner(x):
    """
    Sample Fechner correlation matrix.

    Parameters
    ----------
    x : (n,N) array_like
        Sample of the size n from distribution of the N-dimensional random vector.

    Returns
    -------
    corr : (N,N) ndarray
        Upper triangular sample Fechner correlation matrix.
    """
    x = np.array(x)
    n, N = x.shape
    mean = np.mean(x, axis=0)
    corr = np.zeros((N, N))
    _calculate_fechner(x, corr, mean, n, N)
    return corr


@njit
def _calculate_kruskal(x, corr, med, n, N):
    for i in range(N):
        for j in range(i, N):
            for t in range(n):
                if (x[t][i]-med[i])*(x[t][j]-med[j]) >= 0:
                    corr[i][j] += 1
                else:
                    corr[i][j] -= 1
            corr[i][j] /= n


def kruskal(x):
    """
    Sample Kruskal correlation matrix.

    Parameters
    ----------
    x : (n,N) array_like
        Sample of the size n from distribution of the N-dimensional random vector.

    Returns
    -------
    corr : (N,N) ndarray
        Upper triangular sample Kruskal correlation matrix.
    """
    x = np.array(x)
    n, N = x.shape
    med = np.median(x, axis=0)
    corr = np.zeros((N, N))
    _calculate_kruskal(x, corr, med, n, N)
    return corr

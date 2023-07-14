import numpy as np
from numba import njit


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
        Sample Pearson correlation matrix.
    """
    x = np.array(x).T
    N, n = x.shape
    mean = np.mean(x, axis=1).reshape((N, -1))
    x = x - mean
    corr = np.dot(x, x.T)
    for i in range(N):
        for j in range(i+1, N):
            corr[i][j] /= np.sqrt(corr[i][i] * corr[j][j])
            corr[j][i] = corr[i][j]
    for i in range(N):
        corr[i][i] = 1
    return corr


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
        Sample sign similarity matrix.
    """
    x = np.array(x).T
    N, n = x.shape
    mean = np.mean(x, axis=1).reshape((N, -1))
    x = x - mean
    transformer = np.vectorize(lambda y: 1 if y >= 0 else 0)
    corr = np.sum(transformer(x[..., np.newaxis]*x.T[np.newaxis, ...]), axis=1)
    return corr / n


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
        Sample Fechner correlation matrix.
    """
    x = np.array(x).T
    N, n = x.shape
    mean = np.mean(x, axis=1).reshape((N, -1))
    x = x - mean
    transformer = np.vectorize(lambda y: 1 if y >= 0 else -1)
    corr = np.sum(transformer(x[..., np.newaxis]*x.T[np.newaxis, ...]), axis=1)
    return corr / n


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
        Sample Kruskal correlation matrix.
    """
    x = np.array(x).T
    N, n = x.shape
    med = np.median(x, axis=1).reshape((N, -1))
    x = x - med
    transformer = np.vectorize(lambda y: 1 if y >= 0 else -1)
    corr = np.sum(transformer(x[..., np.newaxis]*x.T[np.newaxis, ...]), axis=1)
    return corr / n


@njit
def _calculate_kendall(x, corr, n, N):
    for i in range(N):
        for j in range(i, N):
            for t in range(n):
                for s in range(t+1, n):
                    if (x[t][i] - x[s][i]) * (x[t][j] - x[s][j]) >= 0:
                        corr[i][j] += 1
                    else:
                        corr[i][j] -= 1
            corr[i][j] *= 2 / (n * (n - 1))
            corr[j][i] = corr[i][j]


def kendall(x):
    """
    Sample Kendall correlation matrix.

    Parameters
    ----------
    x : (n,N) array_like
        Sample of the size n from distribution of the N-dimensional random vector.

    Returns
    -------
    corr : (N,N) ndarray
        Sample Kendall correlation matrix.
    """
    x = np.array(x)
    n, N = x.shape
    corr = np.zeros((N, N))
    _calculate_kendall(x, corr, n, N)
    return corr


@njit
def _calculate_spearman(x, corr, n, N):
    for i in range(N):
        for j in range(i, N):
            for t in range(n):
                for s in range(n):
                    if s != t:
                        for l in range(s+1, n):
                            if l != t:
                                if (x[t][i] - x[s][i]) * (x[t][j] - x[l][j]) >= 0:
                                    corr[i][j] += 1
                                else:
                                    corr[i][j] -= 1
            corr[i][j] *= 6 / (n * (n - 1) * (n - 2))
            corr[j][i] = corr[i][j]


def spearman(x):
    """
    Sample Spearman correlation matrix.

    Parameters
    ----------
    x : (n,N) array_like
        Sample of the size n from distribution of the N-dimensional random vector.

    Returns
    -------
    corr : (N,N) ndarray
        Sample Spearman correlation matrix.
    """
    x = np.array(x)
    n, N = x.shape
    corr = np.zeros((N, N))
    _calculate_spearman(x, corr, n, N)
    return corr


def partial(x):
    """
    Sample Partial Pearson correlation matrix.

    Parameters
    ----------
    x : (n,N) array_like
        Sample of the size n from distribution of the N-dimensional random vector.

    Returns
    -------
    corr : (N,N) ndarray
        Sample Partial Pearson correlation matrix.
    """
    x = np.array(x).T
    N, n = x.shape
    mean = np.mean(x, axis=1).reshape((N, -1))
    x = x - mean
    corr = np.dot(x, x.T)
    corr = corr / (n-1)
    corr = np.linalg.inv(corr)
    for i in range(N):
        for j in range(i+1, N):
            corr[i][j] /= -np.sqrt(corr[i][i] * corr[j][j])
            corr[j][i] = corr[i][j]
    for i in range(N):
        corr[i][i] /= -np.sqrt(corr[i][i] * corr[i][i])
    return corr

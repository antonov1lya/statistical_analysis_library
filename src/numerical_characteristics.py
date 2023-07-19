import numpy as np
from scipy.stats._stats import _kendall_dis


def covariance(x):
    """
    Sample covariance matrix.

    Parameters
    ----------
    x : (n,N) array_like
        Sample of the size n from distribution of the N-dimensional random vector.

    Returns
    -------
    cov : (N,N) ndarray
        Sample covariance matrix.
    """
    x = np.array(x).T
    N, n = x.shape
    mean = np.mean(x, axis=1).reshape((N, -1))
    x = x - mean
    cov = np.zeros((N, N))
    for i in range(N):
        for j in range(i, N):
            cov[i][j] = np.dot(x[i], x[j]) / n
            cov[j][i] = cov[i][j]
    return cov


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
    corr = covariance(x)
    N = corr.shape[0]
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
    corr = np.zeros((N, N))
    transformer = np.vectorize(lambda y: 1 if y >= 0 else 0)
    for i in range(N):
        for j in range(i+1, N):
            corr[i][j] = np.sum(transformer(x[i] * x[j])) / n
            corr[j][i] = corr[i][j]
    for i in range(N):
        corr[i][i] = 1
    return corr


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
    corr = np.zeros((N, N))
    transformer = np.vectorize(lambda y: 1 if y >= 0 else -1)
    for i in range(N):
        for j in range(i+1, N):
            corr[i][j] = np.sum(transformer(x[i] * x[j])) / n
            corr[j][i] = corr[i][j]
    for i in range(N):
        corr[i][i] = 1
    return corr


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
    corr = np.zeros((N, N))
    transformer = np.vectorize(lambda y: 1 if y >= 0 else -1)
    for i in range(N):
        for j in range(i+1, N):
            corr[i][j] = np.sum(transformer(x[i] * x[j])) / n
            corr[j][i] = corr[i][j]
    for i in range(N):
        corr[i][i] = 1
    return corr


def _kendall_pair(x, y):
    p = np.argsort(y, kind='stable')
    x, y = x[p], y[p]
    y = np.r_[True, y[1:] != y[:-1]].cumsum()

    p = np.argsort(x, kind='stable')
    x, y = x[p], y[p]
    x = np.r_[True, x[1:] != x[:-1]].cumsum()

    Q = _kendall_dis(x, y)
    n = x.shape[0]

    return 1 - (4 * Q / (n * (n - 1)))


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
    x = np.array(x).T
    N, n = x.shape
    corr = np.zeros((N, N))
    for i in range(N):
        for j in range(i+1, N):
            corr[i][j] = _kendall_pair(x[i], x[j])
            corr[j][i] = corr[i][j]
    for i in range(N):
        corr[i][i] = 1
    return corr


def _spearman_pair(x, y):
    p = np.argsort(y, kind='stable')
    x, y = x[p], y[p]
    y = np.r_[True, y[1:] != y[:-1]].cumsum()

    y_ord = y

    p = np.argsort(x, kind='stable')
    x, y = x[p], y[p]
    x = np.r_[True, x[1:] != x[:-1]].cumsum()

    x_ord = x

    Q = 0
    n = x.shape[0]
    Q += np.sum(np.searchsorted(x_ord, x - 1, side='right') *
                (n - np.searchsorted(y_ord, y + 1, side='left')))
    Q += np.sum(np.searchsorted(y_ord, y - 1, side='right') *
                (n - np.searchsorted(x_ord, x + 1, side='left')))
    Q -= 2 * _kendall_dis(x, y)

    return 3 - (6 * Q / (n * (n - 1) * (n - 2)))


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
    x = np.array(x).T
    N, n = x.shape
    corr = np.zeros((N, N))
    for i in range(N):
        for j in range(i, N):
            corr[i][j] = _spearman_pair(x[i], x[j])
            corr[j][i] = corr[i][j]
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
    corr = np.linalg.inv(covariance(x))
    N = corr.shape[0]
    for i in range(N):
        for j in range(i+1, N):
            corr[i][j] /= -np.sqrt(corr[i][i] * corr[j][j])
            corr[j][i] = corr[i][j]
    for i in range(N):
        corr[i][i] = -1
    return corr


def kurtosis(x):
    """
    Sample analog of kurtosis.

    Parameters
    ----------
    x : (n,N) array_like
        Sample of the size n from distribution of the N-dimensional random vector.

    Returns
    -------
    kurtosis : float
        Sample analog of kurtosis.
    """
    n, N = x.shape
    S = np.linalg.inv(covariance(x))

    x = np.array(x).T
    mean = np.mean(x, axis=1).reshape((N, -1))
    x = x - mean
    x = x.T

    k = 1 / (n * N * (N + 2)) * np.sum(np.sum(np.dot(x, S) * x, axis=1)**2) - 1
    return k

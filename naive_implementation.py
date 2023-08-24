import numpy as np


def pearson(x: np.ndarray) -> np.ndarray:
    x = np.array(x).T
    N, n = x.shape
    corr = np.cov(x, ddof=0)
    for i in range(N):
        for j in range(i + 1, N):
            corr[i][j] = corr[i][j] / np.sqrt(corr[i][i] * corr[j][j])
            corr[j][i] = corr[i][j]
        corr[i][i] = 1
    return corr


def covariance(x: np.ndarray) -> np.ndarray:
    return np.cov(x, ddof=0, rowvar=False)


def sign_similarity(x: np.ndarray) -> np.ndarray:
    x = np.array(x).T
    N, n = x.shape
    corr = np.zeros((N, N))
    for i in range(N):
        for j in range(i, N):
            m_i = np.mean(x[i])
            m_j = np.mean(x[j])
            for t in range(n):
                if (x[i][t] - m_i) * (x[j][t] - m_j) >= 0:
                    corr[i][j] += 1
            corr[i][j] /= n
            corr[j][i] = corr[i][j]
    return corr


def fechner(x: np.ndarray) -> np.ndarray:
    x = np.array(x).T
    N, n = x.shape
    corr = np.zeros((N, N))
    for i in range(N):
        for j in range(i, N):
            m_i = np.mean(x[i])
            m_j = np.mean(x[j])
            for t in range(n):
                if (x[i][t] - m_i) * (x[j][t] - m_j) >= 0:
                    corr[i][j] += 1
                else:
                    corr[i][j] -= 1
            corr[i][j] /= n
            corr[j][i] = corr[i][j]
    return corr


def kruskal(x: np.ndarray) -> np.ndarray:
    x = np.array(x).T
    N, n = x.shape
    corr = np.zeros((N, N))
    for i in range(N):
        for j in range(i, N):
            m_i = np.median(x[i])
            m_j = np.median(x[j])
            for t in range(n):
                if (x[i][t] - m_i) * (x[j][t] - m_j) >= 0:
                    corr[i][j] += 1
                else:
                    corr[i][j] -= 1
            corr[i][j] /= n
            corr[j][i] = corr[i][j]
    return corr


def kendall(x: np.ndarray) -> np.ndarray:
    x = np.array(x).T
    N, n = x.shape
    corr = np.zeros((N, N))
    for i in range(N):
        for j in range(i, N):
            for t in range(n):
                for s in range(n):
                    if s != t:
                        if (x[i][t] - x[i][s]) * (x[j][t] - x[j][s]) >= 0:
                            corr[i][j] += 1
                        else:
                            corr[i][j] -= 1
            corr[i][j] /= n * (n - 1)
            corr[j][i] = corr[i][j]
    return corr


def spearman(x: np.ndarray) -> np.ndarray:
    x = np.array(x).T
    N, n = x.shape
    corr = np.zeros((N, N))
    for i in range(N):
        for j in range(i, N):
            for t in range(n):
                for s in range(n):
                    if s != t:
                        for l in range(n):
                            if l != t and l != s:
                                if (x[i][t] - x[i][s]) * (x[j][t] - x[j][l]) >= 0:
                                    corr[i][j] += 1
                                else:
                                    corr[i][j] -= 1
            corr[i][j] *= 3 / (n * (n - 1) * (n - 2))
            corr[j][i] = corr[i][j]
    return corr


def kurtosis(x: np.ndarray) -> float:
    x = np.array(x).T
    N, n = x.shape
    mean = np.zeros(N)
    for i in range(N):
        mean[i] = np.mean(x[i])
    S = np.linalg.inv(np.cov(x, ddof=0))
    x = x.T
    sum = 0
    for t in range(n):
        sum += (
            float(np.dot(np.dot(x[t] - mean, S), (x[t] - mean).reshape((N, -1)))) ** 2
        )
    return 1 / (N * (N + 2) * n) * sum - 1


def partial(x: np.ndarray) -> np.ndarray:
    x = np.array(x).T
    N, n = x.shape
    corr = np.linalg.inv(np.cov(x, ddof=0))
    for i in range(N):
        for j in range(i + 1, N):
            corr[i][j] = -corr[i][j] / np.sqrt(corr[i][i] * corr[j][j])
            corr[j][i] = corr[i][j]
        corr[i][i] = 1
    return corr


def pcc(x: np.ndarray) -> np.ndarray:
    x = np.array(x).T
    N, n = x.shape
    corr = np.zeros((N, N))
    for i in range(N):
        for j in range(i, N):
            for t in range(n):
                for s in range(n):
                    if s != t:
                        for l in range(n):
                            if l != t and l != s:
                                first = (x[i][t] - x[i][s]) * (x[j][t] - x[j][s]) >= 0
                                second = (x[i][t] - x[i][l]) * (x[j][t] - x[j][l]) >= 0
                                if first and second:
                                    corr[i][j] += 1
            corr[i][j] /= n * (n - 1) * (n - 2)
            corr[j][i] = corr[i][j]
    return corr

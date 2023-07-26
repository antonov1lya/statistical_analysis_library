import numpy as np


def pearson(x: np.ndarray) -> np.ndarray:
    corr = np.cov(x, ddof=0, rowvar=False)
    N, n = x.shape
    for i in range(n):
        for j in range(i + 1, n):
            corr[i][j] = corr[i][j] / np.sqrt(corr[i][i] * corr[j][j])
            corr[j][i] = corr[i][j]
        corr[i][i] = 1
    return corr


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

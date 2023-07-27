import unittest
import numpy as np
from numpy.random import multivariate_normal

import src.numerical_characteristics as optimal
import naive_implementation as naive

a = np.array([
    [1, 2, 4, 3, 2, 4, 3, 3, 4, 3, 2, 4, 5, 1, 2, 4, 3, 2, 2, 3],
    [4, 2, 3, 4, 3, 5, 3, 3, 2, 3, 2, 4, 2, 4, 2, 4, 3, 1, 2, 4],
    [4, 2, 3, 4, 2, 5, 3, 4, 2, 3, 2, 1, 3, 4, 4, 4, 1, 4, 5, 3],
    [5, 4, 3, 5, 6, 2, 6, 3, 6, 2, 6, 1, 3, 2, 4, 3, 4, 2, 2, 4],
    [2, 4, 5, 3, 4, 5, 3, 5, 6, 2, 6, 4, 3, 4, 5, 5, 4, 5, 3, 5]
]).T

S = [
    [5, 2, 1, 0],
    [2, 2, 0, 1],
    [1, 0, 2, 1],
    [0, 1, 1, 2]
]
b = multivariate_normal([0, 0, 0, 0], S, size=100)


class TestNumericalCharacteristics(unittest.TestCase):
    def test_numerical_characteristics(self):
        first = [naive.pearson, naive.covariance, naive.sign_similarity,
                 naive.fechner, naive.kruskal, naive.kendall,
                 naive.spearman, naive.kurtosis, naive.partial]

        second = [optimal.pearson, optimal.covariance, optimal.sign_similarity,
                  optimal.fechner, optimal.kruskal, optimal.kendall,
                  optimal.spearman, optimal.kurtosis, optimal.partial]

        for i in range(len(first)):
            self.assertEqual(np.round(first[i](a), 10).tolist(),
                             np.round(second[i](a), 10).tolist())
            self.assertEqual(np.round(first[i](b), 10).tolist(),
                             np.round(second[i](b), 10).tolist())


unittest.main()

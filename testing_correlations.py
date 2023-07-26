import unittest
import numpy as np
from numpy.random import multivariate_normal

import src.numerical_characteristics as optimal
import naive_correlations as naive

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

class TestCorrelationMatrices(unittest.TestCase):
    def test_pearson(self):
        self.assertEqual(np.round(naive.pearson(a), 10).tolist(),
                         np.round(optimal.pearson(a), 10).tolist())
        self.assertEqual(np.round(naive.pearson(b), 10).tolist(),
                         np.round(optimal.pearson(b), 10).tolist())

    def test_covariance(self):
        self.assertEqual(np.round(np.cov(a, ddof=0, rowvar=False), 10).tolist(),
                         np.round(optimal.covariance(a), 10).tolist())
        self.assertEqual(np.round(np.cov(b, ddof=0, rowvar=False), 10).tolist(),
                         np.round(optimal.covariance(b), 10).tolist())

    def test_sign_similarity(self):
        self.assertEqual(np.round(naive.sign_similarity(a), 10).tolist(),
                         np.round(optimal.sign_similarity(a), 10).tolist())
        self.assertEqual(np.round(naive.sign_similarity(b), 10).tolist(),
                         np.round(optimal.sign_similarity(b), 10).tolist())
    
    def test_fechner(self):
        self.assertEqual(np.round(naive.fechner(a), 10).tolist(),
                         np.round(optimal.fechner(a), 10).tolist())
        self.assertEqual(np.round(naive.fechner(b), 10).tolist(),
                         np.round(optimal.fechner(b), 10).tolist())
        
    def test_kruskal(self):
        self.assertEqual(np.round(naive.kruskal(a), 10).tolist(),
                         np.round(optimal.kruskal(a), 10).tolist())
        self.assertEqual(np.round(naive.kruskal(b), 10).tolist(),
                         np.round(optimal.kruskal(b), 10).tolist())
    
    def test_kendall(self):
        self.assertEqual(np.round(naive.kendall(a), 10).tolist(),
                         np.round(optimal.kendall(a), 10).tolist())
        self.assertEqual(np.round(naive.kendall(b), 10).tolist(),
                         np.round(optimal.kendall(b), 10).tolist())
        
    def test_spearman(self):
        self.assertEqual(np.round(naive.spearman(a), 10).tolist(),
                         np.round(optimal.spearman(a), 10).tolist())
        self.assertEqual(np.round(naive.spearman(b), 10).tolist(),
                         np.round(optimal.spearman(b), 10).tolist())

unittest.main()

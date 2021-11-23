"""
Tests for pval module
"""

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
import unittest

from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt

from stattest_clusters_approx import stattest_clusters_approx


class TestPvalModule(unittest.TestCase):

    def test_smoke(self):
        """
        simple smoke test to make sure function runs
        :return: nothing
        """
        x = np.array([[5, 3],
                      [10, 15],
                      [15, 12],
                      [24, 10],
                      [30, 30],
                      [85, 70],
                      [71, 80],
                      [60, 78],
                      [70, 55],
                      [80, 91], ])
        cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')
        cluster.fit_predict(x)
        plt.scatter(x[:, 0], x[:, 1], c=cluster.labels_, cmap='rainbow')
        plt.show()
        k1 = 1
        k2 = 2
        stattest_clusters_approx(x, k1, k2)

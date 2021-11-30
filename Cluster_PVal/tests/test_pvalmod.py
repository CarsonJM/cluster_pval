"""
Tests for pval module
"""

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
import unittest

from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt

from src.pval_module.stattest_clusters_approx import stattest_clusters_approx


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
        #plt.scatter(x[:, 0], x[:, 1], c=cluster.labels_, cmap='rainbow')
        #plt.show()
        k1 = 0
        k2 = 1
        stattest_clusters_approx(x, k1, k2, cluster.labels_)


    def test_penguin(self):
        """
        Test using Penguin data used in R tutorial
        :return: same results as shown in R tutorial
        """
        penguin_data = np.genfromtxt('tests/data_for_tests/penguin_data_subset.txt', delimiter=' ',
                                     skip_header=1)
        K = 5
        #set linkage to average to match R script
        positional_arguments = []
        keyword_arguments = {'n_clusters':K, 'affinity':'euclidean', 'linkage':'average'}
        cluster = AgglomerativeClustering(**keyword_arguments)
        cluster.fit_predict(penguin_data)
        #flipped these axes to match figure in R
        #plt.scatter(penguin_data[:, 1], penguin_data[:, 0], c=cluster.labels_, cmap='rainbow')
        #plt.show()
        k1 = 0
        k2 = 1
        stattest_clusters_approx(penguin_data, k1, k2, cluster.labels_, AgglomerativeClustering, positional_arguments,
                                 keyword_arguments)
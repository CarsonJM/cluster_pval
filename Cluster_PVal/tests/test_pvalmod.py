"""
Tests for pval module
"""

import numpy as np
from sklearn.cluster import AgglomerativeClustering
import unittest
import matplotlib.pyplot as plt

from pval_module.stattests import stattest_clusters_approx
from pval_module.stattests import wald_test

class TestPvalModule(unittest.TestCase):

    def test_smoke_stattest_clusters_approx(self):
        """
        simple smoke test to make sure stattest_clusters_approx function runs
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
        K = 2
        cl_fun = AgglomerativeClustering
        positional_arguments = []
        keyword_arguments = {'n_clusters': K, 'affinity': 'euclidean',
                             'linkage': 'average'}
        cluster = cl_fun(*positional_arguments, **keyword_arguments)
        cluster.fit_predict(x)
        k1 = 0
        k2 = 1
        stattest_clusters_approx(x, k1, k2, cluster.labels_, cl_fun,
                             positional_arguments, keyword_arguments,)

    def test_smoke_wald_test(self):
        """
        simple smoke test to make sure wald_test function runs
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
        K = 2
        cl_fun = AgglomerativeClustering
        positional_arguments = []
        keyword_arguments = {'n_clusters': K, 'affinity': 'euclidean',
                             'linkage': 'average'}
        cluster = cl_fun(*positional_arguments, **keyword_arguments)
        cluster.fit_predict(x)
        k1 = 0
        k2 = 1
        wald_test(x, k1, k2, cluster.labels_)

    def test_penguin_gao(self):
        """
        Test using Penguin data used in R tutorial
        :return: same results as shown in R tutorial
        """
        penguin_data = np.genfromtxt(
            'tests/data_for_tests/penguin_data_subset.txt',
            delimiter=' ', skip_header=1)
        K = 5
        #set linkage to average to match R script
        positional_arguments = []
        keyword_arguments = {'n_clusters':K, 'affinity':'euclidean', 'linkage':'average'}
        cluster = AgglomerativeClustering(**keyword_arguments)
        cluster.fit_predict(penguin_data)
        #flipped these axes to match figure in R
        #plt.scatter(penguin_data[:, 1], penguin_data[:, 0],
        # c=cluster.labels_, cmap='rainbow')
        #print (cluster.labels_)
        #plt.show()
        k1 = 0
        k2 = 1
        stat, pval, stderr = stattest_clusters_approx(penguin_data, k1, k2,
                                          cluster.labels_, AgglomerativeClustering, positional_arguments,
                                 keyword_arguments, ndraws=10000)
        passing = True
        assert np.isclose(stat, 10.11433)
        try:
            assert np.isclose(stderr, 0.01084133, atol=.001)
        except AssertionError:
            passing = False
            print("stderr is {}, meant to be within .001 of "
                  "0.01084133".format(stderr))
        try:
            assert np.isclose(pval, 0.6360161, atol=.02)
        except AssertionError:
            passing = False
            print("pval is {}, meant to be within .02 of "
                  "0.6360161".format(pval))
        assert(passing == True)

    def test_penguin_wald(self):
        """
        Test using Penguin data used in R tutorial
        :return: same results as shown when using R wald_test function,
        stat =  10.11433; pval = 0.006226331
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
        stat, pval = wald_test(penguin_data, k1, k2, cluster.labels_)
        assert np.isclose(stat, 10.11433)
        assert np.isclose(pval, 0.006226331)
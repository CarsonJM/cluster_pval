"""
Tests for pval module
"""

import unittest

import numpy as np
from sklearn.cluster import AgglomerativeClustering
# import matplotlib.pyplot as plt

from pval_module.stattests import stattest_clusters_approx
from pval_module.stattests import wald_test


class TestPvalModule(unittest.TestCase):
    """ Unittest class holding tests for pval module

    Args:
        None in addition to those inherited from unittest.Testcase
    Attributes:
        None in addition to those inherited from unittest.Testcase
    Functions:
    """

    def test_smoke_gao(self):
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
        k = 2
        cl_fun = AgglomerativeClustering
        positional_arguments = []
        keyword_arguments = {'n_clusters': k, 'affinity': 'euclidean',
                             'linkage': 'average'}
        cluster = cl_fun(*positional_arguments, **keyword_arguments)
        cluster.fit_predict(x)
        k1 = 0
        k2 = 1
        stattest_clusters_approx(x, k1, k2, cluster.labels_, cl_fun,
                                 positional_arguments, keyword_arguments, )
        self.assertTrue(True)


    def test_smoke_wald(self):
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
        k = 2
        cl_fun = AgglomerativeClustering
        positional_arguments = []
        keyword_arguments = {'n_clusters': k, 'affinity': 'euclidean',
                             'linkage': 'average'}
        cluster = cl_fun(*positional_arguments, **keyword_arguments)
        cluster.fit_predict(x)
        k1 = 0
        k2 = 1
        wald_test(x, k1, k2, cluster.labels_)
        self.assertTrue(True)


    def test_penguin_gao_10000(self):
        """
        One-shot test using Penguin data used in R tutorial with ndraws same
        as shown in R tutorial.
        :return: same results as when using R stattest_clusters_approx function:
        stat = 10.11433, stderr ~ .01084133, pval > .5
        """
        penguin_data = np.genfromtxt(
            'tests/data_for_tests/penguin_data_subset.txt',
            delimiter=' ', skip_header=1)
        k = 5
        # set linkage to average to match R script
        positional_arguments = []
        keyword_arguments = {'n_clusters': k, 'affinity': 'euclidean',
                             'linkage': 'average'}
        cluster = AgglomerativeClustering(**keyword_arguments)
        cluster.fit_predict(penguin_data)
        # flipped these axes to match figure in R
        # plt.scatter(penguin_data[:, 1], penguin_data[:, 0],
        # c=cluster.labels_, cmap='rainbow')
        # print (cluster.labels_)
        # plt.show()
        k1 = 0
        k2 = 1
        stat, pval, stderr = stattest_clusters_approx(penguin_data, k1, k2,
                                                      cluster.labels_,
                                                      AgglomerativeClustering,
                                                      positional_arguments,
                                                      keyword_arguments,
                                                      ndraws=10000)
        passing = True
        assert np.isclose(stat, 10.11433)
        try:
            assert np.isclose(stderr, 0.01084133, atol=.001)
        except AssertionError:
            passing = False
            print("stderr is {}, should be within .001 of "
                  "0.01084133".format(stderr))
        try:
            assert pval > .5
        except AssertionError:
            passing = False
            print("pval is {}, should be > .5".format(pval))
        self.assertTrue(passing)


    def test_penguin_wald(self):
        """
        One-shot test using Penguin data used in R tutorial
        :return: same results as shown when using R wald_test function,
        stat =  10.11433; pval = 0.006226331
        """
        penguin_data = np.genfromtxt(
            'tests/data_for_tests/penguin_data_subset.txt', delimiter=' ',
            skip_header=1)
        k = 5
        positional_arguments = []
        keyword_arguments = {'n_clusters': k, 'affinity': 'euclidean',
                             'linkage': 'average'}
        cluster = AgglomerativeClustering(*positional_arguments,
                                          **keyword_arguments)
        cluster.fit_predict(penguin_data)
        k1 = 0
        k2 = 1
        stat, pval = wald_test(penguin_data, k1, k2, cluster.labels_)
        assert np.isclose(stat, 10.11433)
        assert np.isclose(pval, 0.006226331)
        self.assertTrue(True)


    def test_penguin_gao_200(self):
        """
        One-shot test using Penguin data used in R tutorial with
        consistent parameters except ndraws=200 (to expedite function running
        while testing)
        :return: same results as when using R stattest_clusters_approx function:
        stat = 10.11433; stderr ~ .07; p > .3
        """
        penguin_data = np.genfromtxt(
            'tests/data_for_tests/penguin_data_subset.txt',
            delimiter=' ', skip_header=1)
        k = 5
        # set linkage to average to match R script
        positional_arguments = []
        keyword_arguments = {'n_clusters': k, 'affinity': 'euclidean',
                             'linkage': 'average'}
        cluster = AgglomerativeClustering(**keyword_arguments)
        cluster.fit_predict(penguin_data)
        k1 = 0
        k2 = 1
        stat, pval, stderr = stattest_clusters_approx(penguin_data, k1, k2,
                                                      cluster.labels_,
                                                      AgglomerativeClustering,
                                                      positional_arguments,
                                                      keyword_arguments,
                                                      ndraws=200)
        passing = True
        assert np.isclose(stat, 10.11433)
        try:
            assert np.isclose(stderr, 0.07, atol=.02)
        except AssertionError:
            passing = False
            print("stderr is {}, should be within .02 of "
                  "0.07".format(stderr))
        try:
            assert pval > .3
        except AssertionError:
            passing = False
            print("pval is {}, should be >.3".format(pval))
        self.assertTrue(passing)


    ###### GAO TESTS PERMUTING PARAMETERS
    def test_penguin_gao_isoFalse_sigNone_siginvNone_200(self):
        """
        One-shot test using Penguin data used in R tutorial with
        consistent parameters except iso is False, and ndraws=200 (to expedite
        function running while testing)
        :return: same results as when using R stattest_clusters_approx function:
        stat = 1.223436; stderr ~ .07; p > .3 (with ndraw=200 there can be a
        lot of variability here)
        """
        penguin_data = np.genfromtxt(
            'tests/data_for_tests/penguin_data_subset.txt',
            delimiter=' ', skip_header=1)
        k = 5
        # set linkage to average to match R script
        positional_arguments = []
        keyword_arguments = {'n_clusters': k, 'affinity': 'euclidean',
                             'linkage': 'average'}
        cluster = AgglomerativeClustering(**keyword_arguments)
        cluster.fit_predict(penguin_data)
        k1 = 0
        k2 = 1
        stat, pval, stderr = stattest_clusters_approx(penguin_data, k1, k2,
                                                      cluster.labels_,
                                                      AgglomerativeClustering,
                                                      positional_arguments,
                                                      keyword_arguments,
                                                      iso=False,
                                                      ndraws=200)
        passing = True
        assert np.isclose(stat, 1.223436)
        try:
            assert np.isclose(stderr, 0.07, atol=.03)
        except AssertionError:
            passing = False
            print("stderr is {}, should be within .03 of "
                  "0.07".format(stderr))
        try:
            assert pval > .3
        except AssertionError:
            passing = False
            print("pval is {}, should be greater than .3".format(pval))
        self.assertTrue(passing)


    def test_penguin_gao_isoFalse_sigNone_siginvqxqndarray_200(self):
        """
        One-shot test using Penguin data used in R tutorial with
        consistent parameters except iso is False, ndraws=200 (to expedite
        function running while testing), and siginv provided
        :return: same results as when using R stattest_clusters_approx function
        with these parameters:
        stat = 8.134167; stderr < .009; p < .05 (with ndraw=200 there can be a
        lot of variability here, these may be a bad stderr and pval thresholds)
        """
        penguin_data = np.genfromtxt(
            'tests/data_for_tests/penguin_data_subset.txt',
            delimiter=' ', skip_header=1)
        k = 5
        # set linkage to average to match R script
        positional_arguments = []
        keyword_arguments = {'n_clusters': k, 'affinity': 'euclidean',
                             'linkage': 'average'}
        cluster = AgglomerativeClustering(**keyword_arguments)
        cluster.fit_predict(penguin_data)
        k1 = 0
        k2 = 1
        siginv = np.array([[1, 1], [1, 1]])
        stat, pval, stderr = stattest_clusters_approx(penguin_data, k1, k2,
                                                      cluster.labels_,
                                                      AgglomerativeClustering,
                                                      positional_arguments,
                                                      keyword_arguments,
                                                      iso=False,
                                                      siginv=siginv,
                                                      ndraws=2000)
        passing = True
        assert np.isclose(stat, 8.134167)
        try:
            assert stderr < .009
        except AssertionError:
            passing = False
            print("stderr is {}, should be less than "
                  "0.009".format(stderr))
        try:
            assert pval < .05
        except AssertionError:
            passing = False
            print("pval is {}, should be less than .05".format(pval))
        self.assertTrue(passing)


    def test_penguin_gao_isoTrue_sig5_200(self):
        """
        One-shot test using Penguin data used in R tutorial with
        consistent parameters except ndraws=200 (to expedite function running
        while testing), and sig is 5
        :return: same results as when using R stattest_clusters_approx function
        with these parameters:
        stat = 10.11433; stderr < .1; p > .1 (with ndraw=200 there can be a
        lot of variability here, these may be a bad stderr and pval thresholds)
        """
        penguin_data = np.genfromtxt(
            'tests/data_for_tests/penguin_data_subset.txt',
            delimiter=' ', skip_header=1)
        k = 5
        # set linkage to average to match R script
        positional_arguments = []
        keyword_arguments = {'n_clusters': k, 'affinity': 'euclidean',
                             'linkage': 'average'}
        cluster = AgglomerativeClustering(**keyword_arguments)
        cluster.fit_predict(penguin_data)
        k1 = 0
        k2 = 1
        stat, pval, stderr = stattest_clusters_approx(penguin_data, k1, k2,
                                                      cluster.labels_,
                                                      AgglomerativeClustering,
                                                      positional_arguments,
                                                      keyword_arguments,
                                                      iso=True,
                                                      sig=5,
                                                      ndraws=200)
        passing = True
        assert np.isclose(stat, 10.11433)
        try:
            assert stderr < .1
        except AssertionError:
            passing = False
            print("stderr is {}, should be less than "
                  "0.1".format(stderr))
        try:
            assert pval > .1
        except AssertionError:
            passing = False
            print("pval is {}, should be greater than .1".format(pval))
        self.assertTrue(passing)

    def test_gao_survives0(self):
        """
        One shot test to make sure Runtime Error raised if survives == 0
        Running same code as test_penguin_gao_200 but ndraws = 1, running
        until you get runtime error
        """
        penguin_data = np.genfromtxt(
            'tests/data_for_tests/penguin_data_subset.txt',
            delimiter=' ', skip_header=1)
        k = 5
        # set linkage to average to match R script
        positional_arguments = []
        keyword_arguments = {'n_clusters': k, 'affinity': 'euclidean',
                             'linkage': 'average'}
        cluster = AgglomerativeClustering(**keyword_arguments)
        cluster.fit_predict(penguin_data)
        k1 = 0
        k2 = 1
        passing = False
        for i in range(100):
            try:
                stat, pval, stderr = stattest_clusters_approx(penguin_data, k1, k2,
                                                      cluster.labels_,
                                                      AgglomerativeClustering,
                                                      positional_arguments,
                                                      keyword_arguments,
                                                      ndraws=1)
            except RuntimeError:
                passing = True
                break
        self.assertTrue(passing)

####### EDGE TESTS ###########
    def test_gao_ndraws_valueerror(self):
        """
        Edge test to make sure ValueError raised if ndraws < 0
        Running same code as test_penguin_gao_200 but ndraws = -1
        """
        penguin_data = np.genfromtxt(
            'tests/data_for_tests/penguin_data_subset.txt',
            delimiter=' ', skip_header=1)
        k = 5
        # set linkage to average to match R script
        positional_arguments = []
        keyword_arguments = {'n_clusters': k, 'affinity': 'euclidean',
                             'linkage': 'average'}
        cluster = AgglomerativeClustering(**keyword_arguments)
        cluster.fit_predict(penguin_data)
        k1 = 0
        k2 = 1
        with self.assertRaises(ValueError):
            stattest_clusters_approx(penguin_data, k1, k2, cluster.labels_,
                                                      AgglomerativeClustering,
                                                      positional_arguments,
                                                      keyword_arguments,
                                                      ndraws=-1)


    def test_x(self):
        """
        Edge test to make sure ValueError raised when x is not 2d numpy array
        Tests first with 3d numpy array, then 1d numpy array, than 2d list,
        last checks if x contains any nans
        """
        # 3d numpy array
        x = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        k1 = 0
        k2 = 1
        cluster_labels = np.array([0, 1, 0, 1])
        with self.assertRaises(ValueError):
            wald_test(x, k1, k2, cluster_labels)

        # 1d numpy array
        x = np.array([1, 2, 3, 4, 5, 6, 7, 8])
        cluster_labels = [0, 1, 0, 1, 0, 1, 0, 1]
        with self.assertRaises(ValueError):
            wald_test(x, k1, k2, cluster_labels)

        # 2d list
        x = [[5, 3],
              [10, 15],
              [15, 12],
              [24, 10],
              [30, 30],
              [85, 70],
              [71, 80],
              [60, 78],
              [70, 55],
              [80, 91], ]
        cluster_labels = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
        with self.assertRaises(ValueError):
            wald_test(x, k1, k2, cluster_labels)

        # contains nans
        x = np.array([[5., 3.],
                      [10., 15],
                      [15., 12],
                      [24., 10],
                      [30., 30],
                      [85, 70],
                      [71, 80],
                      [60, 78],
                      [70, 55],
                      [80, 91], ])
        x[1,1] = np.nan
        with self.assertRaises(ValueError):
            wald_test(x, k1, k2, cluster_labels)



    def test_clusterlabels(self):
        """
        Edge test to make sure ValueError raised when cluster_labels is wrong.
        Checks that ValueError is thrown if cluster_labels is too short,
        too long, not a numpy ndarray, not one-dimensional, or contains nans,
        or has fewer than 2 clusters.
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
        k = 2
        cl_fun = AgglomerativeClustering
        positional_arguments = []
        keyword_arguments = {'n_clusters': k, 'affinity': 'euclidean',
                             'linkage': 'average'}
        cluster = cl_fun(*positional_arguments, **keyword_arguments)
        cluster.fit_predict(x)
        k1 = 0
        k2 = 1

        # cluster_labels too short
        with self.assertRaises(ValueError):
            wald_test(x, k1, k2, np.array([1,0]))
        # cluster_labels too long
        with self.assertRaises(ValueError):
            wald_test(x, k1, k2, np.zeros(100))
        # cluster_labels list (not numpy array)
        with self.assertRaises(ValueError):
            wald_test(x, k1, k2, [1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
        # cluster_labels not one dimensional
        with self.assertRaises(ValueError):
            wald_test(x, k1, k2, np.zeros((5,2)))
        #cluster_labels contains nan values
        cluster_labels = np.array([1., 0, 1, 0, 1, 0, 1, 0, 1, 0])
        cluster_labels[0] = np.nan
        with self.assertRaises(ValueError):
            wald_test(x, k1, k2, cluster_labels)
        #cluster_labesl not between 2 and n:
        cluster_labels = np.zeros(10)
        with self.assertRaises(ValueError):
            wald_test(x, k1, k2, cluster_labels)

    def test_iso_bool(self):
        """
        Edge test to make sure ValueError raised when iso isn't boolean
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
        k = 2
        cl_fun = AgglomerativeClustering
        positional_arguments = []
        keyword_arguments = {'n_clusters': k, 'affinity': 'euclidean',
                             'linkage': 'average'}
        cluster = cl_fun(*positional_arguments, **keyword_arguments)
        cluster.fit_predict(x)
        k1 = 0
        k2 = 1
        with self.assertRaises(ValueError):
            wald_test(x, k1, k2, cluster.labels_, iso="hello")

    def test_iso_sig_siginv(self):
        """ Edge test to make sure sig and siginv are correct.
         Checks to make sure errors thrown if iso = True and siginv != None,
         if iso = False and sig != None, if sig is not a float or int,
         and if siginv is not a qxq ndarray.
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
        k = 2
        cl_fun = AgglomerativeClustering
        positional_arguments = []
        keyword_arguments = {'n_clusters': k, 'affinity': 'euclidean',
                             'linkage': 'average'}
        cluster = cl_fun(*positional_arguments, **keyword_arguments)
        cluster.fit_predict(x)
        k1 = 0
        k2 = 1
        siginv = np.array([[1, 1], [1, 1]])

        # iso True siginv not None
        with self.assertRaises(ValueError):
            wald_test(x, k1, k2, cluster.labels_, iso=True, siginv = siginv)

        # iso False, sig not None
        sig = 5
        with self.assertRaises(ValueError):
            wald_test(x, k1, k2, cluster.labels_, iso=False, sig = sig)

        # iso True, siginv None, sig not float or int
        with self.assertRaises(ValueError):
            wald_test(x, k1, k2, cluster.labels_, iso=True, sig="Hello")

        #iso False, sig None, siginv 2x2 list
        thissiginv = [[1, 1], [1, 1]]
        with self.assertRaises(ValueError):
            wald_test(x, k1, k2, cluster.labels_, iso=False, siginv=thissiginv)

        #iso False, sig None, siginv 1x1 numpy array (incorrect dimensions)
        thissiginv = np.array([1,2])
        with self.assertRaises(ValueError):
            wald_test(x, k1, k2, cluster.labels_, iso=False, siginv=thissiginv)

        #iso False, sig None, siginv 2x2x2 numpy array (incorrect dimensions)
        thissiginv = np.array([[1, 1, 1], [1, 1, 1]])
        with self.assertRaises(ValueError):
            wald_test(x, k1, k2, cluster.labels_, iso=False, siginv=thissiginv)

    def test_k1k2(self):
        """
        Edge test to make sure k1 and k2 are ints
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
        k = 2
        cl_fun = AgglomerativeClustering
        positional_arguments = []
        keyword_arguments = {'n_clusters': k, 'affinity': 'euclidean',
                             'linkage': 'average'}
        cluster = cl_fun(*positional_arguments, **keyword_arguments)
        cluster.fit_predict(x)
        k1 = "hello"
        k2 = True
        # k1 not an int
        with self.assertRaises(ValueError):
            stattest_clusters_approx(x, k1, 2, cluster.labels_, cl_fun,
                                 positional_arguments, keyword_arguments)
        # k2 not an int
        with self.assertRaises(ValueError):
            stattest_clusters_approx(x, 1, k2, cluster.labels_, cl_fun,
                                 positional_arguments, keyword_arguments)
        k1 = -1
        k2 = 2
        # k1 not between 0 and k-1
        with self.assertRaises(ValueError):
            stattest_clusters_approx(x, k1, 1, cluster.labels_, cl_fun,
                                 positional_arguments, keyword_arguments)
        # k2 not between 0 and k-1
        with self.assertRaises(ValueError):
            stattest_clusters_approx(x, 0, k2, cluster.labels_, cl_fun,
                                 positional_arguments, keyword_arguments)

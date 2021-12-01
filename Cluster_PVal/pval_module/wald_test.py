""" Implements Wald Test
Tests null hypothesis of no difference in means between clusters.
Copied from Lucy L Gao's implementation of Wald test in R.
Code for original function found:
https://github.com/lucylgao/clusterpval-experiments
clusterpval-experiments/real-data-code/util.R wald_test function
clusterpval package website: https://www.lucylgao.com/clusterpval/
Code written for: https://arxiv.org/abs/2012.02936
"""

import math
import scipy.stats
import numpy as np


def wald_test(x, k1, k2, cluster_labels, iso=True, sig=None, siginv=None):
    """
    Performs Wald Test
    :param x: n by p matrix (np.array), containing numeric data
    :param k1: integer, selects a cluster to test
    :param k2: integer, selects a cluster to test
    :param cluster_labels: numpy.ndarray, labels of each point (row) in X
    :param iso: boolean, if True isotropic covariance matrix model, otherwise
    not
    :param sig: optional scalar specifying sigma,  relevant if iso == True
    :param siginv: optional matrix specifying Sigma^-1, relevant if iso == False
    """

    rows, cols = x.shape
    n = rows
    q = cols

    unique, counts = np.unique(cluster_labels, return_counts=True)
    k1colmeans = np.mean(x[cluster_labels == k1, :], axis=0)
    k2colmeans = np.mean(x[cluster_labels == k2, :], axis=0)
    diff_means = k1colmeans - k2colmeans

    points_per_cluster = dict(zip(unique, counts))
    n1 = points_per_cluster[k1]
    n2 = points_per_cluster[k2]
    squared_norm_nu = (1 / n1) + (1 / n2)

    # NOTE: THIS CODE IS COPIED PASTED FROM STATTEST_CLUSTER_APPROX! Make
    # function
    if iso:
        if sig is None:
            colmeans = np.mean(x, axis=0)
            scaledx = x - colmeans
            sig = math.sqrt(np.sum(scaledx ** 2) / (n * q - q))
        else:
            pass
        scale_factor = squared_norm_nu * (sig ** 2)
        # Compute test statistic
        stat = np.linalg.norm(diff_means)
    else:
        if siginv is None:
            colmeans = np.mean(x, axis=0)
            scaledx = x - colmeans
            sig = np.cov(scaledx, rowvar=False)
            siginv = np.linalg.inv(sig)
        else:
            pass
        scale_factor = squared_norm_nu
        # Compute test statistic
        tdiff_means = np.transpose(diff_means)
        stat = math.sqrt(np.matmul(np.matmul(tdiff_means, siginv), diff_means))

    pval = 1 - scipy.stats.ncx2.cdf(x=(stat ** 2) / scale_factor, df=q, nc=0)
    return stat, pval

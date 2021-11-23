"""
Implements Lucy L Gao's test_clusters_approx function from clusterpval R package
Code for original function found in clusterpval/R/trunc_inf.R
clusterpval website: https://www.lucylgao.com/clusterpval/
Package described in: https://arxiv.org/abs/2012.02936
"""

import numpy as np

def stattest_clusters_approx(X, k1, k2, cluster_labels, iso=True, sig=None, SigInv=None, ndraws=2000):
    """
    Monte-Carlo significance test for any clustering method. This function takes matrix X clustered
    into K clusters and tests the null hypothesis of no difference in means between clusters k1 and
    k2. To account for the fact that the clusters were estimated from the data, the p-values are
    computed conditional on the fact that those clusters were estimated. P-values are approximated
    via importance sampling.
    :param X: n by p matrix, containing numeric data
    :param k1: integer, selects a cluster to test
    :param k2: integer, selects a cluster to test
    :param cluster_labels: numpy.ndarray, labels of each point (row) in X
    :param iso: boolean, if True isotropic covariance matrix model, otherwise not
    :param sig: optional scalar specifying sigma,  relevant if iso == True
    :param SigInv: optional matrix specifying Sigma^-1, relevant if iso == False
    :param ndraws: integer, selects the number of importance samples, default of 2000
    :return:
        - stat - float, the test statistic: Euclidean distance between mean of cluster k1 and mean of
                 cluster k2
        - pval - float, the approximate p value
        - stderr - float, the standard error of the p-value estimate
    """

    #check to make sure X is ndarray
    rows, cols = X.shape
    n = rows
    q = cols

    unique, counts = np.unique(cluster_labels, return_counts=True)
    K = len(unique)
    #check to make sure K (number of clusters) is between 2 and n
    #check to make sure k1 and k2 are between 0 and K-1
    #maybe check to make sure iso is true or false, or set default to true (as is done here)
    points_per_cluster = dict(zip(unique, counts))
    n1 = points_per_cluster[k1]
    n2 = points_per_cluster[k2]
    squared_norm_nu = (1/n1) + (1/n2)

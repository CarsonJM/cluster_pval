""" Implementations of Wald Test and Lucy Gao's test_clusters_approx function
Implements two p-value calculating functions that test the null hypothesis of
no difference between cluster means.

Lucy L Gao's test_clusters_approx function from clusterpval R package.
Code for original function found in clusterpval/R/trunc_inf.R
clusterpval package website: https://www.lucylgao.com/clusterpval/
Package described in: https://arxiv.org/abs/2012.02936

Wald test function copied from Lucy L Gao's Wald test implementation in R.
Code for original function found:
https://github.com/lucylgao/clusterpval-experiments
clusterpval-experiments/real-data-code/util.R wald_test function
Code written for: https://arxiv.org/abs/2012.02936
"""

import math
import scipy
import scipy.stats
import numpy as np
import pandas as pd

def check_inputs(x, k1, k2, cluster_labels, iso, sig, siginv):
    """
    Checks to make sure parameters given to pvalue functions are formatted
    correctly. Ensures x is 2d ndarray, k is between 2 and n, k1 and k2 are
    between 0 and k-1, iso is boolean, sig is float or int, siginv is either
    None or numpy ndarray with dimensions of covariance matrix
    ((number of x cols) by (number of x cols)).

    :param x: n by q matrix (np.array), containing numeric data
    :param k1: integer, selects cluster to test, must be between 0 and k-1
    :param k2: integer, selects cluster to test, must be between 0 and k-1
    :param cluster_labels: numpy.ndarray, labels of each point (row) in X,
    used here to find k
    :param iso: boolean, if True isotrophic covariance matrix model,
    otherwise not
    :param sig: optional scalar specifying sigma,  must be float or int
    :param siginv: optional matrix specifying Sigma^-1, must be either None
    or np.ndarray with dimensions qxq
    """
    # check to make sure types of all parameters are as we expect
    if not type(x) == np.ndarray:
        raise ValueError("x must be 2-dimensional numpy array")
    if not type(k1) == int or not type(k2) == int:
        raise ValueError("k1 and k2 must be integers between 0 and K-1")
    if not type(cluster_labels) == np.ndarray:
        raise ValueError("cluster_labels must be 1-dimensional numpy array")
    if not type(iso) == bool:
        raise ValueError("iso must be 'True' or 'False'")

    # if iso is false and sig is not none, error out and say sig won't be
    # used if iso is false
    if iso is False and not sig is None:
        raise ValueError("If iso = False, sig will not be used. Reevaluate if "
                         "you want iso = True or want to set sig = None")

    # if iso is true and siginv is not none, error out and say siginv won't be
    # used if iso is true
    if iso is True and not siginv is None:
        raise ValueError("If iso = True, siginv will not be used. Reevaluate "
                         "if you want iso = True or want to set siginv = None")

    # check types of sig and siginv
    if not type(sig) == float and not type(sig) == int and not sig is None:
        raise ValueError("sig must be a float or an int scalar")
    if not type(siginv) == np.ndarray and not siginv is None:
        raise ValueError("siginv must be a qxq numpy array (considering x's "
                         "dimensions to be nxq)")

    # check to make sure x is 2D ndarray
    if not x.ndim == 2:
        raise ValueError("x must be 2-dimensional numpy array")
    # check to make sure x has no nan values
    if np.isnan(x).any():
        raise ValueError("x cannot contain nan values")

    # check to make sure siginv has correct dimensions (qxq)
    rows, cols = x.shape
    n = rows
    q = cols
    if not siginv is None:
        try:
            siginv_rows, siginv_cols = siginv.shape
            if siginv_rows != q or siginv_cols != q:
                raise ValueError( "siginv must be a qxq numpy array "
                                  "(considering x's dimensions to be nxq)")
        except ValueError:
            #this will happen if siginv is not 2-dimensional numpy array
            raise ValueError("siginv must be a qxq numpy array (considering "
                             "x's dimensions to be nxq)")

    # check to make sure cluster_labels is 1D ndarray
    if not cluster_labels.ndim == 1:
        raise ValueError("cluster_labels must be 1-dimensional numpy array")
    # check to make sure cluster_labels has no nan values
    if np.isnan(cluster_labels).any():
        raise ValueError("cluster_labels cannot contain nan values")

    # check to make sure cluster_labels length is same as rows of x
    if not len(cluster_labels) == n:
        raise ValueError("cluster_labels must have one label for each point ("
                         "row of x)")

    # check to make sure K (number of clusters in cluster_labels) is between 2
    # and n
    unique, counts = np.unique(cluster_labels, return_counts=True)
    k = len(unique)
    if k < 2 or k > n:
        raise ValueError("cluster_labels must contain at least 2 clusters and "
                         "no more clusters than there are datapoints in x")

    #check to make sure k1 and k2 are between 0 and K-1
    if k1 < 0 or k1 > (k-1) or k2 < 0 or k2 > (k-1):
        raise ValueError("k1 and k2 must be between 0 and k-1")


def calculate_log_gamma_function(q):
    if q/2 <= 0:
        raise ValueError("Your dataset somehow has a negative number of "
                         "columns. Something has gone horribly wrong.")
    #q/2 will always be a positive number
    if scipy.special.gamma(q/2) != np.inf:
        return math.log(scipy.special.gamma(q/2))
    else:
        if q % 2 != 0:
            raise ValueError("Datasets with this many dimensions must have even "
                             "number of cols")
        nums_to_sum = []
        for i in range(1, int(math.floor(q/2))):
            nums_to_sum.append(math.log(i))
        total = sum(nums_to_sum)
        return total


def stattest_clusters_approx(x, k1, k2, cluster_labels, cl_fun,
                             positional_arguments, keyword_arguments,
                             iso=True, sig=None, siginv=None, ndraws=2000):
    """
    Monte-Carlo significance test for any clustering method. This function
    takes matrix X clustered into K clusters and tests the null hypothesis of
    no difference in means between clusters k1 and k2. To account for the
    fact that the clusters were estimated from the data, the p-values are
    computed conditional on the fact that those clusters were estimated.
    P-values are approximated via importance sampling.

    :param x: n by q matrix (np.array), containing numeric data
    :param k1: integer, selects a cluster to test
    :param k2: integer, selects a cluster to test
    :param cluster_labels: numpy.ndarray, labels of each point (row) in X
    :param cl_fun: function used to cluster data in clustering module
    :param positional_arguments: list of positional arguments used by cl_fun
    :param keyword_arguments: dict of keyword argument key:value pairs used by
    cl_fun
    :param iso: boolean, if True isotropic covariance matrix model, otherwise
    not
    :param sig: optional scalar specifying sigma,  relevant if iso is True
    :param siginv: optional matrix specifying Sigma^-1, relevant if
    iso == False
    :param ndraws: integer, selects the number of importance samples, default
    of 2000
    :return:
        - stat - float, the test statistic: Euclidean distance between mean of
                 cluster k1 and mean of cluster k2
        - pval - float, the approximate p value
        - stderr - float, the standard error of the p-value estimate
    """

    check_inputs(x, k1, k2, cluster_labels, iso, sig, siginv)

    #make sure ndraws >= 0
    if ndraws < 0:
        raise ValueError("ndraws must be >= 0")

    rows, cols = x.shape
    q = cols
    unique, counts = np.unique(cluster_labels, return_counts=True)
    points_per_cluster = dict(zip(unique, counts))
    n1 = points_per_cluster[k1]
    n2 = points_per_cluster[k2]
    k1colmeans = np.mean(x[cluster_labels == k1,:], axis=0)
    k2colmeans = np.mean(x[cluster_labels == k2,:], axis=0)
    diff_means = k1colmeans - k2colmeans
    prop_k2 = n2/(n1+n2)

    stat, scale_factor = calculate_scale_factor_and_stat(x, k1, k2,
                                                         cluster_labels, iso,
                                                         sig, siginv)

    scale_factor = math.sqrt(scale_factor)
    log_survives = np.empty(ndraws)
    log_survives[:] = np.NaN
    phi = np.random.normal(size=ndraws)*scale_factor + stat
    k1_constant = prop_k2*diff_means/stat
    k2_constant = (prop_k2 - 1)*diff_means/stat
    orig_k1 = np.transpose(x[cluster_labels == k1,:])
    orig_k2 = np.transpose(x[cluster_labels == k2, :])

    for j in range(ndraws):
        if phi[j] < 0:
            continue
        #compute the perturbed data set
        Xphi = np.copy(x)
        Xphi[cluster_labels == k1, :] = np.transpose(orig_k1[:]) + ((phi[j] - stat) * k1_constant)
        Xphi[cluster_labels == k2, :] = np.transpose(orig_k2[:]) + ((phi[j] - stat) * k2_constant)
        #recluster the perturbed data set
        cl_Xphi = cl_fun(*positional_arguments, **keyword_arguments)
        cl_Xphi.fit_predict(Xphi)

        if(preserve_cl(cluster_labels, cl_Xphi.labels_, k1, k2)):
            first_term = -((phi[j]/scale_factor)**2)/2
            second_term = (q-1)*math.log(phi[j]/scale_factor)
            third_term = (q/2 -1)*math.log(2)
            fourth_term = calculate_log_gamma_function(q)
            fifth_term = math.log(scale_factor)
            sixth_term = scipy.stats.norm.logpdf(phi[j], loc=stat, scale=scale_factor)
            log_survives[j] = (first_term + second_term - third_term - fourth_term - fifth_term - sixth_term)


    #trim down to only survives
    survives_indexes = np.where(~np.isnan(log_survives))
    phi = phi[survives_indexes]
    log_survives = log_survives[~np.isnan(log_survives)]
    survives = len(log_survives)

    #raise runtime error if nothing survives (test by running with 1 until it
    # hits)
    if survives == 0:
        raise RuntimeError("Oops - we didn't generate any samples that "
                           "preserved the clusters! Try re-running with a "
                           "larger value of ndraws")


    #approximate p values
    with np.errstate(invalid="ignore"):
        log_survives_shift = log_survives - max(log_survives)
    props = np.exp(log_survives_shift)/sum(np.exp(log_survives_shift))
    pval = sum(props[phi >= stat])
    var_pval = (1-pval)**2 * sum(props[phi >= stat]**2) + pval**2 * \
               sum(props[phi<stat]**2)
    stderr = math.sqrt(var_pval)

    return stat, pval, stderr


def wald_test(x, k1, k2, cluster_labels, iso=True, sig=None, siginv=None):
    """
    Performs Wald Test

    :param x: n by q matrix (np.array), containing numeric data
    :param k1: integer, selects a cluster to test
    :param k2: integer, selects a cluster to test
    :param cluster_labels: numpy.ndarray, labels of each point (row) in X
    :param iso: boolean, if True isotropic covariance matrix model, otherwise
    not
    :param sig: optional scalar specifying sigma,  relevant if iso == True
    :param siginv: optional matrix specifying Sigma^-1, relevant if iso == False
    :return:
        - stat - float, the test statistic: Euclidean distance between mean of
                 cluster k1 and mean of cluster k2
        - pval - float, the approximate p value
    """
    check_inputs(x, k1, k2, cluster_labels, iso, sig, siginv)
    stat, scale_factor = calculate_scale_factor_and_stat(x, k1, k2,
                                                         cluster_labels, iso,
                                                         sig, siginv)

    rows, cols = x.shape
    q = cols
    pval = 1 - scipy.stats.ncx2.cdf(x=(stat ** 2) / scale_factor, df=q, nc=0)
    return stat, pval


def calculate_scale_factor_and_stat(x, k1, k2, cluster_labels, iso, sig, siginv):
    """ Calculates scale factor and test statistic given iso and sig
    :param x: n by q matrix (np.array), containing numeric data
    :param k1: integer, selects a cluster to test
    :param k2: integer, selects a cluster to test
    :param cluster_labels: numpy.ndarray, labels of each point (row) in X
    :param iso: boolean, if True isotropic covariance matrix model, otherwise
    not
    :param sig: optional scalar specifying sigma,  relevant if iso is True
    :param siginv: optional matrix specifying Sigma^-1, relevant if iso is False
    :return:
        - stat - float, the test statistic: Euclidean distance between mean of
                 cluster k1 and mean of cluster k2
        - scale_factor - float, scale factor for p value calculation
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

    return stat, scale_factor


def preserve_cl(cl, cl_phi, k1, k2):
    """ Helper function for test_clusters_approx
    Checks if Ck, Ck' in C(x'(phi)). Returns True if Ck, Ck' in (C(x'(phi))),
    False otherwise. Checks if clusters in original cluster labelset are the
    same size as clusters in perturbed data labelset

    :param cl: list or numpy.ndarray, cluster labels from initial input
    :param cl_phi: list or numpy.ndarray, cluster labels of perturbed data
    :param k1: integer, index of cluster involved in test
    :param k2: integer, intex of cluster involved in test
    :return: True if Ck, Ck' in (C(x'(phi))), False otherwise
    """
    df = pd.DataFrame({'cl':cl, 'cl_phi':cl_phi})
    tab = pd.crosstab(index=df['cl'], columns=df['cl_phi'])
    k1_in = (sum(tab.iloc[k1, :] != 0) == 1) and (sum(tab.iloc[:, k1] != 0) == 1)
    k2_in = (sum(tab.iloc[k2, :] != 0) == 1) and (sum(tab.iloc[:, k2] != 0) == 1)
    return k1_in and k2_in
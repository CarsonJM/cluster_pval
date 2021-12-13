#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Implements cluster function for hierarchical and kmeans clustering of a
pandas dataframe using Scikit designed to cluster RNA seq data, but could be
applied to other types of datasets too.

The user is requested to submit the dataset, the desired number of clusters,
the cluster method and linkage method, which is standard set on ward.

"""

import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans


def clustering(input_dataset, nr_of_clusters, cluster_method,
                            linkage_method='ward'):
    """
    Function to cluster data based on number of clusters, cluster method and
    linkage method. The function automatically tests whether the input data
    is submitted in the correct data format.

    Parameters:
    :param dataset: pandas dataframe, dataset with RNA seq data
    :param nr_of_clusters: integer, number of cluster that should be consdered
    :param cluster_method: string, string with name of cluster method

    Returns:
    pandas dataframe: input_dataframe (input dataframe with calculated cluster)
    integer: nr_of_clusters (number of clusters)
    function: ccl_fun (used function)
    list: positional_arguments
    dictionary: keyword_arguments
    """
    dataset = input_dataset
    check_value_type1 = isinstance(nr_of_clusters, int)
    if check_value_type1 is False:
        raise ValueError("The number of clusters should an integer.")
    else:
        pass

    check_value_type2 = isinstance(input_dataset, pd.DataFrame)
    if check_value_type2 is False:
        raise ValueError("The dataset should be a pandas dataframe.")
    else:
        pass

    check_value_type3 = isinstance(cluster_method, str)
    if check_value_type3 is False:
        raise ValueError("The cluster method should be a string.")
    else:
        pass

    if cluster_method == "k-means":
        cluster = KMeans(n_clusters=nr_of_clusters)
        dataset['cluster'] = cluster.fit_predict(input_dataset)
        ccl_fun = KMeans
    else:
        pass


    if cluster_method == "hierarchical":
        cluster = AgglomerativeClustering(n_clusters= nr_of_clusters,
                                          affinity='euclidean',
                                          linkage=linkage_method)
        dataset['cluster'] = cluster.fit_predict(dataset)
        ccl_fun = AgglomerativeClustering

    positional_arguments = []
    keyword_arguments = {'n_clusters': nr_of_clusters, 'affinity': 'euclidean',
                         'linkage': 'ward'}

    return dataset, nr_of_clusters, ccl_fun, positional_arguments, keyword_arguments

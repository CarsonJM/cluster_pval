#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 15:01:35 2021

Cluster function for hierarchical clustering
"""

import pandas as pd
from sklearn.cluster import AgglomerativeClustering


def hierarchical_clustering(dataset, nr_of_clusters):
    """
    Function to cluster data hierarchically
    
    Parameters:
    :param1 pandas dataframe dataset: dataframe with RNA seq data
    :param2 integer nr_of_cluster: number of cluster that should be consdered

    Returns:
    float: result value of query
    list: list with calculated distances
   """
    check_value_type1 = isinstance(nr_of_clusters, int)
    if check_value_type1 is False:
        raise ValueError("The number of clusters should an integer.")
    else:
        pass
    check_value_type2 = isinstance(dataset, pd.DataFrame)
    if check_value_type2 is False:
        raise ValueError("The dataset should be a pandas dataframe.")
    else:
        pass
    cluster = AgglomerativeClustering(n_clusters= nr_of_clusters, affinity='euclidean', linkage='ward')
    cluster_method = "AgglomerativeClustering(n_clusters= nr_of_clusters, affinity='euclidean', linkage='ward')"
    dataset['cluster'] = cluster.fit_predict(dataset)
    return dataset, cluster_method, dataset, nr_of_clusters

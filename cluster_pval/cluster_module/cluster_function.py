#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 15:01:35 2021

Cluster function for hierarchical clustering of a pandas dataset based on requested number of clusters
"""

import pandas as pd
from sklearn.cluster import AgglomerativeClustering


def hierarchical_clustering(dataset, nr_of_clusters):
    """
    Function to cluster data hierarchically
    
    Parameters:
    :param dataset: pandas dataframe dataset: dataframe with RNA seq data
    :param nr_of_clusters: integer nr_of_cluster: number of cluster that should be consdered

    Returns:
    pandas dataframe: input_dataframe (input dataframe with calculated cluster)
    integer: nr_of_clusters (number of clusters)
    function: ccl_fun (used function)
    list: positional_arguments  
    dictionary: keyword_arguments
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
    dataset['cluster'] = cluster.fit_predict(dataset)
    ccl_fun = AgglomerativeClustering 
    positional_arguments = []
    keyword_arguments = {'n_clusters': nr_of_clusters, 'affinity': 'euclidean','linkage': 'ward'}
    return dataset, nr_of_clusters, ccl_fun, positional_arguments, keyword_arguments




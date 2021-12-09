#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 15:01:35 2021

Cluster function for hierarchical clustering of a pandas dataset based on requested number of clusters
"""

import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans


def hierarchical_clustering(input_dataset, nr_of_clusters, cluster_method, linkage_method='ward'):
    """
    Function to cluster data hierarchically
    
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
    
    if cluster_method == "KMeans":
        cluster = KMeans(n_clusters=nr_of_clusters)
        dataset['cluster'] = cluster.fit_predict(input_dataset)
        ccl_fun = KMeans
    else:
        pass
    
    if cluster_method == "Hierarchical":
        cluster = AgglomerativeClustering(n_clusters= nr_of_clusters, affinity='euclidean', linkage=linkage_method)
        dataset['cluster'] = cluster.fit_predict(dataset)
        ccl_fun = AgglomerativeClustering 
        
    positional_arguments = []
    keyword_arguments = {'n_clusters': nr_of_clusters, 'affinity': 'euclidean','linkage': 'ward'}
    
    return dataset, nr_of_clusters, ccl_fun, positional_arguments, keyword_arguments




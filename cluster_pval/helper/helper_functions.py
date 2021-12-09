"""
Common
=======
This module serves as a storage location for functions that are used by the app.py dashboard.

Functions
---------
iterate_wald_test: 

iterate_stattest_clusters_approx: 
"""
import pandas as pd
from cluster_pval import pval_module

def iterate_wald_test(x, cluster_labels, iso=True, sig=None, siginv=None):
    comparison_list = []
    wald_pvalue_list = []

    for k1 in range(len(set(cluster_labels))):
         for k2 in range(k1, set(cluster_labels)):
            comparison_list.append(str(k1) + ',' + str(k2))
            stat, wald_pvalue = pval_module.wald_test(x, k1, k2, x['cluster'], iso=iso, sig=sig, siginv=siginv)
            wald_pvalue_list.append(wald_pvalue)
            
    pvalue_df = pd.DataFrame({'comparisons':comparison_list, 'wald_pvalue':wald_pvalue_list})

    return pvalue_df

def iterate_stattest_clusters_approx(wald_pvalue_df, sig_threshold, x, cluster_labels, cl_fun, positional_arguments, keyword_arguments, iso=True, sig=None, siginv=None, ndraws=2000):
    comparison_list = []
    cluster_pvalue_list = []

    wald_pvalue_sig = wald_pvalue_df[wald_pvalue_df['wald_pvalue'] < sig_threshold]
    sig_comparisons = wald_pvalue_sig['comparisons'].tolist()

    for i in sig_comparisons:
        comparison_list.append(i)
        k1, k2 = i.split(',')
        stat, cluster_pvalue = pval_module.wald_test(x, k1, k2, cluster_labels, cl_fun, positional_arguments, keyword_arguments, iso=True, sig=None, siginv=None, ndraws=2000)
        
        cluster_pvalue_list.append(cluster_pvalue)

    cluster_pvalue_df = pd.DataFrame({'comparisons':comparison_list, 'cluster_pvalue':cluster_pvalue_list})

    combined_pvalue_df = wald_pvalue_df.merge(cluster_pvalue_df, how='left', on='comparisons')

    return combined_pvalue_df
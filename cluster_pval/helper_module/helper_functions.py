"""
Common
=======
This module serves as a storage location for functions that are used by the app.py dashboard.

Functions
---------
iterate_wald_test: 

iterate_stattest_clusters_approx: 
"""
import base64
import io

import pandas as pd

from cluster_pval import pval_module
from cluster_pval import display_module

import dash
from dash.dependencies import Input, Output, State
from dash import dcc
from dash import html
from dash import dash_table



available_clustering_methods = ['Hierarchical']
available_linkage_methods = ['ward', 'complete', 'average', 'single']

def iterate_wald_test(x, cluster_labels, iso=True, sig=None, siginv=None):
    """
    Function to perform the wald test
    
    Parameters:
    :param x: pandas dataframe, dataframe with RNA seq data
    :param cluster_labels: pandas series, serie with cluster labels
    :param iso: boolean, if True isotrophic covariance matrix model,
     otherwise not
    :param sig: optional scalar specifying sigma,  must be float or int
    :param siginv: optional matrix specifying Sigma^-1, must be either None
    or np.ndarray with dimensions qxq

    Returns:
    pandas dataframe: pvalue dataframe, 
    """
    comparison_list = []
    wald_pvalue_list = []

    x_np = x.to_numpy()
    cluster_labels_np = cluster_labels.to_numpy()

    for k1 in range(len(set(cluster_labels))):
         for k2 in range(k1+1, len(set(cluster_labels))):
            comparison_list.append(str(k1) + ',' + str(k2))
            stat, wald_pvalue = pval_module.wald_test(x_np, k1, k2, cluster_labels_np, iso=iso, sig=sig, siginv=siginv)
            wald_pvalue_list.append(wald_pvalue)
            
    pvalue_df = pd.DataFrame({'comparisons':comparison_list, 'wald_pvalue':wald_pvalue_list})

    return pvalue_df

def iterate_stattest_clusters_approx(wald_pvalue_df, sig_threshold, x, cluster_labels, cl_fun, positional_arguments, keyword_arguments, iso, sig, siginv, ndraws):
    """
    Function to recalculate the significant pvalues obtained from the wald test to calculate the adjusted pvalue
    
    Parameters:
    :param wald_pvalue_df: pandas dataframe with pvalues from Wald test
    :param sig_threshold: float, threshold that determine significance
    :param x: pandas dataframe, dataframe with RNA seq data
    :param cluster_labels: pandas series, serie with cluster labels
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
    
    Returns:
    pandas dataframe: combined pvalue dataframe
    """
    comparison_list = []
    cluster_pvalue_list = []

    x_np = x.to_numpy()
    cluster_labels_np = cluster_labels.to_numpy()

    wald_pvalue_sig = wald_pvalue_df[wald_pvalue_df['wald_pvalue'] < sig_threshold]
    sig_comparisons = wald_pvalue_sig['comparisons'].tolist()

    for i in sig_comparisons:
        comparison_list.append(i)
        k1, k2 = i.split(',')
        stat, cluster_pvalue, stderr = pval_module.stattest_clusters_approx(x_np, int(k1), int(k2), cluster_labels_np, cl_fun, positional_arguments, keyword_arguments, iso=True, sig=None, siginv=None, ndraws=ndraws)
        
        cluster_pvalue_list.append(cluster_pvalue)

    cluster_pvalue_df = pd.DataFrame({'comparisons':comparison_list, 'cluster_pvalue':cluster_pvalue_list})

    combined_pvalue_df = wald_pvalue_df.merge(cluster_pvalue_df, how='left', on='comparisons')

    return combined_pvalue_df

# output-read-file-status (determine if file is csv, and return status)
def read_file_status(filename):
    """
    Function to upload the CSV file
    
    Parameters:
    :param filename: string with pathway and name of file
    
    returns:
    string: reading file status
    """
    if 'csv' in filename:
        return html.Div([
            # return read file status
            html.Div(children=[
                'Reading file: ', filename
                ]),

            html.Hr(),
        ])
    else:
        return html.Div(children=[
            'The uploaded file must be in csv format'],
            style={
            'textAlign': 'center',
            'fontSize': '30px'
            }
        )

# once file is loaded, return preview, and return input for number of clusters
def preview_file_and_num_clusters_and_cluster_methods(contents, filename):
    """
    Function to give a preview of the uploaded dataset on the dashboard and request input of the user
    
    Parameters:
    :param contents: CSV file with content of input data
    :param filename: string the name of file
    
    returns:
    preview of data
    """    
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)

    if 'csv' in filename:
        try:
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
            
    # return exception if file is not read
        except Exception as e:
            print(e)
            return html.Div(children=[
                'An error occured when reading the file'],
                style={
                'textAlign': 'center',
                'fontSize': '30px'
                }
            )

        # objects to return after reading the file
        return html.Div([
            html.H6(['File preview: ', filename]),
            html.Div(''),
            html.Div(''),
            # return a preview of the file showing first 10 lines
            dash_table.DataTable(
                data=df.to_dict('records'),
                columns=[{'name': i, 'id': i} for i in df.columns],
                page_size=10,
                style_table={'overflowX': 'auto'},
                style_cell={
                'overflow': 'hidden',
                'textOverflow': 'ellipsis',
                'textAlign': 'left'
                }
            ),

            html.Hr(),

            # return option to specify which columns contain data
            html.Div([
                html.H6('Data information:'),
                html.Div(''),
                html.Div(''), 
                #give a preview of the uploaded dataset on the dashboard and request input of the user
                "Input columns containing data to be clustered (first column should denoted with 0):",
                html.Div([dcc.Input(id='min-col', type='number', min=0, step=1),
                ' - ', dcc.Input(id='max-col', type='number', min=0, step=1)
                ]),
            ]),

            html.Hr(),

            # return option to input desired number of clusters
            html.Div([
                html.H6('Number of clusters:'),
                html.Div(''),
                html.Div(''),
                "Input desired number of clusters: ",
                dcc.Input(id='num-clusters', type='number', min=1, step=1)
            ]),

            html.Hr(),

            html.Div([
                html.H6('Clustering method:'),
                html.Div(''),
                html.Div(''),
                "Select a clustering method to use",
                dcc.Dropdown(
                id='cluster-method',
                options=[{'label': i, 'value': i} for i in available_clustering_methods],
            ),

            html.Hr()
            ])
        ])

# once number of clusters and method has been selected, return linkage method and/or cluster button
def linkage(cluster_method):
    """
    Function to ask for a linkage approach
    
    Parameters:
    :param contents: CSV file with content of input data
    :param filename: string the name of file
    
    """    
    if cluster_method == 'Hierarchical':
        # objects to return after reading the file
        return html.Div([
            html.H6('Linkage method:'),
            html.Div(''),
            html.Div(''),
            # return option to select linkage method
            html.Div(['Select a linkage method to be applied to hierarchical clustering',
            dcc.Dropdown(
                id='linkage-method',
                options=[{'label': i, 'value': i} for i in available_linkage_methods],
            )
            ]),

            html.Hr()
        ])
    
    else:
        return html.Div([
            dcc.Input(id='linkage-method', value=0, type='hidden'),
        ])

# once linkage has been seleceted and/or button has been pressed, return clustering status
def cluster_settings_and_submit(filename, min_col, max_col, num_clusters, cluster_method, linkage_method):
    """
    Function to determine cluster settings and submit clustering
    
    Parameters:
    :param filename: string the name of file
    :param min_col: integer with column index in which cluster data starts
    :param max_col: integer with column index in which cluster data ends
    :param num_clusters: integer with the number of desired number of clusters
    :param cluster_methods: string with the name of the cluster method
    :param linkage_method: string with the name of the linkage method
    """    
    if cluster_method == 'Hierarchical':
        # return clustering status
        return html.Div([
            html.H6('Clustering summary:'),
            html.Div(''),
            html.Div(''),
            html.Div(['Clustering file: ', filename, '\n']),
            html.Div(['Columns to be clustered: ', min_col, ' - ', max_col]),
            html.Div(['Number of clusters: ', num_clusters, '\n']),
            html.Div(['Clustering method: ', cluster_method, '\n']),
            html.Div(['Linkage method: ', linkage_method]),

            html.Hr(),

            # button to submit clustering
            html.Button(id='cluster-button', children='Press to submit clustering'),

            html.Hr(),
        ])

# once the options have been selected and button is pressed, cluster the data, and display cluster graph
def cluster_figure(clustered_df):
    """
    Function to show plot with clustered data
    
    Parameters:
    :param clustered_df: dataframe with input data including clusters
    
    returns:
    graph with the clustered data
    """
    # generate clustering figure
    fig = display_module.cluster_plot(clustered_df)

    return html.Div([
        html.H6('Cluster visualization:'),
        html.Div(''),
        html.Div(''),
        # return visualization of clustering
        dcc.Graph(
        figure=fig
        ),

        html.Hr()
        ])

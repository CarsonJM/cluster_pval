"""
Dashboard helper
================
This module serves as a storage location for functions that are used
to generate dashboard outputs for each callback

Functions
---------
:read_file_status: returns html object with read file status
:file_preview_req_num_clusters_req_cluster_method: returns preview of the uploaded file,
and requests number of clusters/clustering method
:req_linkage_method: requests user to input a linkage method
:cluster_settings_req_submit: displays input cluster settings and request final submission
:cluster_figure: returns interactive clustered figure
:req_threshold_req_num_draws: requests input of significance threshold
and number of draws (if applicable)
:iterate_wald_test: iterates wald test over each pairwise cluster comparison
:wald_preview_clusterpval_status: returns preview of wald table, and displays cluster status
:iterate_stattest_clusters_approx: iterates adjusted p-value test over
each significant pairwise cluster comparison (determined via wald)
:cluster_pval_preview: returns a preview of the wald and ajusted p-values
side-by-side

"""
import base64
import io

import pandas as pd

from dash import dcc
from dash import html
from dash import dash_table

from cluster_pval import pval_module
from cluster_pval import display_module

# lists with clustering and linkage methods
available_clustering_methods = ['hierarchical', 'k-means']
available_linkage_methods = ['ward', 'complete', 'average', 'single']

# output-read-file-status (determine if file is csv, and return status)
def read_file_status(filename):
    """
    Function to return the status of the CSV upload

    Parameters:
    :param filename: string with pathway and name of file

    returns:
    display reading file status with filename
    """
    if 'csv' in filename:
        return html.Div([
            # return read file status
            html.Div('Reading file: ' + str(filename),
                style={'font-style': 'italic',
                'font-weight':'bold'}),

            html.Hr(),
        ])
    else:
        return html.Div(children=[
            'The uploaded file must be in csv format'],
            style={
            'textAlign': 'center',
            'fontSize': '30px',
            'color' : 'red'
            }
        )

# once file is loaded, return preview, and return input for number of clusters
def file_preview_req_num_clusters_req_cluster_method(orientation, contents, filename):
    """
    Function to give a preview of the uploaded dataset on
    the dashboard and request input of the user

    Parameters:
    :param contents: CSV file with content of input data
    :param filename: string the name of file

    returns:
    preview of data, request for data info, request number of clusters,
    and request clustering method
    """
    _, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)

    if 'csv' in filename:
        try:
            data_df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
            if orientation == 'columns':
                data_df = data_df.transpose()
            else:
                pass

    # return exception if file is not read
        except Exception as exception_e:
            print(exception_e)
            return html.Div(children=[
                'An error occured when reading the file'],
                style={
                'textAlign': 'center',
                'fontSize': '30px'
                }
            )

        # objects to return after reading the file
        return html.Div([
            html.H6(['File preview: '], style={'font-weight':'bold'}),
            html.Div(''),
            html.Div(''),
            # return a preview of the file showing first 10 lines
            dash_table.DataTable(
                data=data_df.to_dict('records'),
                columns=[
                    {'name': i, 'id': i} for i in data_df.columns] if orientation =='rows' else [
                        {'name': str(i), 'id': str(i)} for i in data_df.columns],
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
                html.H6('Data information:', style={'font-weight':'bold'}),
                html.Div(''),
                html.Div(''),
                #give a preview of the uploaded dataset on the dashboard
                # and request input of the user
                html.Div("Input columns containing data to be clustered",
                style={'font-weight':'bold'}),
                html.Br(),
                html.Div("Leaving empty will select the entire dataframe. ",
                style={'font-style': 'italic'}),
                html.Div("Use 0-based indexes, and leave last number blank if"
                "the rest of the dataframe should be used", style={'font-style': 'italic'}),
                html.Div([dcc.Input(id='min-col', type='number', min=0, step=1),
                ' - ', dcc.Input(id='max-col', type='number', min=0, step=1)
                ]),
            ]),

            html.Hr(),

            # return option to input desired number of clusters
            html.Div([
                html.H6('Number of clusters:', style={'font-weight':'bold'}),
                html.Div(''),
                html.Div(''),
                html.Div("Input desired number of clusters: ", style={'font-weight':'bold'}),
                dcc.Input(id='num-clusters', type='number', min=1, step=1)
            ]),

            html.Hr(),

            html.Div([
                html.H6('Clustering method:', style={'font-weight':'bold'}),
                html.Div(''),
                html.Div(''),
                html.Div("Select a clustering method to use", style={'font-weight':'bold'}),
                dcc.Dropdown(
                id='cluster-method',
                options = [{'label': i, 'value': i} for i in available_clustering_methods],
            ),

            html.Hr()
            ])
        ])

# once number of clusters and method has been selected,
# return linkage method and/or cluster button
def req_linkage_method(cluster_method):
    """
    Function to ask user for a linkage method

    Parameters:
    :param num_clusters: number of clusters input by user
    :param cluster_method: cluster method selected by user

    returns:
    a request for user to input linkage method,
    or proceeds to next step if
    k-means cluster method was selected
    """
    if cluster_method == 'hierarchical':
        # objects to return after reading the file
        return html.Div([
            html.H6('Linkage method:', style={'font-weight':'bold'}),
            html.Div(''),
            html.Div(''),
            # return option to select linkage method
            html.Div('Select a linkage method to be applied to hierarchical clustering',
            style={'font-weight':'bold'}),
            dcc.Dropdown(
                id='linkage-method',
                options=[{'label': i, 'value': i} for i in available_linkage_methods]),

            html.Hr()
        ])

    else:
        return html.Div([
            dcc.Input(id='linkage-method', value=0, type='hidden'),
        ])

# once linkage has been seleceted and/or button has been pressed, return clustering status
def cluster_settings_req_submit(filename, min_col, max_col,
num_clusters, cluster_method, linkage_method):
    """
    Function to determine cluster settings and submit clustering

    Parameters:
    :param filename: string the name of file
    :param min_col: integer with column index in which cluster data starts
    :param max_col: integer with column index in which cluster data ends
    :param num_clusters: integer with the number of desired number of clusters
    :param cluster_methods: string with the name of the cluster method
    :param linkage_method: string with the name of the linkage method

    returns:
    a preview of user selected clustering settings, and a button for submitting
    the clustering job
    """
    if cluster_method == 'hierarchical':
        # return clustering status
        return html.Div([
            html.H6('Clustering summary:', style={'font-weight':'bold'}),
            html.Div(''),
            html.Div(''),
            html.Div(['Clustering file: ', filename, '\n']),
            html.Div(['Columns to be clustered: ', min_col, ' - ', max_col]),
            html.Div(['Number of clusters: ', num_clusters, '\n']),
            html.Div(['Clustering method: ', cluster_method, '\n']),
            html.Div(['Linkage method: ', linkage_method]),

            html.Br(),

            html.Div("Clustering will be performed using scikit-learn's clustering algorithms",
            style={'font-style':'italic'}),
            html.Div("See https://scikit-learn.org/stable/modules/clustering.html for more information",
            style={'font-style':'italic'}),

            html.Hr(),

            # button to submit clustering
            html.Button(id='cluster-button', n_clicks=0, children='Press to submit clustering',
            style={'font-weight':'bold'}),

            html.Hr(),
        ])

    else:
        return html.Div([
            html.H6('Clustering summary:', style={'font-weight':'bold'}),
            html.Div(''),
            html.Div(''),
            html.Div(['Clustering file: ', filename, '\n']),
            html.Div(['Columns to be clustered: ', min_col, ' - ', max_col]),
            html.Div(['Number of clusters: ', num_clusters, '\n']),
            html.Div(['Clustering method: ', cluster_method, '\n']),

            html.Hr(),

            # button to submit clustering
            html.Button(id='cluster-button', n_clicks=0,
            children='Press to submit clustering', style={'font-weight':'bold'}),

            html.Hr()
        ])

# once the options have been selected and button is pressed,
# cluster the data, and display cluster graph
def cluster_figure(clustered_df):
    """
    Function to show plot with clustered data

    Parameters:
    :param clustered_df: dataframe with input data including clusters

    returns:
    figure with the clustered data
    """
    # generate clustering figure
    fig = display_module.cluster_plot(clustered_df)

    return html.Div([
        html.H6('Cluster visualization:', style={'font-weight':'bold'}),
        html.Div("Created using plotly (https://plotly.com) and scikit-learn decompositon"
        "(https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)",
        style={'font-style':'italic'}),
        html.Div(''),
        html.Div(''),
        # return visualization of clustering
        dcc.Graph(
        figure=fig
        ),

        html.Hr()
        ])

# once figure has been displayed, request threshold and num draws if applicable
def req_threshold_req_num_draws():
    """
    Function to request user input of p-value threshold
    and the number of draws to use

    Parameters:
    :None:

    returns:
    request user input of p-value threshold and number of draws
    """
    return html.Div([
        html.H6('P-value calculation information:', style={'font-weight':'bold'}),
        html.Div('The initial p value is calculated using a Wald test.',
        style={'font-style':'italic'}),
        html.Div(['The adjusted p value calculation methods used',
        'here assume an isotropic covariance matrix model'],
        style={'font-style':'italic'}),
        html.Div(['and estimates sigma as described in [Gao et al 2020 section 4.3]',
        '(https://arxiv.org/pdf/2012.02936.pdf).'], style={'font-style':'italic'}),
        html.Div(['Adjusted p value is estimated using Monte Carlo approximation',
        'as described in [Gao et al 2020 section 4.1](https://arxiv.org/pdf/2012.02936.pdf).'],
        style={'font-style':'italic'}),

        html.Br(),
        html.Br(),

        html.Div("Input p-value significance threshold: ",
        style={'font-weight':'bold'}),
        html.Div("Must be a float between 0.0 and 1.0",
        style={'font-style':'italic'}),
        dcc.Input(id='sig-threshold', type='number',
        min=0, max=1),

        html.Br(),
        html.Br(),

        html.Div("Input number of draws to be used in calculating adjusted p-value: ",
        style={'font-weight':'bold'}),
        html.Div("Must be an integer (recommended number of draws is 2,000)",
        style={'font-style':'italic'}),
        dcc.Input(id='num-draws', type='number', min=0),

        html.Hr(),

        html.Button(id='p-value-button', n_clicks=0,
        children='Press to submit p-value calculation',
        style={'font-weight':'bold'}),

        html.Hr()
    ])

# function to iterate wald test over each comparison
def iterate_wald_test(data_df, cluster_labels):
    """
    Function to iterate the wald test over each comparison

    Parameters:
    :param x: pandas dataframe, dataframe with RNA seq data
    :param cluster_labels: pandas series, serie with cluster labels

    Returns:
    pandas dataframe: p-value dataframe containing comparisons and 
    their associated wald p-value
    """
    comparison_list = []
    wald_pvalue_list = []

    x_data = data_df.iloc[: , :-1]
    x_np = x_data.to_numpy()
    cluster_labels_np = cluster_labels.to_numpy()

    for k_1 in range(len(set(cluster_labels))):
        for k_2 in range(k_1 + 1, len(set(cluster_labels))):
            comparison_list.append(str(k_1) + ',' + str(k_2))
            _, wald_pvalue = pval_module.wald_test(x_np, k_1, k_2,
            cluster_labels_np, iso=iso, sig=sig, siginv=siginv)
            wald_pvalue_list.append(wald_pvalue)

    pvalue_df = pd.DataFrame({'comparisons':comparison_list, 'wald_pvalue':wald_pvalue_list})

    return pvalue_df

# once wald is calculated, return preview of the table
def wald_preview_clusterpval_status(wald_json):
    """
    Function to preview the calculated wald p-values

    Parameters:
    :wald_json: pandas df, with results from wald p-values

    Returns:
    pandas dataframe: a display of p-value dataframe
    containing comparisons and their associated wald p-value
    """

    wald_df = pd.read_json(wald_json, orient='split')
    wald_df.loc[:,'comparison_1'] = wald_df['comparisons'].str.split(',', expand=True)[0].astype(int)
    wald_df.loc[:,'comparison_2'] = wald_df['comparisons'].str.split(',', expand=True)[1].astype(int)
    wald_df.loc[:,'comparisons'] = (wald_df['comparison_1'].astype(int) + 1).astype(str) + ',' + (wald_df['comparison_2'].astype(int) + 1).astype(str)
    wald_df = wald_df.iloc[:, :-2]

    return html.Div([
        html.H6(['Wald p-value file preview: '], style={'font-weight':'bold'}),
        html.Div(''),
        html.Div(''),

        # return a preview of the file showing first 10 lines
        dash_table.DataTable(
            data=wald_df.to_dict('records'),
            columns=[{'name': i, 'id': i} for i in wald_df.columns],
            page_size=10,
            style_table={'overflowX': 'auto'},
            style_cell={
            'overflow': 'hidden',
            'textOverflow': 'ellipsis',
            'textAlign': 'left'
            }
        ),

        html.Hr(),

        # return status of cluster pval
        html.Div(["Calculating adjusted p-value for clusters that were"
        "significantly different according to Wald p-value"],
        style={'font-weight':'bold', 'font-style':'italic'}),

        html.Hr()

    ])

# function to iterate wald test over all significant comparisons
def iterate_stattest_clusters_approx(wald_pvalue_df, sig_threshold, data_df,
cluster_labels, cl_fun, positional_arguments, keyword_arguments, n_draws):
    """
    Function to recalculate the significant pvalues obtained from
    the wald test to calculate the adjusted pvalue

    Parameters:
    :param wald_pvalue_df: pandas dataframe with pvalues from Wald test
    :param sig_threshold: float, threshold that determine significance
    :param x: pandas dataframe, dataframe with RNA seq data
    :param cluster_labels: pandas series, serie with cluster labels
    :param cl_fun: function used to cluster data in clustering module
    :param positional_arguments: list of positional arguments used by cl_fun
    :param keyword_arguments: dict of keyword argument key:value pairs used by
    cl_fun
    :param ndraws: integer, selects the number of importance samples, default
    of 2000

    Returns:
    pandas dataframe: combined pvalue dataframe
    """
    comparison_list = []
    cluster_pvalue_list = []

    x_data = data_df.iloc[: , :-1]
    x_np = x_data.to_numpy()
    cluster_labels_np = cluster_labels.to_numpy()

    wald_pvalue_sig = wald_pvalue_df[wald_pvalue_df['wald_pvalue'] < float(sig_threshold)]
    sig_comparisons = wald_pvalue_sig['comparisons'].tolist()

    for i in sig_comparisons:
        comparison_list.append(i)
        k_1, k_2 = i.split(',')
        _, cluster_pvalue, _ = pval_module.stattest_clusters_approx(x_np, int(k_1),
        int(k_2), cluster_labels_np, cl_fun, positional_arguments,
        keyword_arguments, ndraws=n_draws)

        cluster_pvalue_list.append(cluster_pvalue)

    cluster_pvalue_df = pd.DataFrame({'comparisons':comparison_list,
    'cluster_pvalue':cluster_pvalue_list})

    combined_pvalue_df = wald_pvalue_df.merge(cluster_pvalue_df, how='left', on='comparisons')

    return combined_pvalue_df

# function to display preview of clusterpval table
def clusterpval_preview(clusterpval_json, sig_threshold):
    """
    Function to preview the calculated adjusted p-values
    compared to the original wald p-values

    Parameters:
    :wald_pvalue_df: pandas df, with results from wald p-values
    :sig_threshold: significance threshold input by user
    :data_df: pandas df containing data input by user
    :cluster_labels: pandas df containing the clusters
    each sample/cell is assigned to
    :cl_fun: the sklearn function used for clustering
    :positional_arguments: positional_arguments used in clustering
    the data
    :keyword_arguments: keyword_arguments used in clustering the data
    :n_draw: number of draws to use when calculatig the adjusted p-value

    Returns:
    pandas dataframe: a display of p-value dataframe
    containing comparisons and their associated wald
    & adjusted p-values
    """
    clusterpval_df = pd.read_json(clusterpval_json, orient='split')
    clusterpval_df.loc[:,'comparison_1'] = clusterpval_df['comparisons'].str.split(',', expand=True)[0].astype(int)
    clusterpval_df.loc[:,'comparison_2'] = clusterpval_df['comparisons'].str.split(',', expand=True)[1].astype(int)
    clusterpval_df.loc[:,'comparisons'] = (clusterpval_df['comparison_1'].astype(int) + 1).astype(str) + ',' + (clusterpval_df['comparison_2'].astype(int) + 1).astype(str)
    clusterpval_df = clusterpval_df.iloc[:, :-2]

    return html.Div([
        html.H6(['Adjusted p-value file preview: '], style={'font-weight':'bold'}),
        html.Br(),

        html.Div(['Cells in wald_pvalue column that are orange signify',
        'clusters that are significantly'],
        style={'font-style':'italic'}),

        html.Div('different when using traditional Wald test.',
        style={'font-style':'italic'}),
        html.Div(['Cells in cluster_pvalue column that are yellow signify',
        'clusters that are significantly'],
        style={'font-style':'italic'}),

        html.Div('different, even when using the adjusted p value method',
        style={'font-style':'italic'}),

        html.Br(),
        html.Div(''),
        html.Div(''),

        # return a preview of the file showing first 10 lines
        dash_table.DataTable(
            data=clusterpval_df.to_dict('records'),
            columns=[{'name': i, 'id': i} for i in clusterpval_df.columns],
            page_size=10,
            style_table={'overflowX': 'auto'},
            style_cell={
            'overflow': 'hidden',
            'textOverflow': 'ellipsis',
            'textAlign': 'left'
            },
            style_data_conditional=[
            {
            'if': {
                'filter_query': '{{cluster_pvalue}} < {} && {{cluster_pvalue}} >= 0'.format(float(sig_threshold)),
                'column_id': 'cluster_pvalue'
            },
            'backgroundColor': 'yellow',
            'color': 'black'
            },
            {
            'if': {
                'filter_query': "{cluster_pvalue} is nil",
                'column_id': 'cluster_pvalue'
            },
            'backgroundColor': 'white',
            'color': 'black'
            },
            {
            'if': {
                'filter_query':'{{wald_pvalue}} < {}'.format(float(sig_threshold)),
                'column_id': 'wald_pvalue'
            },
            'backgroundColor': 'orange',
            'color': 'black'
            },
            ]
        ),
    ])

"""
App
===
This module serves as the location where the user interface is built,
all of the functions are joined together and used, and the outputs of the
app are encoded

Functions
---------
:output_read_file_status: returns html object with read file status
:output_file_preview_req_num_clusters_req_cluster_method: returns preview 
of the uploaded file,
and requests number of clusters/clustering method
:output_req_linkage: requests user to input a linkage method
:output_cluster_settings_req_submit: displays input cluster settings and 
requests final submission
:output_cluster_status: displays status of data clustering
:output_cluster_df: stores the clustered df for future use
:output_cluster_figure: returns interactive clustered figure
:output_req_threshold_req_num_draws: requests input of significance threshold
and number of draws (if applicable)
:output_wald_status: displays status of pvalue submission
:output_wald_df: stores df containing wald p-values for future use
:output_wald_preview_clusterpval_status: returns preview of wald table,
and displays cluster status
:output_clusterpval_df: iterates adjusted p-value test over
each significant pairwise cluster comparison and stores df for future use
:output_cluster_pval_preview: returns a preview of the wald and ajusted p-values
side-by-side
"""

import base64
import io

import dash
from dash.dependencies import Input, Output, State
from dash import dcc
from dash import html

import pandas as pd

from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans

from cluster_pval import cluster_module
from cluster_pval import helper_module

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

available_clustering_methods = ['hierarchical', 'k-means']
available_linkage_methods = ['ward', 'complete', 'average', 'single']
available_pvalue_methods = ['wald']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server
app.config.suppress_callback_exceptions = True

app.layout = html.Div([
    # title
    html.Header(
        children="Cluster PVal",
        style={
            'textAlign': 'center',
            'background-color': '#4b2e83',
            'fontSize': '60px',
            'color': 'white',
            'padding': '30px',
            'font-weight':'bold'
        }),

    # horizontal line
    html.Hr(),

    # change subtitle to contain a better explanation of the tool
    html.H6(['Tool summary: '], style={'font-weight':'bold'}),
            html.Div(''),
            html.Div(''),

    html.Div(
        children="""
        Comparing traditional and adjusted p-values when comparing differences of means
        between two estimated clusters in a data set.
        """,
        style={
            'textAlign': 'left'
        }),

    html.Hr(),

    html.H6(['Data Import: '], style={'font-weight':'bold'}),
            html.Div(''),
            html.Div(''),

    # file upload
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '99%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
    ),
    # add a line break
    html.Br(),
    html.Br(),

    # request sample orientation
    html.Div("Are the samples organized in columns or rows? ", style={'font-weight':'bold'}),
    html.Br(),
    html.Div("Select option where each row represents one object to be clustered",
    style={'font-style': 'italic'}),
    dcc.Dropdown(
    id='data-orientation',
    options=[{'label': i, 'value': i} for i in ['columns', 'rows']],
    value='rows'
    ),

    html.Hr(),

    # show read file status DONE
    html.Div(id='output-read-file-status'),

    # once file is read, show file preview and present option for choosing number of clusters DONE
    html.Div(id='output-preview-file-req-num-clusters-req-cluster-method'),

    # once number of clusters is selected, return option to select clustering method
    html.Div(id='output-cluster-settings'),

    # once cluster method have been selected, return dropdown for linkages and/or cluster butto DONE
    html.Div(id='output-req-linkage'),

    # after all clustering/linkage methods have been selected,
    # show file clustering status and submit button
    html.Div(id='output-cluster-settings-req-submit'),

    # after submitting, show file clustering status
    html.Div(id='output-cluster-status'),

    # store clustered df for future use
    dcc.Store(id='output-cluster-df'),

    # output clustering figure
    html.Div(id='output-cluster-figure'),

    # after clustering is complete,
    # request sig threshold, num draws (if applicable), and submit button
    html.Div(id='output-req-threshold-req-num-draws'),

    # after submitting, show wald calculation status
    html.Div(id='output-wald-status'),

    # store wald df for future use
    dcc.Store(id='output-wald-df'),

    # show pvalue df with explanation of which pvalues will be recalculated
    html.Div(id='output-wald-preview-clusterpval-status'),

    # store adjusted pvalue df for future use
    dcc.Store(id='output-clusterpval-df'),

    # display pvalue table with changed results highlighted with option to download
    html.Div(id='output-clusterpval-preview')
])

# return reading file status once file is uploaded DONE
@app.callback(Output('output-read-file-status', 'children'),
              Input('upload-data', 'contents'),
              State('upload-data', 'filename'))
def output_read_file_status(contents, filename):
    """
    Function to return the status of the CSV upload

    Parameters:
    :param contents: the actual contents of the input file
    :param filename: string with pathway and name of file

    returns:
    display reading file status with filename
    """
    if contents is not None:
        children = helper_module.read_file_status(filename)

        return children


# return file preview and request num clusters and cluster method input DONE
@app.callback(Output('output-preview-file-req-num-clusters-req-cluster-method', 'children'),
              Input('output-read-file-status', 'children'),
              Input('data-orientation', 'value'),
              State('upload-data', 'contents'),
              State('upload-data', 'filename'))
def output_preview_file_req_num_clusters_req_cluster_method(status,
orientation, contents, filename):
    """
    Function to give a preview of the uploaded dataset on
    the dashboard and request input of the user

    Parameters:
    :param status: the status of reading the file
    :param orientation: the orientation of the data as input by the user
    :param contents: CSV file with content of input data
    :param filename: string the name of file

    returns:
    preview of data, requests for data info, requests number of clusters,
    and requests clustering method
    """
    if status is not None:
        children = helper_module.file_preview_req_num_clusters_req_cluster_method(orientation,
        contents, filename)

        return children


# request linkage methods DONE
@app.callback(Output('output-req-linkage', 'children'),
              Input('num-clusters', 'value'),
              Input('cluster-method', 'value'))
def output_req_linkage(num_clusters, cluster_method):
    """A"""
    if num_clusters is not None and cluster_method is not None:
        children = helper_module.req_linkage_method(num_clusters, cluster_method)

        return children


# once linkage has been selected and/or button has been pressed,
# return clustering status and button to submit DONE
@app.callback(Output('output-cluster-settings-req-submit', 'children'),
              Input('linkage-method', 'value'),
              State('min-col', 'value'),
              State('max-col', 'value'),
              State('num-clusters', 'value'),
              State('cluster-method', 'value'),
              State('upload-data', 'contents'),
              State('upload-data', 'filename'))
def output_cluster_settings_req_submit(linkage_method, min_col, max_col,
num_clusters, cluster_method, contents, filename):
    """
    Function to ask user for a linkage method

    Parameters:
    :param linkage_method: the linkage method input by the user
    :param min_col: the minimum column with data (input by user)
    :param max_col: the maximum column with data (input by user)
    :param num_clusters: number of clusters input by user
    :param cluster_method: cluster method selected by user
    :param contents: the actual contents of the file
    :param filename: the name of the file

    returns:
    a request for user to input linkage method,
    or proceeds to next step if
    k-means cluster method was selected
    """
    if contents is not None and num_clusters is not None:
        if cluster_method is not None and linkage_method is not None:
            children = helper_module.cluster_settings_req_submit(filename, min_col,
            max_col, num_clusters, cluster_method, linkage_method)

            return children


# once button has been pressed, return clustering status DONE
@app.callback(Output('output-cluster-status', 'children'),
              Input('cluster-button', 'n_clicks'),
              State('upload-data', 'filename'))
def output_cluster_status(n_clicks, filename):
    """
    Function to return the status of the clustering

    Parameters:
    :param n_clicks: the number of clicks on the submit button
    :param filename: string with pathway and name of file

    returns:
    display clustering status with filename
    """
    if n_clicks > 0:
        return html.Div([html.Div("Clustering file: " + str(filename),
        style={'font-weight':'bold', 'font-style':'italic'}),
        html.Hr()
        ])


# once button has been pressed, cluster and store data DONE
@app.callback(Output('output-cluster-df', 'data'),
              Input('output-cluster-status', 'children'),
              State('data-orientation', 'value'),
              State('min-col', 'value'),
              State('max-col', 'value'),
              State('linkage-method', 'value'),
              State('num-clusters', 'value'),
              State('cluster-method', 'value'),
              State('upload-data', 'contents'))
def output_cluster_df(status, orientation, min_col, max_col,
linkage_method, num_clusters, cluster_method, contents):
    """
    Function to cluster the data and store the results

    Parameters:
    :param status: status of the clustering to display
    :param orientation: the orientation of the data as input
    by the user
    :param min_col: the minimum column with data as input by the user
    :param max_col: the maximum column with data as input by the user
    :param linkage_method: the linkage method as input by the user
    :param num_clusters: the number of clusters to from as input
    by the user
    :param cluster_method: the method of clustering to use as input
    by the user
    :param contents: the actual contents of the file input by the user

    returns:
    stores the clustered file for future use
    """
    if status is not None:
        _, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        data_df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        if max_col is None:
            df_col = data_df.iloc[:,min_col:max_col+1]
        else:
            df_col = data_df.iloc[:,min_col:]
        if orientation == 'columns':
            data_df = data_df.transpose()
        else:
            pass

        clustered_df, _, _, _, _ = cluster_module.clustering(df_col,
        num_clusters, cluster_method, linkage_method=linkage_method)

        return clustered_df.to_json(orient='split')


# once clustering has taken place and df is stored, return clustering figure DONE
@app.callback(Output('output-cluster-figure', 'children'),
              Input('output-cluster-df', 'data'))
def output_cluster_figure(clustered_json):
    """
    Function to show plot with clustered data

    Parameters:
    :param clustered_json: dataframe with input data including clusters

    returns:
    figure with the clustered data
    """
    if clustered_json is not None:
        clustered_df = pd.read_json(clustered_json, orient='split')
        clustered_df['cluster'] = clustered_df['cluster'] + 1
        children = helper_module.cluster_figure(clustered_df)
        return children



# request significance threshold and num draws if appropriate
@app.callback(Output('output-req-threshold-req-num-draws', 'children'),
              Input('output-cluster-figure', 'children'),
              State('cluster-method', 'value'),
              State('linkage-method', 'value'))
def output_req_threshold_req_num_draws(clustered_figure, cluster_method, linkage_method):
    """
    Function to request user input of p-value threshold
    and the number of draws to use

    Parameters:
    :param clustered_figure: the display of the clustered figure
    to ensure that clustering is complete
    :param cluster_method: the method of clustering as input by the user
    :param linkage_method: the linkage method as input by the user

    returns:
    request user input of p-value threshold and number of draws
    """
    if clustered_figure is not None:
        children = helper_module.req_threshold_req_num_draws(clustered_figure,
        cluster_method, linkage_method)
        return children


# once image is generated, return wald pvalue status DONE
@app.callback(Output('output-wald-status', 'children'),
              Input('p-value-button', 'n_clicks'),
              Input('sig-threshold', 'value'),
              State('num-draws', 'value'),
              State('output-cluster-figure', 'children'))
def output_wald_status(n_clicks, sig_threshold):
    """
    Function to display status of wald p-value calculation
    to the user

    Parameters:
    :param n_clicks: the number of clicks on the button
    :param sig_threshold: the significance threshold as input
    by the user

    returns:
    a status update of the wald p-value calcualtion
    """
    if n_clicks > 0:
        return html.Div([
            html.Div("Calculating wald p-value with threshold: " + str(sig_threshold),
            style={'font-weight':'bold', 'font-style':'italic'}),

            html.Hr()
        ])


# once wald status has been returned, store wald pvalue df DONE
@app.callback(Output('output-wald-df', 'data'),
              Input('output-wald-status', 'children'),
              State('output-cluster-df', 'data'))
def output_wald_df(status, clustered_json):
    """
    Function to store the df containing the calculated wald
    p-values for future use

    Parameters:
    :param status: status of wald p-value calculation to
    ensure that it has been displayed
    :param clustered_json: a df of the clustered data

    returns:
    stores the wald pvalue df for future use
    """
    if status is not None:
        clustered_df = pd.read_json(clustered_json, orient='split')
        wald_df = helper_module.iterate_wald_test(clustered_df, clustered_df['cluster'],
        iso=True, sig=None, siginv=None)

        return wald_df.to_json(orient='split')


# once wald df has been stored, return a preview of pvalue table and return adj pvalue status DONE
@app.callback(Output('output-wald-preview-clusterpval-status', 'children'),
              Input('output-wald-df', 'data'))
def output_wald_preview_and_cluserpval_status(wald_json):
    """
    Function to preview the calculated wald p-values

    Parameters:
    :wald_json: pandas df, with results from wald p-values

    Returns:
    pandas dataframe: a display of p-value dataframe
    containing comparisons and their associated wald p-value
    """
    if wald_json is not None:
        children = helper_module.wald_preview_clusterpval_status(wald_json)
        return children


# # once clusterpval status has been returned, store clusterpval df DONE
@app.callback(Output('output-clusterpval-df', 'data'),
              Input('output-wald-preview-clusterpval-status', 'children'),
              State('output-cluster-df', 'data'),
              State('output-wald-df', 'data'),
              State('linkage-method', 'value'),
              State('num-clusters', 'value'),
              State('cluster-method', 'value'),
              State('sig-threshold', 'value'),
              State('num-draws', 'value'))
def output_clusterpval_df(status, clustered_json, wald_json,
linkage_method, num_clusters, cluster_method, sig_threshold, num_draws):
    """
    Function to store the df containing the adjusted
    p-values for future use

    Parameters:
    :param status: status of wald p-value calculation to
    ensure that it has been displayed
    :param clustered_json: a df of the clustered data
    :param wald_json: a df of the calculated wald p-values
    :param linkage_method: the linkage method as input by the user
    :param num_clusters: the number of clusters as input by the user
    :param cluster_method: the clustering method as input by the user
    :param sig_threshold: the significance threshold as input by the user
    :param num_draws: the number of draws as input by the user

    returns:
    stores the adusted pvalue df for future use
    """
    if status is not None:
        clustered_df = pd.read_json(clustered_json, orient='split')
        wald_df = pd.read_json(wald_json, orient='split')

        if cluster_method == 'hierarchical':
            clfun=AgglomerativeClustering
            clusterpval_df = helper_module.iterate_stattest_clusters_approx(wald_df,
            sig_threshold, clustered_df, clustered_df['cluster'], cl_fun=clfun,
            positional_arguments=[], keyword_arguments={'n_clusters': num_clusters,
            'affinity': 'euclidean', 'linkage': linkage_method},
            n_draws=num_draws)
        else:
            clfun=KMeans
            clusterpval_df = helper_module.iterate_stattest_clusters_approx(wald_df,
            sig_threshold, clustered_df, clustered_df['cluster'], cl_fun=clfun,
            positional_arguments=[], keyword_arguments={'n_clusters': num_clusters},
            n_draws=num_draws)

        return clusterpval_df.to_json(orient='split')


# once clusterpval df has been stored, return a preview of pvalue table DONE
@app.callback(Output('output-clusterpval-preview', 'children'),
              Input('output-clusterpval-df', 'data'),
              State('sig-threshold', 'value'))
def output_cluserpval_preview(clusterpval_json, sig_threshold):
    """
    Function to recalculate the significant pvalues obtained from
    the wald test to calculate the adjusted pvalue

    Parameters:
    :param clusterpval_json: pandas dataframe with adjusted pvalues
    :param sig_threshold: float, threshold that determine significance

    Returns:
    pandas dataframe: combined pvalue dataframe
    """
    if clusterpval_json is not None:
        children = helper_module.clusterpval_preview(clusterpval_json, sig_threshold)
        return children


if __name__ == '__main__':
    app.run_server(debug=True)

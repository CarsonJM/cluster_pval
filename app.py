"""
App
===

Contents
--------

"""

import base64
import io

import dash
from dash.dependencies import Input, Output, State
from dash import dcc
from dash import html
from dash import dash_table

import pandas as pd
import plotly.express as px

from cluster_pval import pval_module
from cluster_pval import cluster_module
from cluster_pval import display_module
from cluster_pval import helper_module

from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans

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
            'padding': '30px'
        }),

    # horizontal line
    html.Hr(),

    # change subtitle to contain a better explanation of the tool
    html.Div(
        children="""
        Comparing traditional and adjusted p-values when comparing differences of means
        between two estimated clusters in a data set.
        """,
        style={
            'textAlign': 'center'
        }),
    
    html.Hr(), 

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
    "Are the samples organized in columns or rows?",
    "(Select option where each row represents an object to be clusterd)",
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

    # show file clustering status and submit button
    html.Div(id='output-cluster-settings-req-submit'),

    # show file clustering status and submit button
    html.Div(id='output-cluster-status'),

    # store clustered df
    dcc.Store(id='output-cluster-df'),

    # output clustering figure with option to download
    html.Div(id='output-cluster-figure'),

    # request sig threshold and num draws (if applicable)
    html.Div(id='output-req-threshold-req-num-draws'),

    # show wald calculation status
    html.Div(id='output-wald-status'),

    # store wald df
    dcc.Store(id='output-wald-df'),

    # show pvalue df with explanation of which pvalues will be recalculated
    html.Div(id='output-wald-preview-clusterpval-status'),

    # store wald pvalue df
    dcc.Store(id='output-clusterpval-df'),

    # display pvalue table with changed results highlighted with option to download
    html.Div(id='output-clusterpval-preview')
])

# return reading file status once file is uploaded DONE
@app.callback(Output('output-read-file-status', 'children'),
              Input('upload-data', 'contents'),
              State('upload-data', 'filename')     
)
def output_read_file_status(contents, filename):
     if contents is not None:
        children = helper_module.read_file_status(filename)
    
        return children


# return file preview and request num clusters and cluster method input DONE
@app.callback(Output('output-preview-file-req-num-clusters-req-cluster-method', 'children'),
              Input('output-read-file-status', 'children'),
              Input('data-orientation', 'value'),
              State('upload-data', 'contents'),
              State('upload-data', 'filename')
)
def output_preview_file_req_num_clusters_req_cluster_method(status, orientation, contents, filename):
     if status is not None:
        children = helper_module.file_preview_req_num_clusters_req_cluster_method(orientation, contents, filename)
    
        return children


# request linkage methods DONE
@app.callback(Output('output-req-linkage', 'children'),
              Input('num-clusters', 'value'),
              Input('cluster-method', 'value')    
)
def output_req_linkage_req_cluster(num_clusters, cluster_method):
     if num_clusters is not None and cluster_method is not None:
        children = helper_module.req_linkage_method(num_clusters, cluster_method)
    
        return children


# once linkage has been selected and/or button has been pressed, return clustering status and button to submit DONE
@app.callback(Output('output-cluster-settings-req-submit', 'children'),
              Input('linkage-method', 'value'),
              State('min-col', 'value'),
              State('max-col', 'value'),
              State('num-clusters', 'value'),
              State('cluster-method', 'value'),
              State('upload-data', 'contents'),
              State('upload-data', 'filename')
)
def output_cluster_settings_req_submit(linkage_method, min_col, max_col, num_clusters, cluster_method, contents, filename):
     if contents is not None and num_clusters is not None and cluster_method is not None and linkage_method is not None:
        children = helper_module.cluster_settings_req_submit(filename, min_col, max_col, num_clusters, cluster_method, linkage_method)
    
        return children


# once button has been pressed, return clustering status DONE
@app.callback(Output('output-cluster-status', 'children'),
              Input('cluster-button', 'n_clicks'),
              State('upload-data', 'filename')
)
def output_cluster_status(n_clicks, filename):
     if n_clicks > 0:
        return html.Div([
            "Clustering file: ", filename,

            html.Hr()
        ])


# once button has been pressed, cluster and store data DONE
@app.callback(Output('output-cluster-df', 'data'),
              Input('output-cluster-status', 'children'),
              State('data-orientation', 'value'),
              State('min-col', 'value'),
              State('max-col', 'value'),
              State('cluster-button', 'n_clicks'),
              State('linkage-method', 'value'),
              State('num-clusters', 'value'),
              State('cluster-method', 'value'),
              State('upload-data', 'contents'),
              State('upload-data', 'filename')
)
def output_cluster_df(status, orientation, min_col, max_col, n_clicks, linkage_method, num_clusters, cluster_method, contents, filename):
     if status is not None:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        if max_col != None:
            df_col = df.iloc[:,min_col:max_col+1]
        else:
            df_col = df.iloc[:,min_col:]
        if orientation == 'columns':
                df = df.transpose()
        else:
            pass

        clustered_df, nr_of_clusters, ccl_fun, positional_arguments, keyword_arguments = cluster_module.hierarchical_clustering(df_col, num_clusters, cluster_method, linkage_method=linkage_method)

        return clustered_df.to_json(orient='split')


# once clustering has taken place and df is stored, return clustering figure DONE
@app.callback(Output('output-cluster-figure', 'children'),
              Input('output-cluster-df', 'data')
)
def output_cluster_figure(clustered_json):
    if clustered_json is not None:
        clustered_df = pd.read_json(clustered_json, orient='split')
        children = helper_module.cluster_figure(clustered_df)
    
        return children


# request significance threshold and num draws if appropriate
@app.callback(Output('output-req-threshold-req-num-draws', 'children'),
              Input('output-cluster-figure', 'children'),
              State('cluster-method', 'value'),
              State('linkage-method', 'value')
)
def output_req_threshold_req_num_draws(clustered_figure, cluster_method, linkage_method):
    if clustered_figure is not None:
        children = helper_module.req_threshold_req_num_draws(clustered_figure, cluster_method, linkage_method)
        return children


# once image is generated, return wald pvalue status DONE
@app.callback(Output('output-wald-status', 'children'),
              Input('p-value-button', 'n_clicks'),
              Input('sig-threshold', 'value'),
              State('num-draws', 'value'),
              State('output-cluster-figure', 'children')
)
def output_wald_status(n_clicks, sig_threshold, num_draws, clustered_figure):
    if n_clicks > 0:
        return html.Div([
            "Calculating wald p-value with threshold: ", sig_threshold,

            html.Hr()
        ])


# once wald status has been returned, store wald pvalue df DONE
@app.callback(Output('output-wald-df', 'data'),
              Input('output-wald-status', 'children'),
              State('sig-threshold', 'value'),
              State('num-draws', 'value'),
              State('output-cluster-figure', 'children'),
              State('output-cluster-df', 'data')      
)
def output_wald_df(status, sig_threshold, num_draws, figure, clustered_json):
     if status is not None:
        clustered_df = pd.read_json(clustered_json, orient='split')
        wald_df = helper_module.iterate_wald_test(clustered_df, clustered_df['cluster'], iso=True, sig=None, siginv=None)

        return wald_df.to_json(orient='split')


# once wald df has been stored, return a preview of pvalue table and return adj pvalue status DONE
@app.callback(Output('output-wald-preview-clusterpval-status', 'children'),
              Input('output-wald-df', 'data'),
              State('sig-threshold', 'value'),
              State('num-draws', 'value')
)
def output_wald_preview_and_cluserpval_status(wald_json, sig_threshold, num_draws):
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
              State('num-draws', 'value')
)
def output_clusterpval_df(status, clustered_json, wald_json, linkage_method, num_clusters, cluster_method, sig_threshold, num_draws):
     if status is not None:
        clustered_df = pd.read_json(clustered_json, orient='split')
        wald_df = pd.read_json(wald_json, orient='split')

        if cluster_method == 'hierarchical':
            clfun=AgglomerativeClustering
            if linkage_method == 'ward' or 'average':
                clusterpval_df = helper_module.new_pvalue_test(wald_df, sig_threshold, clustered_df, clustered_df['cluster'], cl_fun=clfun, positional_arguments=[], keyword_arguments={'n_clusters': num_clusters, 'affinity': 'euclidean', 'linkage': linkage_method}, iso=True, sig=None, siginv=None, n_draws=num_draws)
            else:
                clusterpval_df = helper_module.iterate_stattest_clusters_approx(wald_df, sig_threshold, clustered_df, clustered_df['cluster'], cl_fun=clfun, positional_arguments=[], keyword_arguments={'n_clusters': num_clusters, 'affinity': 'euclidean', 'linkage': linkage_method}, iso=True, sig=None, siginv=None, n_draws=num_draws)
        if cluster_method == 'k-means':
            clfun=KMeans
            clusterpval_df = helper_module.iterate_stattest_clusters_approx(wald_df, sig_threshold, clustered_df, clustered_df['cluster'], cl_fun=clfun, positional_arguments=[], keyword_arguments={'n_clusters': num_clusters}, iso=True, sig=None, siginv=None, n_draws=num_draws)

        return clusterpval_df.to_json(orient='split')


# once clusterpval df has been stored, return a preview of pvalue table DONE
@app.callback(Output('output-clusterpval-preview', 'children'),
              Input('output-clusterpval-df', 'data'),
              State('sig-threshold', 'value'),
              State('num-draws', 'value')
)
def output_cluserpval_preview(clusterpval_json, sig_threshold, num_draws):
     if clusterpval_json is not None:
        children = helper_module.clusterpval_preview(clusterpval_json, sig_threshold, num_draws)
        return children


if __name__ == '__main__':
    app.run_server(debug=True)
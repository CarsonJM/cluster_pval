# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

import base64
import io
import json

import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import dash_table

import pandas as pd
import plotly.express as px

from cluster_pval import pval_module
from cluster_pval import cluster_module
from cluster_pval import display
from cluster_pval import helper

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

available_clustering_methods = ['Hierarchical']
available_linkage_methods = ['ward', 'complete', 'average', 'single']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

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

    html.Hr(),

    # show read file status DONE
    html.Div(id='output-read-file-status'),

    # once file is read, show file preview and present option for choosing number of clusters DONE
    html.Div(id='output-preview-file'),

    # once number of clusters is selected, return option to select clustering method
    html.Div(id='output-cluster-method'),

    # once cluster method have been selected, return dropdown for linkages and/or cluster butto DONE
    html.Div(id='output-linkage-cluster'),

    # show file clustering status and submit button
    html.Div(id='output-cluster-settings'),

    # show file clustering status and submit button
    html.Div(id='output-cluster-status'),

    # store clustered df
    dcc.Store(id='output-cluster-df'),

    # output clustering figure with option to download
    html.Div(id='output-cluster-figure'),

    # show pvalue calculation status
    #html.Div(id='output-pvalue-status'),

    # show pvalue df with explanation of which pvalues will be recalculated
    #html.Div(id='output-wald-pvalue-preview'),

    # store wald pvalue df
    #dcc.Store(id='intermediate-wald-pvalue-df'),

    # display pvalue table with changed results highlighted with option to download
    #html.Div(id='output-pvalue-table')
])

# output-read-file-status (determine if file is csv, and return status)
def read_file_status(filename):
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

# cluster and return cluster df
def cluster_df(contents, num_clusters, cluster_method, linkage_method):
    
    return html.Div([
        dcc.Store(id='cluster-df'),
        dcc.Store(id='nr-of-clusters'),
        dcc.Store(id='ccl-fun'),
        dcc.Store(id='positional_arguments'),
        dcc.Store(id='keyword_arguments')
    ])
    clustered_df, nr_of_clusters, ccl_fun, positional_arguments, keyword_arguments

# once the options have been selected and button is pressed, cluster the data, and display cluster graph
def cluster_figure(clustered_df):
    # generate clustering figure
    fig = display.cluster_plot(clustered_df)

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

# return a preview of the file showing first 10 lines
        # dash_table.DataTable(
        #     data=pvalue_df.to_dict('records'),
        #     columns=[{'name': i, 'id': i} for i in df.columns],
        #     page_size=50,
        #     style_table={'overflowX': 'auto'},
        #     style_cell={
        #     'overflow': 'hidden',
        #     'textOverflow': 'ellipsis',
        #     'textAlign': 'left'
        #     }
        # ),

# # generate scatter plot figure
# fig = display.cluster_plot(clustered_df)
# # calculate wald pvalue for all pairwise comparisons
# wald_pvalue_df = helper.iterate_wald_test(clustered_df, clustered_df['cluster'], iso=True, sig=None, siginv=None)
# # calculate approximate pvalue for all significant wald comparisons
# pvalue_df = helper.iterate_stattest_clusters_approx(wald_pvalue_df, clustered_df, clustered_df['cluster'], ccl_fun, positional_arguments, keyword_arguments, iso=True, sig=None, siginv=None, ndraws=2000)

# return reading file status once file is uploaded DONE
@app.callback(Output('output-read-file-status', 'children'),
              Input('upload-data', 'contents'),
              State('upload-data', 'filename')     
)
def output_read_file_status(contents, filename):
     if contents is not None:
        children = read_file_status(filename)
    
        return children

# return file preview and num clusters and cluster method input DONE
@app.callback(Output('output-preview-file', 'children'),
              Input('output-read-file-status', 'children'),
              State('upload-data', 'contents'),
              State('upload-data', 'filename')
)
def output_preview_file(status, contents, filename):
     if status is not None:
        children = preview_file_and_num_clusters_and_cluster_methods(contents, filename)
    
        return children
            
# return linkage methods and/or cluster button once num_cluster and cluster method have been selected DONE
@app.callback(Output('output-linkage-cluster', 'children'),
              Input('num-clusters', 'value'),
              Input('cluster-method', 'value')    
)
def output_linkage_and_cluster(num_clusters, cluster_method):
     if num_clusters is not None and cluster_method is not None:
        children = linkage(cluster_method)
    
        return children

# once linkage has been selected and/or button has been pressed, return clustering status and button to submit DONE
@app.callback(Output('output-cluster-settings', 'children'),
              Input('linkage-method', 'value'),
              State('min-col', 'value'),
              State('max-col', 'value'),
              State('num-clusters', 'value'),
              State('cluster-method', 'value'),
              State('upload-data', 'contents'),
              State('upload-data', 'filename')
)
def output_cluster_settings_and_submit(linkage_method, min_col, max_col, num_clusters, cluster_method, contents, filename):
     if contents is not None and num_clusters is not None and cluster_method is not None and linkage_method is not None:
        children = cluster_settings_and_submit(filename, min_col, max_col, num_clusters, cluster_method, linkage_method)
    
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
              State('min-col', 'value'),
              State('max-col', 'value'),
              State('cluster-button', 'n_clicks'),
              State('linkage-method', 'value'),
              State('num-clusters', 'value'),
              State('cluster-method', 'value'),
              State('upload-data', 'contents'),
              State('upload-data', 'filename')
)
def output_cluster_df(status, min_col, max_col, n_clicks, linkage_method, num_clusters, cluster_method, contents, filename):
     if status is not None:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        df_col = df.iloc[:,min_col:max_col+1]

        clustered_df, nr_of_clusters, ccl_fun, positional_arguments, keyword_arguments = cluster_module.hierarchical_clustering(df_col, num_clusters, linkage_method)

        return clustered_df.to_json(orient='split')

# once clustering has taken place and df is stored, return clustering figure DONE
@app.callback(Output('output-cluster-figure', 'children'),
              Input('output-cluster-df', 'data')
)
def output_cluster_figure(clustered_json):
    if clustered_json is not None:
        clustered_df = pd.read_json(clustered_json, orient='split')
        children = cluster_figure(clustered_df)
    
        return children


# once image is generated, calculate wald pvalue
@app.callback(Output('output-wald-df', 'children'),
              Input('output-cluster-df', 'data')
)
def output_cluster_figure(clustered_json):
    if clustered_json is not None:
        clustered_df = pd.read_json(clustered_json, orient='split')
        children = cluster_figure(clustered_df)
    
        return children

if __name__ == '__main__':
    app.run_server(debug=True)
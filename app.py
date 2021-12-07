# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

import base64
import datetime
import io

import dash
from dash.dependencies import Input, Output, State
from dash import dcc
from dash import html
from dash import dash_table

import pandas as pd
import plotly.express as px
import umap

from cluster_pval import pval_module
from cluster_pval import cluster_module
from cluster_pval import display

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

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

    # subtitle
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

    # data upload object
    html.Div(id='output-data-upload')
])

def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)

    # try reading the file
    try:
        if 'csv' in filename:
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
            standard_embedding = umap.UMAP(random_state=42).fit_transform(df)
            df_clustered, cluster_method, df_clustered, nr_of_clusters = cluster_module.hierarchical_clustering(df, 3)
            df_clustered = df_clustered.sort_values(['cluster'], ascending=True)
            df_clustered['cluster'] = df_clustered['cluster'].astype(str)
            fig = px.scatter(x=standard_embedding[:, 0], y=standard_embedding[:, 1], color=df_clustered['cluster'], 
            labels={'x': "UMAP_1", 'y': "UMAP_2", 'color': "Cluster"}, 
            title="Scatter plot of clustered cells", 
            template="simple_white")
        elif 'csv' not in filename:
            raise TypeError
            

    # return exception if file is not read
    except Exception as e:
        print(e)
        return html.Div(children=[
            'The input file is not in csv format'],
            style={
            'textAlign': 'center',
            'fontSize': '30px'
            }
        )

    # objects to return after reading the file
    return html.Div([
        html.Hr(),

        # return file name
        html.Div(children=[
            'Filename: ', filename
            ]),

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

        # return visualization of clustering
        dcc.Graph(
        id='example-graph',
        figure=fig
        )
    ])

@app.callback(Output('output-data-upload', 'children'),
              Input('upload-data', 'contents'),
              State('upload-data', 'filename')
              )
def update_output(contents, filename):
    if contents is not None:
        children = [
            parse_contents(contents, filename)]
        return children

if __name__ == '__main__':
    app.run_server(debug=True)
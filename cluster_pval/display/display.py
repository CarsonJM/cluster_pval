"""
Display
=======

Functions
---------
"""
import umap
from cluster_pval import cluster_module
import plotly.express as px

def cluster_plot(df, clustered_df):    
    standard_embedding = umap.UMAP(random_state=42).fit_transform(df)

    clustered_df = clustered_df.sort_values(['cluster'], ascending=True)
    clustered_df['cluster'] = clustered_df['cluster'].astype(str)
    fig = px.scatter(x=standard_embedding[:, 0], y=standard_embedding[:, 1], color=clustered_df['cluster'], 
    labels={'x': "UMAP_1", 'y': "UMAP_2", 'color': "Cluster"}, 
    title="Scatter plot of clustered cells", 
    template="simple_white")

    return clustered_df
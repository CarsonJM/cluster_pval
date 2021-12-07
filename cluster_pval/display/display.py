"""
Display
=======

Functions
---------
"""

def cluster_plot(df, x, y, color, hover):    
    standard_embedding = umap.UMAP(random_state=42).fit_transform(df)

    df_clustered, nr_of_clusters, ccl_fun, positional_arguments, keyword_arguments = cluster_module.hierarchical_clustering(df, 3)

    df_clustered = df_clustered.sort_values(['cluster'], ascending=True)
    df_clustered['cluster'] = df_clustered['cluster'].astype(str)
    fig = px.scatter(x=standard_embedding[:, 0], y=standard_embedding[:, 1], color=df_clustered['cluster'], 
    labels={'x': "UMAP_1", 'y': "UMAP_2", 'color': "Cluster"}, 
    title="Scatter plot of clustered cells", 
    template="simple_white")

    return df_clustered
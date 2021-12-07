"""
Display
=======

Functions
---------
"""
import umap
from cluster_pval import cluster_module
import plotly.express as px

def cluster_plot(clustered_df):
    # pca2 = PCA(n_components=2)
    # principalComponents = pca2.fit_transform(X_Scale)

    # principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])

    # finalDf = pd.concat([principalDf, df[['class']]], axis = 1)
    # finalDf.head()

    standard_embedding = umap.UMAP(random_state=42).fit_transform(clustered_df)

    clustered_df = clustered_df.sort_values(['cluster'], ascending=True)
    clustered_df['cluster'] = clustered_df['cluster'].astype(str)
    fig = px.scatter(x=standard_embedding[:, 0], y=standard_embedding[:, 1], color=clustered_df['cluster'], 
    labels={'x': "UMAP_1", 'y': "UMAP_2", 'color': "Cluster"}, 
    title="Scatter plot of clustered cells", 
    template="simple_white")

    return fig
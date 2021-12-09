"""
Module containing function for plotting clustered data onto a two-dimensional scatterplot using
principal component analysis
"""
import pandas as pd
from sklearn.decomposition import PCA
import plotly.express as px


def cluster_plot(clustered_df):
    """
    Function to plot first two principal components of clustered data

    Parameters:
    :param clustered_df: pandas dataframe of clustered RNA seq data
    :return: fig: two-dimensional scatterplot of dimensionally reduced data color-coded by cluster
    """
    # check for correct datatype
    check_value_type = isinstance(clustered_df, pd.DataFrame)
    if check_value_type is False:
        raise ValueError("The dataset should be a pandas dataframe.")

    # perform PCA and reduce data to two dimensions
    pca_2 = PCA(n_components=2)
    principal_components = pca_2.fit_transform(clustered_df)
    principal_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])

    # plot data in scatterplot
    clustered_df = clustered_df.sort_values(['cluster'], ascending=True)
    clustered_df['cluster'] = clustered_df['cluster'].astype(str)
    fig = px.scatter(x=principal_df['PC1'], y=principal_df['PC2'], color=clustered_df['cluster'],
                     labels={'x': "First Principal Component", 'y': "Second Principal Component", 'color': "Cluster"},
                     title="Scatter plot of clustered cells",
                     template="simple_white")

    return fig




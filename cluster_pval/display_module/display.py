"""
This is a module containing a function for plotting clustered data onto a two-dimensional scatter plot using
principal component analysis and a function for calculating percent explained variance of each of the first two
principal components.
"""
import pandas as pd
from sklearn.decomposition import PCA
import plotly.express as px


def cluster_plot(clustered_df):
    """
    This is a function to plot first two principal components of clustered data.

    Parameters:
    :param clustered_df: pandas dataframe of clustered data
    :return fig: two-dimensional scatter plot of dimensionally reduced data color-coded by cluster
    """
    # check for correct datatype
    check_value_type = isinstance(clustered_df, pd.DataFrame)
    if check_value_type is False:
        raise ValueError("The dataset should be a pandas dataframe.")

    # perform PCA and reduce data to two dimensions
    pca_2 = PCA(n_components=2)
    principal_components = pca_2.fit_transform(clustered_df)
    principal_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
    var_per_array = pca_2.explained_variance_ratio_
    var_1 = 'PC1 (' + str(round(var_per_array[0] * 100, 2)) + '%)'
    var_2 = 'PC2 (' + str(round(var_per_array[1] * 100, 2)) + '%)'

    # plot data in scatter plot
    clustered_df = clustered_df.sort_values(['cluster'], ascending=True)
    clustered_df['cluster'] = clustered_df['cluster'].astype(str)
    fig = px.scatter(x=principal_df['PC1'], y=principal_df['PC2'], color=clustered_df['cluster'],
                     labels={'x': var_1, 'y': var_2, 'color': "Cluster"},
                     title="Scatter plot of clustered cells",
                     template="simple_white")

    return fig


def pca_var_per(clustered_df):
    """This is a function to calculate percent explained variance of the first two principal components.

    Parameters:
    :param clustered_df: pandas dataframe of clustered data
    :return var_per: pandas dataframe of percent variance explained by each of the first two principal components
    """
    # check for correct datatype
    check_value_type = isinstance(clustered_df, pd.DataFrame)
    if check_value_type is False:
        raise ValueError("The dataset should be a pandas dataframe.")

    # calculate first two principal components
    pca_2 = PCA(n_components=2)
    pca_2.fit_transform(clustered_df)

    # calculate percent explained variance and put into a dataframe
    var_per_array = pca_2.explained_variance_ratio_
    var_per = pd.DataFrame(data=var_per_array, columns=['% variance explained by PC1', '% variance explained by PC2'])
    return var_per

"""
Display
=======

Functions
---------
"""

def cluster_plot(df, x, y, color, hover):
    import plotly.express as px
    
    px.scatter(df, x=x, y=y, color=color, hover_name=hover,
    labels={
        x: "UMAP_1",
        y: "UMAP_2",
        color: "Cluster"
    },
    title="Scatter plot of clustered cells",
    template="simple_white")
    
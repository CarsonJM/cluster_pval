"""
Tests for functions in display module
"""
import unittest
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import plotly.express as px
from display_module.display import cluster_plot

class PlotFunctionTest(unittest.TestCase):
    """Tests for cluster_plot function"""
    def test_smoke_clusterplot(self):
        """Smoke test to make sure function runs"""
        test_array = np.array([[3, 1, 230, 0], [6, 2, 745, 0], [6, 6, 1080, 1], [4, 3, 495, 0], [2, 5, 260, 0]])
        test_df = pd.DataFrame(data=test_array, columns=[0, 1, 2, 'cluster'])
        cluster_plot(test_df)
        return

    def test_exception_clusterplot(self):
        """Test to see if exception is raised for wrong input type"""
        test_array = np.array([[3, 1, 230, 0], [6, 2, 745, 0], [6, 6, 1080, 1], [4, 3, 495, 0], [2, 5, 260, 0]])
        with self.assertRaises(ValueError):
            cluster_plot(test_array)
        return

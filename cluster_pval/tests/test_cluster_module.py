
"""
tests for cluster_module function
"""
import unittest
import numpy as np
import pandas as pd
from cluster_module.cluster_function import hierarchical_clustering


class ClusterModuleTest(unittest.TestCase):
    """
    tests for knn function
    """
    @classmethod
    def test_smoke(cls):
        """
        tests if function gives a result
        """
        data = np.array([[3, 1, 230],[6, 2, 745],[6, 6, 1080],[4, 3, 495],[2, 5, 260]])
        dataset=pd.DataRrame(data)
        nr_of_clusters = 3
        result = hierarchical_clustering(dataset, nr_of_clusters)
        return result


    def test_result_is_integer(self):
        """
        tests if result is pandas dataframe
        """
        data = np.array([[3, 1, 230],[6, 2, 745],[6, 6, 1080],[4, 3, 495],[2, 5, 260]])
        dataset=pd.DataRrame(data)
        nr_of_clusters = 3
        result = hierarchical_clustering(dataset, nr_of_clusters)
        self.assertIsInstance((result), pd.DataFrame)


    def test_error_n_neighbors(self):
        """
        tests if value error is raised when wrong type of input data is entered
        """
        dataset = np.array([[3, 1, 230],[6, 2, 745],[6, 6, 1080],[4, 3, 495],[2, 5, 260]])
        nr_of_clusters = 3
        with self.assertRaises(ValueError):
            result = hierarchical_clustering(dataset, nr_of_clusters)
        return result


    def test_error_label(self):
        """
        tests if value error is raised when wrong type of nr_of_clusters is entered
        """
        data = np.array([[3, 1, 230],[6, 2, 745],[6, 6, 1080],[4, 3, 495],[2, 5, 260]])
        dataset=pd.DataRrame(data)
        nr_of_clusters = 1.2
        with self.assertRaises(ValueError):
            result = hierarchical_clustering(dataset, nr_of_clusters)
        return result

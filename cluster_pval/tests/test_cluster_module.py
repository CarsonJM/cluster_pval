
"""
tests for cluster_module function
"""
import unittest
import numpy as np
import pandas as pd
from cluster_module.cluster_function import clustering


class ClusterModuleTest(unittest.TestCase):
    """
    tests for knn function
    """
    @classmethod
    def test_smoke_hierarchical(cls):
        """
        tests if function gives a result for hierarchical clustering
        """
        data = np.array([[3, 1, 230],[6, 2, 745],[6, 6, 1080],[4, 3, 495],[2, 5, 260]])
        dataset=pd.DataFrame(data)
        nr_of_clusters = 3
        clustering(dataset, nr_of_clusters, "hierarchical")
        return
    
    @classmethod
    def test_smoke_Kmeans(cls):
        """
        tests if function gives a result for Kmeans clustering
        """
        data = np.array([[3, 1, 230],[6, 2, 745],[6, 6, 1080],[4, 3, 495],[2, 5, 260]])
        dataset=pd.DataFrame(data)
        nr_of_clusters = 3
        clustering(dataset, nr_of_clusters, "KMeans")
        return


    def test_result_is_dataframe(self):
        """
        tests if result is pandas dataframe
        """
        data = np.array([[3, 1, 230],[6, 2, 745],[6, 6, 1080],[4, 3, 495],[2, 5, 260]])
        dataset=pd.DataFrame(data)
        nr_of_clusters = 3
        result = clustering(dataset, nr_of_clusters, "hierarchical")
        self.assertIsInstance((result[0]), pd.DataFrame)


    def test_error_n_neighbors(self):
        """
        tests if value error is raised when wrong type of input data is entered
        """
        dataset = np.array([[3, 1, 230],[6, 2, 745],[6, 6, 1080],[4, 3, 495],[2, 5, 260]])
        nr_of_clusters = 3
        with self.assertRaises(ValueError):
            clustering(dataset, nr_of_clusters, "hierarchical")
        return


    def test_error_label(self):
        """
        tests if value error is raised when wrong type of nr_of_clusters is entered
        """
        data = np.array([[3, 1, 230],[6, 2, 745],[6, 6, 1080],[4, 3, 495],[2, 5, 260]])
        dataset=pd.DataFrame(data)
        nr_of_clusters = 1.2
        with self.assertRaises(ValueError):
            clustering(dataset, nr_of_clusters, "hierarchical")
        return

"""
This code is the format data module.
"C:/USERS/annam/CSE/clusterclub/Cluster_PVal/Cluster_PVal/tests/data_for_tests/GSE158761_matrix.mtx.gz"
"""

import scipy.io
import pandas as pd

def format_data(genes_in_col, path_to_file): 
    EXPR_MTX_PATH = path_to_file
    print("mm starting")
    expr_spmat = scipy.io.mmread(EXPR_MTX_PATH) #specific format in with RNA data came
    print("mm read done")
    y = pd.DataFrame.sparse.from_spmatrix(expr_spmat) #create dataframe from this specific format
    if genes_in_col == True: pass
    elif genes_in_col == False:
        y = y.T #changes to columns to lines and lines to columns. This is contingent in genes_in_col. 
    y.to_csv('C:/USERS/annam/CSE/clusterclub/Cluster_PVal/Cluster_PVal/tests/data_for_tests/out.csv')  #places the dataframe in a CSV file
    dataset = 'C:/USERS/annam/CSE/clusterclub/project/tests/data_for_tests/GSE158761_matrix.mtx.gz'
    data = pd.read_csv(dataset, sep=",", header=[0]) #creating a dataframe based on input file with comma seperation
    dataset = 'C:/USERS/annam/CSE/clusterclub/project/tests/data_for_tests/GSE158761_matrix.mtx.gz'
    data = pd.read_csv(dataset, sep=" ", header=[0]) #penguin file in dataframe

    return y


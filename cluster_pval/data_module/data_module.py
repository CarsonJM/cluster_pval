"""
This code is the format data module.
"C:/USERS/annam/CSE/clusterclub/Cluster_PVal/Cluster_PVal/tests/data_for_tests/GSE158761_matrix.mtx.gz"
"""

import scipy.io
import pandas as pd

def ask_user():
    """Ask how the user would like to format the dataframe
    """
    print("Do you want genes in columns of genes in rows?")
    response = ''
    while response.lower() not in {"columns", "rows"}:
        response = input("Please enter columns or rows: ")
    return response.lower() == "columns"

def format_data(genes_in_col, path_to_file): 
    """Passing data through format module. If user inputs genes in columns, it passes. 
    If user selects False, it will flip data to genes in rows
    
    :param genes_in_col: boolean that user decides. If true, genes will be in 
    columns. If false, genes will be in rows.
    :param path_to_file: string that tells code where to find the input data
    : return: return y as dataframe
    """
    EXPR_MTX_PATH = path_to_file
    print("mm starting")
    expr_spmat = scipy.io.mmread(EXPR_MTX_PATH) #specific format in with RNA data came
    print("mm read done")
    y = pd.DataFrame.sparse.from_spmatrix(expr_spmat) #create dataframe from this specific format
    if genes_in_col == True: pass
    elif genes_in_col == False:
        y = y.T #changes to columns to lines and lines to columns. This is contingent in genes_in_col. 
    return y


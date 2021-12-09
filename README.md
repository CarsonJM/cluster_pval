[![Build Status](https://app.travis-ci.com/Cluster-Club/Cluster_PVal.svg?branch=master)](https://app.travis-ci.com/Cluster-Club/Cluster_PVal)

# cluster_pval

Clustering is a common means of sorting cells in RNAseq datasets into different cell types. Tests for differences in means between cell type clusters do not take into account the fact that the clusters were inferred from the data, this “double dipping” inflates Type 1 error. This is considered one of the greatest challenges currently facing single cell data science. Gao et al (2021) proposed a new method for calculating p values when measuring differences in means between clusters that controls for type 1 error

This tool calculates the difference in means between RNAseq clusters and reports the p-value calculated using the wald test and the method proposed in L. Gao, J. Bien & D. Witten [2021]. Users upload scRNAseq datasets and input metadata about the datasets. The tool then uses hierarchical clustering to assign each cell to a cluster and calculates both p values for the difference in means between each cluster pair.

Envisioned users include Users researchers that work with scRNAseq datasets, statisticians interested in the p value calculation methods, and datascientists who may want to calculate adjusted p values for other types of data.


## Installation and Requirements
This package is not pip installable. To install please git clone the directory. 

This package requires the packages numpy, pandas, scikit-learn, 
scipy, umap-learn, and plotly.

## Directory Layout

```bash

├── app.py
├── cluster_pval
│   ├── cluster_module
│   │   ├── cluster_function.py
│   │   ├── __init__.py
│   ├── data_module
│   │   ├── data_module.py
│   │   └── __init__.py
│   ├── display_module
│   │   ├── display.py
│   │   ├── __init__.py
│   ├── helper_module
│   │   ├── helper_functions.py
│   │   ├── __init__.py
│   ├── __init__.py
│   ├── pval_module
│   │   ├── __init__.py
│   │   ├── stattests.py
│   │   └── trunc_sets.py
│   └── tests
│       ├── data_for_tests
│       │   ├── 200tcells_200bcells_200memorycells.csv
│       │   ├── 600tcells.csv
│       │   ├── GSE158761_matrix.mtx.gz
│       │   ├── out.csv
│       │   ├── penguin_data_subset.txt
│       │   ├── penguin_data_subset_with_species.csv
│       │   ├── penguin_data_subset_with_species.txt
│       │   └── SigInv1.csv
│       ├── data_module_test.py
│       ├── __init__.py
│       ├── test_cluster_module.py
│       ├── test_display.py
│       └── test_pvalmod.py
├── doc
│   ├── Component_Diagram.png
│   ├── images_for_Readme
│   │   ├── GUI_Home_Page.PNG
│   │   └── Open_File_Navigator.PNG
│   ├── USE_CASES.md
│   └── USER_STORIES.md
├── environment.yml
├── LICENSE
├── Procfile
├── README.md
└── requirements.txt

```

Base Directory: Contains files related to git, travis (for continuous 
integration), the app's dashboard, our License, and README

app.py: Contains code for organizing the cluster pval dashboard and incorporating
the functions from all other modules.

doc: Contains documents related to tool design (component diagrams, user 
stories & use cases)

cluster_pval: Contains modules required to build tool as descrived in the 
component diagram in doc folder. Contains the following folders:

cluster_module: contains module responsible for clustering data

data_format: contains module responsible for formatting input data

display: contains module responsible for displaying clustered data and p 
value after p value calculation

Helper_module: contains module containing functions used in app.py

pval_module: contains module responsible for calculating wald and adjusted p 
values

tests: contains unit tests for each module and folder containing data for 
unit tests.

## Usage

INSERT GUI INSTRUCTIONS AND SCREEN GRABS HERE


The p value calculation functions can be used with any clustering algorithm in sklearn as follows
```python
from pval_module.stattests import stattest_clusters_approx
from pval_module.stattests import wald_test

# np array holding points to cluster
x = np.array([[5, 3],
                      [10, 15],
                      [15, 12],
                      [24, 10],
                      [30, 30],
                      [85, 70],
                      [71, 80],
                      [60, 78],
                      [70, 55],
                      [80, 91], ])
# number of clusters
k = 2
# function used to cluster
cl_fun = sklearn.cluster.AgglomerativeClustering
# positional arguments given to function used to cluster
positional_arguments = []
# keyword arguments given to function to cluster
keyword_arguments = {'n_clusters': k, 'affinity': 'euclidean',
                             'linkage': 'average'}
cluster = cl_fun(*positional_arguments, **keyword_arguments)
cluster.fit_predict(x)
# k1 and k2 are the clusters being tested for significant differences
# in means
k1 = 0
k2 = 1
# running adjusted pvalue function
stat, pval, stderr = stattest_clusters_approx(x, k1, k2, cluster.labels_, 
                                              cl_fun, positional_arguments, 
                                              keyword_arguments)
#running wald test function
wald_test(x, k1, k2, cluster.labels_)
```

## Contributing
Project completed by S. Jannetty, C. Miller, A. Mounsey, S. Pollack & L. Droog. 

Clustering functions based on functions implemented by Lucy L Gao in the R package [clusterpval](https://www.lucylgao.com/clusterpval/).

Pull requests are welcome. 

## License
[BSD 2-Clause "Simplified" License](https://choosealicense.com/licenses/bsd-2-clause/)


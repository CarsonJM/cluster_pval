![Build Status](https://github.com/Cluster-Club/cluster_pval/actions/workflows/python-app.yml/badge.svg)
![Coverage Badge](doc/images_for_README/pea_logo.png)

# [Open the webapp here!](https://cluster-pval.herokuapp.com/)

![Logo image](doc/images_for_README/coverage-badge.svg)

# cluster_pval

Clustering is a common means of sorting cells in RNAseq datasets into different cell types. Tests for differences in means between cell type clusters do not take into account the fact that the clusters were inferred from the data, this “double dipping” inflates Type 1 error. This is considered one of the greatest challenges currently facing single cell data science. Gao et al (2021) proposed a new method for calculating p values when measuring differences in means between clusters that controls for type 1 error

This tool calculates the difference in means between RNAseq clusters and reports the p-value calculated using the wald test and the method proposed in L. Gao, J. Bien & D. Witten [2021]. Users upload scRNAseq datasets and input metadata about the datasets. The tool then uses hierarchical clustering to assign each cell to a cluster and calculates both p values for the difference in means between each cluster pair.

Envisioned users include researchers that work with scRNAseq datasets, statisticians interested in the p value calculation methods, and data scientists who may want to calculate adjusted p values for other types of data.


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

data_module: contains module responsible for formatting input data

display_module: contains module responsible for plotting clustered data

helper_module: contains module containing functions used in app.py

pval_module: contains module responsible for calculating wald and adjusted p 
values

tests: contains unit tests for each module and folder containing data for 
unit tests.

## Usage

To open from your terminal:
1. Import the Cluster_Pval package by:
	- opening your terminal
	- assuming that  Git is installed, type: Git Clone "[link to repository]"
2. While being in the main repository directory, 
   open the web app by typing in your terminal: python app.py
3. Use the link that is showed in the Terminal to go to the webapp

Or open directly from link in README

4. Drag or upload the dataset (in CSV format) in the web app
5. You'll see a preview of your data, and you will be requested to submit the following data:
	- Data information; the first and last columns containing data to be clustered. 
		Please note that this is in zero-index format, 
	  meaning that the first column of the dataset is denoted with 0. 
	- Number of clusters
	- Clustering method; choose to apply hierarchical of K-means clustering
	- linkage method
6. Click on "Press to submit clustering"
7. You will see the following results:
	- A cluster visualization with the option to download the figure
	- A preview of the Wald p-value
	- A preview of the adjusted p-value
	
## Contributing
Project completed by S. Jannetty, C. Miller, A. Mounsey, S. Pollack & L. Droog. 

Clustering functions based on functions implemented by Lucy L Gao in the R package [clusterpval](https://www.lucylgao.com/clusterpval/).

Pull requests are welcome. 

## License
[BSD 2-Clause "Simplified" License](https://choosealicense.com/licenses/bsd-2-clause/)

## Acknowledgement
This project is conducted as part of the CSE 583 - Software development for Data Scientists at the University of Washington. Therefore, we want to thank prof. D. Beck and A. Mittal for the valuable lessons and guidance that helped to create this application. 


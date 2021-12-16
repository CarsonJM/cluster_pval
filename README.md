![Build Status](https://github.com/Cluster-Club/cluster_pval/actions/workflows/python-app.yml/badge.svg) 
![Coverage Button](doc/images_for_README/coverage-badge.svg)


# [Open the webapp here!](https://cluster-pval.herokuapp.com/)


![Logo image](doc/images_for_README/pea_logo.png)



# cluster_pval

Clustering is a common means of sorting cells in RNAseq datasets into different cell types. Tests for differences in means between cell type clusters do not take into account the fact that the clusters were inferred from the data, this “double dipping” inflates Type 1 error. This is considered one of the greatest challenges currently facing single cell data science. Gao et al (2021) proposed a new method for calculating p values when measuring differences in means between clusters that controls for type 1 error.

This tool calculates the difference in means between RNAseq clusters and reports the p-value calculated using the wald test and the method proposed in L. Gao, J. Bien & D. Witten [2021]. Users upload scRNAseq datasets and input metadata about the datasets. The tool then uses hierarchical clustering to assign each cell to a cluster and calculates both p values for the difference in means between each cluster pair.

Envisioned users include researchers that work with scRNAseq datasets, statisticians interested in the p value calculation methods, and data scientists who may want to calculate adjusted p values for other types of data.


## Installation and Requirements
This package is not pip installable. To use this tool, please either use the [webapp](https://cluster-pval.herokuapp.com/) or git clone the directory. The webapp will time out if clustering or p-value calculation takes more than 30 seconds, so if you are clustering a large dataset or you would like to calculate the adjusted p-value with a large number of Monte-Carlo draws, please git clone the directory and run the app locally. See instructions in **Usage** section of README below.

Required packages can be found in requirements.txt

## Directory Layout

```bash
cluster_pval
├── CODE_OF_CONDUCT.md
├── LICENSE
├── Procfile
├── README.md
├── app.py
├── cluster_pval
│   ├── README.md
│   ├── __init__.py
│   ├── cluster_module
│   │   ├── __init__.py
│   │   └── cluster_function.py
│   ├── data_module
│   │   ├── __init__.py
│   │   └── data_module.py
│   ├── display_module
│   │   ├── __init__.py
│   │   └── display.py
│   ├── helper_module
│   │   ├── __init__.py
│   │   └── helper_functions.py
│   ├── pval_module
│   │   ├── __init__.py
│   │   └── stattests.py
│   └── tests
│       ├── __init__.py
│       ├── data_for_tests
│       │   ├── 200tcells_200bcells_200memorycells.csv
│       │   ├── 600tcells.csv
│       │   ├── GSE158761_matrix.mtx.gz
│       │   ├── README.md
│       │   ├── SigInv1_600tcells.csv
│       │   ├── SigInv2_200t_200b_200mem.csv
│       │   ├── penguin_data_subset.txt
│       │   ├── penguin_data_subset_with_species.csv
│       │   └── penguin_data_subset_with_species.txt
│       ├── test_cluster_module.py
│       ├── test_datamod.py
│       ├── test_display.py
│       └── test_pvalmod.py
├── doc
│   ├── Component_Diagram.png
│   ├── README.md
│   ├── USER_STORIES.md
│   ├── USE_CASES.md
│   ├── final_presentation.pdf
│   └── images_for_README
│       ├── coverage-badge.svg
│       └── pea_logo.png
└── requirements.txt

10 directories, 39 files
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

**NOTE**: If using a large dataset (50+ columns), the dataset MUST have an even number of 
columns (dimensions). This is a quirk that currently allows us to avoid overflowing a float.
We are working on addressing this issue.

To open from your terminal and run locally (which you should do if working with a large dataset):
1. Import the Cluster_Pval package by:
	- opening your terminal
	- assuming that  Git is installed, type: `git clone https://github.com/Cluster-Club/cluster_pval.git`
2. While being in the main repository directory, (`cd cluster_pval`), type: `pip install -r requirements.txt`
   or, if using conda, create a new environment for running the app by typing: 
   `conda create -n cluster_pval python=3.9` and then `pip install -r requirements.txt`
3. Open the web app by typing in your terminal: python app.py
4. Use the link that is showed in the Terminal to go to the webapp

Or open directly from link in README

![image](https://user-images.githubusercontent.com/91644573/146062855-d2d5733e-139e-42b5-afd2-1c87d1b6e513.png)

5. Drag or upload the dataset (in CSV format) in the web app
6. You'll see a preview of your data, and you will be requested to submit the following data:
	- Organization of samples (rows vs. columns)
	- Data information; the first and last columns containing data to be clustered. 
		Please note that this is in zero-index format, 
	  meaning that the first column of the dataset is denoted with 0. 
	- Number of clusters
	- Clustering method; choose to apply hierarchical of K-means clustering
	- linkage method

![image](https://user-images.githubusercontent.com/91644573/146283192-420bb6d5-217b-4fad-9d8a-2dfbf65bbc37.png)

7. Click on "Press to submit clustering"
8. You will a cluster visualization with the option to download the figure
 ![image](https://user-images.githubusercontent.com/91644573/146061937-7cb3c3ac-0a5a-4951-87e5-c2d3915c5338.png)

9. You will be requested to submit the following information for the calculation of the p-value:
	- A threshold for significance
	- An input number of draws to be used in calculating adjusted p-value
![image](https://user-images.githubusercontent.com/91644573/146062313-6404e2e4-45ff-4f2d-93bf-dda89aadb8a4.png)

10. Click on "Press to submit p-value calculation"
11. You will see the following results:
	- A preview of the wald p-value file
	- A preview of the adjusted p-value file 
![image](https://user-images.githubusercontent.com/91644573/146283251-3c89e0fc-ea3f-4538-be92-aa31c739e8ac.png)

	
## Contributing
Project completed by S. Jannetty, C. Miller, A. Mounsey, S. Pollack & L. Droog. 

Clustering functions based on functions implemented by Lucy L Gao in the R package [clusterpval](https://www.lucylgao.com/clusterpval/).

Pull requests are welcome. 

## License
[BSD 2-Clause "Simplified" License](https://choosealicense.com/licenses/bsd-2-clause/)

## Acknowledgement
This project is conducted as part of the CSE 583 - Software development for Data Scientists at the University of Washington. Therefore, we want to thank prof. D. Beck and A. Mittal for the valuable lessons and guidance that helped to create this application. 


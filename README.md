[![Build Status](https://app.travis-ci.com/Cluster-Club/Cluster_PVal.svg?branch=master)](https://app.travis-ci.com/Cluster-Club/Cluster_PVal?branch=master)

Welcome to the project page of group 10 - scRNAseq Clustering Issues

S. Jannetty, C. Miller, A. Mounsey, S. Pollack & L. Droog

Background: 
Clustering is a common means of sorting cells in RNAseq datasets into different cell types. Tests for differences in means between cell type clusters do not take into account the fact that the clusters were inferred from the data, this “double dipping” inflates Type 1 error. This is considered one of the greatest challenges currently facing single cell data science. Gao et al (2021) proposed a new method for calculating p values when measuring differences in means between clusters that controls for type 1 error

Goal: 
Calculate difference in means between RNAseq clusters and report the p-value based on L. Gao, J. Bien & D. Witten [2021]and to compare the adjusted p value to published p values.

Users:      
Researchers that work with scRNAseq dataset
Statistician interested in the methods
Datascientist that edit the code



DIRECTORY LAYOUT

doc: contains documents (component diagrams, user stories & use cases)


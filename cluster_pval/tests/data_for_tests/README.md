## Data Files used in unit tests
**200tcells_200bcells_200memorycells.csv** - scRNAseq data from [Zheng et 
al 2017](https://pubmed.ncbi.nlm.nih.gov/28091601/) preprocessed as 
described in [Gao et al 2021](https://arxiv.org/pdf/2012.02936.pdf). 
Contains expression of top 500 genes from 200 memory t cells, 200 b cells, and 
200 monocytes. Clustering these data into 3 clusters should yield 
statistically significant clusters.

**600tcells.csv** - scRNAseq data from 
[Zheng et al 2017](https://pubmed.ncbi.nlm.nih.gov/28091601/) preprocessed as 
described in [Gao et al 2021](https://arxiv.org/pdf/2012.02936.pdf). 
Contains expression of top 500 genes from 600 memory t cells. Clustering 
these data into 3 clusters should not yield statistically significant clusters.

**GSE158761_matric_mtx_gz** - Arabidopsis scRNAseq data with genes in 
columns, used to test data_module.

**SigInv1_600tcells.csv** -  SigInv matrix used in 
[Gao et al 2021](https://arxiv.org/pdf/2012.02936.pdf) when running 
test_hier_clusters_exact on Zheng scRNAseq data subset consisting of 
expression data from 600 t cells. Used to test stattests module.

**SigInv2_200t_200b_200mem.csv** -  SigInv matrix used in 
[Gao et al 2021](https://arxiv.org/pdf/2012.02936.pdf) when running 
test_hier_clusters_exact on Zheng scRNAseq data subset consisting of
expression data from 200 t cells, 200 b cells, and 200 monocytes. 
Used to test stattests module.

**penguin_data_subset.txt** - Palmer Penguins dataset preprocessed as 
described in [Gao et al 2021](https://arxiv.org/pdf/2012.02936.pdf) without 
species data.

**penguin_data_subset_with_species.csv** - Palmer Penguins dataset 
preprocessed as described in 
[Gao et al 2021](https://arxiv.org/pdf/2012.02936.pdf) with 
species data saved as a csv.

**penguin_data_subset_with_species.txt** - Palmer Penguins dataset 
preprocessed as described in 
[Gao et al 2021](https://arxiv.org/pdf/2012.02936.pdf) with 
species data.

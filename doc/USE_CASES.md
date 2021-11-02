# Use Cases

## Researcher
- **Tool:** Requests data in specified formats
- **Researcher:** Import pre-processed scRNAseq dataset along with associated metadata file. Additionally, they specify the number of clusters they want and the clustering method(s).
- **Tool:** Imports and formats data. Then runs clustering via all selected methods. Finally, the tool calculates P values using Wald p-value method and Gao P value method. (Should the tool display means that are different using traditional p-value, but not different using modified p-value?)
- **Tool:** Displays clustering network, and associated associated p-values. Additionally the tool asks if the results should be downloaded.
- **Researcher:** Inputs desired directory for download.
- **Tool:** Downloads the data.

## Statistician
- **Tool:** Requests data in specified formats
- **Statistician:** Imports multiple pre-processed scRNAseq datasets along with associated metadata files in order to perform statistical analyses on traditional vs modified p-values. Additionally, they specify the number of clusters they want and the clustering method(s) for each dataset.
- **Tool:** Imports and formats data. Then runs clustering via all selected methods. Finally, the tool calculates P values using Wald p-value method and Gao P value method. (Should the tool display means that are different using traditional p-value, but not different using modified p-value?)
- **Tool:** Displays clustering network, and associated associated p-values. Additionally the tool asks if the results should be merged and downloaded.
- **Statistician:** Inputs desired directory for download.
- **Tool:** Results from each run are merged, then the tool downloads the data.
- **Statistician:** Is interested in new P value method. Seeks out additional information in our documentation about inner workings.

## Data Scientist
- **Data Scientist:** Talks to researcher and identify new clustering method they would like to incorperate. Download repo and add new clustering method.
- **Tool:** Modular structure allows for easy incorporation of the new method. Also, provides a small test dataset to ensure the tool is running properly.
- **Data Scientist:** Tests new clustering method using test data set (same process as for the researcher)
- **Tool:** Runs test dataset and displays the results
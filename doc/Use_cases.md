# Use Cases

## Researcher
- **Researcher:** Import pre-processed scRNAseq dataset. Specifies number of clusters they want and clustering method(s).
- **Tool:** Displays "run test" 
- **Tool:** Runs clustering with all selected methods. Calculates P values using Wald p-value method and Gao P value method. Presents output of each.

## Statistician
- **Statistician:** Import pre-processed scRNAseq dataset. Specifies number of clusters they want and clustering method(s). 
- **Tool:** Displays "run test"
**Tool:** Runs clustering with all selected methods. Calculates P values using Wald p-value method and Gao P value method. Presents output of each.
- **Statistician:** Is interested in new P value method. Seeks out additional information in our documentation about inner workings.
- **Something to consider** Maybe allow for multiple datasets to be uploaded at a time.

## Data Scientist
- **Data Scientist** Talk to researcher and identify new clustering method they would like to incorperate. Download repo and add new clustering method.
- **Tool** Easily adapted to incorperate new method. If there is a GUI, change is reflected in GUI by adding new option (potentially to drop down?).
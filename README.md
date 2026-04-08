
# SDMGAE: A novel cancer driver genes identification method based on self-supervised dual masked graph autoencoder.

## Dependencies
Higher versions should be also available.

+ numpy==1.21.6
+ torch==1.12.1+cu102
+ torch-cluster==1.6.0
+ torch_geometric>=2.4.0
+ torch-scatter==2.0.9
+ torch-sparse==0.6.14
+ scipy==1.7.3
+ texttable==1.6.2
+ CUDA 10.2
+ CUDNN 7.6.0

## Clone this project
```bash
git clone https://github.com/liwenjun111/SDMGAE.git
cd SDMGAE
```

## Installation

```bash
pip install -r requirements.txt
```

## Dataset Description

The dataset consists of three main components:

### 1. Feature (Node Features)
- Feature is an N×F dimensional matrix, where N represents the number of genes and F represents the feature dimension. It is constructed by concatenating gene expression, gene mutation, copy number variation (CNV), and DNA methylation data, all derived from the TCGA database.

### 2. CPDB (Graph Structure)
- CPDB is derived from ConsensusPathDB and is used to construct graph structures, reflecting which genes are connected by edges.
<table>
  <tr>
    <td>source</td>
    <td>target</td>
  </tr>
  <tr>
    <td>gene1</td>
    <td>gene3</td>
  </tr>
  <tr>
    <td>gene2</td>
    <td>gene4</td>
  </tr>
</table>
### 3. Label (Node Labels)
- The label indicates whether each gene is a cancer‑associated gene.


## User Guide

### Step 1: Prepare Dataset
Modify the code according to your needs.

GAE_model/train.py:
```bash
data, clf_data = get_PPIdataset('D:/SDMGAE-main/data/PANCER/', 'feature.csv', 'CPDB.csv', 'label.csv')
```
SAGAE_main/datasets/data_util.py:
```bash
features_df = pd.read_csv("D:/SDMGAE-main/data/PANCER/feature.csv")
edges_df = pd.read_csv("D:/SDMGAE-main/data/PANCER/CPDB.csv")
```
SAGAE_main/models/evaluation_5cv.py:
```bash
labels_df = pd.read_csv("D:/SDMGAE-main/data/PANCER/label.csv")
```


### Step 2: Pretraining and Evaluation
Run the code:
```bash
python main_transductive.py
```
The model will be pre-trained, and the obtained embeddings will be evaluated using ten rounds of 5-fold cross-validation. Finally, the average AUROC and AUPRC values will be output.

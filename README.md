
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
<table>
  <tr>
    <td>gene symbol</td>
    <td>CNV-sample1</td>
    <td>CNV-sample2</td>
    <td>...</td>
    <td>EXP-sample1</td>
    <td>EXP-sample2</td>
    <td>...</td>
    <td>MET-sample1</td>
    <td>MET-sample2</td>
    <td>...</td>
    <td>MUT-sample1</td>
    <td>MUT-sample2</td>
    <td>...</td>
  </tr>
  <tr>
    <td>gene1</td>
    <td>cnv</td>
    <td>cnv</td>
    <td>...</td>
    <td>exp</td>
    <td>exp</td>
    <td>...</td>
    <td>meth</td>
    <td>meth</td>
    <td>...</td>
    <td>mut</td>
    <td>mut</td>
    <td>...</td>
  </tr>
  <tr>
    <td>gene2</td>
    <td>cnv</td>
    <td>cnv</td>
    <td>...</td>
    <td>exp</td>
    <td>exp</td>
    <td>...</td>
    <td>meth</td>
    <td>meth</td>
    <td>...</td>
    <td>mut</td>
    <td>mut</td>
    <td>...</td>
  </tr>
  <tr>
    <td>gene3</td>
    <td>cnv</td>
    <td>cnv</td>
    <td>...</td>
    <td>exp</td>
    <td>exp</td>
    <td>...</td>
    <td>meth</td>
    <td>meth</td>
    <td>...</td>
    <td>mut</td>
    <td>mut</td>
    <td>...</td>
  </tr>
</table>

### 2. PPI network (Graph Structure)
- PPI network is derived from ConsensusPathDB and is used to construct graph structures, reflecting which genes are connected by edges.
<table>
  <tr>
    <td>source gene</td>
    <td>target gene</td>
  </tr>
  <tr>
    <td>gene1-index</td>
    <td>gene3-index</td>
  </tr>
  <tr>
    <td>gene2-index</td>
    <td>gene4-index</td>
  </tr>
</table>

### 3. Label (Node Labels)
- The label indicates whether each gene is a cancer‑associated gene.
<table>
  <tr>
    <td>gene symbol</td>
    <td>index</td>
    <td>label</td>
  </tr>
  <tr>
    <td>gene1</td>
    <td>gene1-index</td>
    <td>1</td>
  </tr>
  <tr>
    <td>gene2</td>
    <td>gene2-index</td>
    <td>0</td>
  </tr>
</table>

## User Guide

### Step 1: Prepare Dataset
Modify the code according to your needs.

GAE_model/train.py:
```bash
data, clf_data = get_PPIdataset('./data/PANCER/', 'feature.csv', 'CPDB.csv', 'label.csv')
```
SAGAE_main/datasets/data_util.py:
```bash
features_df = pd.read_csv("./data/PANCER/feature.csv")
edges_df = pd.read_csv("./data/PANCER/CPDB.csv")
```
SAGAE_main/models/evaluation_5cv.py:
```bash
labels_df = pd.read_csv("./data/PANCER/label.csv")
```


### Step 2: Pretraining and Evaluation
Run the code:
```bash
python main_transductive.py
```
The model will be pre-trained, and the obtained embeddings will be evaluated using ten rounds of 5-fold cross-validation. Finally, the average AUROC and AUPRC values will be output.

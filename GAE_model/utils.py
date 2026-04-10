import sys
import os
import pandas as pd
import torch
import random
import numpy as np
from texttable import Texttable
import torch_geometric.transforms as T
from torch_geometric.datasets import Amazon, Coauthor, Planetoid, Reddit
from torch_geometric.data import Data
from torch_geometric.utils import index_to_mask
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def set_seed(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False





def get_PPIdataset(root: str, node_feature_file: str, edge_index_file: str, target_label_file: str):
    import os
    import numpy as np
    import pandas as pd
    import torch
    from torch_geometric.data import Data
    from torch_geometric.utils import index_to_mask
    from sklearn.model_selection import train_test_split

    feat_df = pd.read_csv(os.path.join(root, node_feature_file))

    if "gene" not in feat_df.columns:
        raise ValueError("feature.csv 缺少 gene 列")

    genes = feat_df["gene"].astype(str).tolist()
    local_id = np.arange(len(genes), dtype=np.int64)
    gene2local = dict(zip(genes, local_id))

    x_df = feat_df.drop(columns=["gene"])
    x_df = x_df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    x = torch.tensor(x_df.values, dtype=torch.float32)
    x = scale_feats(x)
    num_nodes = x.size(0)


    label_df = pd.read_csv(os.path.join(root, target_label_file))
    if not {"gene", "index", "label"}.issubset(label_df.columns):
        raise ValueError("label.csv 需要包含 gene, index, label 三列")

    label_df["gene"] = label_df["gene"].astype(str)


    merged = label_df.merge(
        pd.DataFrame({"gene": genes, "local_id": local_id}),
        on="gene",
        how="inner"
    )


    global2local = dict(zip(merged["index"].astype(int).tolist(), merged["local_id"].astype(int).tolist()))

    y = torch.full((num_nodes,), -1, dtype=torch.long)
    for _, r in merged.iterrows():
        y[int(r["local_id"])] = int(r["label"])


    edge_df = pd.read_csv(os.path.join(root, edge_index_file), header=None)
    u = edge_df.iloc[:, 0].astype(int)
    v = edge_df.iloc[:, 1].astype(int)

    mask = u.isin(global2local) & v.isin(global2local)
    u_local = u[mask].map(global2local).to_numpy()
    v_local = v[mask].map(global2local).to_numpy()

    edge_index = torch.tensor(np.vstack([u_local, v_local]), dtype=torch.long)


    assert edge_index.min().item() >= 0
    assert edge_index.max().item() < num_nodes, (edge_index.max().item(), num_nodes)

    data = Data(x=x, edge_index=edge_index, y=y, num_nodes=num_nodes)


    labeled_mask = (y != -1)
    filtered_indices = labeled_mask.nonzero(as_tuple=False).view(-1)  # local_id
    labeled_x = x[filtered_indices]
    labeled_y = y[filtered_indices]

    clf_data = Data(
        labeled_x=labeled_x,
        labeled_y=labeled_y,
        labeled_num_nodes=labeled_x.size(0),
        filtered_indices=filtered_indices
    )

    # train/val/test mask
    train_idx, test_idx = train_test_split(range(clf_data.labeled_num_nodes), test_size=0.2, random_state=42)
    val_idx, test_idx = train_test_split(test_idx, test_size=0.5, random_state=42)
    clf_data.train_mask = index_to_mask(torch.tensor(train_idx), clf_data.labeled_num_nodes)
    clf_data.val_mask   = index_to_mask(torch.tensor(val_idx), clf_data.labeled_num_nodes)
    clf_data.test_mask  = index_to_mask(torch.tensor(test_idx), clf_data.labeled_num_nodes)

    return data, clf_data



def tab_printer(args):
    """Function to print the logs in a nice tabular format.

    Note
    ----
    Package `Texttable` is required.
    Run `pip install Texttable` if was not installed.

    Parameters
    ----------
    args: Parameters used for the model.
    """
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable()
    t.add_rows([["Parameter", "Value"]] + [[k, str(args[k])] for k in keys if not k.startswith('__')])
    return t.draw()


def scale_feats(x):

    scaler = StandardScaler()
    feats = x.numpy()
    scaler.fit(feats)
    feats = torch.from_numpy(scaler.transform(feats)).float()
    return feats
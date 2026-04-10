
from collections import namedtuple, Counter
import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F

import dgl
from sklearn.preprocessing import StandardScaler



def preprocess(graph):
    feat = graph.ndata["feat"]

    graph = dgl.remove_self_loop(graph)

    graph = dgl.to_simple(graph)
    graph = dgl.to_bidirected(graph)
    graph.ndata["feat"] = feat

    graph = graph.remove_self_loop().add_self_loop()
    graph.create_formats_()
    return graph


def scale_feats(x):

    scaler = StandardScaler()
    feats = x.numpy()
    scaler.fit(feats)
    feats = torch.from_numpy(scaler.transform(feats)).float()
    return feats

def load_dataset():
    import pandas as pd
    import torch
    import dgl


    features_df = pd.read_csv("./data/PANCER/feature.csv")

    if "gene" in features_df.columns:
        features_df = features_df.drop(columns=["gene"])

    features_df = features_df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    feat_bio = torch.tensor(features_df.values, dtype=torch.float32)
    feat_bio = scale_feats(feat_bio)

    num_nodes = feat_bio.shape[0]  # 9110
    num_features = feat_bio.shape[1]
    print("特征维度:", num_features)


    edges_df = pd.read_csv("./data/PANCER/CPDB.csv")
    src_nodes = edges_df.iloc[:, 0].to_numpy()
    dst_nodes = edges_df.iloc[:, 1].to_numpy()

    mask = (src_nodes >= 0) & (src_nodes < num_nodes) & (dst_nodes >= 0) & (dst_nodes < num_nodes)
    src_nodes = src_nodes[mask]
    dst_nodes = dst_nodes[mask]


    graph = dgl.graph((src_nodes, dst_nodes), num_nodes=num_nodes)


    graph.ndata["feat"] = feat_bio


    graph = preprocess(graph)
    feat = scale_feats(graph.ndata["feat"])
    graph.ndata["feat"] = feat

    num_classes = 2
    return graph, (num_features, num_classes)






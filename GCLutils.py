import math
import numpy as np
import scipy.sparse as sp
from scipy.special import iv
from scipy.sparse.linalg import eigsh
import os.path as osp
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.manifold import SpectralEmbedding
# from libKMCUDA import kmeans_cuda
from tqdm import tqdm
from matplotlib import cm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.optim import Adam
from torch.utils.data import random_split
from torch_geometric.nn import GCNConv, SGConv, SAGEConv, GATConv, GraphConv, GINConv
from torch_geometric.utils import sort_edge_index, degree, add_remaining_self_loops, remove_self_loops, get_laplacian, \
    to_undirected, to_dense_adj, to_networkx
from torch_geometric.datasets import KarateClub
from torch_scatter import scatter
import torch_sparse

import networkx as nx
import matplotlib.pyplot as plt


def get_base_model(name: str):
    print("name:",name) #name: GCNConv
    print("in_channels: ",in_channels)#in_channels:  6805---coauthor-cs数据集的特征维度
    print("out_channels:",out_channels)#out_channels: 256("num_hidden": 256--b编码器GCN的隐层维度）
    def gat_wrapper(in_channels, out_channels):
        model=GATConv(
            in_channels=in_channels,
            out_channels=out_channels // 4,
            heads=4
        )
        print("model:",model)
        return GATConv(
            in_channels=in_channels,
            out_channels=out_channels // 4,
            heads=4
        )
      
    def gin_wrapper(in_channels, out_channels):
        mlp = nn.Sequential(
            nn.Linear(in_channels, 2 * out_channels),
            nn.ELU(),
            nn.Linear(2 * out_channels, out_channels)
        )
        print("mlp: ",mlp)
        print("GINConv(mlp):", GINConv(mlp))
        return GINConv(mlp)

    base_models = {
        'GCNConv': GCNConv, #GCN卷积模型
        'SGConv': SGConv,
        'SAGEConv': SAGEConv,
        'GATConv': gat_wrapper,
        'GraphConv': GraphConv,
        'GINConv': gin_wrapper
    }
    print("base_models: ",base_models)#base_models:  {'GCNConv': <class 'torch_geometric.nn.conv.gcn_conv.GCNConv'>, 'SGConv': <class 'torch_geometric.nn.conv.sg_conv.SGConv'>, 'SAGEConv': <class 'torch_geometric.nn.conv.sage_conv.SAGEConv'>, 'GATConv': <function get_base_model.<locals>.gat_wrapper at 0x00000198655BFF28>, 'GraphConv': <class 'torch_geometric.nn.conv.graph_conv.GraphConv'>, 'GINConv': <function get_base_model.<locals>.gin_wrapper at 0x00000198655BFEA0>}
    print("base_models[name] :",base_models[name]) 
    #base_models[name] : <class 'torch_geometric.nn.conv.gcn_conv.GCNConv'>
    return base_models[name]


def get_activation(name: str):
    print("name: ",name)#name:  rrelu
    activations = {
        'relu': F.relu,
        'hardtanh': F.hardtanh,
        'elu': F.elu,
        'leakyrelu': F.leaky_relu,
        'prelu': torch.nn.PReLU(),
        'rrelu': F.rrelu
    }
    print("activations[name]: ",activations[name])#activations[name]:  <function rrelu at 0x0000019864E3CB70>
    return activations[name]


def compute_pr(edge_index, damp: float = 0.85, k: int = 10):#网页排名中心性+计算特征权重,实现特征增强

    num_nodes = edge_index.max().item() + 1
    deg_out = degree(edge_index[0])
    x = torch.ones((num_nodes, )).to(edge_index.device).to(torch.float32)
    for i in range(k):
        edge_msg = x[edge_index[0]] / deg_out[edge_index[0]]
        agg_msg = scatter(edge_msg, edge_index[1], reduce='sum')
        x = (1 - damp) * x + damp * agg_msg
    return x


def eigenvector_centrality(data):#特征向量中心性----+计算特征权重，实现特征增强
    graph = to_networkx(data)
    x = nx.eigenvector_centrality_numpy(graph)
    x = [x[i] for i in range(data.num_nodes)]
    return torch.tensor(x, dtype=torch.float32).to(data.edge_index.device)

def generate_split(num_samples: int, train_ratio: float, val_ratio: float):

    train_len = int(num_samples * train_ratio)
    val_len = int(num_samples * val_ratio)
    test_len = num_samples - train_len - val_len

    train_set, test_set, val_set = random_split(torch.arange(0, num_samples), (train_len, test_len, val_len))

    idx_train, idx_test, idx_val = train_set.indices, test_set.indices, val_set.indices
    #【0，18332】区间的数据集下标----替换成我的数据集对应的划分
    train_mask = torch.zeros((num_samples,)).to(torch.bool)
    test_mask = torch.zeros((num_samples,)).to(torch.bool)
    val_mask = torch.zeros((num_samples,)).to(torch.bool)
    train_mask[idx_train] = True #将训练集的idx部分设置为True，其它所有idx部分的值设置为false
    test_mask[idx_test] = True
    val_mask[idx_val] = True #验证集下标对应的部分为True,非验证集外的idx设置为false

    return train_mask, test_mask, val_mask


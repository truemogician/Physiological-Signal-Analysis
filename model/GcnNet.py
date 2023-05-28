import sys
from typing import Union

import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import Module
from torch.nn.parameter import Parameter
from numpy.typing import NDArray

sys.path.append("..")
from utils.torch import get_device


class Gcn(Module):  # GCN为：relu(A@X@B)=>((X.T@A.T).T@B)
    def __init__(self, 
        adj_mat_array: NDArray,
        node_emb_dim: int):
        node_num = adj_mat_array.shape[0]
        assert len(adj_mat_array.shape) == 2 and node_num == adj_mat_array.shape[1], "adj_mat_array must be a square matrix"
        super(Gcn, self).__init__()
        self.node_emb_dim = node_emb_dim
        self.linear1 = nn.Linear(node_num, node_num, device = get_device())
        self.linear1.weight = Parameter(torch.from_numpy(adj_mat_array.T))
        self.linear2: Union[nn.Linear, None] = None
        self.ReLU = nn.ReLU()

    def forward(self, node_array: Tensor):
        x: Tensor = self.linear1(node_array.T).T
        x = x.reshape(x.shape[0], -1)
        if self.linear2 is None:
            self.linear2 = nn.Linear(x.shape[1], self.node_emb_dim, device = x.device)
        x = self.ReLU(self.linear2(x))
        return x


class GcnNet(Module):
    def __init__(self, 
        node_emb_dims: int,
        adj_mat_array: NDArray,
        num_classes: int):
        assert len(adj_mat_array.shape) == 2 and adj_mat_array.shape[0] == adj_mat_array.shape[1], "adj_mat_array must be a square matrix"
        super(GcnNet, self).__init__()
        self.node_emb_dims = node_emb_dims
        self.adj_mat_array = adj_mat_array
        self.node_num = adj_mat_array.shape[0]
        self.num_classes = num_classes
        device = get_device()
        self.gcn_layer = Gcn(self.adj_mat_array, self.node_emb_dims)
        self.conv1 = nn.Conv1d(self.node_num, self.node_num, 3, device=device)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.linear1 = nn.Linear(self.node_emb_dims * self.node_num, self.node_emb_dims, device=device)
        self.linear2 = nn.Linear(self.node_emb_dims, self.num_classes, device=device)

    def forward(self, node_att_array: Tensor):
        shape = node_att_array.shape
        batch_size = shape[0]
        x = torch.empty((0, self.node_num, self.node_emb_dims)).to(get_device())
        for i in range(batch_size):
            temp = self.gcn_layer(node_att_array[i])
            temp = torch.unsqueeze(temp, dim=0)
            x = torch.cat([x, temp], dim=0)

        # x = self.conv1(x)
        x = x.reshape(x.shape[0], -1)

        x = self.relu(self.linear1(x))
        self.dropout(x)
        x = self.linear2(x)

        return x
    
    def get_matrix(self):
        return self.gcn_layer.linear1.weight.detach()

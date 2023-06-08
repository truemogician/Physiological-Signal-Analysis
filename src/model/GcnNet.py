from typing import Union

import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import Module
from torch.nn.parameter import Parameter
from numpy.typing import NDArray

from .utils import get_device


class Gcn(Module):  # GCN：relu(A@X@B)=>((X.T@A.T).T@B)
    def __init__(self, node_emb_dim: int, adjacent_matrix: NDArray, dtype = torch.float32):
        node_num = adjacent_matrix.shape[0]
        assert len(adjacent_matrix.shape) == 2 and node_num == adjacent_matrix.shape[1], "adj_mat_array must be a square matrix"
        super(Gcn, self).__init__()
        self.node_emb_dim = node_emb_dim
        self.dtype = dtype
        self.linear1 = nn.Linear(node_num, node_num, dtype=dtype, device=get_device())
        self.linear1.weight = Parameter(torch.from_numpy(adjacent_matrix.T).to(dtype))
        self.linear2: Union[nn.Linear, None] = None
        self.ReLU = nn.ReLU()

    def forward(self, tensor: Tensor):
        x: Tensor = self.linear1(tensor.T).T
        if len(x.shape) > 2:
            x = x.view(x.shape[0], -1)
        if self.linear2 is None:
            self.linear2 = nn.Linear(x.shape[1], self.node_emb_dim, dtype=self.dtype, device=x.device)
        x = self.ReLU(self.linear2(x))
        return x


class GcnNet(Module):
    def __init__(self, node_embedding_dims: int, class_num: int, adjacent_matrix: NDArray, dtype=torch.float32):
        assert len(adjacent_matrix.shape) == 2 and adjacent_matrix.shape[0] == adjacent_matrix.shape[1], "adj_mat_array must be a square matrix"
        super(GcnNet, self).__init__()
        self.node_embedding_dims = node_embedding_dims
        self.node_num = adjacent_matrix.shape[0]
        self.class_num = class_num
        device = get_device()
        self.gcn_layer = Gcn(self.node_embedding_dims, adjacent_matrix, dtype)
        # self.conv1 = nn.Conv1d(self.node_num, self.node_num, 3, dtype=dtype, device=device)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.linear1 = nn.Linear(self.node_embedding_dims * self.node_num, self.node_embedding_dims, dtype=dtype, device=device)
        self.linear2 = nn.Linear(self.node_embedding_dims, self.class_num, dtype=dtype, device=device)

    def forward(self, tensor: Tensor):
        shape = tensor.shape
        batch_size = shape[0]
        x = torch.empty((0, self.node_num, self.node_embedding_dims)).to(get_device())
        for i in range(batch_size):
            temp = self.gcn_layer(tensor[i])
            temp = torch.unsqueeze(temp, dim=0)
            x = torch.cat([x, temp], dim=0)
        # x = self.conv1(x)
        if len(x.shape) > 2:
            x = x.view(x.shape[0], -1)
        x = self.relu(self.linear1(x))
        self.dropout(x)
        x = self.linear2(x)
        return x
    
    def get_matrix(self):
        return self.gcn_layer.linear1.weight.detach()

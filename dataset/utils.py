from typing import Union

import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split
import torch
from torch import Tensor
import torch.utils.data as Data


def create_data_loader(
    data: Union[NDArray, Tensor], 
    labels: Union[NDArray, Tensor],
    batch_size = 1,
    shuffle = True):
    assert data.shape[0] == labels.shape[0], "data and labels must have the same length"
    if type(data) is not Tensor:
        data = torch.from_numpy(data)
    if type(labels) is not Tensor:
        labels = torch.from_numpy(labels)
    return Data.DataLoader(Data.TensorDataset(data, labels), batch_size=batch_size, shuffle=shuffle)

def create_train_test_loader(
    data: Union[NDArray, Tensor],
    labels: Union[NDArray, Tensor],
    test_size = 0.25,
    batch_size = 1,
    shuffle = True):
    assert data.shape[0] == labels.shape[0], "data and labels must have the same length"
    assert test_size is None or 0 < test_size < 1, "test_size must be in (0, 1)"
    train_data, test_data, train_label, test_label = \
        train_test_split(data, labels, test_size=test_size)
    train_loader = create_data_loader(train_data, train_label, batch_size, shuffle)
    test_loader = create_data_loader(test_data, test_label, batch_size, shuffle)
    return train_loader, test_loader
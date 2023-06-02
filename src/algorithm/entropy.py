import math
from typing import Any, Union, Tuple

import numpy as np
from nptyping import NDArray, Shape, Floating


List = Union[NDArray[Shape["*"], Any], list]

def shannon_entropy(freq_list: List):
    ''' This function computes the shannon entropy of a given frequency distribution.
    USAGE: shannon_entropy(freq_list)
    ARGS: freq_list = Numeric vector representing the frequency distribution
    OUTPUT: A numeric value representing shannon's entropy'''
    return -math.fsum([f * np.log(f) for f in freq_list if f != 0])

def permutation_entropy(ord_freq_list: List):
    max_entropy = math.log(len(ord_freq_list))
    if not isinstance(ord_freq_list, np.ndarray):
        ord_freq_list = np.array(ord_freq_list)
    p = np.divide(ord_freq_list, ord_freq_list.sum())
    return shannon_entropy(p) / max_entropy

def ordinal_pattern_frequency(time_series: List, window_size: int, window_step: int = 1):
    ''' This function computes the ordinal patterns of a time series for a given embedding dimension and embedding delay.
    USAGE: ordinal_patterns(time_series, window_size, window_step)
    ARGS: time_series = Numeric vector representing the time series,
    window_size = length of the time window, preferred range is [3, 7], window_step = time window step
    OUPTUT: A numeric vector representing frequencies of ordinal patterns'''
    n_states = math.factorial(window_size)
    lst = list()
    for i in range(0, len(time_series) - window_size, window_step):
        lst.append(np.argsort(time_series[i : i + window_size]))
    _, freq = np.unique(np.stack(lst, 0), return_counts=True, axis=0)
    freq = list(freq)
    if len(freq) < n_states:
        freq += [0] * (n_states - len(freq))
    return freq

def joint_ordinal_pattern_frequency(time_series: Tuple[List, List], window_size: int, window_step: int = 1):
    ''' This function computes the ordinal patterns of two time series for a given embedding dimension and embedding delay.
    USAGE: joint_ordinal_patterns(time_series, window_size, window_step)
    ARGS: time_series = Numeric vector representing two time series, 
    window_size = length of the time window, preferred range is [3, 7], window_step = time window step
    OUPTUT: A numeric vector representing frequencies of ordinal patterns'''
    assert len(time_series) == 2 and len(time_series[0]) == len(time_series[1]), "ts must be a tuple of two lists with same length"
    s1, s2 = time_series
    n_states = math.factorial(window_size) * 6
    l1, l2 = list(), list()
    for i in range(0, len(s1) - window_size, window_step):
        l1.append(list(np.argsort(s1[i : i + window_size])))
        l2.append(list(np.argsort(s2[i : i + window_size])))
    lst = np.concatenate((np.stack(l1, 0), np.stack(l2, 0)), axis=1)
    _, freq = np.unique(lst, return_counts=True, axis=0)
    freq = list(freq)
    if len(freq) < n_states:
        freq += [0] * (n_states - len(freq))
    return (freq)

def complexity(ord_freq_list: list):
    ''' This function computes the complexity of a time series defined as: Comp_JS = Q_o * JSdivergence * pe
    Q_o = Normalizing constant
    JSdivergence = Jensen-Shannon divergence
    pe = permutation entopry
    ARGS: ordinal pattern'''
    pe = permutation_entropy(ord_freq_list)
    length = len(ord_freq_list)
    constant1 = (0.5 + ((1 - 0.5) / length)) * math.log(0.5 + ((1 - 0.5) / length))
    constant2 = (1 - 0.5) / length * math.log((1 - 0.5) / length) * (length - 1)
    constant3 = 0.5 * math.log(length)
    Q_o = -1 / (constant1 + constant2 + constant3)

    temp_op_prob = np.divide(ord_freq_list, sum(ord_freq_list))
    temp_op_prob2 = (0.5 * temp_op_prob) + (0.5 * (1 / length))
    JSdivergence = shannon_entropy(temp_op_prob2) - 0.5 * shannon_entropy(temp_op_prob) - 0.5 * math.log(length)
    Comp_JS = Q_o * JSdivergence * pe
    return Comp_JS

def pmi(epoch: NDArray[Shape["*, *"], Floating], window_size: int, window_step: int = 1):
    ''' This function computes the PMI for an epoch.
    USAGE: PMI_1epoch(epoch, window_size, window_step)
    ARGS: epoch = Numpy shape = (n_channels,ts), 
    window_size = length of the time window, preferred range is [3, 7], window_step = time window step
    OUPTUT: PMI matrix
    '''
    assert len(epoch.shape) == 2, "epoch must be a 2D array"
    pmi = np.zeros([epoch.shape[0], epoch.shape[0]])
    ops = [ordinal_pattern_frequency(epoch[i], window_size, window_step) for i in range(epoch.shape[0])]
    entropies = [permutation_entropy(op) for op in ops]
    for i in range(epoch.shape[0]):
        for j in np.arange(i, epoch.shape[0]):
            op_xy = joint_ordinal_pattern_frequency((epoch[i], epoch[j]), window_size, window_step)
            p_x, p_y = entropies[i], entropies[j]
            p_xy = permutation_entropy(op_xy)
            pmi[i, j] = pmi[j, i] = p_x + p_y - p_xy
    return pmi

def pmi_all(epochs: NDArray[Shape["*, *, *"], Floating], window_size: int, window_step: int = 1):
    ''' This function computes the PMI for epochs.
    USAGE: PMI_epochs(epoch, window_size, window_step)
    ARGS: epochs = Numpy, shape = (n_epochs, n_channels, ts), 
    window_size = length of the time window, preferred range is [3, 7], window_step = time window step
    OUPTUT: PMI matrices
    '''
    pmis = np.zeros([epochs.shape[0], epochs.shape[1], epochs.shape[1]])
    for i in range(epochs.shape[0]):
        pmis[i] = pmi(epochs[i], window_size, window_step)
    return pmis

def spmi(epoch: NDArray[Shape["*, *"], Floating], window_size: int, window_step: int = 1):
    ''' This function computes the SPMI for an epoch.
    USAGE: SPMI_1epoch(epoch, window_size, window_step)
    ARGS: epoch = Numpy, shape = (n_channels, ts), 
    window_size = length of the time window, preferred range is [3, 7], window_step = time window step
    OUPTUT: PMI matrix
    '''
    assert len(epoch.shape) == 2, "epoch must be a 2D array"
    spmi = np.zeros([epoch.shape[0], epoch.shape[0]])
    ops = [ordinal_pattern_frequency(epoch[i], window_size, window_step) for i in range(epoch.shape[0])]
    entropies = [permutation_entropy(op) for op in ops]
    for i in range(epoch.shape[0]):
        for j in np.arange(i, epoch.shape[0]):
            op_xy = joint_ordinal_pattern_frequency((epoch[i], epoch[j]), window_size, window_step)
            p_x, p_y = entropies[i], entropies[j]
            p_xy = permutation_entropy(op_xy)
            spmi[i, j] = spmi[j, i] = (p_x + p_y) / p_xy - 1
    return spmi

def spmi_all(epochs: NDArray[Shape["*, *, *"], Floating], window_size: int, window_step: int = 1):
    ''' This function computes the SPMI for epochs.
    USAGE: SPMI_epochs(epochs, window_size, window_step)
    ARGS: epochs = Numpy, shape = (n_epochs, n_channels, ts), 
    window_size = length of the time window, preferred range is [3, 7], window_step = time window step
    OUPTUT: SPMI matrices
    '''
    spmis = np.zeros([epochs.shape[0], epochs.shape[1], epochs.shape[1]])
    for i in range(epochs.shape[0]):
        spmis[i] = spmi(epochs[i], window_size, window_step)
    return spmis
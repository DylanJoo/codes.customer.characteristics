import torch
import numpy as np

def npmapping(arr, mapping, reverse=False):
    if reverse:
        mapping = {v:k for (k, v) in mapping.items()}
    if ~isinstance(arr, np.ndarray):
        arr = np.array(arr)
    u, inv = np.unique(arr, return_inverse=True)
    arr_mapped = np.array([mapping[x] for x in u])[inv].reshape(arr.shape)
    return arr_mapped

def torchmapping(arr, mapping, reverse=False):
    if reverse:
        mapping = {v:k for (k, v) in mapping.items()}
    u, inv = torch.unique(arr, return_inverse=True)
    arr_mapped = np.array([mapping[x] for x in u])[inv].reshape(arr.shape)
    return arr_mapped

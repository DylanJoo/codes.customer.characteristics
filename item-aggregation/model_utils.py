import torch
import numpy as np
from transformers.utils import ModelOutput
from typing import List, Optional, Tuple, Union

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

class CustomizedOutput(ModelOutput):
    """
    Base class for outputs of sentence classification models.
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Classification (or regression if config.num_labels==1) loss.
        logits (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    importances: Optional[Tuple[torch.FloatTensor]] = None
    last_hidden_state: Optional[Tuple[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

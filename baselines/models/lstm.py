"""
file containing the definitions for the baseline LSTM model
"""
import torch
import torch.nn as nn
import models.model_utils as utils

from baselines.models.model_components.base_components import LSTM


device = "cuda" if torch.cuda.is_available() else "cpu"


class BaselineLSTM(nn.Module):
    def __init__(
        self,
        n_layers=6,
        input_dim=384,
        hidden_dim=20,
        kg_node_dim=100,
        dropout=0.1,
    ):
        pass

"""
file containing the definitions for the baseline BERT model
"""
import torch
import torch.nn as nn
import models.model_utils as utils

device = "cuda" if torch.cuda.is_available() else "cpu"


class BaselineBERT(nn.Module):
    """
    baseline BERT model
    """

    def __init__(
        self,
        n_heads=16,
        n_layers=6,
        input_dim=384,
        hidden_dim=20,
        dropout=0.1,
    ):
        nn.Module.__init__(self)
        # embed into hidden dim
        full_hidden_dim = hidden_dim * n_heads
        self.embedder = nn.Linear(input_dim, full_hidden_dim)
        # decider decides which sentences are most important in deciding how "incomplete" story is
        self.decider = nn.Transformer(
            nhead=n_heads,
            d_model=full_hidden_dim,
            batch_first=True,
            num_encoder_layers=n_layers,
            num_decoder_layers=n_layers,
            dropout=dropout,
        )
        # project feature space to single probability
        self.proj = nn.Linear(full_hidden_dim, 1)
        # sigmoid function to determine percentage of story cut off
        self.sigmoid = nn.Sigmoid()
        print(
            f"initialized unresolvedBERT with {utils.get_model_size(self)} parameters."
        )

    def forward(self, x):
        """
        :param x: sequence of sentence encodings from a story with shape (batch_size, seq_len, input_dim)
        :returns: single logit determining percentage of story that was left out
        """
        batch_size = x.shape[0]

        # embed input
        x = self.embedder(x)

        # obtain decider output
        x = self.decider(x, torch.zeros([x.shape[0], 1, x.shape[-1]]).to(device))

        # pass all output into projection layer
        x = self.proj(x)
        x = x.reshape([x.shape[0]])
        return self.sigmoid(x)

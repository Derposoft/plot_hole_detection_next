"""
file containing the definitions for the baseline LSTM model
"""
import torch
import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "cpu"


class BaselineLSTM(nn.Module):
    def __init__(
        self,
        n_layers=1,
        input_dim=384,
        hidden_dim=20,
        bidirectional=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.linear = nn.Linear(2 * hidden_dim if bidirectional else hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor, **kargs):
        batch_size = x.shape[0]
        x, h = self.lstm(x)
        x = self.linear(x)
        x = self.sigmoid(x)
        return x.reshape(batch_size, -1)

import torch
from torch import nn


class LSTMWorldModel(nn.Module):
    """Predict the next continuous network state from recent states."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.output = nn.Linear(
            hidden_size,
            input_size,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs, _ = self.lstm(inputs)
        return self.output(outputs[:, -1, :])

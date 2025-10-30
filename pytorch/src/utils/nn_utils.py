import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    Multi-layer perceptron network.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_size: int,
        hidden_layers: int,
        *,
        dropout_p: float = 0.0,
        dtype: torch.dtype,
    ):
        """
        Parameters
        ----------
            in_features : int
                Number of input features.

            out_features : int
                Number of output features.

            hidden_size : int
                Number of hidden units in each hidden layer.

            hidden_layers : int
                Number of hidden layers. Must be at least 1.

            dropout_p : float
                Dropout probability.    

            dtype : torch.dtype
                Data type of the linear layers. Must be a floating-point type.
        """
        super().__init__()
        assert hidden_layers >= 1
        assert dtype.is_floating_point

        layers = []

        # First hidden layer
        layers.append(nn.Linear(in_features, hidden_size, dtype=dtype))
        layers.append(nn.ReLU())
        if dropout_p > 0:
            layers.append(nn.Dropout(dropout_p))

        # Additional hidden layers
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size, dtype=dtype))
            layers.append(nn.ReLU())
            if dropout_p > 0:
                layers.append(nn.Dropout(dropout_p))

        # Output layer
        layers.append(nn.Linear(hidden_size, out_features, dtype=dtype))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

'''GNN architecture for QEC decoding:
Message passing on the Tanner graph.

Notations:
- hc[i,t] is the hidden state of CN i at iteration t.
- hv[j,t] is the hidden state of VN j at iteration t.
- mcv[i,j,t] is the message from CN i to VN j at iteration t.
- mvc[j,i,t] is the message from VN j to CN i at iteration t.
- mc[i,t] is the aggregated message at CN i at iteration t.
- mv[j,t] is the aggregated message at VN j at iteration t.

Initialization (t = 0): hc[i,0] = hv[j,0] = 0.

In each iteration t (1 <= t <= T), the model executes the following steps:
1. Calculate the messages:
   mcv[i,j,t] = MLP_CV(hc[i,t-1], hv[j,t-1])
   mvc[j,i,t] = MLP_VC(hc[i,t-1], hv[j,t-1])
2. Aggregate the messages:
   mc[i,t] = sum(mvc[j,i,t] for j ∈ N(i))
   mv[j,t] = sum(mcv[i,j,t] for i ∈ N(j))
3. Update the hidden states of the CNs and VNs:
   hc[i,t] = GRU_C0(mc[i,t], hc[i,t-1]) if synd[i] == 0
             GRU_C1(mc[i,t], hc[i,t-1]) if synd[i] == 1
   hv[j,t] = GRU_V(mv[j,t], hv[j,t-1])
where MLP_CV and MLP_VC are MLP networks, and GRU_C0, GRU_C1, and GRU_V are GRU cells.

Final prediction: For each VN j, output llr[j] = Linear(hv[j,T]).
'''

import numpy as np
import torch
import torch.nn as nn

FLOAT_DTYPE = torch.float32


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
        dropout_p: float = 0.0,
    ):
        super().__init__()
        assert hidden_layers >= 1

        layers = []

        # First hidden layer
        layers.append(nn.Linear(in_features, hidden_size, dtype=FLOAT_DTYPE))
        layers.append(nn.ReLU())
        if dropout_p > 0:
            layers.append(nn.Dropout(dropout_p))

        # Additional hidden layers
        for _ in range(hidden_layers - 1):
            layers.append(
                nn.Linear(hidden_size, hidden_size, dtype=FLOAT_DTYPE))
            layers.append(nn.ReLU())
            if dropout_p > 0:
                layers.append(nn.Dropout(dropout_p))

        # Output layer
        layers.append(nn.Linear(hidden_size, out_features, dtype=FLOAT_DTYPE))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class GNNDecoder(nn.Module):
    """
    Graph neural network-based decoder.
    """

    def __init__(
        self,
        pcm: np.ndarray,
        *,
        num_iters: int,
        node_features: int,
        edge_features: int,
        mlp_hidden_size: int,
        mlp_hidden_layers: int,
        mlp_dropout_p: float = 0.05,
        gru_dropout_p: float = 0.05,
    ):
        super().__init__()
        assert isinstance(pcm, np.ndarray)
        assert np.issubdtype(pcm.dtype, np.integer) or np.issubdtype(pcm.dtype, np.bool_)
        assert pcm.ndim == 2
        self.num_chks, self.num_vars = pcm.shape
        self.num_iters = num_iters
        self.node_features = node_features
        self.edge_features = edge_features
        self.mlp_hidden_size = mlp_hidden_size
        self.mlp_hidden_layers = mlp_hidden_layers
        self.mlp_dropout_p = mlp_dropout_p
        self.gru_dropout_p = gru_dropout_p

        # Register endpoints of the Tanner graph edges.
        # The i-th edge connects the CN chk_endpts[i] and the VN var_endpts[i].
        chk_endpts, var_endpts = np.nonzero(pcm)
        self.num_edges = len(chk_endpts)
        self.register_buffer("chk_endpts", torch.as_tensor(chk_endpts, dtype=torch.long))  # (num_edges,)
        self.register_buffer("var_endpts", torch.as_tensor(var_endpts, dtype=torch.long))  # (num_edges,)

        # Build MLP networks for message passing.
        # Input shape: (any, ..., 2 * node_features)
        # Output shape: (any, ..., edge_features)
        self.mlp_msg_v2c = MLP(
            in_features=2 * self.node_features,
            out_features=self.edge_features,
            hidden_size=self.mlp_hidden_size,
            hidden_layers=self.mlp_hidden_layers,
            dropout_p=self.mlp_dropout_p,
        )
        self.mlp_msg_c2v = MLP(
            in_features=2 * self.node_features,
            out_features=self.edge_features,
            hidden_size=self.mlp_hidden_size,
            hidden_layers=self.mlp_hidden_layers,
            dropout_p=self.mlp_dropout_p,
        )

        # Build GRU cells for node updates.
        # Input shape: (any, edge_features)
        # Hidden shape: (any, node_features)
        #
        # Note: Dropout only applies when we pass the hidden vector to downstream layers (i.e., node prediction).
        self.gru_v = nn.GRUCell(
            edge_features, node_features, dtype=FLOAT_DTYPE)
        self.gru_c0 = nn.GRUCell(
            edge_features, node_features, dtype=FLOAT_DTYPE)
        self.gru_c1 = nn.GRUCell(
            edge_features, node_features, dtype=FLOAT_DTYPE)
        self.gru_dropout = nn.Dropout(gru_dropout_p)

        # Build prediction layers.
        # Input shape: (any, ..., node_features)
        # Output shape: (any, ..., 1)
        self.pred = nn.Linear(node_features, 1, dtype=FLOAT_DTYPE)

    def forward(self, syndromes: torch.Tensor) -> list[torch.Tensor]:
        """
        Parameters
        ----------
            syndromes : torch.Tensor
                Syndrome bits ∈ {0,1}, shape=(batch_size, num_chks), int

        Returns
        -------
            var2llrs : list[torch.Tensor]
                A Python list of tensors, one for each VN, that stores the LLR values at all iterations. More 
                specifically, `var2llrs[j]` is a tensor of shape (batch_size, num_iters), such that `var2llrs[j][:, t]` 
                is the batch of LLR values for VN `j` at iteration `t`.
        """
        device = syndromes.device
        batch_size = syndromes.shape[0]
        syndromes_flat = syndromes.view(batch_size * self.num_chks)
        mask0 = syndromes_flat == 0  # (batch_size * num_chks,), torch.bool
        mask1 = syndromes_flat == 1  # (batch_size * num_chks,), torch.bool

        # A nested list that will store the LLR values for each VN at each iteration.
        # The outer list is indexed by VN, and the inner list is indexed by iteration.
        # Each element will be a tensor of shape (batch_size,).
        var2iter2llrs: list[list[torch.Tensor]] = [
            [
                None  # placeholder; will be a tensor of shape (batch_size,)
                for _ in range(self.num_iters)
            ]
            for _ in range(self.num_vars)
        ]

        # Initialize hidden states to be all zeros at t=0
        hc = torch.zeros(batch_size, self.num_chks, self.node_features,
                         device=device, dtype=FLOAT_DTYPE)
        hv = torch.zeros(batch_size, self.num_vars, self.node_features,
                         device=device, dtype=FLOAT_DTYPE)

        for t in range(self.num_iters):
            # Gather a pair of node features for each edge.
            paired_hc_hv = torch.cat([
                torch.index_select(hc, dim=1, index=self.chk_endpts),
                torch.index_select(hv, dim=1, index=self.var_endpts)
            ], dim=-1)  # (batch_size, num_edges, 2 * node_features)

            # Calculate messages along edges in both directions using MLP networks.
            mvc = self.mlp_msg_v2c(paired_hc_hv)  # (batch_size, num_edges, edge_features)
            mcv = self.mlp_msg_c2v(paired_hc_hv)  # (batch_size, num_edges, edge_features)

            # Aggregate messages by summing over all incoming edges for each node.
            mv = torch.zeros(batch_size, self.num_vars, self.edge_features,
                             device=device, dtype=FLOAT_DTYPE)
            mv.index_add_(1, self.var_endpts, mcv)  # mv[:, var_endpts[e], :] += mcv[:, e, :] for all edges e.
            mc = torch.zeros(batch_size, self.num_chks, self.edge_features,
                             device=device, dtype=FLOAT_DTYPE)
            mc.index_add_(1, self.chk_endpts, mvc)  # mc[:, chk_endpts[e], :] += mvc[:, e, :] for all edges e.

            # Update variable node features using GRU cells.
            mv_flat = mv.view(batch_size * self.num_vars, self.edge_features)
            hv_flat = hv.view(batch_size * self.num_vars, self.node_features)
            hv_flat: torch.Tensor = self.gru_v(mv_flat, hv_flat)
            hv = hv_flat.view(batch_size, self.num_vars, self.node_features)

            # Update check node features using syndrome-conditioned GRU cells.
            mc_flat = mc.view(batch_size * self.num_chks, self.edge_features)
            hc_flat = hc.view(batch_size * self.num_chks, self.node_features)
            tmp = torch.zeros_like(hc_flat)
            if mask0.any():
                tmp[mask0] = self.gru_c0(mc_flat[mask0], hc_flat[mask0])
            if mask1.any():
                tmp[mask1] = self.gru_c1(mc_flat[mask1], hc_flat[mask1])
            hc = tmp.view(batch_size, self.num_chks, self.node_features)

            # Prediction at variable nodes.
            hv_drop = self.gru_dropout(hv)
            llrs = self.pred(hv_drop).squeeze(-1)  # (batch_size, num_vars)
            for j, llr in enumerate(torch.unbind(llrs, dim=1)):
                var2iter2llrs[j][t] = llr

        # Convert the nested list var2iter2llrs into a list of tensors, one for each VN.
        # Each tensor has shape (batch_size, num_iters).
        var2llrs = [
            torch.stack(var2iter2llrs[j], dim=1)
            for j in range(self.num_vars)
        ]
        return var2llrs


__all__ = [
    "GNNDecoder",
]

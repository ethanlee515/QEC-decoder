from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from qecdec.utils import build_tanner_graph

FLOAT_DTYPE = torch.float32
EPS = 1e-6


class _NodeMLP(nn.Module):
    """Linear -> GeLU -> Linear with configurable hidden expansion."""
    def __init__(self, in_dim: int, out_dim: int, hidden_mult: int = 4):
        super().__init__()
        hidden = max(1, hidden_mult * in_dim)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Learned_HighDimBPDecoder(nn.Module):
    """
    Learned high-dimensional message passing on a fixed Tanner graph.

    - Edge messages are vectors in R^d.
    - Each check node i has its own MLP: (deg_i * d) -> (deg_i * d).
    - Each variable node j has its own MLP:
          (deg_j * d + 1) -> (deg_j * d + 1),
      where the extra scalar is the LLR.
    """

    def __init__(
        self,
        pcm: np.ndarray,
        prior: np.ndarray,
        *,
        msg_dim: int,
        hidden_mult: int = 4,
        num_iters: int = 5,
        init_scheme: str = "prior_first_component",
    ):
        super().__init__()

        assert pcm.ndim == 2
        self.num_chks, self.num_vars = pcm.shape
        assert prior.shape == (self.num_vars,)

        self.msg_dim = msg_dim
        self.hidden_mult = hidden_mult
        self.default_num_iters = num_iters
        self.init_scheme = init_scheme

        # Tanner graph
        self.chk_nbrs, self.var_nbrs, self.chk_nbr_pos, self.var_nbr_pos = \
            build_tanner_graph(pcm)

        # Prior LLRs
        prior = np.clip(prior, min=EPS, max=1.0 - EPS)
        prior_llr = np.log((1.0 - prior) / prior)
        self.register_buffer(
            "prior_llr",
            torch.as_tensor(prior_llr, dtype=FLOAT_DTYPE),
        )

        d = msg_dim

        # Per-check-node MLPs
        self.chk_mlps = nn.ModuleList()
        for i in range(self.num_chks):
            deg = len(self.chk_nbrs[i])
            self.chk_mlps.append(
                _NodeMLP(deg * d, deg * d, hidden_mult)
            )

        # Per-variable-node MLPs (+1 for LLR)
        self.var_mlps = nn.ModuleList()
        for j in range(self.num_vars):
            deg = len(self.var_nbrs[j])
            self.var_mlps.append(
                _NodeMLP(deg * d + 1, deg * d + 1, hidden_mult)
            )

    def _init_edge_message(
        self,
        batch_size: int,
        device: torch.device,
        llr_scalar: torch.Tensor,
    ) -> torch.Tensor:
        d = self.msg_dim
        if self.init_scheme == "zeros":
            return torch.zeros((batch_size, d), device=device, dtype=FLOAT_DTYPE)

        # Put prior LLR in first component
        msg = torch.zeros((batch_size, d), device=device, dtype=FLOAT_DTYPE)
        msg[:, 0] = llr_scalar.to(FLOAT_DTYPE)
        return msg

    def forward(
        self,
        syndromes: torch.Tensor,
        *,
        num_iters: int | None = None,
    ) -> torch.Tensor:
        """
        syndromes: (batch_size, num_chks) in {0,1}
        returns: (num_iters, batch_size, num_vars) LLRs
        """
        T = num_iters if num_iters is not None else self.default_num_iters

        device = syndromes.device
        batch_size = syndromes.shape[0]
        d = self.msg_dim

        synd_sgn = (1 - 2 * syndromes).to(FLOAT_DTYPE)  # (B, M)

        iter_llrs = torch.empty(
            (T, batch_size, self.num_vars),
            device=device,
            dtype=FLOAT_DTYPE,
        )

        # Initialize messages VN->CN
        chk_inmsg: list[list[torch.Tensor]] = []
        for i in range(self.num_chks):
            row = []
            for j in self.chk_nbrs[i]:
                llr_j = self.prior_llr[j].expand(batch_size)
                row.append(self._init_edge_message(batch_size, device, llr_j))
            chk_inmsg.append(row)

        # CN->VN messages
        var_inmsg: list[list[torch.Tensor]] = [
            [torch.zeros((batch_size, d), device=device, dtype=FLOAT_DTYPE)
             for _ in self.var_nbrs[j]]
            for j in range(self.num_vars)
        ]

        for t in range(T):
            # ---------- Check node update ----------
            for i in range(self.num_chks):
                deg = len(self.chk_nbrs[i])
                if deg == 0:
                    continue

                x = torch.stack(chk_inmsg[i], dim=1).reshape(batch_size, deg * d)
                y = self.chk_mlps[i](x).reshape(batch_size, deg, d)
                y = y * synd_sgn[:, i].view(batch_size, 1, 1)

                for k, (j, pos) in enumerate(
                    zip(self.chk_nbrs[i], self.chk_nbr_pos[i])
                ):
                    var_inmsg[j][pos] = y[:, k, :]

            # ---------- Variable node update ----------
            for j in range(self.num_vars):
                deg = len(self.var_nbrs[j])
                if deg == 0:
                    iter_llrs[t, :, j] = self.prior_llr[j]
                    continue

                inc = torch.stack(var_inmsg[j], dim=1).reshape(batch_size, deg * d)
                prior_j = self.prior_llr[j].expand(batch_size).unsqueeze(1)
                x = torch.cat([inc, prior_j], dim=1)

                y = self.var_mlps[j](x)
                msg_block = y[:, : deg * d].reshape(batch_size, deg, d)
                llr = y[:, deg * d]

                iter_llrs[t, :, j] = llr

                if t < T - 1:
                    for k, (i, pos) in enumerate(
                        zip(self.var_nbrs[j], self.var_nbr_pos[j])
                    ):
                        chk_inmsg[i][pos] = msg_block[:, k, :]

        return iter_llrs

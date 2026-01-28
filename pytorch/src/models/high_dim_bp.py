from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
import torch.nn as nn

from qecdec.utils import build_tanner_graph  # same as Yingkang's code

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


@dataclass(frozen=True)
class LearnedNodeMLPBPConfig:
    msg_dim: int                 # d
    num_iters: int               # T
    init_scheme: Literal["prior_first_component", "zeros"] = "prior_first_component"
    hidden_mult: int = 4         # "4x" as you suggested


class Learned_NodeMLPBPDecoder(nn.Module):
    """
    Learned message passing on a fixed Tanner graph:
      - Edge messages are vectors in R^d.
      - Each check node i has its own MLP: (deg_i*d) -> (deg_i*d).
      - Each variable node j has its own MLP: (deg_j*d + 1) -> (deg_j*d + 1),
        where the +1 scalar is the LLR (we feed in prior_llr; we output posterior llr).
    """

    def __init__(self, pcm: np.ndarray, prior: np.ndarray, *, cfg: LearnedNodeMLPBPConfig):
        super().__init__()
        assert isinstance(pcm, np.ndarray) and isinstance(prior, np.ndarray)
        assert pcm.ndim == 2
        self.num_chks, self.num_vars = pcm.shape
        assert prior.shape == (self.num_vars,)
        self.cfg = cfg

        # Build Tanner graph (same fields as Yingkang's implementation).
        self.chk_nbrs, self.var_nbrs, self.chk_nbr_pos, self.var_nbr_pos = build_tanner_graph(pcm)

        # Register prior LLRs.
        prior = np.clip(prior, min=EPS, max=1.0 - EPS)
        prior_llr = np.log((1.0 - prior) / prior)
        self.register_buffer("prior_llr", torch.as_tensor(prior_llr, dtype=FLOAT_DTYPE))  # (num_vars,)

        d = cfg.msg_dim

        # Per-check-node MLPs, degree-dependent input/output sizes.
        self.chk_mlps = nn.ModuleList()
        for i in range(self.num_chks):
            deg = len(self.chk_nbrs[i])
            in_dim = deg * d
            out_dim = deg * d
            self.chk_mlps.append(_NodeMLP(in_dim, out_dim, hidden_mult=cfg.hidden_mult))

        # Per-variable-node MLPs, degree-dependent (+1 scalar for LLR).
        self.var_mlps = nn.ModuleList()
        for j in range(self.num_vars):
            deg = len(self.var_nbrs[j])
            in_dim = deg * d + 1
            out_dim = deg * d + 1
            self.var_mlps.append(_NodeMLP(in_dim, out_dim, hidden_mult=cfg.hidden_mult))

    def _init_edge_message(self, batch_size: int, device: torch.device, llr_scalar: torch.Tensor) -> torch.Tensor:
        """
        Initialize an edge message vector in R^d for each sample in batch.
        llr_scalar: (batch_size,)
        returns: (batch_size, d)
        """
        d = self.cfg.msg_dim
        if self.cfg.init_scheme == "zeros":
            return torch.zeros((batch_size, d), device=device, dtype=FLOAT_DTYPE)

        # Default: put prior LLR in the first component, rest zeros.
        msg = torch.zeros((batch_size, d), device=device, dtype=FLOAT_DTYPE)
        msg[:, 0] = llr_scalar.to(FLOAT_DTYPE)
        return msg

    def forward(self, syndromes: torch.Tensor) -> torch.Tensor:
        """
        syndromes: (batch_size, num_chks) with entries in {0,1} (int/bool ok)
        returns: llrs over iterations, shape (num_iters, batch_size, num_vars)
        """
        device = syndromes.device
        batch_size = syndromes.shape[0]
        d = self.cfg.msg_dim
        T = self.cfg.num_iters

        # Syndrome sign in {+1, -1}.
        synd_sgn = (1 - 2 * syndromes).to(FLOAT_DTYPE)  # (B, M)

        # Storage for outputs (T, B, N).
        iter_llrs = torch.empty((T, batch_size, self.num_vars), device=device, dtype=FLOAT_DTYPE)

        # Messages:
        #   chk_inmsg[i][k] : message VN->CN on edge (i <- nbrs[k]), shape (B, d)
        #   var_inmsg[j][k] : message CN->VN on edge (j <- nbrs[k]), shape (B, d)
        chk_inmsg: list[list[torch.Tensor]] = []
        for i in range(self.num_chks):
            row = []
            for j in self.chk_nbrs[i]:
                llr_j = self.prior_llr[j].expand(batch_size)  # (B,)
                row.append(self._init_edge_message(batch_size, device, llr_j))
            chk_inmsg.append(row)

        var_inmsg: list[list[torch.Tensor]] = [
            [torch.zeros((batch_size, d), device=device, dtype=FLOAT_DTYPE) for _ in self.var_nbrs[j]]
            for j in range(self.num_vars)
        ]

        # Main iterations.
        for t in range(T):
            # ------------------ CN update ------------------
            for i in range(self.num_chks):
                deg = len(self.chk_nbrs[i])
                if deg == 0:
                    continue

                # (B, deg, d) -> (B, deg*d)
                x = torch.stack(chk_inmsg[i], dim=1).reshape(batch_size, deg * d)
                y = self.chk_mlps[i](x).reshape(batch_size, deg, d)  # (B, deg, d)

                # Inject syndrome influence without changing dimensions.
                y = y * synd_sgn[:, i].view(batch_size, 1, 1)

                # Write CN->VN messages into var_inmsg using the precomputed neighbor positions.
                for k, (j, pos_in_vn) in enumerate(zip(self.chk_nbrs[i], self.chk_nbr_pos[i])):
                    var_inmsg[j][pos_in_vn] = y[:, k, :]

            # ------------------ VN update ------------------
            for j in range(self.num_vars):
                deg = len(self.var_nbrs[j])
                if deg == 0:
                    # Isolated variable node: posterior just prior.
                    iter_llrs[t, :, j] = self.prior_llr[j]
                    continue

                # Flatten incoming CN->VN messages: (B, deg, d) -> (B, deg*d)
                inc = torch.stack(var_inmsg[j], dim=1).reshape(batch_size, deg * d)

                # Append prior LLR scalar (B,1)
                prior_j = self.prior_llr[j].expand(batch_size).to(FLOAT_DTYPE).unsqueeze(1)
                x = torch.cat([inc, prior_j], dim=1)  # (B, deg*d + 1)

                y = self.var_mlps[j](x)  # (B, deg*d + 1)
                msg_block = y[:, : deg * d].reshape(batch_size, deg, d)  # (B, deg, d)
                llr = y[:, deg * d]  # (B,)

                iter_llrs[t, :, j] = llr

                # Outgoing VN->CN messages for next CN update (skip last iter).
                if t < T - 1:
                    for k, (i, pos_in_cn) in enumerate(zip(self.var_nbrs[j], self.var_nbr_pos[j])):
                        chk_inmsg[i][pos_in_cn] = msg_block[:, k, :]

        return iter_llrs

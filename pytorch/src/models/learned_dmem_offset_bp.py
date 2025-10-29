from typing import Literal, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from qecdec.utils import build_tanner_graph

EPS = 1e-6
BIG = 1e8
FLOAT_DTYPE = torch.float32


class Learned_DMemOffsetBPDecoder(nn.Module):
    """
    A PyTorch Module that implements a Disordered Memory Offset normalized min-sum BP decoder with 
    trainable memory strength, trainable offset parameters, and trainable normalization factors.
    """

    def __init__(
        self,
        pcm: np.ndarray,
        prior: np.ndarray,
        *,
        num_iters: int,
        train_offset: bool = True,
        train_nf: bool = True,
        min_impl_method: Literal["smooth", "hard"] = "smooth",
        sign_impl_method: Literal["smooth", "hard"] = "smooth",
        gamma_init: Optional[np.ndarray] = None,
    ):
        """
        Parameters
        ----------
            pcm : ndarray
                Parity-check matrix ∈ {0,1}, shape=(num_chks, num_vars), integer or bool

            prior : ndarray
                Prior probabilities of errors, shape=(num_vars,), float

            num_iters : int
                Number of BP iterations.

            train_offset : bool
                Whether to train the offset parameters. If False, the offset parameters are fixed to 0.0.

            train_nf : bool
                Whether to train the normalization factors. If False, the normalization factors are fixed to 1.0.

            min_impl_method : Literal["smooth", "hard"]
                Implementation method of the min function. Can be "smooth" (based on softmin) or "hard" (using torch.amin).

            sign_impl_method : Literal["smooth", "hard"]
                Implementation method of the sign function. Can be "smooth" (based on tanh) or "hard" (using torch.sign).

            gamma_init : ndarray | None
                Initial memory strength, shape=(num_vars,), float. If None, the memory strength is initialized to 0.0.
        """
        super().__init__()
        assert isinstance(pcm, np.ndarray) and isinstance(prior, np.ndarray)
        assert np.issubdtype(pcm.dtype, np.integer) or np.issubdtype(pcm.dtype, np.bool_)
        assert np.issubdtype(prior.dtype, np.floating)
        assert pcm.ndim == 2
        self.num_chks, self.num_vars = pcm.shape
        assert prior.shape == (self.num_vars,)
        self.num_iters = num_iters
        self.train_offset = train_offset
        self.train_nf = train_nf

        if min_impl_method == "smooth":
            from ..utils.tensor_ops import smooth_min
            self.min_func = smooth_min
        elif min_impl_method == "hard":
            self.min_func = torch.amin
        else:
            raise ValueError(f"Invalid min_impl_method: {min_impl_method}")

        if sign_impl_method == "smooth":
            from ..utils.tensor_ops import smooth_sign
            self.sign_func = smooth_sign
        elif sign_impl_method == "hard":
            self.sign_func = torch.sign
        else:
            raise ValueError(f"Invalid sign_impl_method: {sign_impl_method}")

        # Build Tanner graph.
        # - chk_nbrs[i] = list of all VNs connected to CN i, sorted in increasing order.
        # - var_nbrs[j] = list of all CNs connected to VN j, sorted in increasing order.
        # - chk_nbr_pos[i][k] = position of CN i in the list of neighbors of the VN chk_nbrs[i][k].
        #       i.e., var_nbrs[chk_nbrs[i][k]][chk_nbr_pos[i][k]] = i.
        # - var_nbr_pos[j][k] = position of VN j in the list of neighbors of the CN var_nbrs[j][k].
        #       i.e., chk_nbrs[var_nbrs[j][k]][var_nbr_pos[j][k]] = j.
        self.chk_nbrs, self.var_nbrs, self.chk_nbr_pos, self.var_nbr_pos = build_tanner_graph(pcm)

        # Register prior LLRs.
        prior = np.clip(prior, min=EPS, max=1 - EPS)
        prior_llr = np.log((1 - prior) / prior)
        self.register_buffer("prior_llr",
                             torch.as_tensor(prior_llr, dtype=FLOAT_DTYPE))  # (num_vars,)

        # Register trainable parameters.
        if gamma_init is None:
            self.gamma = nn.ParameterList([
                nn.Parameter(torch.zeros((), dtype=FLOAT_DTYPE))
                for _ in range(self.num_vars)
            ])  # (num_vars,)
        else:
            assert isinstance(gamma_init, np.ndarray)
            assert np.issubdtype(gamma_init.dtype, np.floating)
            assert gamma_init.shape == (self.num_vars,)
            self.gamma = nn.ParameterList([
                nn.Parameter(torch.as_tensor(gamma_init[i], dtype=FLOAT_DTYPE))
                for i in range(self.num_vars)
            ])  # (num_vars,)
        if train_offset:
            self.offset = nn.ParameterList([
                nn.Parameter(torch.zeros(len(self.chk_nbrs[i]), dtype=FLOAT_DTYPE))
                for i in range(self.num_chks)
            ])
        if train_nf:
            self.nf = nn.ParameterList([
                nn.Parameter(torch.ones(len(self.chk_nbrs[i]), dtype=FLOAT_DTYPE))
                for i in range(self.num_chks)
            ])

    def forward(self, syndromes: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
            syndromes : torch.Tensor
                Syndrome bits ∈ {0,1}, shape=(batch_size, num_chks), int

        Returns
        -------
            var2llrs : list[torch.Tensor]
                A Python list of tensors, one for each VN, that stores the posterior LLRs at all BP iterations. More 
                specifically, `var2llrs[j]` is a tensor of shape (batch_size, num_iters), such that `var2llrs[j][:, t]` 
                is the batch of posterior LLRs for VN `j` at BP iteration `t`.
        """
        device = syndromes.device
        batch_size = syndromes.shape[0]
        synd_sgn = (1 - 2 * syndromes).to(FLOAT_DTYPE)  # (batch_size, num_chks) ∈ {+1,-1}

        # A nested list that will store the posterior LLRs for each VN at each BP iteration.
        # The outer list is indexed by VN, and the inner list is indexed by BP iteration.
        # Each element will be a tensor of shape (batch_size,).
        var2iter2llrs: list[list[torch.Tensor]] = [
            [
                None  # placeholder; will be a tensor of shape (batch_size,)
                for _ in range(self.num_iters)
            ]
            for _ in range(self.num_vars)
        ]

        # Initialize messages.
        # chk_inmsg[i][k] = incoming message at CN i from its k-th neighbor, shape=(batch_size,)
        # var_inmsg[j][k] = incoming message at VN j from its k-th neighbor, shape=(batch_size,)
        chk_inmsg: list[list[torch.Tensor]] = [
            [
                torch.full((batch_size,), self.prior_llr[j],
                           device=device, dtype=FLOAT_DTYPE)
                for j in self.chk_nbrs[i]
            ]
            for i in range(self.num_chks)
        ]
        var_inmsg: list[list[torch.Tensor]] = [
            [
                None  # placeholder; will be a tensor of shape (batch_size,)
                for _ in self.var_nbrs[j]
            ]
            for j in range(self.num_vars)
        ]

        # Main BP iteration loop
        for t in range(self.num_iters):
            # ------------------ CN update ------------------
            for i in range(self.num_chks):
                nbrs = self.chk_nbrs[i]
                nbr_pos = self.chk_nbr_pos[i]
                num_nbrs = len(nbrs)

                # Gather incoming messages at CN i.
                msgs = torch.stack(chk_inmsg[i], dim=1)  # (batch_size, num_nbrs)
                msgs_abs = msgs.abs()  # (batch_size, num_nbrs)
                msgs_sgn = self.sign_func(msgs)  # (batch_size, num_nbrs)

                # For each neighboring VN, compute product over msgs_sgn excluding that VN.
                # We achieve leave-one-out by masking the corresponding entry with 1.0.
                msgs_sgn_repeated = msgs_sgn \
                    .unsqueeze(dim=1) \
                    .repeat(1, num_nbrs, 1)  # (batch_size, num_nbrs, num_nbrs)
                mask = torch.eye(num_nbrs, device=device, dtype=torch.bool) \
                    .unsqueeze(dim=0)  # (1, num_nbrs, num_nbrs)
                msgs_sgn_masked = msgs_sgn_repeated.masked_fill(mask, 1.0)  # (batch_size, num_nbrs, num_nbrs)
                msgs_sgn_prod_excl = msgs_sgn_masked.prod(dim=2)  # (batch_size, num_nbrs)

                # For each neighboring VN, compute min over msgs_abs excluding that VN.
                # We achieve leave-one-out by masking the corresponding entry with a large number.
                msgs_abs_repeated = msgs_abs \
                    .unsqueeze(dim=1) \
                    .repeat(1, num_nbrs, 1)  # (batch_size, num_nbrs, num_nbrs)
                msgs_abs_masked = msgs_abs_repeated.masked_fill(mask, BIG)  # (batch_size, num_nbrs, num_nbrs)
                msgs_abs_min_excl = self.min_func(msgs_abs_masked, dim=2)  # (batch_size, num_nbrs)

                # Compute outgoing messages at CN i and update var_inmsg.
                out = torch.unbind(
                    synd_sgn[:, i].unsqueeze(dim=1)
                    * msgs_sgn_prod_excl
                    * (
                        F.relu(msgs_abs_min_excl -
                               self.offset[i].unsqueeze(dim=0))
                        if self.train_offset else msgs_abs_min_excl
                    ) * (
                        self.nf[i].unsqueeze(dim=0)
                        if self.train_nf else 1.0
                    ),
                    dim=1
                )  # tuple of num_nbrs tensors, each of shape (batch_size,)
                for k, (j, pos) in enumerate(zip(nbrs, nbr_pos)):
                    var_inmsg[j][pos] = out[k]

            # ------------------ VN update ------------------
            for j in range(self.num_vars):
                # Gather and sum incoming messages at VN j.
                incoming_sum = torch.stack(var_inmsg[j], dim=1).sum(dim=1)  # (batch_size,)

                # Calculate posterior LLR at VN j for the current BP iteration.
                if t == 0:
                    var2iter2llrs[j][t] = incoming_sum + self.prior_llr[j]
                else:
                    var2iter2llrs[j][t] = incoming_sum + \
                        (1 - self.gamma[j]) * self.prior_llr[j] + \
                        self.gamma[j] * var2iter2llrs[j][t - 1]

                # Compute outgoing messages at VN j and update chk_inmsg.
                if t < self.num_iters - 1:  # no need to update in the last iteration
                    for k, (i, pos) in enumerate(zip(self.var_nbrs[j], self.var_nbr_pos[j])):
                        chk_inmsg[i][pos] = var2iter2llrs[j][t] - var_inmsg[j][k]

        # Convert the nested list var2iter2llrs into a list of tensors, one for each VN.
        # Each tensor has shape (batch_size, num_iters).
        var2llrs = [
            torch.stack(var2iter2llrs[j], dim=1)
            for j in range(self.num_vars)
        ]
        return var2llrs


__all__ = [
    "Learned_DMemOffsetBPDecoder",
]

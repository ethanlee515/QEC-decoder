from enum import Enum

import numpy as np
import torch
import torch.nn.functional as F

EPS = 1e-6
FLOAT_DTYPE = torch.float32


class LossType(Enum):
    OBS_ONLY = 0
    SYND_ONLY = 1
    HYBRID = 2


class IterativeDecodingLoss:
    """
    A PyTorch Module that implements a loss function for training iterative QEC decoders.

    Given a check matrix `chkmat` and an observable matrix `obsmat`, the loss function consists of two parts:
    1. The first part quantifies how the estimated error pattern recovers the syndrome (i.e., are we back to the code space?).
    2. The second part quantifies how the estimated error pattern predicts the observable (i.e., is there a logical error?).

    More specifically, suppose we want to calculate the loss for a single shot (`llr`, `syndrome`, `observable`). The loss from part 1 is 
    `loss1 = mean(loss_syn)`, where `loss_syn[i] = BCEWithLogitsLoss(-syndrome_pred_llr[i], syndrome[i])`, where `syndrome_pred_llr[i]` is the 
    LLR value of the `i`-th syndrome bit calculated from the LLR values of those error bits corresponding to the `i`-th row of `chkmat`. 
    Similarly, the loss from part 2 is `loss2 = mean(loss_obs)`, where `loss_obs[i] = BCEWithLogitsLoss(-observable_pred_llr[i], observable[i])`, 
    where `observable_pred_llr[i]` is the LLR value of the `i`-th observable bit calculated from the LLR values of those error bits corresponding 
    to the `i`-th row of `obsmat`. Finally, the total loss is `loss = ß * loss1 + (1-ß) * loss2`, where `ß` ∈ [0,1] is a hyperparameter that 
    controls the relative importance of the two parts.

    To calculate the loss for a batch of shots, we average the loss of each shot over the batch.
    """

    def __init__(
        self,
        chkmat: np.ndarray,
        obsmat: np.ndarray,
        *,
        beta: float = 1.0,
        skip_iters: int = 0,
    ):
        """
        Parameters
        ----------
            chkmat : ndarray
                Check matrix ∈ {0,1}, shape=(num_chks, num_vars), integer or bool

            obsmat : ndarray
                Observable matrix ∈ {0,1}, shape=(num_obsers, num_vars), integer or bool

            beta : float
                Hyperparameter that balances the contribution of the two parts of the loss function.
                Default is 1.0, meaning only the first part (corresponding to the syndrome recovery) is considered.

            skip_iters : int
                The first `skip_iters` iterations are skipped in the calculation of the loss.
                Default is 0, meaning that the LLRs output from all iterations contribute to the loss.
        """
        assert isinstance(chkmat, np.ndarray) and isinstance(obsmat, np.ndarray)
        assert np.issubdtype(chkmat.dtype, np.integer) or np.issubdtype(chkmat.dtype, np.bool_)
        assert np.issubdtype(obsmat.dtype, np.integer) or np.issubdtype(obsmat.dtype, np.bool_)
        assert chkmat.ndim == 2 and obsmat.ndim == 2
        assert chkmat.shape[1] == obsmat.shape[1]
        assert 0 <= beta <= 1
        self.num_chks, self.num_vars = chkmat.shape
        self.num_obsers = obsmat.shape[0]
        if beta < EPS:
            self.type = LossType.OBS_ONLY
        elif beta > 1 - EPS:
            self.type = LossType.SYND_ONLY
        else:
            self.type = LossType.HYBRID
            self.beta = beta
        self.skip_iters = skip_iters

        # Obtain the support of each row of the check matrix and observable matrix.
        # - chk_supp[i] = support of the i-th row of chkmat
        # - obs_supp[i] = support of the i-th row of obsmat
        self.chk_supp = [
            [j for j in range(self.num_vars) if chkmat[i, j]]
            for i in range(self.num_chks)
        ]
        self.obs_supp = [
            [j for j in range(self.num_vars) if obsmat[i, j]]
            for i in range(self.num_obsers)
        ]

    def forward(
        self,
        llrs: torch.Tensor,
        syndromes: torch.Tensor,
        observables: torch.Tensor
    ) -> torch.Tensor:
        """
        Parameters
        ----------
            llrs : torch.Tensor
                LLR values at all iterations, shape=(num_iters, batch_size, num_vars), float

            syndromes : torch.Tensor
                Syndrome bits ∈ {0,1}, shape=(batch_size, num_chks), int

            observables : torch.Tensor
                Observable bits ∈ {0,1}, shape=(batch_size, num_obsers), int

        Returns
        -------
            loss : torch.Tensor
                Loss, shape=(), float
        """
        if self.skip_iters > 0:
            assert self.skip_iters < llrs.shape[0]
            llrs = llrs[self.skip_iters:, :, :]

        tanhhalfllrs = torch.tanh(llrs / 2.)  # (num_iters, batch_size, num_vars)

        # Unbind the third dimension to get a tuple of tensors, one for each VN.
        # Each tensor has shape (num_iters, batch_size).
        var2tanhhalfllrs = torch.unbind(tanhhalfllrs, dim=2)

        # Compute loss from two parts.
        if self.type != LossType.OBS_ONLY:
            loss1 = self._get_syndrome_loss(var2tanhhalfllrs, syndromes)  # (num_iters, batch_size)
        if self.type != LossType.SYND_ONLY:
            loss2 = self._get_observable_loss(var2tanhhalfllrs, observables)  # (num_iters, batch_size)

        # Compute total loss.
        if self.type == LossType.SYND_ONLY:
            loss = loss1
        elif self.type == LossType.OBS_ONLY:
            loss = loss2
        else:
            loss = self.beta * loss1 + (1 - self.beta) * loss2
        return loss.mean()

    def _get_syndrome_loss(
        self,
        var2tanhhalfllrs: tuple[torch.Tensor, ...],  # tuple of num_vars tensors, each of shape (num_iters, batch_size)
        syndromes: torch.Tensor  # (batch_size, num_chks), int, values ∈ {0,1}
    ) -> torch.Tensor:
        synd_pred_llr = []
        for i in range(self.num_chks):
            synd_i_pred_llr = 2 * (
                torch.stack([var2tanhhalfllrs[j] for j in self.chk_supp[i]], dim=2)
                .prod(dim=2)
                .clamp(min=-1 + EPS, max=1 - EPS)
                .atanh()
            )  # (num_iters, batch_size)
            synd_pred_llr.append(synd_i_pred_llr)
        synd_pred_llr = torch.stack(synd_pred_llr, dim=2)  # (num_iters, batch_size, num_chks)

        loss = F.binary_cross_entropy_with_logits(
            -synd_pred_llr,
            syndromes.to(FLOAT_DTYPE).unsqueeze(dim=0).expand_as(synd_pred_llr),
            reduction="none"
        ).mean(dim=2)  # (num_iters, batch_size)
        return loss

    def _get_observable_loss(
        self,
        var2tanhhalfllrs: tuple[torch.Tensor, ...],  # tuple of num_vars tensors, each of shape (num_iters, batch_size)
        observables: torch.Tensor  # (batch_size, num_obsers), int, values ∈ {0,1}
    ) -> torch.Tensor:
        obs_pred_llr = []
        for i in range(self.num_obsers):
            obs_i_pred_llr = 2 * (
                torch.stack([var2tanhhalfllrs[j] for j in self.obs_supp[i]], dim=2)
                .prod(dim=2)
                .clamp(min=-1 + EPS, max=1 - EPS)
                .atanh()
            )  # (num_iters, batch_size)
            obs_pred_llr.append(obs_i_pred_llr)
        obs_pred_llr = torch.stack(obs_pred_llr, dim=2)  # (num_iters, batch_size, num_obsers)

        loss = F.binary_cross_entropy_with_logits(
            -obs_pred_llr,
            observables.to(FLOAT_DTYPE).unsqueeze(dim=0).expand_as(obs_pred_llr),
            reduction="none"
        ).mean(dim=2)  # (num_iters, batch_size)
        return loss

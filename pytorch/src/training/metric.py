import numpy as np
import torch
from torchmetrics import Metric

INT_DTYPE = torch.int32


class DecodingMetric(Metric):
    """
    A PyTorch Metric that calculates the performance metrics of the decoder.
    """

    def __init__(
        self,
        chkmat: np.ndarray,
        obsmat: np.ndarray,
    ):
        """
        Parameters
        ----------
            chkmat : ndarray
                Check matrix ∈ {0,1}, shape=(num_chks, num_vars), integer or bool

            obsmat : ndarray
                Observable matrix ∈ {0,1}, shape=(num_obsers, num_vars), integer or bool
        """
        super().__init__()
        assert isinstance(chkmat, np.ndarray) and isinstance(obsmat, np.ndarray)
        assert np.issubdtype(chkmat.dtype, np.integer) or np.issubdtype(chkmat.dtype, np.bool_)
        assert np.issubdtype(obsmat.dtype, np.integer) or np.issubdtype(obsmat.dtype, np.bool_)
        assert chkmat.ndim == 2 and obsmat.ndim == 2
        assert chkmat.shape[1] == obsmat.shape[1]

        self.register_buffer("chkmat", torch.as_tensor(chkmat, dtype=INT_DTYPE))
        self.register_buffer("obsmat", torch.as_tensor(obsmat, dtype=INT_DTYPE))

        self.add_state("wrong_syndrome", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("wrong_observable", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("wrong_either", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(
        self,
        var2llrs: list[torch.Tensor],
        syndromes: torch.Tensor,
        observables: torch.Tensor
    ):
        """
        Parameters
        ----------
            var2llrs : list[torch.Tensor]
                A Python list of tensors, one for each VN, that stores the posterior LLRs at all iterations. More 
                specifically, `var2llrs[j]` is a tensor of shape (batch_size, num_iters), such that `var2llrs[j][:, t]` 
                is the batch of posterior LLRs for VN `j` at iteration `t`.

            syndromes : torch.Tensor
                Syndrome bits ∈ {0,1}, shape=(batch_size, num_chks), int

            observables : torch.Tensor
                Observable bits ∈ {0,1}, shape=(batch_size, num_obsers), int
        """
        all_llrs = torch.stack(var2llrs, dim=2)  # (batch_size, num_iters, n)
        batch_size, num_iters, num_vars = all_llrs.shape

        # For each shot, check if the decoder converges, i.e., whether the syndrome is matched at any iteration
        hard_decisions = (all_llrs < 0).to(INT_DTYPE)  # (batch_size, num_iters, num_vars), int, 0/1
        synd_pred = torch.matmul(hard_decisions, self.chkmat.T) % 2  # (batch_size, num_iters, num_chks), int, 0/1
        synd_matched_mask = torch.all(synd_pred == syndromes.unsqueeze(dim=1), dim=2)  # (batch_size, num_iters), bool
        converged_mask = torch.any(synd_matched_mask, dim=1)  # (batch_size,), bool

        # For each shot, find which iteration is the overall output of the decoder:
        # If the decoder converges, this is the first iteration where the syndrome is matched;
        # If the decoder does not converge, this is the last iteration.
        output_iters = torch.where(
            converged_mask,
            synd_matched_mask.int().argmax(dim=1),
            num_iters - 1
        )  # (batch_size,), int

        # Get the output error pattern for each shot
        index = output_iters.reshape(batch_size, 1, 1).expand(batch_size, 1, num_vars)
        ehat = hard_decisions.gather(dim=1, index=index).squeeze(1)  # (batch_size, num_vars), int, 0/1

        # For each shot, check if the decoder predicts the observables correctly
        obs_pred = torch.matmul(ehat, self.obsmat.T) % 2  # (batch_size, num_obsers), int, 0/1
        obs_correct_mask = torch.all(obs_pred == observables, dim=1)  # (batch_size,), bool

        # Update states
        self.wrong_syndrome += torch.sum(~converged_mask)
        self.wrong_observable += torch.sum(~obs_correct_mask)
        self.wrong_either += torch.sum(~converged_mask | ~obs_correct_mask)
        self.total += batch_size

    def compute(self) -> dict[str, float]:
        return {
            "wrong_syndrome_rate": self.wrong_syndrome.float() / self.total.float(),
            "wrong_observable_rate": self.wrong_observable.float() / self.total.float(),
            "failure_rate": self.wrong_either.float() / self.total.float(),
        }


__all__ = [
    "DecodingMetric",
]

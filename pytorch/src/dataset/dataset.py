from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

INT_DTYPE = torch.int32


class DecodingDataset(Dataset):
    """
    A PyTorch Dataset. Each item is a (syndrome, observable) pair with integer dtype.
    """

    def __init__(
        self,
        syndromes: np.ndarray | torch.Tensor,
        observables: np.ndarray | torch.Tensor,
    ):
        """
        Parameters
        ----------
            syndromes : np.ndarray | torch.Tensor
                Syndrome bits ∈ {0,1}, shape=(num_shots, num_chks), integer or bool

            observables : np.ndarray | torch.Tensor
                Observable bits ∈ {0,1}, shape=(num_shots, num_obsers), integer or bool
        """
        assert isinstance(syndromes, np.ndarray) or isinstance(syndromes, torch.Tensor)
        assert isinstance(observables, np.ndarray) or isinstance(observables, torch.Tensor)
        assert syndromes.ndim == 2 and observables.ndim == 2
        assert syndromes.shape[0] == observables.shape[0]

        self.syndromes = torch.as_tensor(syndromes, dtype=INT_DTYPE)
        self.observables = torch.as_tensor(observables, dtype=INT_DTYPE)

    @classmethod
    def load_from_file(cls, file: str | Path):
        """
        Load the dataset from a file.
        """
        if isinstance(file, str):
            file = Path(file)
        if not file.exists():
            raise FileNotFoundError(f"File {file} does not exist")

        syndromes, observables = torch.load(file)
        return cls(syndromes, observables)

    def save_to_file(self, file: str | Path, overwrite_ok: bool = False):
        """
        Save the dataset to a file.
        """
        if isinstance(file, str):
            file = Path(file)
        if file.exists() and not overwrite_ok:
            raise FileExistsError(f"File {file} already exists, and overwrite_ok is set to False")

        file.parent.mkdir(parents=True, exist_ok=True)
        torch.save((self.syndromes, self.observables), file)

    def __len__(self):
        return len(self.syndromes)

    def __getitem__(self, idx):
        return self.syndromes[idx], self.observables[idx]

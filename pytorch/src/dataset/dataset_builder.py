from typing import Optional
from itertools import combinations

import numpy as np
from qecdec.experiments import MemoryExperiment

from .dataset import DecodingDataset


def build_decoding_datasets(
    expmt: MemoryExperiment,
    *,
    train_shots: int,
    val_shots: int,
    seed: Optional[int] = None,
    incl_all_wt1_errors_in_train: bool = True,
    incl_all_wt2_errors_in_train: bool = True,
    remove_trivial_syndromes: bool = True,
    verbose: bool = True,
) -> tuple[DecodingDataset, DecodingDataset]:
    """
    Parameters
    ----------
        expmt : MemoryExperiment
            The MemoryExperiment object from which to sample data

        train_shots : int
            Number of sampling shots for building `train_dataset`

        val_shots : int
            Number of sampling shots for building `val_dataset`

        seed : int | None
            Random seed used for sampling

        incl_all_wt1_errors_in_train : bool
            Whether to include all weight-1 errors in `train_dataset`

        incl_all_wt2_errors_in_train : bool
            Whether to include all weight-2 errors in `train_dataset`

        remove_trivial_syndromes : bool
            Whether to filter out trivial (i.e., all-zero) syndromes in `train_dataset` and `val_dataset`

        verbose : bool
            Whether to print verbose output

    Returns
    -------
        train_dataset : DecodingDataset
            Training dataset

        val_dataset : DecodingDataset
            Validation dataset
    """
    def filter(syndromes, observables):
        if remove_trivial_syndromes:
            mask = np.any(syndromes != 0, axis=1)
            syndromes = syndromes[mask]
            observables = observables[mask]
        return syndromes, observables

    n = expmt.num_error_mechanisms

    # =============================== sample shots from noisy circuit ===============================
    if verbose:
        print("Sampling shots from the noisy circuit...")
    sampler = expmt.dem.compile_sampler(seed=seed)
    synds, obsers, _ = sampler.sample(train_shots + val_shots)
    synds = synds.astype(np.int32)
    obsers = obsers.astype(np.int32)
    train_syndromes, train_observables = filter(
        synds[:train_shots], obsers[:train_shots])
    val_syndromes, val_observables = filter(
        synds[train_shots:], obsers[train_shots:])
    if verbose:
        print(
            f"Added {len(train_syndromes)} samples to the training dataset.")
        print(
            f"Added {len(val_syndromes)} samples to the validation dataset.")

    # =============================== weight-1 errors ===============================
    if incl_all_wt1_errors_in_train:
        if verbose:
            print("Generating all weight-1 errors...")
        errors = np.eye(n, dtype=np.int32)
        train_syndromes_from_wt1_errors, train_observables_from_wt1_errors = filter(
            (errors @ expmt.chkmat.T) % 2, (errors @ expmt.obsmat.T) % 2)
        train_syndromes = np.concatenate(
            [train_syndromes, train_syndromes_from_wt1_errors])
        train_observables = np.concatenate(
            [train_observables, train_observables_from_wt1_errors])
        if verbose:
            print(
                f"Added {len(train_syndromes_from_wt1_errors)} samples to the training dataset.")

    # =============================== weight-2 errors ===============================
    if incl_all_wt2_errors_in_train:
        if verbose:
            print("Generating all weight-2 errors...")
        errors = np.zeros(((n * (n - 1)) // 2, n), dtype=np.int32)
        for row, cols in enumerate(combinations(range(n), 2)):
            errors[row, cols] = 1
        train_syndromes_from_wt2_errors, train_observables_from_wt2_errors = filter(
            (errors @ expmt.chkmat.T) % 2, (errors @ expmt.obsmat.T) % 2)
        train_syndromes = np.concatenate(
            [train_syndromes, train_syndromes_from_wt2_errors])
        train_observables = np.concatenate(
            [train_observables, train_observables_from_wt2_errors])
        if verbose:
            print(
                f"Added {len(train_syndromes_from_wt2_errors)} samples to the training dataset.")

    # =============================== collect all instances ===============================
    train_dataset = DecodingDataset(train_syndromes, train_observables)
    val_dataset = DecodingDataset(val_syndromes, val_observables)

    if verbose:
        print(f"Size of train_dataset: {len(train_dataset)}")
        print(f"Size of val_dataset: {len(val_dataset)}")

    return train_dataset, val_dataset

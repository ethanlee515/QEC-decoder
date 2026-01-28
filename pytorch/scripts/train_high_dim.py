from datetime import datetime
import sys
from pathlib import Path
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
from qecdec import RotatedSurfaceCode_Memory


if __name__ == "__main__":
    # Parse command line arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--distance", type=int, required=True, help="code distance")
    parser.add_argument("-p", "--per", type=float, default=0.01, help="physical error rate (optional, default=0.01)")

    args = parser.parse_args()
    d: int = args.distance  # code distance
    p: float = args.per  # physical error rate
    rounds: int = d  # number of rounds of stabilizer measurements

    # Import modules.
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from src.dataset import DecodingDataset
    from src.models import Learned_HighDimBPDecoder
    from src.training import *

    # Set training parameters. # TODO: make config file
    num_epochs = 20
    batch_size = 256
    decoder_kwargs = dict(num_iters=5, msg_dim=16)
    optimizer_kwargs = dict(lr=0.002)
    loss_fn_kwargs = dict(beta=0.8)
    lr_scheduler_kwargs = dict(factor=0.2, patience=3, threshold=1e-3, threshold_mode="abs")
    early_stopper_kwargs = dict(patience=5, min_delta=1e-3)

    # Set up dataloaders.
    dataset_dir = Path(__file__).resolve().parent.parent / "datasets" / "rotated_surface_code_memory_Z" / f"d={d}_rounds={rounds}_p={p}"
    train_dataset = DecodingDataset.load_from_file(dataset_dir / "train_dataset.pt")
    val_dataset = DecodingDataset.load_from_file(dataset_dir / "val_dataset.pt")
    print(f"Size of train_dataset: {len(train_dataset)}")
    print(f"Size of val_dataset: {len(val_dataset)}")
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Set up decoding task.
    expmt = RotatedSurfaceCode_Memory(
        d=d,
        rounds=rounds,
        basis='Z',
        data_qubit_error_rate=p,
        meas_error_rate=p,
    )
    print("Number of error mechanisms:", expmt.num_error_mechanisms)
    print("Number of detectors:", expmt.num_detectors)
    print("Number of observables:", expmt.num_observables)

    # Set up training components.
    decoder = Learned_HighDimBPDecoder(expmt.chkmat, expmt.prior, **decoder_kwargs)
    loss_fn = IterativeDecodingLoss(expmt.chkmat, expmt.obsmat, **loss_fn_kwargs)
    metric = DecodingMetric(expmt.chkmat, expmt.obsmat)
    optimizer = torch.optim.Adam(decoder.parameters(), **optimizer_kwargs)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **lr_scheduler_kwargs)
    early_stopper = EarlyStopper(**early_stopper_kwargs)

    # Train decoder.
    train_decoder(
        decoder,
        train_dataloader,
        val_dataloader,
        loss_fn,
        metric,
        optimizer,
        num_epochs=num_epochs,
        # device="cpu",
        lr_scheduler=lr_scheduler,
        early_stopper=early_stopper,
        progress_bar=True,
    )

    # Save learned parameters.
    # gamma = torch.stack([x.detach() for x in decoder.gamma]).cpu().numpy().astype(np.float64)
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    save_dir = Path(__file__).resolve().parent
    # np.save(save_dir / f"dmembp_d{d}_{timestamp}_gamma.npy", gamma)
    torch.save(decoder.state_dict(), save_dir / f"high_dim_d{d}_{timestamp}.pt")


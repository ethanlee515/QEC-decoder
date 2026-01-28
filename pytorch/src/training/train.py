from typing import Optional
from pathlib import Path

from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from ..training import IterativeDecodingLoss, DecodingMetric, EarlyStopper


def train_decoder(
    decoder: nn.Module,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    loss_fn: IterativeDecodingLoss,
    metric: DecodingMetric,
    optimizer: torch.optim.Optimizer,
    *,
    num_epochs: int,
    device: Optional[str] = None,
    lr_scheduler: Optional[ReduceLROnPlateau] = None,
    early_stopper: Optional[EarlyStopper] = None,
    checkpoint_dir: Optional[str | Path] = None,
    progress_bar: bool = True,
):
    """
    Parameters
    ----------
        decoder : nn.Module
            The decoder to be trained, whose input is an integer tensor of shape (batch_size, num_chks) 
            representing syndrome bits, and output is a float tensor of shape (num_iters, batch_size, num_vars) 
            representing LLR values for all variable nodes at all iterations.

        train_dataloader : DataLoader
            The dataloader for the training dataset.

        val_dataloader : DataLoader
            The dataloader for the validation dataset.

        loss_fn : IterativeDecodingLoss
            The loss function.

        metric : DecodingMetric
            The metric to be evaluated.

        optimizer : torch.optim.Optimizer
            The optimizer.

        num_epochs : int
            The number of epochs.

        device : str | None
            The device to train on. If None, let PyTorch determine the device automatically.

        lr_scheduler : ReduceLROnPlateau | None
            The learning rate scheduler. If None, do not use learning rate scheduler.

        early_stopper : EarlyStopper | None
            The early stopper. If None, do not use early stopping.

        checkpoint_dir : str | Path | None
            The directory to save/load checkpoints. If None, do not save/load checkpoints.

        progress_bar : bool
            Whether to show a progress bar.
    """
    if device is None:
        if torch.accelerator.is_available():
            device = torch.accelerator.current_accelerator().type
        else:
            device = "cpu"
    print(f"Using {device} device")

    start_epoch = 0
    best_val_loss = float("inf")  # <-- minimal fix #1

    if checkpoint_dir is not None:
        if isinstance(checkpoint_dir, str):
            checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_file = checkpoint_dir / "last_checkpoint.pt"
        best_model_file = checkpoint_dir / "best_model.pt"
        # Load last checkpoint if it exists.
        if checkpoint_file.exists():
            print(f"Loading last checkpoint from {checkpoint_file}...")
            chkpt = torch.load(checkpoint_file)
            decoder.load_state_dict(chkpt["model_state_dict"])
            optimizer.load_state_dict(chkpt["optimizer_state_dict"])
            if lr_scheduler is not None:
                lr_scheduler.load_state_dict(chkpt["lr_scheduler_state_dict"])
            start_epoch = chkpt["epoch"] + 1
            best_val_loss = chkpt["best_val_loss"]
        else:
            print("No checkpoint found, starting from scratch...")

    decoder = decoder.to(device)
    metric = metric.to(device)

    # Train model.
    for epoch in range(start_epoch, num_epochs):
        # Training phase.
        decoder.train()
        running_loss = 0.0
        pbar = tqdm(
            train_dataloader,
            desc=f"Epoch {epoch+1}/{num_epochs}",
            total=len(train_dataloader),
            disable=not progress_bar
        )
        for syndromes, observables in pbar:
            syndromes = syndromes.to(device)
            observables = observables.to(device)
            optimizer.zero_grad()

            # Forward pass.
            llrs = decoder(syndromes)
            loss = loss_fn(llrs, syndromes, observables)
            running_loss += loss.item()

            # Backpropagation.
            loss.backward()
            if progress_bar:
                grad_norm = nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=float('inf'))
                pbar.set_postfix({
                    "avg_loss": f"{running_loss / (pbar.n + 1):.6f}",
                    "grad_norm": f"{grad_norm:.6f}"
                })
            optimizer.step()
        avg_train_loss = running_loss / len(train_dataloader)

        # Validation phase.
        decoder.eval()
        metric.reset()
        running_loss = 0.0
        with torch.no_grad():
            for syndromes, observables in val_dataloader:
                syndromes = syndromes.to(device)
                observables = observables.to(device)

                # Forward pass
                llrs = decoder(syndromes)
                loss = loss_fn(llrs, syndromes, observables)
                running_loss += loss.item()
                metric.update(llrs, syndromes, observables)
        avg_val_loss = running_loss / len(val_dataloader)
        val_metrics = metric.compute()

        # Update learning rate scheduler.
        if lr_scheduler is not None:
            lr_scheduler.step(avg_val_loss)

        # Print epoch summary.
        print(f"Epoch {epoch+1} Summary:")
        print(f"  Avg Train Loss: {avg_train_loss:.6f}")
        print(f"  Avg Val Loss: {avg_val_loss:.6f}")
        for key, value in val_metrics.items():
            print(f"  {key}: {value:.6f}")
        if lr_scheduler is not None:
            print(f"  Learning Rate: {lr_scheduler.get_last_lr()[0]:.6f}")
        print()

        # Save best model.
        if checkpoint_dir is not None and avg_val_loss < best_val_loss:  # <-- minimal fix #2
            best_val_loss = avg_val_loss
            torch.save(decoder.state_dict(), best_model_file)
            print(f"New best model saved to {best_model_file}.")

        # Save checkpoint.
        if checkpoint_dir is not None:
            torch.save({
                'epoch': epoch,
                'model_state_dict': decoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'lr_scheduler_state_dict': lr_scheduler.state_dict() if lr_scheduler is not None else None,
                'best_val_loss': best_val_loss
            }, checkpoint_file)
            print(f"Checkpoint saved to {checkpoint_file}.")

        # Update early stopper.
        if early_stopper is not None:
            early_stopper.update(avg_val_loss)
            if early_stopper.check():
                print("Early stopping triggered")
                break

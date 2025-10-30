class EarlyStopper:
    """
    A class that implements early stopping.
    """

    def __init__(
        self,
        *,
        patience: int,
        min_delta: float = 0.0,
    ):
        """
        Parameters
        ----------
            patience : int
                Number of epochs with no improvement after which training will be stopped

            min_delta : float
                Minimum change in the monitored quantity to qualify as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False

    def update(self, val_loss: float):
        if self.early_stop:
            raise RuntimeError("Early stopping has been triggered")

        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter > self.patience:
                self.early_stop = True

    def check(self) -> bool:
        return self.early_stop

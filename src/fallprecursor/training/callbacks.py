class EarlyStopping:
    """Track validation loss and signal when training should stop."""

    def __init__(self, patience: int) -> None:
        self.patience = patience
        self.best_loss = float("inf")
        self.num_bad_epochs = 0

    def should_stop(self, val_loss: float) -> bool:
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.num_bad_epochs = 0
            return False

        self.num_bad_epochs += 1
        return self.num_bad_epochs > self.patience
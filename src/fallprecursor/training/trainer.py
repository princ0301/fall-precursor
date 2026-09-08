import copy

import torch
from torch import nn
from torch.utils.data import DataLoader

from fallprecursor.training.callbacks import EarlyStopping


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    """Run one training epoch and return the mean loss across batches."""
    model.train()
    total_loss = 0.0

    for windows, labels in dataloader:
        windows, labels = windows.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(windows)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> tuple[float, torch.Tensor, torch.Tensor]:
    """Return (mean loss, all predictions, all true labels) over a dataloader."""
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_labels = []

    for windows, labels in dataloader:
        windows, labels = windows.to(device), labels.to(device)
        logits = model(windows)
        loss = loss_fn(logits, labels)
        total_loss += loss.item()
        all_predictions.append(logits.argmax(dim=1).cpu())
        all_labels.append(labels.cpu())

    mean_loss = total_loss / len(dataloader)
    return mean_loss, torch.cat(all_predictions), torch.cat(all_labels)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    max_epochs: int,
    patience: int,
    scoring_fn=None,
) -> dict:
    """Train with early stopping. Returns per-epoch loss and score history.

    scoring_fn(predictions, labels) -> float, higher is better, is used to
    pick which epoch's weights to keep and when to stop. It defaults to
    negative validation loss when not given. Minimizing loss alone can
    favor a model that is more conservative on rare classes without
    actually improving the metric that matters (e.g. precursor F1), so
    pass a scoring_fn built on evaluation.metrics for imbalanced settings
    rather than relying on the loss-based default.
    """
    model.to(device)
    early_stopping = EarlyStopping(patience)
    history = {"train_loss": [], "val_loss": [], "val_score": []}
    best_score = float("-inf")
    best_state_dict = copy.deepcopy(model.state_dict())

    for epoch in range(max_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss, predictions, labels = evaluate(model, val_loader, loss_fn, device)
        score = scoring_fn(predictions.numpy(), labels.numpy()) if scoring_fn is not None else -val_loss

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_score"].append(score)
        print(f"epoch {epoch + 1}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_score={score:.4f}")

        if score > best_score:
            best_score = score
            best_state_dict = copy.deepcopy(model.state_dict())

        if early_stopping.should_stop(-score):
            print(f"stopping early at epoch {epoch + 1}")
            break

    model.load_state_dict(best_state_dict)
    return history
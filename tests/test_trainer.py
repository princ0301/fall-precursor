import numpy as np
import torch
from torch.utils.data import DataLoader

from fallprecursor.data.dataset import SkeletonWindowDataset
from fallprecursor.models.lstm import LSTMClassifier
from fallprecursor.training.trainer import train_one_epoch, evaluate, train_model


def make_toy_loaders() -> tuple[DataLoader, DataLoader]:
    num_samples = 16
    windows = np.random.rand(num_samples, 5, 33, 3).astype(np.float32)
    labels = np.array([i % 3 for i in range(num_samples)])
    dataset = SkeletonWindowDataset(windows, labels, indices=np.arange(num_samples))
    loader = DataLoader(dataset, batch_size=4)
    return loader, loader


def test_train_one_epoch_returns_a_finite_loss():
    train_loader, _ = make_toy_loaders()
    model = LSTMClassifier(num_landmarks=33, num_channels=3, hidden_size=8, num_classes=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()

    loss = train_one_epoch(model, train_loader, optimizer, loss_fn, torch.device("cpu"))
    assert np.isfinite(loss)


def test_evaluate_returns_matching_prediction_and_label_counts():
    _, val_loader = make_toy_loaders()
    model = LSTMClassifier(num_landmarks=33, num_channels=3, hidden_size=8, num_classes=3)
    loss_fn = torch.nn.CrossEntropyLoss()

    loss, predictions, labels = evaluate(model, val_loader, loss_fn, torch.device("cpu"))
    assert np.isfinite(loss)
    assert predictions.shape == labels.shape == (16,)


def test_train_model_history_has_one_entry_per_epoch_when_no_early_stop():
    train_loader, val_loader = make_toy_loaders()
    model = LSTMClassifier(num_landmarks=33, num_channels=3, hidden_size=8, num_classes=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()

    history = train_model(
        model, train_loader, val_loader, optimizer, loss_fn,
        torch.device("cpu"), max_epochs=3, patience=10,
    )
    assert len(history["train_loss"]) == 3
    assert len(history["val_loss"]) == 3


def test_returned_model_matches_best_val_loss_not_final_epoch():
    train_loader, val_loader = make_toy_loaders()
    model = LSTMClassifier(num_landmarks=33, num_channels=3, hidden_size=8, num_classes=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = torch.nn.CrossEntropyLoss()

    history = train_model(
        model, train_loader, val_loader, optimizer, loss_fn,
        torch.device("cpu"), max_epochs=15, patience=10,
    )

    restored_val_loss, _, _ = evaluate(model, val_loader, loss_fn, torch.device("cpu"))
    best_logged_val_loss = min(history["val_loss"])
    assert abs(restored_val_loss - best_logged_val_loss) < 1e-4


def test_scoring_fn_overrides_loss_for_checkpoint_selection():
    train_loader, val_loader = make_toy_loaders()
    model = LSTMClassifier(num_landmarks=33, num_channels=3, hidden_size=8, num_classes=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = torch.nn.CrossEntropyLoss()

    call_count = [0]

    def always_improving_score(predictions, labels):
        call_count[0] += 1
        return float(call_count[0])

    history = train_model(
        model, train_loader, val_loader, optimizer, loss_fn,
        torch.device("cpu"), max_epochs=5, patience=10,
        scoring_fn=always_improving_score,
    )

    # the score increases every epoch, so the last epoch's weights must be the ones kept.
    restored_val_loss, _, _ = evaluate(model, val_loader, loss_fn, torch.device("cpu"))
    assert abs(restored_val_loss - history["val_loss"][-1]) < 1e-4
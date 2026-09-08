import torch

from fallprecursor.models.lstm import LSTMClassifier


def test_output_shape_matches_batch_and_num_classes():
    model = LSTMClassifier(num_landmarks=33, num_channels=3, hidden_size=16, num_classes=3)
    window = torch.rand(4, 30, 33, 3)
    logits = model(window)
    assert logits.shape == (4, 3)


def test_gradients_flow_through_the_model():
    model = LSTMClassifier(num_landmarks=33, num_channels=3, hidden_size=16, num_classes=3)
    window = torch.rand(2, 10, 33, 3)
    labels = torch.tensor([0, 2])

    logits = model(window)
    loss = torch.nn.functional.cross_entropy(logits, labels)
    loss.backward()

    assert model.classifier.weight.grad is not None
    assert torch.any(model.classifier.weight.grad != 0)
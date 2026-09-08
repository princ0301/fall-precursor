import torch
from torch import nn


class LSTMClassifier(nn.Module):
    """Bidirectional LSTM over flattened per-frame skeleton coordinates."""

    def __init__(self, num_landmarks: int, num_channels: int, hidden_size: int, num_classes: int) -> None:
        super().__init__()
        input_size = num_landmarks * num_channels
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, window: torch.Tensor) -> torch.Tensor:
        """window has shape (batch, T, K, C). Returns logits of shape (batch, num_classes)."""
        batch_size, num_frames = window.shape[0], window.shape[1]
        flattened = window.reshape(batch_size, num_frames, -1)
        _, (final_hidden, _) = self.lstm(flattened)
        forward_hidden, backward_hidden = final_hidden[-2], final_hidden[-1]
        combined_hidden = torch.cat([forward_hidden, backward_hidden], dim=1)
        return self.classifier(combined_hidden)
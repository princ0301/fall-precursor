from torch import nn

from fallprecursor.models.lstm import LSTMClassifier
from fallprecursor.models.stgcn import STGCN
from fallprecursor.pose.schema import NUM_LANDMARKS, NUM_CHANNELS


def build_model(model_name: str, hidden_size: int, num_classes: int = 3) -> nn.Module:
    if model_name == "lstm":
        return LSTMClassifier(NUM_LANDMARKS, NUM_CHANNELS, hidden_size, num_classes)
    if model_name == "stgcn":
        return STGCN(in_channels=NUM_CHANNELS, num_classes=num_classes, hidden_channels=hidden_size)
    raise ValueError(f"unknown model: {model_name}")
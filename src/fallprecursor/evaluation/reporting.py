import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

PRECURSOR_CLASS = 1


@torch.no_grad()
def compute_precursor_scores(model: torch.nn.Module, dataloader: DataLoader, device: torch.device) -> np.ndarray:
    """Return the model's predicted probability of the precursor class for every window, in order."""
    model.to(device)
    model.eval()
    scores = []
    for windows, _ in dataloader:
        windows = windows.to(device)
        probabilities = F.softmax(model(windows), dim=1)
        scores.append(probabilities[:, PRECURSOR_CLASS].cpu().numpy())
    return np.concatenate(scores)


def print_lead_time_curve(curve: list[dict]) -> None:
    print(f"{'threshold':>10} {'false_alarm_rate':>17} {'detection_rate':>15} {'mean_lead_time_s':>17}")
    for point in curve:
        lead_time = point["mean_lead_time_seconds"]
        lead_time_str = f"{lead_time:.2f}" if lead_time is not None else "n/a"
        print(f"{point['threshold']:>10.2f} {point['false_alarm_rate']:>17.2f} {point['detection_rate']:>15.2f} {lead_time_str:>17}")
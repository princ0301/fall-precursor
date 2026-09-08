import numpy as np
import torch
from torch import nn

from fallprecursor.pose.schema import NUM_LANDMARKS, POSE_CONNECTIONS


def build_normalized_adjacency(num_joints: int, connections: list[tuple[int, int]]) -> torch.Tensor:
    """Build a symmetrically normalized joint adjacency matrix with self-loops.

    Self-loops let a joint's own features pass through the graph convolution
    unchanged in addition to its neighbors'. Normalization by degree prevents
    high-degree joints (e.g. hips, connected to more neighbors) from
    dominating the aggregated signal purely due to connection count.
    """
    adjacency = np.eye(num_joints, dtype=np.float32)
    for joint_a, joint_b in connections:
        adjacency[joint_a, joint_b] = 1.0
        adjacency[joint_b, joint_a] = 1.0

    degree = adjacency.sum(axis=1)
    degree_inv_sqrt = np.zeros_like(degree)
    degree_inv_sqrt[degree > 0] = np.power(degree[degree > 0], -0.5)
    normalized = degree_inv_sqrt[:, None] * adjacency * degree_inv_sqrt[None, :]
    return torch.tensor(normalized, dtype=torch.float32)


class STGCNBlock(nn.Module):
    """One spatial graph convolution followed by one temporal convolution, with a residual connection."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        adjacency: torch.Tensor,
        temporal_kernel: int = 9,
    ) -> None:
        super().__init__()
        self.register_buffer("adjacency", adjacency)
        self.spatial_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.spatial_norm = nn.BatchNorm2d(out_channels)

        temporal_padding = temporal_kernel // 2
        self.temporal_conv = nn.Conv2d(
            out_channels, out_channels, kernel_size=(temporal_kernel, 1), padding=(temporal_padding, 0)
        )
        self.temporal_norm = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU()
        self.residual = (
            nn.Identity() if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x has shape (N, C_in, T, V). Returns (N, C_out, T, V)."""
        residual = self.residual(x)

        spatial = torch.einsum("nctv,vw->nctw", x, self.adjacency)
        spatial = self.relu(self.spatial_norm(self.spatial_conv(spatial)))

        temporal = self.temporal_norm(self.temporal_conv(spatial))
        return self.relu(temporal + residual)


class STGCN(nn.Module):
    """Spatial-temporal graph convolutional network for skeleton window classification."""

    def __init__(self, in_channels: int, num_classes: int, hidden_channels: int = 32) -> None:
        super().__init__()
        adjacency = build_normalized_adjacency(NUM_LANDMARKS, POSE_CONNECTIONS)

        self.block1 = STGCNBlock(in_channels, hidden_channels, adjacency)
        self.block2 = STGCNBlock(hidden_channels, hidden_channels, adjacency)
        self.block3 = STGCNBlock(hidden_channels, hidden_channels * 2, adjacency)

        self.classifier = nn.Linear(hidden_channels * 2, num_classes)

    def forward(self, window: torch.Tensor) -> torch.Tensor:
        """window has shape (batch, T, V, C_in). Returns logits of shape (batch, num_classes)."""
        x = window.permute(0, 3, 1, 2)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        pooled = x.mean(dim=[2, 3])
        return self.classifier(pooled)
import numpy as np
import torch

from fallprecursor.models.stgcn import build_normalized_adjacency, STGCNBlock, STGCN


def test_adjacency_is_symmetric():
    connections = [(0, 1), (1, 2)]
    adjacency = build_normalized_adjacency(num_joints=3, connections=connections)
    assert torch.allclose(adjacency, adjacency.T)


def test_adjacency_has_nonzero_self_loops():
    connections = [(0, 1)]
    adjacency = build_normalized_adjacency(num_joints=2, connections=connections)
    assert adjacency[0, 0] > 0
    assert adjacency[1, 1] > 0


def test_adjacency_zero_for_unconnected_joints():
    connections = [(0, 1)]
    adjacency = build_normalized_adjacency(num_joints=3, connections=connections)
    assert adjacency[0, 2] == 0.0
    assert adjacency[2, 0] == 0.0


def test_stgcn_block_preserves_temporal_and_joint_dimensions():
    adjacency = build_normalized_adjacency(num_joints=5, connections=[(0, 1), (1, 2)])
    block = STGCNBlock(in_channels=3, out_channels=8, adjacency=adjacency)
    x = torch.rand(2, 3, 10, 5)
    output = block(x)
    assert output.shape == (2, 8, 10, 5)


def test_stgcn_output_shape_matches_batch_and_num_classes():
    model = STGCN(in_channels=3, num_classes=3, hidden_channels=8)
    window = torch.rand(4, 30, 33, 3)
    logits = model(window)
    assert logits.shape == (4, 3)


def test_stgcn_gradients_flow():
    model = STGCN(in_channels=3, num_classes=3, hidden_channels=8)
    window = torch.rand(2, 10, 33, 3)
    labels = torch.tensor([0, 2])

    logits = model(window)
    loss = torch.nn.functional.cross_entropy(logits, labels)
    loss.backward()

    assert model.classifier.weight.grad is not None
    assert torch.any(model.classifier.weight.grad != 0)
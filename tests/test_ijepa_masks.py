import torch

from ssldet.methods.ijepa import sample_target_blocks, sinusoidal_2d_position


def test_target_blocks_are_nonempty():
    masks = sample_target_blocks(2, 8, 8, 4, (0.1, 0.25), (0.75, 1.5), torch.device("cpu"))
    assert masks.shape == (2, 4, 8, 8)
    assert masks.flatten(2).any(dim=2).all()


def test_position_embedding_shape():
    position = sinusoidal_2d_position(7, 7, 64, torch.device("cpu"))
    assert position.shape == (1, 49, 64)


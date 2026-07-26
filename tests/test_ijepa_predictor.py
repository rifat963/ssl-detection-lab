import torch

from ssldet.methods.ijepa import LatentPredictor, sinusoidal_2d_position


def _predict_masked(positions: list[int], grid: int = 4, dim: int = 16) -> torch.Tensor:
    torch.manual_seed(0)
    predictor = LatentPredictor(dim, depth=2, heads=4).eval()
    tokens = torch.randn(1, grid * grid, dim)
    position = sinusoidal_2d_position(grid, grid, dim, torch.device("cpu"))
    target_union = torch.zeros(1, grid * grid, dtype=torch.bool)
    target_union[0, positions] = True
    with torch.no_grad():
        return predictor(tokens, target_union, position)[0, positions]


def test_masked_positions_receive_distinct_predictions():
    """The predictor must stay position-aware where the mask token is substituted.

    Adding the position embedding before masking lets torch.where discard it, leaving
    every target holding the same bare mask token. Self-attention is permutation
    equivariant, so the predictor would emit one averaged latent for every block.
    """

    predictions = _predict_masked([2, 5, 9])
    assert not torch.allclose(predictions[0], predictions[1], atol=1e-6)
    assert not torch.allclose(predictions[0], predictions[2], atol=1e-6)
    assert not torch.allclose(predictions[1], predictions[2], atol=1e-6)


def test_unmasked_context_tokens_are_still_positioned():
    grid, dim = 4, 16
    torch.manual_seed(0)
    predictor = LatentPredictor(dim, depth=2, heads=4).eval()
    position = sinusoidal_2d_position(grid, grid, dim, torch.device("cpu"))
    tokens = torch.zeros(1, grid * grid, dim)
    empty_mask = torch.zeros(1, grid * grid, dtype=torch.bool)
    with torch.no_grad():
        output = predictor(tokens, empty_mask, position)
    # Identical context values must still be separated by their position embedding.
    assert not torch.allclose(output[0, 0], output[0, 7], atol=1e-6)

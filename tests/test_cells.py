import torch
import pytest

@pytest.mark.parametrize("cell", [RHNCell, MGUCell, LiGRUCell, LightRUCell])
def test_cell(cell):
    batch_size = 4
    seq_len = 5
    input_size = 8
    hidden_size = 16
    num_layers = 2

    layer = cell(
        input_size, hidden_size, num_layers=num_layers, dropout=0.5, batch_first=True
    )
    input_tensor = torch.randn(batch_size, seq_len, input_size)

    output, state = layer(input_tensor)

    # (batch_size, seq_len, hidden_size) with batch_first=True
    assert output.shape == (
        batch_size,
        seq_len,
        hidden_size,
    ),  f"Output shape mismatch in {cell.__name__} layer"

    # (num_layers, batch_size, hidden_size)
    assert state.shape == (
        num_layers,
        batch_size,
        hidden_size,
    ), f"State shape mismatch in {cell.__name__} layer"

@pytest.mark.parametrize("cell_class", [NASCell, RANCell])
def test_gated_cell(cell):
    batch_size = 4
    seq_len = 5
    input_size = 8
    hidden_size = 16
    num_layers = 2

    layer = cell(
        input_size, hidden_size, num_layers=num_layers, dropout=0.5, batch_first=True
    )
    input_tensor = torch.randn(batch_size, seq_len, input_size)
    output, (hidden_state, cell_state) = layer(input_tensor)

    # (batch_size, seq_len, hidden_size) with batch_first=True
    assert output.shape == (
        batch_size,
        seq_len,
        hidden_size,
    ), f"Output shape mismatch in {cell.__name__} layer"

    # (num_layers, batch_size, hidden_size)
    assert hidden_state.shape == (
        num_layers,
        batch_size,
        hidden_size,
    ), f"Hidden state shape mismatch in {cell.__name__} layer"

    # (num_layers, batch_size, hidden_size)
    assert cell_state.shape == (
        num_layers,
        batch_size,
        hidden_size,
    ), f"Cell state shape mismatch in {cell.__name__} layer"

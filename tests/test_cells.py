import pytest
import torch
from torch import Tensor
from torchrecurrent import (
    MGUCell, IndRNNCell, LiGRUCell, NASCell,
    PeepholeLSTMCell, RANCell#, SCRNCell
)

# Parameterize over the cell classes and how many states they use
CELL_CASES = [
    # (CellClass, input_size, hidden_size, uses_double_state)
    (MGUCell,         4,  8,  False),
    (IndRNNCell,      3,  5,  False),
    (LiGRUCell,       6, 12,  False),
    (NASCell,         7,  7,  True),
    (PeepholeLSTMCell,5, 10,  True),
    (RANCell,         4,  9,  True),
    #(SCRNCell,        8, 16,  False),
]

@pytest.mark.parametrize("Cell, in_size, hid_size, double", CELL_CASES)
def test_cell_output_and_state_shapes(Cell, in_size, hid_size, double):
    """Each cell should accept both 1D and 2D inputs, init state if None, and return correct shapes."""
    # instantiate
    cell = Cell(in_size, hid_size, bias=False)

    # 1D input (single timestep, no batch)
    x1 = torch.randn(in_size)
    out1 = cell(x1) if not double else cell(x1, (None, None))
    if double:
        h1, c1 = out1
        assert isinstance(h1, Tensor) and isinstance(c1, Tensor)
        assert h1.shape == (hid_size,)
        assert c1.shape == (hid_size,)
    else:
        h1 = out1
        assert isinstance(h1, Tensor)
        assert h1.shape == (hid_size,)

    # 2D input (batch of size B)
    B = 4
    x2 = torch.randn(B, in_size)
    if double:
        h2, c2 = cell(x2, (None, None))
        assert h2.shape == (B, hid_size)
        assert c2.shape == (B, hid_size)
    else:
        h2 = cell(x2)
        assert h2.shape == (B, hid_size)

    # feeding in previous state should keep batch dimension
    if double:
        h3, c3 = cell(x2, (h2, c2))
        assert h3.shape == (B, hid_size)
        assert c3.shape == (B, hid_size)
    else:
        h3 = cell(x2, h2)
        assert h3.shape == (B, hid_size)

@pytest.mark.parametrize("Cell, in_size, hid_size, _", CELL_CASES)
def test_cell_gradients(Cell, in_size, hid_size, _):
    """A quick smoke test: outputs should be differentiable wrt parameters."""
    cell = Cell(in_size, hid_size, bias=False)
    params = [p for p in cell.parameters() if p.requires_grad]
    # simple scalar loss from summed outputs
    x = torch.randn(2, in_size, requires_grad=True)
    out = cell(x) if not getattr(cell, "uses_double_state", lambda: False)() else cell(x, (None, None))[0]
    loss = out.sum()
    loss.backward()
    # ensure each param got a grad
    for p in params:
        assert p.grad is not None

def test_repr_shows_key_args():
    """Check that __repr__ matches PyTorch style for a sample layer."""
    from torchrecurrent.layers import RNN  # assuming you have RNN subclass
    rnn1 = RNN(3, 5)  
    assert repr(rnn1) == "RNN(3, 5)"
    rnn2 = RNN(3, 5, num_layers=2, dropout=0.3, batch_first=True)
    # arguments out of default should appear in repr
    s = repr(rnn2)
    assert "RNN(3, 5" in s
    assert "num_layers=2" in s
    assert "dropout=0.3" in s
    assert "batch_first=True" in s
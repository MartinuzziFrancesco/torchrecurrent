import sys

import pytest
import torch
from torch import Tensor

from torchrecurrent import (
    AntisymmetricRNNCell,
    ATRCell,
    NBRCell,
    BRCell,
    CFNCell,
    DSGUCell,
    coRNNCell,
    FastRNNCell,
    FastGRNNCell,
    JANETCell,
    LEMCell,
    GatedAntisymmetricRNNCell,
    MGUCell,
    IndRNNCell,
    IntersectionRNNCell,
    LiGRUCell,
    LightRUCell,
    MCLSTMCell,
    MultiplicativeLSTMCell,
    MUT1Cell,
    MUT2Cell,
    MUT3Cell,
    NASCell,
    OriginalLSTMCell,
    PeepholeLSTMCell,
    RANCell,
    ResLSTMCell,
    SCRNCell,
    SGUCell,
    SGRNCell,
    STARCell,
    tauGRUCell,
    TRNNCell,
    TGRUCell,
    UGRNNCell,
    UnICORNNCell,
    WMCLSTMCell,
)

skip_windows = pytest.mark.skipif(
    sys.platform == "win32",
    reason="torch.compile requires Triton, which is not supported on Windows",
)

CELL_CASES = [
    # (CellClass, input_size, hidden_size, uses_double_state)
    (AntisymmetricRNNCell, 3, 5, False),
    (ATRCell, 3, 5, False),
    (NBRCell, 3, 5, False),
    (BRCell, 3, 5, False),
    (CFNCell, 3, 5, False),
    (DSGUCell, 3, 5, False),
    (coRNNCell, 3, 5, True),
    (FastRNNCell, 3, 5, False),
    (FastGRNNCell, 3, 5, False),
    (GatedAntisymmetricRNNCell, 3, 5, False),
    (JANETCell, 3, 5, True),
    (LEMCell, 3, 5, True),
    (MGUCell, 4, 8, False),
    (IndRNNCell, 3, 5, False),
    (LiGRUCell, 6, 12, False),
    (LightRUCell, 3, 5, False),
    (MultiplicativeLSTMCell, 3, 5, True),
    (MUT1Cell, 3, 5, False),
    (MUT2Cell, 3, 5, False),
    (MUT3Cell, 3, 5, False),
    (NASCell, 7, 7, True),
    (PeepholeLSTMCell, 5, 10, True),
    (OriginalLSTMCell, 3, 5, True),
    (RANCell, 4, 9, True),
    (ResLSTMCell, 4, 9, True),
    (SCRNCell, 3, 5, True),
    (SGUCell, 3, 5, False),
    (SGRNCell, 3, 5, False),
    (STARCell, 3, 5, False),
    (tauGRUCell, 3, 5, False),
    (TRNNCell, 3, 5, False),
    (UGRNNCell, 3, 5, False),
    (UnICORNNCell, 3, 5, True),
    (WMCLSTMCell, 3, 5, True),
]


@pytest.mark.parametrize("Cell, in_size, hid_size, double", CELL_CASES)
def test_cell_output_and_state_shapes(Cell, in_size, hid_size, double):
    """Each cell should accept both 1D and 2D inputs, init state if None, and
    return correct shapes.
    """
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


def test_reslstm_cell_parameter_shapes():
    cell = ResLSTMCell(4, 9)

    assert cell.weight_ih.shape == (36, 4)
    assert cell.weight_hh.shape == (36, 9)
    assert cell.weight_proj.shape == (9, 9)
    assert cell.weight_res.shape == (9, 4)
    assert cell.weight_ph.shape == (27,)


def test_taugru_cell_parameter_shapes():
    cell = tauGRUCell(4, 9)

    assert cell.weight_ih.shape == (36, 4)
    assert cell.weight_hh.shape == (36, 9)
    assert cell.bias_ih.shape == (36,)
    assert cell.bias_hh.shape == (36,)


def test_cornn_cell_defaults_are_damped():
    cell = coRNNCell(4, 9)

    assert cell.dt == pytest.approx(0.1)
    assert cell.gamma == pytest.approx(1.0)
    assert cell.epsilon == pytest.approx(1.0)


def test_cornn_cell_matches_official_explicit_update():
    cell = coRNNCell(
        2,
        2,
        bias=False,
        recurrent_bias=False,
        cell_bias=False,
        dt=0.5,
        gamma=2.0,
        epsilon=3.0,
    )
    with torch.no_grad():
        cell.weight_ih.copy_(torch.tensor([[0.1, 0.2], [0.3, 0.4]]))
        cell.weight_hh.copy_(torch.tensor([[0.5, 0.6], [0.7, 0.8]]))
        cell.weight_ch.copy_(torch.tensor([[0.9, 1.0], [1.1, 1.2]]))

    x = torch.tensor([[0.2, -0.3]])
    h = torch.tensor([[0.4, -0.5]])
    z = torch.tensor([[0.6, -0.7]])

    new_h, new_z = cell(x, (h, z))
    act = torch.tanh(
        x @ cell.weight_ih.t() + h @ cell.weight_hh.t() + z @ cell.weight_ch.t()
    )
    expected_z = z + 0.5 * (act - 2.0 * h - 3.0 * z)
    expected_h = h + 0.5 * expected_z

    assert torch.allclose(new_z, expected_z)
    assert torch.allclose(new_h, expected_h)


def test_unicornn_cell_parameter_shapes_match_independent_recurrence():
    cell = UnICORNNCell(4, 9)

    assert cell.weight_ih.shape == (9, 4)
    assert cell.weight_hh.shape == (9,)
    assert cell.weight_ch.shape == (9,)


def test_unicornn_cell_matches_official_independent_update():
    cell = UnICORNNCell(2, 2, bias=False, recurrent_bias=False, dt=0.25, alpha=0.5)
    with torch.no_grad():
        cell.weight_ih.copy_(torch.tensor([[0.1, 0.2], [0.3, 0.4]]))
        cell.weight_hh.copy_(torch.tensor([0.5, 0.6]))
        cell.weight_ch.copy_(torch.tensor([0.7, -0.8]))

    x = torch.tensor([[0.2, -0.3]])
    h = torch.tensor([[0.4, -0.5]])
    z = torch.tensor([[0.6, -0.7]])

    new_h, new_z = cell(x, (h, z))
    step = 0.25 * torch.sigmoid(cell.weight_ch)
    candidate = torch.tanh(x @ cell.weight_ih.t() + h * cell.weight_hh)
    expected_z = z - step * (candidate + 0.5 * h)
    expected_h = h + step * expected_z

    assert torch.allclose(new_z, expected_z)
    assert torch.allclose(new_h, expected_h)


def test_taugru_cell_uses_delayed_state():
    cell = tauGRUCell(1, 1)
    with torch.no_grad():
        cell.weight_ih.zero_()
        cell.weight_hh.zero_()
        cell.bias_ih.zero_()
        cell.bias_hh.zero_()
        cell.weight_hh[1, 0] = 1.0
        cell.bias_hh[2] = 20.0
        cell.bias_hh[3] = 20.0

    x = torch.zeros(1, 1)
    h = torch.zeros(1, 1)
    delayed = torch.ones(1, 1)

    out = cell(x, h, delayed)

    assert torch.allclose(out, torch.tanh(delayed), atol=1e-4)


def test_trnn_cell_has_no_recurrent_weight():
    """T-RNN's forget gate depends only on x(t): there is no weight_hh at all."""
    cell = TRNNCell(4, 9)

    assert cell.weight_ih.shape == (18, 4)
    assert cell.bias_ih.shape == (18,)
    assert not hasattr(cell, "weight_hh")
    assert not hasattr(cell, "bias_hh")


def test_trnn_cell_matches_paper_update():
    cell = TRNNCell(2, 2, bias=False)
    with torch.no_grad():
        cell.weight_ih.copy_(torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]]))

    x = torch.tensor([[0.2, -0.3]])
    h = torch.tensor([[0.4, -0.5]])

    new_h = cell(x, h)
    latent, forget_pre = (x @ cell.weight_ih.t()).chunk(2, 1)
    forget_gate = torch.sigmoid(forget_pre)
    expected = forget_gate * h + (1 - forget_gate) * latent

    assert torch.allclose(new_h, expected)


def test_tgru_cell_parameter_shapes():
    """weight_hh acts on the previous *input*, so it shares weight_ih's shape."""
    cell = TGRUCell(4, 9)

    assert cell.weight_ih.shape == (27, 4)
    assert cell.weight_hh.shape == (27, 4)
    assert cell.bias_ih.shape == (27,)
    assert cell.bias_hh.shape == (27,)


def test_tgru_cell_default_state_uses_input_size_not_hidden_size():
    """The second state component is x(t-1): shaped like the input, not like h."""
    cell = TGRUCell(4, 9)
    x = torch.randn(3, 4)

    h, x_prev = cell(x)

    assert h.shape == (3, 9)
    assert x_prev.shape == (3, 4)
    assert torch.equal(x_prev, x)


def test_tgru_cell_matches_paper_update():
    cell = TGRUCell(2, 2, bias=False, recurrent_bias=False)
    with torch.no_grad():
        cell.weight_ih.copy_(torch.arange(12, dtype=torch.float32).reshape(6, 2) * 0.1)
        cell.weight_hh.copy_(torch.arange(12, dtype=torch.float32).reshape(6, 2) * -0.1)

    x = torch.tensor([[0.2, -0.3]])
    h = torch.tensor([[0.4, -0.5]])
    x_prev = torch.tensor([[-0.1, 0.6]])

    new_h, new_x_prev = cell(x, (h, x_prev))
    gates = x @ cell.weight_ih.t() + x_prev @ cell.weight_hh.t()
    latent, forget_pre, out_pre = gates.chunk(3, 1)
    forget_gate = torch.sigmoid(forget_pre)
    candidate = torch.tanh(out_pre)
    expected_h = forget_gate * h + latent * candidate

    assert torch.allclose(new_h, expected_h)
    assert torch.equal(new_x_prev, x)


def test_tgru_cell_accepts_per_component_none_state():
    """Matches the (None, None) sentinel convention every other double-state
    cell in this library supports (see tests/test_cells.py's CELL_CASES
    harness), even though TGRU isn't itself in that harness."""
    cell = TGRUCell(3, 5)
    x = torch.randn(3)

    h, x_prev = cell(x, (None, None))
    assert h.shape == (5,)
    assert x_prev.shape == (3,)

    h2, x_prev2 = cell(x, (h, None))
    assert h2.shape == (5,)
    assert x_prev2.shape == (3,)

    x_batched = torch.randn(2, 3)
    h3, x_prev3 = cell(x_batched, (None, None))
    assert h3.shape == (2, 5)
    assert x_prev3.shape == (2, 3)

    h4, x_prev4 = cell(x_batched, (None, x_prev3))
    assert h4.shape == (2, 5)
    assert x_prev4.shape == (2, 3)


def test_intersectionrnn_cell_requires_equal_sizes():
    with pytest.raises(ValueError, match="input_size == hidden_size"):
        IntersectionRNNCell(3, 5)


def test_intersectionrnn_cell_shapes():
    cell = IntersectionRNNCell(4, 4)

    assert cell.weight_ih.shape == (16, 4)
    assert cell.weight_hh.shape == (16, 4)
    assert cell.bias_ih.shape == (16,)
    assert cell.bias_hh.shape == (16,)


def test_intersectionrnn_cell_output_differs_from_state():
    """y(t) and h(t) are genuinely different tensors, not aliases."""
    cell = IntersectionRNNCell(4, 4)

    x_unbatched = torch.randn(4)
    y_u, h_u = cell(x_unbatched)
    assert y_u.shape == h_u.shape == (4,)
    assert not torch.allclose(y_u, h_u)

    x = torch.randn(3, 4)
    y, h = cell(x)

    assert y.shape == h.shape == (3, 4)
    assert not torch.allclose(y, h)


def test_intersectionrnn_cell_gradients():
    cell = IntersectionRNNCell(4, 4, bias=False)
    x = torch.randn(2, 4, requires_grad=True)

    y, h = cell(x)
    (y.sum() + h.sum()).backward()

    assert x.grad is not None
    for p in cell.parameters():
        if p.requires_grad:
            assert p.grad is not None


def test_intersectionrnn_cell_matches_paper_update():
    cell = IntersectionRNNCell(2, 2, bias=False, recurrent_bias=False)
    with torch.no_grad():
        cell.weight_ih.copy_(torch.arange(16, dtype=torch.float32).reshape(8, 2) * 0.1)
        cell.weight_hh.copy_(torch.arange(16, dtype=torch.float32).reshape(8, 2) * -0.1)

    x = torch.tensor([[0.2, -0.3]])
    h = torch.tensor([[0.4, -0.5]])

    y, new_h = cell(x, h)
    gates = x @ cell.weight_ih.t() + h @ cell.weight_hh.t()
    depth_pre, recurrent_pre, gy_pre, gh_pre = gates.chunk(4, 1)
    depth_candidate = torch.relu(depth_pre)
    recurrent_candidate = torch.tanh(recurrent_pre)
    gate_y = torch.sigmoid(gy_pre)
    gate_h = torch.sigmoid(gh_pre)
    expected_y = gate_y * x + (1 - gate_y) * depth_candidate
    expected_h = gate_h * h + (1 - gate_h) * recurrent_candidate

    assert torch.allclose(y, expected_y)
    assert torch.allclose(new_h, expected_h)


def test_mclstm_cell_shapes():
    cell = MCLSTMCell(4, 9)

    assert cell.weight_ih.shape == (45, 4)
    assert cell.weight_hh.shape == (45, 9)
    assert cell.bias_ih.shape == (45,)
    assert cell.bias_hh.shape == (45,)


def test_mclstm_cell_output_differs_from_state():
    """h(t) is a pure output: it must differ from both v(t) and c(t)."""
    cell = MCLSTMCell(4, 4)

    x_unbatched = torch.randn(4)
    h_u, (v_u, c_u) = cell(x_unbatched)
    assert h_u.shape == v_u.shape == c_u.shape == (4,)
    assert not torch.allclose(h_u, v_u)
    assert not torch.allclose(h_u, c_u)

    x = torch.randn(3, 4)
    h, (v, c) = cell(x)
    assert h.shape == v.shape == c.shape == (3, 4)
    assert not torch.allclose(h, v)


def test_mclstm_cell_accepts_per_component_none_state():
    cell = MCLSTMCell(3, 5)
    x = torch.randn(3)

    h, (v, c) = cell(x, (None, None))
    assert h.shape == v.shape == c.shape == (5,)

    h2, (v2, c2) = cell(x, (v, None))
    assert h2.shape == v2.shape == c2.shape == (5,)

    x_batched = torch.randn(2, 3)
    h3, (v3, c3) = cell(x_batched, (None, None))
    assert h3.shape == v3.shape == c3.shape == (2, 5)

    h4, (v4, c4) = cell(x_batched, (None, c3))
    assert h4.shape == v4.shape == c4.shape == (2, 5)


def test_mclstm_cell_gradients():
    cell = MCLSTMCell(4, 5, bias=False)
    x = torch.randn(2, 4, requires_grad=True)

    h, (v, c) = cell(x)
    (h.sum() + v.sum() + c.sum()).backward()

    assert x.grad is not None
    for p in cell.parameters():
        if p.requires_grad:
            assert p.grad is not None


def test_mclstm_cell_matches_paper_update():
    cell = MCLSTMCell(2, 2, bias=False, recurrent_bias=False)
    with torch.no_grad():
        cell.weight_ih.copy_(torch.arange(20, dtype=torch.float32).reshape(10, 2) * 0.1)
        cell.weight_hh.copy_(torch.arange(20, dtype=torch.float32).reshape(10, 2) * -0.1)

    x = torch.tensor([[0.2, -0.3]])
    v = torch.tensor([[0.1, -0.2]])
    c = torch.tensor([[0.4, -0.5]])

    h, (new_v, new_c) = cell(x, (v, c))
    gates = x @ cell.weight_ih.t() + v @ cell.weight_hh.t()
    f_pre, i_pre, o_pre, m_pre, n_pre = gates.chunk(5, 1)
    forget_gate = torch.sigmoid(f_pre)
    input_gate = torch.sigmoid(i_pre)
    output_gate = torch.sigmoid(o_pre)
    control_gate = torch.sigmoid(m_pre)
    candidate = torch.tanh(n_pre)

    expected_c = forget_gate * c + input_gate * candidate
    expected_h = output_gate * torch.tanh(expected_c)
    expected_v = control_gate * torch.tanh(expected_c)

    assert torch.allclose(new_c, expected_c)
    assert torch.allclose(h, expected_h)
    assert torch.allclose(new_v, expected_v)


@pytest.mark.parametrize("Cell, in_size, hid_size, _", CELL_CASES)
def test_cell_gradients(Cell, in_size, hid_size, _):
    """A quick smoke test: outputs should be differentiable wrt parameters."""
    cell = Cell(in_size, hid_size, bias=False)
    params = [p for p in cell.parameters() if p.requires_grad]
    x = torch.randn(2, in_size, requires_grad=True)
    out = (
        cell(x)
        if not getattr(cell, "uses_double_state", lambda: False)()
        else cell(x, (None, None))[0]
    )
    loss = out.sum()
    loss.backward()
    # ensure each param got a grad
    for p in params:
        assert p.grad is not None


@pytest.mark.parametrize("Cell, in_size, hid_size, double", CELL_CASES)
def test_cell_runs_on_device(Cell, in_size, hid_size, double, device):
    """Every cell should forward, init state, and backward on each available
    device (cpu plus any accelerator: cuda, mps, xpu)."""
    cell = Cell(in_size, hid_size, bias=False).to(device)

    B = 4
    x = torch.randn(B, in_size, device=device, requires_grad=True)

    if double:
        h, c = cell(x, (None, None))
        assert h.device.type == device.type
        assert c.device.type == device.type
        assert h.shape == (B, hid_size)
        assert c.shape == (B, hid_size)
        # explicit state already living on the device should be accepted
        h2, c2 = cell(x, (h, c))
        assert h2.device.type == device.type
        assert c2.device.type == device.type
        out = h
    else:
        h = cell(x)
        assert h.device.type == device.type
        assert h.shape == (B, hid_size)
        h2 = cell(x, h)
        assert h2.device.type == device.type
        out = h

    out.sum().backward()
    for p in cell.parameters():
        if p.requires_grad:
            assert p.grad is not None
            assert p.grad.device.type == device.type


@skip_windows
@pytest.mark.parametrize("Cell, in_size, hid_size, double", CELL_CASES)
def test_cell_compile(Cell, in_size, hid_size, double):
    """Every cell should be compilable via torch.compile."""
    cell = Cell(in_size, hid_size)
    compiled = torch.compile(cell, fullgraph=True)

    B = 4
    x = torch.randn(B, in_size)
    out_eager = cell(x)
    out_compiled = compiled(x)

    if double:
        h_e, c_e = out_eager
        h_c, c_c = out_compiled
        assert h_c.shape == h_e.shape
        assert c_c.shape == c_e.shape
    else:
        assert out_compiled.shape == out_eager.shape


@skip_windows
@pytest.mark.parametrize("Cell, in_size, hid_size, double", CELL_CASES)
def test_cell_compile_with_state(Cell, in_size, hid_size, double):
    """Compiled cells should accept explicit state."""
    cell = Cell(in_size, hid_size)
    compiled = torch.compile(cell, fullgraph=True)

    B = 4
    x = torch.randn(B, in_size)
    if double:
        h0 = torch.randn(B, hid_size)
        c0 = torch.randn(B, hid_size)
        h, c = compiled(x, (h0, c0))
        assert h.shape == (B, hid_size)
        assert c.shape == (B, hid_size)
    else:
        h0 = torch.randn(B, hid_size)
        h = compiled(x, h0)
        assert h.shape == (B, hid_size)

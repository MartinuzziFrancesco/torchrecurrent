import pytest
import torch
from torch import Tensor

from torchrecurrent import (
    AntisymmetricRNN,
    ATR,
    NBR,
    BR,
    CFN,
    DSGU,
    coRNN,
    FastRNN,
    FastGRNN,
    GatedAntisymmetricRNN,
    IndRNN,
    LiGRU,
    LightRU,
    MGU,
    MultiplicativeLSTM,
    MUT1,
    MUT2,
    MUT3,
    NAS,
    OriginalLSTM,
    PeepholeLSTM,
    RAN,
    ResLSTM,
    SCRN,
    SGU,
    SGRN,
    STAR,
    tauGRU,
    TRNN,
    TGRU,
    UGRNN,
    UnICORNN,
    WMCLSTM,
)

LAYER_CLASSES = [
    AntisymmetricRNN,
    ATR,
    NBR,
    BR,
    CFN,
    DSGU,
    coRNN,
    FastRNN,
    FastGRNN,
    GatedAntisymmetricRNN,
    IndRNN,
    LiGRU,
    LightRU,
    MGU,
    MultiplicativeLSTM,
    MUT1,
    MUT2,
    MUT3,
    NAS,
    OriginalLSTM,
    PeepholeLSTM,
    RAN,
    ResLSTM,
    SCRN,
    SGU,
    SGRN,
    STAR,
    tauGRU,
    TRNN,
    TGRU,
    UGRNN,
    UnICORNN,
    WMCLSTM,
]

# (LayerClass, is_double_state)
LAYER_CASES = [
    (AntisymmetricRNN, False),
    (ATR, False),
    (NBR, False),
    (BR, False),
    (CFN, False),
    (DSGU, False),
    (coRNN, True),
    (FastRNN, False),
    (FastGRNN, False),
    (GatedAntisymmetricRNN, False),
    (IndRNN, False),
    (LiGRU, False),
    (LightRU, False),
    (MGU, False),
    (MultiplicativeLSTM, True),
    (MUT1, False),
    (MUT2, False),
    (MUT3, False),
    (NAS, True),
    (OriginalLSTM, True),
    (PeepholeLSTM, True),
    (RAN, True),
    (ResLSTM, True),
    (SCRN, True),
    (SGU, False),
    (SGRN, False),
    (STAR, False),
    (tauGRU, False),
    (TRNN, False),
    (UGRNN, False),
    (UnICORNN, True),
    (WMCLSTM, True),
]


@pytest.mark.parametrize("Layer, is_double", LAYER_CASES)
def test_layer_shapes_and_state(Layer, is_double):
    input_size, hidden_size = 5, 7
    seq_len, batch_size = 4, 3
    num_layers = 2

    # Pass bias=False for simplicity
    layer = Layer(
        input_size,
        hidden_size,
        num_layers=num_layers,
        dropout=0.0,
        batch_first=False,
        bias=False,
    )

    # Unbatched input: (seq_len, batch_size, input_size)
    x = torch.randn(seq_len, batch_size, input_size)
    out, state = layer(x)

    # 1) Output shape
    assert out.shape == (seq_len, batch_size, hidden_size)

    # 2) State shape & type
    if is_double:
        h, c = state
        assert isinstance(state, tuple) and len(state) == 2
        assert h.shape == (num_layers, batch_size, hidden_size)
        assert c.shape == (num_layers, batch_size, hidden_size)
    else:
        assert isinstance(state, Tensor)
        assert state.shape == (num_layers, batch_size, hidden_size)

    # Now test batch_first=True
    layer_bf = Layer(
        input_size,
        hidden_size,
        num_layers=num_layers,
        dropout=0.0,
        batch_first=True,
        bias=False,
    )
    x_bf = torch.randn(batch_size, seq_len, input_size)
    out_bf, state_bf = layer_bf(x_bf)

    # Output with batch_first
    assert out_bf.shape == (batch_size, seq_len, hidden_size)
    if is_double:
        h2, c2 = state_bf
        assert h2.shape == (num_layers, batch_size, hidden_size)
        assert c2.shape == (num_layers, batch_size, hidden_size)
    else:
        assert state_bf.shape == (num_layers, batch_size, hidden_size)


@pytest.mark.parametrize("Layer, is_double", LAYER_CASES)
def test_layer_runs_on_device(Layer, is_double, device):
    """Every stacked layer should forward and backward on each available device
    (cpu plus any accelerator: cuda, mps, xpu)."""
    input_size, hidden_size = 5, 7
    seq_len, batch_size, num_layers = 4, 3, 2

    layer = Layer(
        input_size,
        hidden_size,
        num_layers=num_layers,
        dropout=0.0,
        batch_first=False,
        bias=False,
    ).to(device)

    x = torch.randn(seq_len, batch_size, input_size, device=device, requires_grad=True)
    out, state = layer(x)

    assert out.device.type == device.type
    assert out.shape == (seq_len, batch_size, hidden_size)

    if is_double:
        h, c = state
        assert h.device.type == device.type
        assert c.device.type == device.type
        assert h.shape == (num_layers, batch_size, hidden_size)
        assert c.shape == (num_layers, batch_size, hidden_size)
    else:
        assert state.device.type == device.type
        assert state.shape == (num_layers, batch_size, hidden_size)

    out.sum().backward()
    for p in layer.parameters():
        if p.requires_grad:
            assert p.grad is not None
            assert p.grad.device.type == device.type


def test_tgru_single_layer_supports_differing_sizes():
    """TGRU's second state (x(t-1)) is input_size-shaped, not hidden_size-shaped:
    a single layer works fine even when input_size != hidden_size."""
    input_size, hidden_size = 5, 7
    seq_len, batch_size = 4, 3

    layer = TGRU(input_size, hidden_size, bias=False)
    x = torch.randn(seq_len, batch_size, input_size)
    out, (hn, xn) = layer(x)

    assert out.shape == (seq_len, batch_size, hidden_size)
    assert hn.shape == (1, batch_size, hidden_size)
    assert isinstance(xn, tuple) and len(xn) == 1
    assert xn[0].shape == (batch_size, input_size)


def test_tgru_stacked_layers_have_per_layer_previous_input_shapes():
    """Layer 0's previous-input state is input_size-shaped; every later layer's
    is hidden_size-shaped, since it receives the previous layer's output."""
    input_size, hidden_size = 5, 7
    seq_len, batch_size, num_layers = 4, 3, 2

    layer = TGRU(input_size, hidden_size, num_layers=num_layers, bias=False)
    x = torch.randn(seq_len, batch_size, input_size)
    out, (hn, xn) = layer(x)

    assert out.shape == (seq_len, batch_size, hidden_size)
    assert hn.shape == (num_layers, batch_size, hidden_size)
    assert isinstance(xn, tuple) and len(xn) == num_layers
    assert xn[0].shape == (batch_size, input_size)
    assert xn[1].shape == (batch_size, hidden_size)


def test_tgru_previous_input_state_holds_last_seen_input():
    """x_n[k] should be the actual last input layer k saw, not just correctly shaped."""
    input_size, hidden_size = 5, 7
    seq_len, batch_size, num_layers = 4, 3, 2

    layer = TGRU(input_size, hidden_size, num_layers=num_layers, bias=False)
    x = torch.randn(seq_len, batch_size, input_size)
    out, (hn, xn) = layer(x)

    assert torch.equal(xn[0], x[-1])

    # layer 1's own input at each timestep is layer 0's output at that timestep;
    # recompute layer 0's per-timestep outputs manually to check xn[1].
    h0 = torch.zeros(batch_size, hidden_size)
    x0_prev = torch.zeros(batch_size, input_size)
    for t in range(seq_len):
        h0, x0_prev = layer.cells[0](x[t], (h0, x0_prev))
    assert torch.allclose(xn[1], h0)


def test_tgru_state_continuity_matches_single_call():
    """Splitting a sequence in two and carrying state across calls must match
    running the whole sequence in one call."""
    input_size, hidden_size = 5, 7
    seq_len, batch_size, num_layers = 6, 3, 2
    split = 2

    layer = TGRU(input_size, hidden_size, num_layers=num_layers, bias=False)
    x = torch.randn(seq_len, batch_size, input_size)

    out_full, _ = layer(x)

    out1, state1 = layer(x[:split])
    out2, _ = layer(x[split:], state1)
    out_chunked = torch.cat([out1, out2], dim=0)

    assert torch.allclose(out_chunked, out_full, atol=1e-6)


def test_tgru_batch_first_matches_seq_first():
    input_size, hidden_size = 5, 7
    seq_len, batch_size, num_layers = 4, 3, 2

    layer = TGRU(input_size, hidden_size, num_layers=num_layers, bias=False)
    layer_bf = TGRU(
        input_size, hidden_size, num_layers=num_layers, bias=False, batch_first=True
    )
    layer_bf.load_state_dict(layer.state_dict())

    x = torch.randn(seq_len, batch_size, input_size)
    out, (hn, xn) = layer(x)
    out_bf, (hn_bf, xn_bf) = layer_bf(x.transpose(0, 1))

    assert torch.allclose(out_bf, out.transpose(0, 1))
    assert torch.allclose(hn_bf, hn)
    for a, b in zip(xn_bf, xn):
        assert torch.allclose(a, b)
    assert xn[1].shape == (batch_size, hidden_size)


@pytest.mark.parametrize("Layer", LAYER_CLASSES)
def test_default_repr_shows_input_hidden(Layer):
    # Default repr should exactly match "Class(input_size, hidden_size)"
    r = repr(Layer(3, 5))
    assert r == f"{Layer.__name__}(3, 5)"


@pytest.mark.parametrize("Layer", LAYER_CLASSES)
def test_repr_includes_nondefault_kwargs(Layer):
    # num_layers != 1
    r = repr(Layer(3, 5, num_layers=2))
    assert "num_layers=2" in r

    # dropout != 0.0
    r = repr(Layer(3, 5, dropout=0.5))
    assert "dropout=0.5" in r

    # batch_first=True
    r = repr(Layer(3, 5, batch_first=True))
    assert "batch_first=True" in r

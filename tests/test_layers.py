import pytest
import torch
from torch import Tensor

from torchrecurrent import (
    AntisymmetricRNN,
    ATR,
    NBR,
    BR,
    CFN,
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
    SCRN,
    SGRN,
    STAR,
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
    SCRN,
    SGRN,
    STAR,
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
    (SCRN, True),
    (SGRN, False),
    (STAR, False),
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

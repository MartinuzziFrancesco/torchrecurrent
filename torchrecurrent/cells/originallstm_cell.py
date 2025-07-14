import torch
from torch import Tensor
import torch.nn as nn
from typing import Optional, Callable, Union, Tuple
from ..base import BaseDoubleRecurrentLayer, BaseDoubleRecurrentCell


class OriginalLSTM(BaseDoubleRecurrentLayer):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        batch_first: bool = False,
        **kwargs,
    ):
        super(OriginalLSTM, self).__init__(
            input_size, hidden_size, num_layers, dropout, batch_first
        )
        self.initialize_cells(OriginalLSTMCell, **kwargs)


class OriginalLSTMCell(BaseDoubleRecurrentCell):
    r"""Original formulation of the LSTM cell with no forget gate.

    .. math::

        \begin{aligned}
        \mathbf{g}(t) &= \mathbf{W}_{ih}\,\mathbf{x}(t) + \mathbf{b}_{ih}
            + \mathbf{W}_{hh}\,\mathbf{h}(t-1) + \mathbf{b}_{hh}, \\
        \mathbf{i}(t) &= \sigma\bigl(g_i(t)\bigr), \\
        \tilde{\mathbf{c}}(t) &= \tanh\bigl(g_c(t)\bigr), \\
        \mathbf{o}(t) &= \sigma\bigl(g_o(t)\bigr), \\
        \mathbf{c}(t) &= \mathbf{c}(t-1)
            + \mathbf{i}(t)\,\odot\,\tilde{\mathbf{c}}(t), \\
        \mathbf{h}(t) &= \mathbf{o}(t)\,\odot\,\tanh\bigl(\mathbf{c}(t)\bigr),
        \end{aligned}

    where :math:`\odot` denotes
    element‐wise multiplication.

    Args:
        input_size (int):        Dimensionality of the input features.
        hidden_size (int):       Dimensionality of the hidden and cell states.
        bias (bool):             If False, no bias terms are used. Default: True.
        activation_fn (Callable):
                                    Activation for candidate cell (default: tanh).
        gate_activation_fn (Callable):
                                    Activation for gates (default: sigmoid).
        kernel_init (Callable):  Initializer for input‐to‐hidden weights.
        recurrent_kernel_init (Callable):
                                    Initializer for hidden‐to‐hidden weights.
        bias_init (Callable):    Initializer for input biases.
        recurrent_bias_init (Callable):
                                    Initializer for hidden biases.
        device (torch.device, optional):
                                    Device on which to allocate parameters.
        dtype (torch.dtype, optional):
                                    Data type for parameters.

    Inputs:
        - **inp** (Tensor):      Shape `(batch, input_size)` or `(input_size,)`.
        - **state** (Tuple[Tensor, Tensor], optional):
            Previous `(h, c)` each of shape `(batch, hidden_size)` or
            `(hidden_size,)`. Defaults to zeros if not provided.

    Outputs:
        - **new_h** (Tensor):    Updated hidden state, same shape as `h`.
        - **new_c** (Tensor):    Updated cell state, same shape as `c`.

    Attributes:
        weight_ih (Tensor):      Input‐to‐hidden weight matrix of shape
                                    `(3*hidden_size, input_size)`.
        weight_hh (Tensor):      Hidden‐to‐hidden weight matrix of shape
                                    `(hidden_size, hidden_size)`.
        weight_ph (Tensor):      Additional hidden‐to‐hidden weight for
                                    candidate update, shape `(hidden_size, hidden_size)`.
        bias_ih   (Tensor):      Input bias, shape `(3*hidden_size,)`.
        bias_hh   (Tensor):      Hidden bias, shape `(3*hidden_size,)`.
        bias_ph   (Tensor):      Additional hidden bias, shape `(hidden_size,)`.

    Examples::
        >>> cell = OriginalLSTMCell(10, 20)
        >>> x = torch.randn(5, 10)          # (batch=5, input_size=10)
        >>> h0 = torch.zeros(5, 20)
        >>> c0 = torch.zeros(5, 20)
        >>> h1, c1 = cell(x, (h0, c0))
    """

    weight_ih: Tensor
    weight_hh: Tensor
    weight_ph: Tensor
    bias_ih: Tensor
    bias_hh: Tensor

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        activation_fn: Callable = torch.tanh,
        gate_activation_fn: Callable = torch.sigmoid,
        kernel_init=nn.init.xavier_uniform_,
        recurrent_kernel_init=nn.init.xavier_uniform_,
        bias_init=nn.init.zeros_,
        recurrent_bias_init=nn.init.zeros_,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super(OriginalLSTMCell, self).__init__(
            input_size, hidden_size, bias, device=device, dtype=dtype
        )
        self.kernel_init = kernel_init
        self.recurrent_kernel_init = recurrent_kernel_init
        self.bias_init = bias_init
        self.recurrent_bias_init = recurrent_bias_init

        self._default_register_tensors(
            input_size, hidden_size, ih_mult=3, hh_mult=3, bias=bias
        )
        self.init_weights()

    def forward(
        self, inp: Tensor, state: Optional[Union[Tensor, Tuple[Tensor, ...]]] = None
    ) -> Tuple[Tensor, Tensor]:
        state, c_state = self._check_states(state)
        self._validate_input(inp)
        self._validate_states((state, c_state))
        inp, state, c_state, is_batched = self._preprocess_states(inp, (state, c_state))

        gates = (
            inp @ self.weight_ih.t()
            + self.bias_ih
            + state @ self.weight_hh.t()
            + self.bias_hh
        )
        input_gate, cell_gate, output_gate = gates.chunk(3, 1)
        new_cstate = c_state + torch.sigmoid(input_gate) + torch.tanh(cell_gate)
        new_state = torch.sigmoid(output_gate) * torch.tanh(new_cstate)

        if not is_batched:
            new_state = new_state.squeeze(0)
            new_cstate = new_cstate.squeeze(0)

        return new_state, new_cstate

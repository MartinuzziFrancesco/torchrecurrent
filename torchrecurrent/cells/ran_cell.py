import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Callable, Tuple, Union
from ..base import BaseDoubleRecurrentLayer, BaseDoubleRecurrentCell


class RAN(BaseDoubleRecurrentLayer):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        batch_first: bool = False,
        **kwargs,
    ):
        super(RAN, self).__init__(input_size, hidden_size, num_layers, dropout, batch_first)
        self.initialize_cells(RANCell, **kwargs)


class RANCell(BaseDoubleRecurrentCell):
    r"""A Recurrent Additive Network (RAN) cell.

    Implements the RAN update from
    “Recurrent Additive Networks” <https://arxiv.org/pdf/1705.07393>_.

    .. math::

        \begin{aligned}
          \tilde{\mathbf{c}}(t) &= \mathbf{W}_{ih}^{c}\,\mathbf{x}(t)
             + \mathbf{b}_{ih}^{c}, \\
          \mathbf{i}(t) &= \sigma\bigl(
             \mathbf{W}_{ih}^{i}\,\mathbf{x}(t) + \mathbf{b}_{ih}^{i}
             + \mathbf{W}_{hh}^{i}\,\mathbf{h}(t-1) + \mathbf{b}_{hh}^{i}
          \bigr), \\
          \mathbf{f}(t) &= \sigma\bigl(
             \mathbf{W}_{ih}^{f}\,\mathbf{x}(t) + \mathbf{b}_{ih}^{f}
             + \mathbf{W}_{hh}^{f}\,\mathbf{h}(t-1) + \mathbf{b}_{hh}^{f}
          \bigr), \\
          \mathbf{c}(t) &= \mathbf{i}(t)\circ\tilde{\mathbf{c}}(t)
             + \mathbf{f}(t)\circ\mathbf{c}(t-1), \\
          \mathbf{h}(t) &= \tanh\bigl(\mathbf{c}(t)\bigr)
        \end{aligned}

    where :math:`\circ` denotes element‐wise multiplication and :math:`\sigma`
    is the sigmoid function.

    Args:
        input_size (int): Number of expected features in the input `inp`.
        hidden_size (int): Number of features in the hidden/cell states.
        bias (bool, optional): If ``False``, disables :math:`\mathbf{b}_{ih}`.
            Default: ``True``.
        recurrent_bias (bool, optional): If ``False``, disables :math:`\mathbf{b}_{hh}`.
            Default: ``True``.
        kernel_init (Callable): Initializer for input‐to‐hidden weights
                                (default: `nn.init.xavier_uniform_`).
        recurrent_kernel_init (Callable):
                                Initializer for hidden‐to‐hidden weights
                                (default: `nn.init.xavier_uniform_`).
        bias_init (Callable): Initializer for input biases
                              (default: `nn.init.zeros_`).
        recurrent_bias_init (Callable):
                                Initializer for hidden biases
                                (default: `nn.init.zeros_`).
        device (torch.device, optional): Device for parameters.
        dtype (torch.dtype, optional): Data type for parameters.

    Inputs:
        - **inp** (Tensor): shape `(batch, input_size)` or `(input_size,)`.
        - **state** (Tuple[Tensor, Tensor], optional):
          Previous `(h, c)` each of shape `(batch, hidden_size)` or
          `(hidden_size,)`. If not provided, defaults to zeros.

    Outputs:
        - **new_h** (Tensor): Next hidden state, same shape as `h`.
        - **new_c** (Tensor): Next cell state, same shape as `c`.

    Attributes:
        weight_ih (Tensor): Input‐to‐hidden weights for content, input &
                            forget gates, shape `(3*hidden_size, input_size)`.
        weight_hh (Tensor): Hidden‐to‐hidden weights for input & forget gates,
                            shape `(2*hidden_size, hidden_size)`.
        bias_ih   (Tensor): Input biases for input & forget gates,
                            shape `(2*hidden_size,)`.
        bias_hh   (Tensor): Hidden biases for input & forget gates,
                            shape `(2*hidden_size,)`.

    .. note::
        RANs omit a separate output gate, applying a simple `tanh` to the
        updated cell state to produce the hidden state.

    Examples::
        >>> cell = RANCell(16, 32)
        >>> x = torch.randn(5, 16)           # batch=5, input_size=16
        >>> h0 = torch.zeros(5, 32)          # batch=5, hidden_size=32
        >>> c0 = torch.zeros(5, 32)
        >>> h1, c1 = cell(x, (h0, c0))
    """

    __constants__ = [
        "input_size",
        "hidden_size",
        "bias",
        "recurrent_bias",
        "kernel_init",
        "recurrent_kernel_init",
        "bias_init",
        "recurrent_bias_init",
    ]

    weight_ih: Tensor
    weight_hh: Tensor
    bias_ih: Tensor
    bias_hh: Tensor

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        recurrent_bias: bool = True,
        kernel_init: Callable = nn.init.xavier_uniform_,
        recurrent_kernel_init: Callable = nn.init.xavier_uniform_,
        bias_init: Callable = nn.init.zeros_,
        recurrent_bias_init: Callable = nn.init.zeros_,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super(RANCell, self).__init__(
            input_size, hidden_size, bias, device=device, dtype=dtype
        )
        self.kernel_init = kernel_init
        self.recurrent_kernel_init = recurrent_kernel_init
        self.bias_init = bias_init
        self.recurrent_bias_init = recurrent_bias_init

        self._register_tensors(
            {
                "weight_ih": ((3 * hidden_size, input_size), True),
                "weight_hh": ((2 * hidden_size, hidden_size), True),
                "bias_ih": ((2 * hidden_size,), bias),
                "bias_hh": ((2 * hidden_size,), recurrent_bias),
            }
        )
        self.init_weights()

    def forward(
        self, inp: Tensor, state: Optional[Union[Tensor, Tuple[Tensor, ...]]] = None
    ) -> Tuple[Tensor, Tensor]:
        state, c_state = self._check_states(state)
        self._validate_input(inp)
        self._validate_states((state, c_state))
        inp, state, c_state, is_batched = self._preprocess_states(inp, (state, c_state))

        weight_ih_c, weight_ih_i, weight_ih_f = self.weight_ih.chunk(3, 0)
        weight_hh_i, weight_hh_f = self.weight_hh.chunk(2, 0)
        bias_ih_i, bias_ih_f = self.bias_ih.chunk(2, 0)
        bias_hh_i, bias_hh_f = self.bias_hh.chunk(2, 0)

        content_layer = inp @ weight_ih_c.t()
        ig = inp @ weight_ih_i.t() + bias_ih_i + state @ weight_hh_i.t() + bias_hh_i
        fg = inp @ weight_ih_f.t() + bias_ih_f + state @ weight_hh_f.t() + bias_hh_f
        input_gate = torch.sigmoid(ig)
        forget_gate = torch.sigmoid(fg)
        new_cstate = input_gate * content_layer + forget_gate * c_state
        new_state = torch.tanh(new_cstate)

        if not is_batched:
            new_state = new_state.squeeze(0)
            new_cstate = new_cstate.squeeze(0)

        return new_state, new_cstate

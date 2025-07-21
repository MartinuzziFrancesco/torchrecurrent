import torch
from torch import Tensor
import torch.nn as nn
from typing import Optional, Callable, Union, Tuple
from ..base import BaseDoubleRecurrentLayer, BaseDoubleRecurrentCell


class JANET(BaseDoubleRecurrentLayer):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        batch_first: bool = False,
        **kwargs,
    ):
        super(JANET, self).__init__(
            input_size, hidden_size, num_layers, dropout, batch_first
        )
        self.initialize_cells(JANETCell, **kwargs)


class JANETCell(BaseDoubleRecurrentCell):
    r"""A JANET (Just Another NETwork) recurrent cell.

    Implements the JANET update from
    “Just Another NETwork” <https://arxiv.org/abs/1804.04849>_.

    .. math::

        \begin{aligned}
          \mathbf{s}(t) &= \mathbf{W}_{ih}^{f}\,\mathbf{x}(t)
             + \mathbf{b}_{ih}^{f}
             + \mathbf{W}_{hh}^{f}\,\mathbf{h}(t-1)
             + \mathbf{b}_{hh}^{f}, \\
          \tilde{\mathbf{c}}(t) &= \tanh\Bigl(
             \mathbf{W}_{ih}^{c}\,\mathbf{x}(t) + \mathbf{b}_{ih}^{c}
             + \mathbf{W}_{hh}^{c}\,\mathbf{h}(t-1)
             + \mathbf{b}_{hh}^{c}\Bigr), \\
          \mathbf{c}(t) &= \sigma\bigl(\mathbf{s}(t)\bigr)\circ \mathbf{c}(t-1)
             \;+\;\bigl(1 - \sigma\bigl(\mathbf{s}(t) - \beta\bigr)\bigr)
             \circ \tilde{\mathbf{c}}(t), \\
          \mathbf{h}(t) &= \mathbf{c}(t)
        \end{aligned}

    where :math:`\sigma` is the sigmoid function and :math:`\circ` is the
    Hadamard product.

    Args:
        input_size (int):  Number of expected features in the input tensor.
        hidden_size (int): Number of features in the hidden and cell states.
        bias (bool, optional): If ``False``, disables :math:`\mathbf{b}_{ih}`.
            Default: ``True``.
        recurrent_bias (bool, optional): If ``False``, disables :math:`\mathbf{b}_{hh}`.
            Default: ``True``.
        kernel_init (Callable):
                           Initializer for input‐to‐hidden weights
                           (default: ``nn.init.xavier_uniform_``).
        recurrent_kernel_init (Callable):
                           Initializer for hidden‐to‐hidden weights
                           (default: ``nn.init.xavier_uniform_``).
        bias_init (Callable):
                           Initializer for input biases
                           (default: ``nn.init.zeros_``).
        recurrent_bias_init (Callable):
                           Initializer for hidden biases
                           (default: ``nn.init.zeros_``).
        beta (float):      Threshold shift for the update gate.
                           Default: 1.0.
        device (torch.device, optional): Device on which to place parameters.
        dtype (torch.dtype, optional):   Data type for parameters.

    Inputs:
        - **inp** (Tensor): shape `(batch, input_size)` or `(input_size,)`
        - **state** (Tensor or Tuple[Tensor, Tensor], optional):
          previous `(h, c)` each of shape `(batch, hidden_size)` or
          `(hidden_size,)`. If not provided, defaults to zeros.

    Outputs:
        - **new_state** (Tensor): Updated hidden state, same shape as `h`.
        - **new_cstate** (Tensor): Updated cell state, same shape as `c`.

    Attributes:
        weight_ih (Tensor): Learnable input‐to‐hidden weights for gates and candidate,
                             shape `(2*hidden_size, input_size)`.
        weight_hh (Tensor): Learnable hidden‐to‐hidden weights for gates and candidate,
                             shape `(2*hidden_size, hidden_size)`.
        bias_ih   (Tensor): Learnable input biases, shape `(2*hidden_size,)`.
        bias_hh   (Tensor): Learnable hidden biases, shape `(2*hidden_size,)`.
        beta      (Parameter): Learnable threshold shift for update gating.

    .. note::
        JANET simplifies the LSTM by using a single gate computation
        and tying the hidden and output states to the cell state.

    Examples::
        >>> cell = JANETCell(10, 20, beta=0.5)
        >>> x = torch.randn(5, 10)      # batch=5, input_size=10
        >>> h0 = torch.zeros(5, 20)
        >>> c0 = torch.zeros(5, 20)
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
        beta: float = 1.0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super(JANETCell, self).__init__(
            input_size, hidden_size, bias, device=device, dtype=dtype
        )
        self.kernel_init = kernel_init
        self.recurrent_kernel_init = recurrent_kernel_init
        self.bias_init = bias_init
        self.recurrent_bias_init = recurrent_bias_init

        self._default_register_tensors(
            input_size,
            hidden_size,
            ih_mult=2,
            hh_mult=2,
            bias=bias,
            recurrent_bias=recurrent_bias,
        )
        self.beta = nn.Parameter(torch.tensor(beta))
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
        s_t, s_c = gates.chunk(2, 1)
        forget_gate = torch.sigmoid(s_t)
        candidate_state = torch.tanh(s_c)
        update_gate = torch.sigmoid(s_t - self.beta)
        new_cstate = forget_gate * c_state + (1.0 - update_gate) * candidate_state
        new_state = new_cstate

        if not is_batched:
            new_state = new_state.squeeze(0)
            new_cstate = new_cstate.squeeze(0)

        return new_state, new_cstate

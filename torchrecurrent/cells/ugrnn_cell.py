import torch
from torch import nn
from torch import Tensor
from typing import Optional, Callable, Tuple, Union
from ..base import BaseSingleRecurrentLayer, BaseSingleRecurrentCell


class UGRNN(BaseSingleRecurrentLayer):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        batch_first: bool = False,
        **kwargs,
    ):
        super(UGRNN, self).__init__(
            input_size, hidden_size, num_layers, dropout, batch_first
        )
        self.initialize_cells(UGRNNCell, **kwargs)


class UGRNNCell(BaseSingleRecurrentCell):
    r"""Update Gate recurrent cell.

    A streamlined gated cell that computes a single update gate
    and a candidate update.

    .. math::

        \begin{aligned}
            \mathbf{c}(t) &= s\left( \mathbf{W}_{hh}^c \, \mathbf{h}(t-1) +
                \mathbf{b}_{hh}^c + \mathbf{W}_{xh}^c \, \mathbf{x}(t) +
                \mathbf{b}_{ih}^c \right), \\
            \mathbf{g}(t) &= \sigma\left( \mathbf{W}_{hh}^g \, \mathbf{h}(t-1) +
                \mathbf{b}_{hh}^g + \mathbf{W}_{xh}^g \, \mathbf{x}(t) +
                \mathbf{b}_{ih}^g \right), \\
            \mathbf{h}(t) &= \mathbf{g}(t) \circ \mathbf{h}(t-1) +
                \left( 1 - \mathbf{g}(t) \right) \circ \mathbf{c}(t).
        \end{aligned}

    where :math:`\sigma` is the sigmoid function and :math:`\circ`
    denotes element-wise multiplication.

    Args:
        input_size (int):  Size of the input feature vector.
        hidden_size (int): Size of the hidden state.
        bias (bool, optional): If ``False``, disables :math:`\mathbf{b}_{ih}`.
            Default: ``True``.
        recurrent_bias (bool, optional): If ``False``, disables :math:`\mathbf{b}_{hh}`.
            Default: ``True``.
        kernel_init (Callable):
                            Initializer for input-to-hidden weights
                            (default: ``nn.init.xavier_uniform_``).
        recurrent_kernel_init (Callable):
                            Initializer for hidden-to-hidden weights
                            (default: ``nn.init.xavier_uniform_``).
        bias_init (Callable):
                            Initializer for input biases
                            (default: ``nn.init.zeros_``).
        recurrent_bias_init (Callable):
                            Initializer for hidden biases
                            (default: ``nn.init.zeros_``).
        device (torch.device, optional): Device for parameters.
        dtype (torch.dtype, optional):   Data type for parameters.

    Inputs:
        - **inp** (Tensor): shape `(batch, input_size)` or `(input_size,)`.
        - **state** (Tensor, optional): previous hidden state of shape
            `(batch, hidden_size)` or `(hidden_size,)`. Defaults to zero.

    Outputs:
        - **new_state** (Tensor): Updated hidden state, same shape as `state`.

    Attributes:
        weight_ih (Tensor): Input-to-hidden weights, shape `(2*hidden_size, input_size)`.
        weight_hh (Tensor): Hidden-to-hidden weights, shape `(hidden_size, hidden_size)`.
        bias_ih   (Tensor): Input bias, shape `(2*hidden_size,)`.
        bias_hh   (Tensor): Hidden bias, shape `(hidden_size,)`.

    Examples::
        >>> cell = UGRNNCell(8, 16)
        >>> x = torch.randn(4, 8)   # batch=4, input_size=8
        >>> h0 = torch.zeros(4, 16)
        >>> h1 = cell(x, h0)
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
        super(UGRNNCell, self).__init__(
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
        self.init_weights()

    def forward(
        self, inp: Tensor, state: Optional[Union[Tensor, Tuple[Tensor, ...]]] = None
    ) -> Tensor:
        state = self._check_state(state)
        self._validate_input(inp)
        self._validate_state(state)
        inp, state, is_batched = self._preprocess_input_and_state(inp, state)

        gates = (
            inp @ self.weight_ih.t()
            + self.bias_ih
            + state @ self.weight_hh.t()
            + self.bias_hh
        )
        gate1, gate2 = gates.chunk(2, 1)

        candidate_state = torch.tanh(gate1)
        update_gate = torch.sigmoid(gate2)
        new_state = update_gate * state + (1.0 - update_gate) * candidate_state

        if not is_batched:
            new_state = new_state.squeeze(0)

        return new_state

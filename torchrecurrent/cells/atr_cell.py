import torch
from torch import Tensor
import torch.nn as nn
from typing import Callable, Optional, Tuple, Union
from ..base import BaseSingleRecurrentLayer, BaseSingleRecurrentCell


class ATR(BaseSingleRecurrentLayer):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        batch_first: bool = False,
        **kwargs,
    ):
        super(ATR, self).__init__(
            input_size, hidden_size, num_layers, dropout, batch_first
        )
        self.initialize_cells(ATRCell, **kwargs)


class ATRCell(BaseSingleRecurrentCell):
    r"""An Additive–Transform Recurrent (ATR) cell.

    This cell maintains a single hidden state and computes two gates over
    input and hidden projections to produce the next state:

    .. math::

        \mathbf{p}(t) &= \mathbf{W}_{ih}\,\mathbf{x}(t) + \mathbf{b}_{ih}, \\
        \mathbf{q}(t) &= \mathbf{W}_{hh}\,\mathbf{h}(t-1) + \mathbf{b}_{hh}, \\
        \mathbf{i}(t) &= \sigma\bigl(\mathbf{p}(t) + \mathbf{q}(t)\bigr), \\
        \mathbf{f}(t) &= \sigma\bigl(\mathbf{p}(t) - \mathbf{q}(t)\bigr), \\
        \mathbf{h}(t) &= \mathbf{i}(t) \,\circ\, \mathbf{p}(t)
                        + \mathbf{f}(t)\,\circ\, \mathbf{h}(t-1)\,.

    Args:
        input_size (int):  Size of the input vector :math:`\mathbf{x}(t)`.
        hidden_size (int): Size of the hidden state :math:`\mathbf{h}(t)`.
        bias (bool, optional): If ``False``, disables both :math:`\mathbf{b}_{ih}` and
            :math:`\mathbf{b}_{hh}`. Default: ``True``.
        activation_fn (Callable, optional): Nonlinearity to use for any candidate
            transforms (not used directly here but stored for consistency). Default: ``torch.tanh``.
        kernel_init (Callable, optional): Initializer for :math:`\mathbf{W}_{ih}`.
            Default: ``nn.init.xavier_uniform_``.
        recurrent_kernel_init (Callable, optional): Initializer for :math:`\mathbf{W}_{hh}`.
            Default: ``nn.init.normal_``.
        bias_init (Callable, optional): Initializer for :math:`\mathbf{b}_{ih}` when
            ``bias=True``. Default: ``nn.init.zeros_``.
        recurrent_bias_init (Callable, optional): Initializer for :math:`\mathbf{b}_{hh}`
            when ``bias=True``. Default: ``nn.init.zeros_``.
        device (torch.device, optional): Device of the parameters. Default: ``None`` (CPU).
        dtype (torch.dtype, optional): Data type of the parameters. Default: ``None``.

    Inputs: input, hidden
        - **input** (Tensor): shape `(H_in,)` or `(N, H_in)`, where `H_in = input_size`.
        - **hidden** (Tensor, optional): previous hidden state of shape
            `(H_out,)` or `(N, H_out)`, where `H_out = hidden_size`. Defaults to zero if not provided.

    Outputs: h’
        - **h’** (Tensor): next hidden state, same shape as **hidden**.

    Shape:
        - input: :math:`(N, H_{in})` or :math:`(H_{in})`.
        - hidden: :math:`(N, H_{out})` or :math:`(H_{out})`.
        - output: :math:`(N, H_{out})` or :math:`(H_{out})`.

    Attributes:
        weight_ih (Tensor): Learnable input‐to‐hidden weights of shape
            `(hidden_size, input_size)`.
        weight_hh (Tensor): Learnable hidden‐to‐hidden weights of shape
            `(hidden_size, hidden_size)`.
        bias_ih (Tensor): Learnable input bias of shape `(hidden_size,)`, if enabled.
        bias_hh (Tensor): Learnable hidden bias of shape `(hidden_size,)`, if enabled.

    Examples:
        >>> cell = ATRCell(10, 20)
        >>> x = torch.randn(5, 10)
        >>> h0 = torch.zeros(20)
        >>> hx = h0
        >>> outputs = []
        >>> for t in range(x.size(0)):
        ...     hx = cell(x[t], hx)
        ...     outputs.append(hx)
    """
    weight_ih: Tensor
    weight_hh: Tensor
    bias_ih: Tensor
    bias_hh: Tensor

    def __init__(self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        activation_fn: Callable = torch.tanh,
        kernel_init: Callable = nn.init.xavier_uniform_,
        recurrent_kernel_init: Callable = nn.init.normal_,
        bias_init: Callable = nn.init.zeros_,
        recurrent_bias_init: Callable = nn.init.zeros_,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super(ATRCell, self).__init__(
            input_size, hidden_size, bias, device = device, dtype = dtype
        )
        self.activation_fn = activation_fn
        self.kernel_init = kernel_init
        self.recurrent_kernel_init = recurrent_kernel_init
        self.bias_init = bias_init
        self.recurrent_bias_init = recurrent_bias_init

        self._default_register_tensors(input_size, hidden_size, bias=bias)
        self.init_weights()

    def forward(self,
        inp: Tensor,
        state: Optional[Union[Tensor, Tuple[Tensor, ...]]] = None
    ) -> Tensor:
        state = self._check_state(state)
        self._validate_input(inp)
        self._validate_state(state)
        inp, state, is_batched = self._preprocess_input_and_state(inp, state)

        pt = inp @ self.weight_ih.t() + self.bias_ih
        qt = state @ self.weight_hh.t() + self.bias_hh
        it = torch.sigmoid(pt + qt)
        ft = torch.sigmoid(pt - qt)
        new_state = it * pt + ft * state

        if not is_batched:
            new_state = new_state.squeeze(0)

        return new_state

import torch
from torch import nn
from torch import Tensor
from typing import Optional, Callable, Tuple, Union
from ..base import BaseSingleRecurrentLayer, BaseSingleRecurrentCell


class SGRN(BaseSingleRecurrentLayer):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        batch_first: bool = False,
        **kwargs,
    ):
        super(SGRN, self).__init__(
            input_size, hidden_size, num_layers, dropout, batch_first
        )
        self.initialize_cells(SGRNCell, **kwargs)


class SGRNCell(BaseSingleRecurrentCell):
    r"""A simple gated recurrent network (SGRN) cell.

    Based on the “Simple gated recurrent network” described in:
    Galeja, J., & Maignan, C. (2018). Simple gated recurrent network.
    IET Conference Publication. https://doi.org/10.1049/gtd2.12056

    The cell updates its hidden state according to:

    .. math::

        \begin{aligned}
            \mathbf{f}(t) &= \sigma\bigl(
                \mathbf{W}_{ih}\mathbf{x}(t) + \mathbf{b}_{ih} +
                \mathbf{W}_{hh}\mathbf{h}(t-1) + \mathbf{b}_{hh}\bigr), \\
            \mathbf{i}(t) &= 1 - \mathbf{f}(t), \\
            \mathbf{h}(t) &= \tanh\bigl(
                \mathbf{i}(t) \circ (\mathbf{W}_{ih}\mathbf{x}(t) + \mathbf{b}_{ih}) +
                \mathbf{f}(t) \circ \mathbf{h}(t-1)\bigr)
        \end{aligned}

    where :math:`\sigma` is the sigmoid function and :math:`\circ` denotes
    elementwise (Hadamard) multiplication.

    Args:
        input_size (int):  Number of features in the input :math:`\mathbf{x}(t)`.
        hidden_size (int): Number of features in the hidden state :math:`\mathbf{h}(t)`.
        bias (bool, optional): If ``False``, the cell does not use bias terms
            :math:`\mathbf{b}_{ih}` and :math:`\mathbf{b}_{hh}`. Default: ``True``.
        kernel_init (Callable, optional): Initialization function for
            input-to-hidden weights :math:`\mathbf{W}_{ih}`.
            Default: ``nn.init.xavier_uniform_``.
        recurrent_kernel_init (Callable, optional): Initialization for
            hidden-to-hidden weights :math:`\mathbf{W}_{hh}`.
            Default: ``nn.init.xavier_uniform_``.
        bias_init (Callable, optional): Initialization for input bias
            :math:`\mathbf{b}_{ih}`. Default: ``nn.init.zeros_``.
        recurrent_bias_init (Callable, optional): Initialization for
            hidden bias :math:`\mathbf{b}_{hh}`. Default: ``nn.init.zeros_``.
        device (torch.device, optional): Device on which to place the weights.
        dtype (torch.dtype, optional): Data type for weights and biases.

    Inputs:
        - **inp** (Tensor): Input tensor at current time step,
            of shape :math:`(N, input\_size)` or :math:`(input\_size)`.
        - **state** (Tensor or Tuple[Tensor,…], optional): Previous hidden
            state :math:`\mathbf{h}(t-1)`, of shape :math:`(N, hidden\_size)`
            or :math:`(hidden\_size)`. Defaults to zero if not provided.

    Outputs:
        - **new_state** (Tensor): Updated hidden state
            :math:`\mathbf{h}(t)`, of shape :math:`(N, hidden\_size)`
            or :math:`(hidden\_size)`.

    Attributes:
        weight_ih (Tensor): Learnable input-to-hidden weights,
            of shape `(hidden_size, input_size)`.
        weight_hh (Tensor): Learnable hidden-to-hidden weights,
            of shape `(hidden_size, hidden_size)`.
        bias_ih (Tensor): Learnable input bias, of shape `(hidden_size,)`.
        bias_hh (Tensor): Learnable hidden bias, of shape `(hidden_size,)`.

    Examples::
        >>> cell = SGRNCell(input_size=10, hidden_size=20)
        >>> x = torch.randn(5, 10)     # sequence length 5, batch size inferred
        >>> h = torch.zeros(20)        # initial state
        >>> out = []
        >>> for t in range(5):
        ...     h = cell(x[t], h)
        ...     out.append(h)
    """

    weight_ih: Tensor
    weight_hh: Tensor
    bias_ih: Tensor
    bias_hh: Tensor

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        kernel_init: Callable = nn.init.xavier_uniform_,
        recurrent_kernel_init: Callable = nn.init.xavier_uniform_,
        bias_init: Callable = nn.init.zeros_,
        recurrent_bias_init: Callable = nn.init.zeros_,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super(SGRNCell, self).__init__(
            input_size, hidden_size, bias, device=device, dtype=dtype
        )
        self.kernel_init = kernel_init
        self.recurrent_kernel_init = recurrent_kernel_init
        self.bias_init = bias_init
        self.recurrent_bias_init = recurrent_bias_init

        self._default_register_tensors(
            input_size, hidden_size, ih_mult=1, hh_mult=1, bias=bias
        )
        self.init_weights()

    def forward(
        self, inp: Tensor, state: Optional[Union[Tensor, Tuple[Tensor, ...]]] = None
    ) -> Tensor:
        state = self._check_state(state)
        self._validate_input(inp)
        self._validate_state(state)
        inp, state, is_batched = self._preprocess_input_and_state(inp, state)

        inp_expanded = inp @ self.weight_ih.t() + self.bias_ih
        state_expanded = state @ self.weight_hh.t() + self.bias_hh
        forget_gate = torch.sigmoid(inp_expanded + state_expanded)
        input_gate = 1.0 - forget_gate
        new_state = torch.tanh(input_gate * inp_expanded + forget_gate * state)

        if not is_batched:
            new_state = new_state.squeeze(0)

        return new_state

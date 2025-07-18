import torch
from torch import nn
from torch import Tensor
from typing import Optional, Callable, Tuple, Union
from ..base import BaseSingleRecurrentLayer, BaseSingleRecurrentCell


class STAR(BaseSingleRecurrentLayer):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        batch_first: bool = False,
        **kwargs,
    ):
        super(STAR, self).__init__(
            input_size, hidden_size, num_layers, dropout, batch_first
        )
        self.initialize_cells(STARCell, **kwargs)


class STARCell(BaseSingleRecurrentCell):
    r"""A stackable recurrent cell (STAR) cell.

    Based on the “Stackable recurrent cell” proposed in arXiv:1911.11033.

    The cell computes its update as:

    .. math::

        \begin{aligned}
            \mathbf{z}(t) &= \tanh\bigl(\mathbf{W}_{ih}^{z}\,\mathbf{x}(t) +
                \mathbf{b}_{ih}^{z}\bigr), \\
            \mathbf{k}(t) &= \sigma\bigl(\mathbf{W}_{ih}^{k}\,\mathbf{x}(t) +
                \mathbf{b}_{ih}^{k} +
                \mathbf{W}_{hh}^{k}\,\mathbf{h}(t-1) +
                \mathbf{b}_{hh}^{k}\bigr), \\
            \mathbf{h}(t) &= \tanh\bigl((1 - \mathbf{k}(t)) \circ \mathbf{h}(t-1) +
                \mathbf{k}(t) \circ \mathbf{z}(t)\bigr),
        \end{aligned}

    where :math:`\sigma` is the sigmoid function and :math:`\circ` denotes
    elementwise multiplication.

    Args:
        input_size (int):  Number of features in the input :math:`\mathbf{x}(t)`.
        hidden_size (int): Number of features in the hidden state :math:`\mathbf{h}(t)`.
        bias (bool, optional): If ``False``, the cell does not use bias terms
            :math:`\mathbf{b}_{ih}` and :math:`\mathbf{b}_{hh}`. Default: ``True``.
        kernel_init (Callable, optional): Initialization for input-to-hidden
            weights :math:`\mathbf{W}_{ih}` (both z and k parts).
            Default: ``nn.init.xavier_uniform_``.
        recurrent_kernel_init (Callable, optional): Initialization for
            hidden-to-hidden weights :math:`\mathbf{W}_{hh}^{k}`.
            Default: ``nn.init.xavier_uniform_``.
        bias_init (Callable, optional): Initialization for input biases
            :math:`\mathbf{b}_{ih}^{z}` and :math:`\mathbf{b}_{ih}^{k}`.
            Default: ``nn.init.zeros_``.
        recurrent_bias_init (Callable, optional): Initialization for
            hidden bias :math:`\mathbf{b}_{hh}^{k}`. Default: ``nn.init.zeros_``.
        device (torch.device, optional): Device for parameters.
        dtype (torch.dtype, optional): Data type for parameters.

    Inputs:
        - **inp** (Tensor): Input at current time step,
            shape :math:`(N, input\_size)` or :math:`(input\_size)`.
        - **state** (Tensor or Tuple[Tensor,…], optional): Previous hidden
            state :math:`\mathbf{h}(t-1)`, shape :math:`(N, hidden\_size)`
            or :math:`(hidden\_size)`. Defaults to zero if not provided.

    Outputs:
        - **new_state** (Tensor): Updated hidden state
            :math:`\mathbf{h}(t)`, shape :math:`(N, hidden\_size)`
            or :math:`(hidden\_size)`.

    Attributes:
        weight_ih (Tensor): Input-to-hidden weights, shape `(2*hidden_size, input_size)`,
            where the first hidden_size rows are :math:`W_{ih}^{z}` and the next
            hidden_size rows are :math:`W_{ih}^{k}`.
        weight_hh (Tensor): Hidden-to-hidden weights for gate k,
            shape `(hidden_size, hidden_size)`.
        bias_ih (Tensor): Input biases, shape `(2*hidden_size,)`,
            concatenated :math:`b_{ih}^{z}` and :math:`b_{ih}^{k}`.
        bias_hh (Tensor): Hidden bias for gate k, shape `(hidden_size,)`.

    Examples::
        >>> cell = STARCell(input_size=16, hidden_size=32)
        >>> seq = torch.randn(10, 8, 16)    # seq length 10, batch size 8
        >>> h = torch.zeros(8, 32)          # initial state
        >>> outputs = []
        >>> for t in range(10):
        ...     h = cell(seq[t], h)
        ...     outputs.append(h)
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
        super(STARCell, self).__init__(
            input_size, hidden_size, bias, device=device, dtype=dtype
        )
        self.kernel_init = kernel_init
        self.recurrent_kernel_init = recurrent_kernel_init
        self.bias_init = bias_init
        self.recurrent_bias_init = recurrent_bias_init

        self._default_register_tensors(
            input_size, hidden_size, ih_mult=2, hh_mult=1, bias=bias
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
        gxs1, gxs2 = inp_expanded.chunk(2, 1)

        nonlinear_inpe = torch.tanh(gxs1)
        input_gate = torch.sigmoid(gxs2 + state_expanded)
        new_state = torch.tanh((1.0 - input_gate) * state + input_gate * nonlinear_inpe)

        if not is_batched:
            new_state = new_state.squeeze(0)

        return new_state

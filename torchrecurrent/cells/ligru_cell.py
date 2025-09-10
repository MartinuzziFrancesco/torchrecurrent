import torch
from torch import nn
from torch import Tensor
from typing import Optional, Callable, Tuple, Union
from ..base import BaseSingleRecurrentLayer, BaseSingleRecurrentCell


class LiGRU(BaseSingleRecurrentLayer):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        batch_first: bool = False,
        **kwargs,
    ):
        super(LiGRU, self).__init__(
            input_size, hidden_size, num_layers, dropout, batch_first
        )
        self.initialize_cells(LiGRUCell, **kwargs)


class LiGRUCell(BaseSingleRecurrentCell):
    r"""A Light Gated Recurrent Unit (LiGRU) cell.

    [`arXiv <https://arxiv.org/abs/1803.10225>`_]

    .. math::

        \mathbf{z}(t) &= \sigma\bigl(
            \mathbf{W}_{ih}^{z}\,\mathbf{x}(t)
            + \mathbf{b}_{ih}^{z}
            + \mathbf{W}_{hh}^{z}\,\mathbf{h}(t-1)
            + \mathbf{b}_{hh}^{z}
        \bigr), \\[6pt]
        \tilde{\mathbf{h}}(t) &= \mathrm{ReLU}\bigl(
            \mathbf{W}_{ih}^{h}\,\mathbf{x}(t)
            + \mathbf{b}_{ih}^{h}
            + \mathbf{W}_{hh}^{h}\,\mathbf{h}(t-1)
            + \mathbf{b}_{hh}^{h}
        \bigr), \\[6pt]
        \mathbf{h}(t) &= \mathbf{z}(t)\,\circ\,\mathbf{h}(t-1)
            \;+\;\bigl(1 - \mathbf{z}(t)\bigr)\,\circ\,\tilde{\mathbf{h}}(t),

    where :math:`\circ` denotes element‐wise multiplication.

    Args:
        input_size: The number of expected features in the input ``x``.
        hidden_size: The number of features in the hidden state ``h``.
        bias: If ``False``, the layer does not use input-side biases.
            Default: ``True``.
        recurrent_bias: If ``False``, the layer does not use recurrent biases.
            Default: ``True``.
        nonlinearity: Activation for the candidate :math:`\tilde{h}`.
            Default: :func:`torch.relu`.
        gate_nonlinearity: Activation for the update gate :math:`z`.
            Default: :func:`torch.sigmoid`.
        kernel_init: Initializer for ``W_{ih}``.
            Default: :func:`torch.nn.init.xavier_uniform_`.
        recurrent_kernel_init: Initializer for ``W_{hh}``.
            Default: :func:`torch.nn.init.xavier_uniform_`.
        bias_init: Initializer for ``b_{ih}`` when ``bias=True``.
            Default: :func:`torch.nn.init.zeros_`.
        recurrent_bias_init: Initializer for ``b_{hh}`` when
            ``recurrent_bias=True``. Default: :func:`torch.nn.init.zeros_`.
        device: The desired device of parameters.
        dtype: The desired floating point type of parameters.

    Inputs: input, h_0
        - **input** of shape ``(batch, input_size)`` or ``(input_size,)``:
          Tensor containing input features.
        - **h_0** of shape ``(batch, hidden_size)`` or ``(hidden_size,)``:
          Tensor containing the initial hidden state.

        If **h_0** is not provided, it defaults to zero.

    Outputs: h_1
        - **h_1** of shape ``(batch, hidden_size)`` or ``(hidden_size,)``:
          Tensor containing the next hidden state.

    Variables:
        weight_ih: The learnable input–hidden weights,
            of shape ``(2*hidden_size, input_size)``.
        weight_hh: The learnable hidden–hidden weights,
            of shape ``(2*hidden_size, hidden_size)``.
        bias_ih: The learnable input–hidden biases,
            of shape ``(2*hidden_size)`` if ``bias=True``.
        bias_hh: The learnable hidden–hidden biases,
            of shape ``(2*hidden_size)`` if ``recurrent_bias=True``.

    Examples::

        >>> cell = LiGRUCell(10, 20)
        >>> x = torch.randn(5, 3, 10)    # (time_steps, batch, input_size)
        >>> h = torch.zeros(3, 20)       # (batch, hidden_size)
        >>> out = []
        >>> for t in range(x.size(0)):
        ...     h = cell(x[t], h)
        ...     out.append(h)
        >>> out = torch.stack(out, dim=0)  # (time_steps, batch, hidden_size)
    """

    __constants__ = [
        "input_size",
        "hidden_size",
        "bias",
        "recurrent_bias",
        "nonlinearity",
        "gate_nonlinearity",
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
        nonlinearity: Callable = torch.relu,
        gate_nonlinearity: Callable = torch.sigmoid,
        kernel_init: Callable = nn.init.xavier_uniform_,
        recurrent_kernel_init: Callable = nn.init.xavier_uniform_,
        bias_init: Callable = nn.init.zeros_,
        recurrent_bias_init: Callable = nn.init.zeros_,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super(LiGRUCell, self).__init__(
            input_size, hidden_size, bias, recurrent_bias, device=device, dtype=dtype
        )
        self.nonlinearity = nonlinearity
        self.gate_nonlinearity = gate_nonlinearity
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
        ug, cg = gates.chunk(2, 1)

        update_gate = self.gate_nonlinearity(ug)
        candidate_state = self.nonlinearity(cg)
        new_state = (1.0 - update_gate) * candidate_state + update_gate * state

        if not is_batched:
            new_state = new_state.squeeze(0)

        return new_state

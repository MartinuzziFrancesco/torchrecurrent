import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Callable, Tuple, Union
from ..base import BaseSingleRecurrentLayer, BaseSingleRecurrentCell


class MGU(BaseSingleRecurrentLayer):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        batch_first: bool = False,
        **kwargs,
    ):
        super(MGU, self).__init__(input_size, hidden_size, num_layers, dropout, batch_first)
        self.initialize_cells(MGUCell, **kwargs)


class MGUCell(BaseSingleRecurrentCell):
    r"""A Minimal Gated Unit (MGU) cell.

    [`arXiv <https://arxiv.org/abs/1603.09420>`_]

    .. math::

        \mathbf{f}(t) &= \sigma\bigl(
            \mathbf{W}_{ih}^{f}\,\mathbf{x}(t)
            + \mathbf{b}_{ih}^{f}
            + \mathbf{W}_{hh}^{f}\,\mathbf{h}(t-1)
            + \mathbf{b}_{hh}^{f}
        \bigr), \\[6pt]
        \tilde{\mathbf{h}}(t) &= \phi\bigl(
            \mathbf{W}_{ih}^{h}\,\mathbf{x}(t)
            + \mathbf{b}_{ih}^{h}
            + \mathbf{W}_{hh}^{h}\bigl(\mathbf{f}(t)\circ\mathbf{h}(t-1)\bigr)
            + \mathbf{b}_{hh}^{h}
        \bigr), \\[6pt]
        \mathbf{h}(t) &= \bigl(1 - \mathbf{f}(t)\bigr)\circ\mathbf{h}(t-1)
            \;+\;\mathbf{f}(t)\circ\tilde{\mathbf{h}}(t),

    where :math:`\circ` is element‐wise product and :math:`\phi` is a
    pointwise nonlinearity (e.g., tanh).

    Args:
        input_size: The number of expected features in the input ``x``.
        hidden_size: The number of features in the hidden state ``h``.
        bias: If ``False``, the layer does not use input-side biases.
            Default: ``True``.
        recurrent_bias: If ``False``, the layer does not use recurrent biases.
            Default: ``True``.
        nonlinearity: Nonlinearity :math:`\phi` for the candidate.
            Default: :func:`torch.tanh`.
        gate_nonlinearity: Activation for the forget gate.
            Default: :func:`torch.sigmoid`.
        kernel_init: Initializer for ``W_{ih}^*``.
            Default: :func:`torch.nn.init.xavier_uniform_`.
        recurrent_kernel_init: Initializer for ``W_{hh}^*``.
            Default: :func:`torch.nn.init.xavier_uniform_`.
        bias_init: Initializer for input-side biases when ``bias=True``.
            Default: :func:`torch.nn.init.zeros_`.
        recurrent_bias_init: Initializer for recurrent biases when
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
            of shape ``(2*hidden_size, input_size)`` (forget & candidate parts).
        weight_hh: The learnable hidden–hidden weights,
            of shape ``(2*hidden_size, hidden_size)`` (forget & candidate parts).
        bias_ih: The learnable input–hidden biases,
            of shape ``(2*hidden_size)`` if ``bias=True``.
        bias_hh: The learnable hidden–hidden biases,
            of shape ``(2*hidden_size)`` if ``recurrent_bias=True``.

    Examples::

        >>> cell = MGUCell(10, 20)
        >>> x = torch.randn(5, 3, 10)     # (time_steps, batch, input_size)
        >>> h = torch.zeros(3, 20)        # (batch, hidden_size)
        >>> out = []
        >>> for t in range(x.size(0)):
        ...     h = cell(x[t], h)
        ...     out.append(h)
        >>> out = torch.stack(out, dim=0) # (time_steps, batch, hidden_size)
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
        nonlinearity: Callable = torch.tanh,
        gate_nonlinearity: Callable = torch.sigmoid,
        kernel_init: Callable = nn.init.xavier_uniform_,
        recurrent_kernel_init: Callable = nn.init.xavier_uniform_,
        bias_init: Callable = nn.init.zeros_,
        recurrent_bias_init: Callable = nn.init.zeros_,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super(MGUCell, self).__init__(
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

        weight_ih_f, weight_ih_h = self.weight_ih.chunk(2, 0)
        weight_hh_f, weight_hh_h = self.weight_hh.chunk(2, 0)
        bias_ih_f, bias_ih_h = self.bias_ih.chunk(2, 0)
        bias_hh_f, bias_hh_h = self.bias_hh.chunk(2, 0)

        fg = inp @ weight_ih_f.t() + bias_ih_f + state @ weight_hh_f.t() + bias_hh_f
        forget_gate = self.gate_nonlinearity(fg)
        hidden_modulated = forget_gate * state
        ch = (
            inp @ weight_ih_h.t()
            + bias_ih_h
            + hidden_modulated @ weight_hh_h.t()
            + bias_hh_h
        )
        candidate_hidden = self.nonlinearity(ch)
        new_state = forget_gate * candidate_hidden + (1.0 - forget_gate) * state

        if not is_batched:
            new_state = new_state.squeeze(0)

        return new_state

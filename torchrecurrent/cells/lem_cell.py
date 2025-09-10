import torch
from torch import Tensor
import torch.nn as nn
from typing import Optional, Callable, Union, Tuple
from ..base import BaseDoubleRecurrentLayer, BaseDoubleRecurrentCell


class LEM(BaseDoubleRecurrentLayer):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        batch_first: bool = False,
        **kwargs,
    ):
        super(LEM, self).__init__(input_size, hidden_size, num_layers, dropout, batch_first)
        self.initialize_cells(LEMCell, **kwargs)


class LEMCell(BaseDoubleRecurrentCell):
    r"""A Long Expressive Memory (LEM) recurrent cell.

    [`arXiv <https://arxiv.org/pdf/2110.04744>`_]

    .. math::

        \begin{aligned}
        \boldsymbol{\Delta t}(t) &= \Delta t \,\hat{\sigma}\bigl(
            \mathbf{W}_{ih}^{1}\,\mathbf{x}(t) + \mathbf{b}_{ih}^{1}
            + \mathbf{W}_{hh}^{1}\,\mathbf{h}(t-1) + \mathbf{b}_{hh}^{1}
        \bigr), \\
        \overline{\boldsymbol{\Delta t}}(t) &= \Delta t \,\hat{\sigma}\bigl(
            \mathbf{W}_{ih}^{2}\,\mathbf{x}(t) + \mathbf{b}_{ih}^{2}
            + \mathbf{W}_{hh}^{2}\,\mathbf{h}(t-1) + \mathbf{b}_{hh}^{2}
        \bigr), \\
        \mathbf{c}(t) &= \bigl(1 - \boldsymbol{\Delta t}(t)\bigr)\circ\mathbf{c}(t-1)
            + \boldsymbol{\Delta t}(t)\circ\sigma\bigl(
                \mathbf{W}_{ih}^{c}\,\mathbf{x}(t) + \mathbf{b}_{ih}^{c}
                + \mathbf{W}_{hh}^{c}\,\mathbf{h}(t-1) + \mathbf{b}_{hh}^{c}
            \bigr), \\
        \mathbf{h}(t) &= \bigl(1 - \boldsymbol{\Delta t}(t)\bigr)\circ\mathbf{h}(t-1)
            + \boldsymbol{\Delta t}(t)\circ\sigma\bigl(
                \mathbf{W}_{ih}^{h}\,\mathbf{x}(t) + \mathbf{b}_{ih}^{h}
                + \mathbf{W}_{ch}\,\mathbf{c}(t) + \mathbf{b}_{ch}
            \bigr)
        \end{aligned}

    where :math:`\hat{\sigma}` is the sigmoid function and
    :math:`\circ` denotes element-wise multiplication.

    Args:
        input_size: The number of expected features in the input ``x``.
        hidden_size: The number of features in the hidden/cell states ``h`` and ``c``.
        bias: If ``False``, the layer does not use input-side bias ``b_{ih}``.
            Default: ``True``.
        recurrent_bias: If ``False``, the layer does not use recurrent bias
            ``b_{hh}``. Default: ``True``.
        cell_bias: If ``False``, the layer does not use cell bias ``b_{ch}``.
            Default: ``True``.
        kernel_init: Initializer for ``W_{ih}``.
        recurrent_kernel_init: Initializer for ``W_{hh}``.
        cell_kernel_init: Initializer for ``W_{ch}``.
        bias_init: Initializer for ``b_{ih}``.
        recurrent_bias_init: Initializer for ``b_{hh}``.
        cell_bias_init: Initializer for ``b_{ch}``.
        dt: Integration time step :math:`\Delta t`. Default: ``1.0``.
        device: The desired device of parameters.
        dtype: The desired floating point type of parameters.

    Inputs: input, (h_0, c_0)
        - **input** of shape ``(batch, input_size)`` or ``(input_size,)``:
          Tensor containing input features.
        - **h_0** of shape ``(batch, hidden_size)`` or ``(hidden_size,)``:
          Tensor containing the initial hidden state.
        - **c_0** of shape ``(batch, hidden_size)`` or ``(hidden_size,)``:
          Tensor containing the initial cell state.

        If ``(h_0, c_0)`` is not provided, both default to zeros.

    Outputs: (h_1, c_1)
        - **h_1** of shape ``(batch, hidden_size)`` or ``(hidden_size,)``:
          Tensor containing the next hidden state.
        - **c_1** of shape ``(batch, hidden_size)`` or ``(hidden_size,)``:
          Tensor containing the next cell state.

    Variables:
        weight_ih: The learnable input–hidden weights,
            of shape ``(4*hidden_size, input_size)``.
        weight_hh: The learnable hidden–hidden weights,
            of shape ``(3*hidden_size, hidden_size)``.
        weight_ch: The learnable cell–hidden weights,
            of shape ``(hidden_size, hidden_size)``.
        bias_ih: The learnable input–hidden bias,
            of shape ``(4*hidden_size)``.
        bias_hh: The learnable hidden–hidden bias,
            of shape ``(3*hidden_size)``.
        bias_ch: The learnable cell–hidden bias,
            of shape ``(hidden_size)``.

    Examples::

        >>> cell = LEMCell(16, 32, dt=0.5)
        >>> x = torch.randn(5, 3, 16)      # (time_steps, batch, input_size)
        >>> h, c = torch.zeros(3, 32), torch.zeros(3, 32)
        >>> out_h = []
        >>> for t in range(x.size(0)):
        ...     h, c = cell(x[t], (h, c))
        ...     out_h.append(h)
        >>> out_h = torch.stack(out_h, dim=0)  # (time_steps, batch, hidden_size)
    """

    __constants__ = [
        "input_size",
        "hidden_size",
        "bias",
        "recurrent_bias",
        "cell_bias",
        "kernel_init",
        "recurrent_kernel_init",
        "cell_kernel_init",
        "bias_init",
        "recurrent_bias_init",
        "cell_bias_init",
        "dt",
    ]

    weight_ih: Tensor
    weight_hh: Tensor
    weight_ch: Tensor
    bias_ih: Tensor
    bias_hh: Tensor
    bias_ch: Tensor

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        recurrent_bias: bool = True,
        cell_bias: bool = True,
        kernel_init: Callable = nn.init.xavier_uniform_,
        recurrent_kernel_init: Callable = nn.init.xavier_uniform_,
        cell_kernel_init: Callable = nn.init.xavier_uniform_,
        bias_init: Callable = nn.init.zeros_,
        recurrent_bias_init: Callable = nn.init.zeros_,
        cell_bias_init: Callable = nn.init.zeros_,
        dt: float = 1.0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super(LEMCell, self).__init__(
            input_size, hidden_size, bias, recurrent_bias, dt=dt, device=device, dtype=dtype
        )
        self.kernel_init = kernel_init
        self.recurrent_kernel_init = recurrent_kernel_init
        self.cell_kernel_init = cell_kernel_init
        self.bias_init = bias_init
        self.recurrent_bias_init = recurrent_bias_init
        self.cell_bias_init = cell_bias_init
        self.dt = dt

        self._register_tensors(
            {
                "weight_ih": ((4 * hidden_size, input_size), True),
                "weight_hh": ((3 * hidden_size, hidden_size), True),
                "weight_ch": ((hidden_size, hidden_size), True),
                "bias_ih": ((4 * hidden_size,), bias),
                "bias_hh": ((3 * hidden_size,), recurrent_bias),
                "bias_ch": ((hidden_size,), cell_bias),
            }
        )
        self.init_weights()

    def init_weights(self):
        for name, param in self.named_parameters():
            if "weight_ih" in name:
                self.kernel_init(param)
            elif "weight_hh" in name:
                self.recurrent_kernel_init(param)
            elif "weight_ch" in name:
                self.cell_kernel_init(param)
            elif "bias_ih" in name:
                self.bias_init(param)
            elif "bias_hh" in name:
                self.recurrent_bias_init(param)
            elif "bias_ch" in name:
                self.cell_bias_init(param)

    def forward(
        self, inp: Tensor, state: Optional[Union[Tensor, Tuple[Tensor, ...]]] = None
    ) -> Tuple[Tensor, Tensor]:
        state, c_state = self._check_states(state)
        self._validate_input(inp)
        self._validate_states((state, c_state))
        inp, state, c_state, is_batched = self._preprocess_states(inp, (state, c_state))

        inp_expanded = inp @ self.weight_ih.t() + self.bias_ih
        state_expanded = state @ self.weight_hh.t() + self.bias_hh
        gxs1, gxs2, gxs3, gxs4 = inp_expanded.chunk(4, 1)
        ghs1, ghs2, ghs3 = state_expanded.chunk(3, 1)

        msdt_bar = self.dt * torch.sigmoid(gxs1 + ghs1)
        msdt = self.dt * torch.sigmoid(gxs2 + ghs2)
        new_cstate = (1.0 - msdt) * c_state + msdt * torch.tanh(gxs3 + ghs3)
        new_state = (1.0 - msdt_bar) * state + msdt_bar * torch.tanh(
            gxs4 + c_state @ self.weight_ch.t() + self.bias_ch
        )

        if not is_batched:
            new_state = new_state.squeeze(0)
            new_cstate = new_cstate.squeeze(0)

        return new_state, new_cstate

import torch
from torch import nn
from torch import Tensor
from typing import Optional, Callable, Tuple, Union
from ..base import BaseSingleRecurrentLayer, BaseSingleRecurrentCell


class CFN(BaseSingleRecurrentLayer):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        batch_first: bool = False,
        **kwargs,
    ):
        super(CFN, self).__init__(input_size, hidden_size, num_layers, dropout, batch_first)
        self.initialize_cells(CFNCell, **kwargs)


class CFNCell(BaseSingleRecurrentCell):
    r"""A Chaos Free Network (CFN) cell.

    [`arXiv <https://arxiv.org/abs/1612.06212`_]

    .. math::

        \boldsymbol{\theta}(t) &= \sigma\bigl(
            \mathbf{W}_{ih}^{\theta}\,\mathbf{x}(t)
            + \mathbf{b}_{ih}^{\theta}
            + \mathbf{W}_{hh}^{\theta}\,\mathbf{h}(t-1)
            + \mathbf{b}_{hh}^{\theta}
        \bigr), \\
            \boldsymbol{\eta}(t) &= \sigma\bigl(
            \mathbf{W}_{ih}^{\eta}\,\mathbf{x}(t)
            + \mathbf{b}_{ih}^{\eta}
            + \mathbf{W}_{hh}^{\eta}\,\mathbf{h}(t-1)
            + \mathbf{b}_{hh}^{\eta}
        \bigr), \\
            \mathbf{h}(t) &= \boldsymbol{\theta}(t)\,\circ\,
            \tanh\bigl(\mathbf{h}(t-1)\bigr)
            \;+\;\boldsymbol{\eta}(t)\,\circ\,\tanh\bigl(
                \mathbf{W}_{ih}^{h}\,\mathbf{x}(t)
                + \mathbf{b}_{ih}^{h}
            \bigr)\,.

    Args:
        input_size: The number of expected features in the input ``x``
        hidden_size: The number of features in the hidden state ``h``
        bias: If ``False``, the layer does not use input-side biases.
            Default: ``True``
        recurrent_bias: If ``False``, the layer does not use recurrent biases.
            Default: ``True``
        kernel_init: Initializer for input–hidden weights ``W_{ih}^*``.
            Default: :func:`torch.nn.init.xavier_uniform_`
        recurrent_kernel_init: Initializer for hidden–hidden weights ``W_{hh}^*``.
            Default: :func:`torch.nn.init.xavier_uniform_`
        bias_init: Initializer for input-side biases ``b_{ih}^*`` when ``bias=True``.
            Default: :func:`torch.nn.init.zeros_`
        recurrent_bias_init: Initializer for hidden biases ``b_{hh}^*``
            when ``recurrent_bias=True``. Default: :func:`torch.nn.init.zeros_`
        device: The desired device of parameters.
        dtype: The desired floating point type of parameters.

    Inputs: input, h_0
        - **input** of shape ``(batch, input_size)`` or ``(input_size,)``:
          tensor containing input features
        - **h_0** of shape ``(batch, hidden_size)`` or ``(hidden_size,)``:
          tensor containing the initial hidden state

        If **h_0** is not provided, it defaults to zero.

    Outputs: h_1
        - **h_1** of shape ``(batch, hidden_size)`` or ``(hidden_size,)``:
          tensor containing the next hidden state

    Variables:
        weight_ih: the learnable input–hidden weights,
            of shape ``(3*hidden_size, input_size)`` (split into θ/η/h parts)
        weight_hh: the learnable hidden–hidden weights,
            of shape ``(2*hidden_size, hidden_size)`` (split into θ/η parts)
        bias_ih: the learnable input–hidden biases,
            of shape ``(3*hidden_size)``
        bias_hh: the learnable hidden–hidden biases,
            of shape ``(2*hidden_size)``

    Examples::

        >>> cell = CFNCell(10, 20)              # (input_size, hidden_size)
        >>> x = torch.randn(5, 3, 10)           # (time_steps, batch, input_size)
        >>> h = torch.zeros(3, 20)              # (batch, hidden_size)
        >>> out = []
        >>> for t in range(x.size(0)):
        ...     h = cell(x[t], h)
        ...     out.append(h)
        >>> out = torch.stack(out, dim=0)       # (time_steps, batch, hidden_size)
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
        super(CFNCell, self).__init__(
            input_size, hidden_size, bias, recurrent_bias, device=device, dtype=dtype
        )
        self.kernel_init = kernel_init
        self.recurrent_kernel_init = recurrent_kernel_init
        self.bias_init = bias_init
        self.recurrent_bias_init = recurrent_bias_init

        self._default_register_tensors(
            input_size,
            hidden_size,
            ih_mult=3,
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

        input_exp = inp @ self.weight_ih.t() + self.bias_ih
        rec_exp = state @ self.weight_hh.t() + self.bias_hh
        input_exp_1, input_exp_2, input_exp_3 = input_exp.chunk(3, 1)
        rec_exp_1, rec_exp_2 = rec_exp.chunk(2, 1)

        horizontal_gate = torch.sigmoid(input_exp_1 + rec_exp_1)
        vertical_gate = torch.sigmoid(input_exp_2 + rec_exp_2)
        new_state = horizontal_gate * torch.tanh(state) + vertical_gate * torch.tanh(
            input_exp_3
        )

        if not is_batched:
            new_state = new_state.squeeze(0)

        return new_state

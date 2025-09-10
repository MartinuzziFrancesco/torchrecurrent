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
    r"""A Stackable Recurrent (STAR) cell.

    [`arXiv <https://arxiv.org/abs/1911.11033>`_]

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

    where :math:`\sigma` is the sigmoid function
    and :math:`\circ` denotes element-wise multiplication.

    Args:
        input_size: Number of features in the input :math:`\mathbf{x}(t)`
        hidden_size: Number of features in the hidden state :math:`\mathbf{h}(t)`
        bias: If ``False``, the layer does not use input-side bias
            :math:`\mathbf{b}_{ih}`. Default: ``True``
        recurrent_bias: If ``False``, the layer does not use recurrent
            bias :math:`\mathbf{b}_{hh}`. Default: ``True``
        kernel_init: Initializer for ``weight_ih``.
            Default: :func:`torch.nn.init.xavier_uniform_`
        recurrent_kernel_init: Initializer for ``weight_hh``.
            Default: :func:`torch.nn.init.xavier_uniform_`
        bias_init: Initializer for ``bias_ih`` when ``bias=True``.
            Default: :func:`torch.nn.init.zeros_`
        recurrent_bias_init: Initializer for ``bias_hh`` when ``recurrent_bias=True``.
            Default: :func:`torch.nn.init.zeros_`
        device: The desired device of parameters
        dtype: The desired floating point type of parameters

    Inputs: input, hidden
        - **input** of shape ``(batch, input_size)`` or ``(input_size,)``:
          tensor containing input features
        - **hidden** of shape ``(batch, hidden_size)`` or ``(hidden_size,)``:
          tensor containing the previous hidden state

        If **hidden** is not provided, it defaults to zero.

    Outputs: h_1
        - **h_1** of shape ``(batch, hidden_size)`` or ``(hidden_size,)``:
          tensor containing the next hidden state

    Variables:
        weight_ih: input–hidden weights,
            of shape ``(2*hidden_size, input_size)``
            (first half ``W_{ih}^z``, second half ``W_{ih}^k``)
        weight_hh: hidden–hidden weights for gate ``k``,
            of shape ``(hidden_size, hidden_size)``
        bias_ih: input biases ``[b_{ih}^z, b_{ih}^k]``,
            of shape ``(2*hidden_size,)`` if ``bias=True``
        bias_hh: hidden bias for gate ``k``,
            of shape ``(hidden_size,)`` if ``recurrent_bias=True``

    Examples::

        >>> cell = STARCell(16, 32)
        >>> seq = torch.randn(10, 8, 16)   # (time, batch, input_size)
        >>> h = torch.zeros(8, 32)         # (batch, hidden_size)
        >>> outs = []
        >>> for t in range(seq.size(0)):
        ...     h = cell(seq[t], h)
        ...     outs.append(h)
        >>> outs = torch.stack(outs, dim=0)  # (time, batch, hidden_size)
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
        super(STARCell, self).__init__(
            input_size, hidden_size, bias, recurrent_bias, device=device, dtype=dtype
        )
        self.kernel_init = kernel_init
        self.recurrent_kernel_init = recurrent_kernel_init
        self.bias_init = bias_init
        self.recurrent_bias_init = recurrent_bias_init

        self._default_register_tensors(
            input_size,
            hidden_size,
            ih_mult=2,
            hh_mult=1,
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

        inp_expanded = inp @ self.weight_ih.t() + self.bias_ih
        state_expanded = state @ self.weight_hh.t() + self.bias_hh
        gxs1, gxs2 = inp_expanded.chunk(2, 1)

        nonlinear_inpe = torch.tanh(gxs1)
        input_gate = torch.sigmoid(gxs2 + state_expanded)
        new_state = torch.tanh((1.0 - input_gate) * state + input_gate * nonlinear_inpe)

        if not is_batched:
            new_state = new_state.squeeze(0)

        return new_state

import torch
from torch import nn
from torch import Tensor
from typing import Optional, Callable, Tuple, Union
from ..base import BaseSingleRecurrentLayer, BaseSingleRecurrentCell


class LightRU(BaseSingleRecurrentLayer):
    r"""Multi-layer light recurrent unit neural network.

    [`pub <https://www.mdpi.com/2079-9292/13/16/3204>`_]

    Each layer consists of a :class:`LightRUCell`, which updates the hidden
    state according to:

    .. math::
        \begin{aligned}
            \tilde{h}_t &= \tanh(W_{ih}^h x_t + b_{ih}^h), \\
            f_t &= \sigma(W_{ih}^f x_t + b_{ih}^f
                   + W_{hh}^f h_{t-1} + b_{hh}^f), \\
            h_t &= (1 - f_t) \circ h_{t-1} + f_t \circ \tilde{h}_t
        \end{aligned}

    where :math:`h_t` is the hidden state at time `t`, :math:`x_t` is the input
    at time `t`, :math:`\sigma` is the sigmoid function, and :math:`\circ`
    denotes elementwise multiplication.

    In a multilayer LightRU, the input :math:`x^{(l)}_t` of the :math:`l`-th
    layer (:math:`l \ge 2`) is the hidden state :math:`h^{(l-1)}_t` of the
    previous layer multiplied by dropout :math:`\delta^{(l-1)}_t`, where each
    :math:`\delta^{(l-1)}_t` is a Bernoulli random variable which is 0 with
    probability :attr:`dropout`.

    Args:
        input_size: The number of expected features in the input `x`.
        hidden_size: The number of features in the hidden state `h`.
        num_layers: Number of recurrent layers. E.g., setting ``num_layers=2`` would
            mean stacking two LightRU layers, with the second receiving the outputs
            of the first. Default: 1
        dropout: If non-zero, introduces a `Dropout` layer on the outputs of each
            layer except the last layer, with dropout probability equal to
            :attr:`dropout`. Default: 0
        batch_first: If ``True``, then the input and output tensors are provided as
            `(batch, seq, feature)` instead of `(seq, batch, feature)`. Default: False
        bias: If ``False``, then the layer does not use input-side bias `b_ih`.
            Default: True
        recurrent_bias: If ``False``, then the layer does not use recurrent bias
            `b_hh`. Default: True
        kernel_init: Initializer for `W_{ih}`. Default:
            :func:`torch.nn.init.xavier_uniform_`
        recurrent_kernel_init: Initializer for `W_{hh}`. Default:
            :func:`torch.nn.init.xavier_uniform_`
        bias_init: Initializer for `b_{ih}`. Default: :func:`torch.nn.init.zeros_`
        recurrent_bias_init: Initializer for `b_{hh}`. Default:
            :func:`torch.nn.init.zeros_`
        device: The desired device of parameters.
        dtype: The desired floating point type of parameters.

    Inputs: input, h_0
        - **input**: tensor of shape :math:`(L, H_{in})` for unbatched input,
          :math:`(L, N, H_{in})` when ``batch_first=False`` or
          :math:`(N, L, H_{in})` when ``batch_first=True`` containing the features of
          the input sequence. The input can also be a packed variable length
          sequence. See :func:`torch.nn.utils.rnn.pack_padded_sequence` or
          :func:`torch.nn.utils.rnn.pack_sequence` for details.
        - **h_0**: tensor of shape :math:`(\text{num_layers}, H_{out})` for unbatched
          input or :math:`(\text{num_layers}, N, H_{out})` containing the initial
          hidden state for each element in the input sequence. Defaults to zeros if
          not provided.

        where:

        .. math::
            \begin{aligned}
                N ={} & \text{batch size} \\
                L ={} & \text{sequence length} \\
                H_{in} ={} & \text{input\_size} \\
                H_{out} ={} & \text{hidden\_size}
            \end{aligned}

    Outputs: output, h_n
        - **output**: tensor of shape :math:`(L, H_{out})` for unbatched input,
          :math:`(L, N, H_{out})` when ``batch_first=False`` or
          :math:`(N, L, H_{out})` when ``batch_first=True`` containing the output
          features `(h_t)` from the last layer of the LightRU, for each `t`. If a
          :class:`torch.nn.utils.rnn.PackedSequence` has been given as the input,
          the output will also be a packed sequence.
        - **h_n**: tensor of shape :math:`(\text{num_layers}, H_{out})` for unbatched
          input or :math:`(\text{num_layers}, N, H_{out})` containing the final
          hidden state for each element in the sequence.

    Attributes:
        cells.{k}.weight_ih : the learnable input-hidden weights of the :math:`k`-th
            layer, of shape `(2*hidden_size, input_size)` for `k = 0`. Otherwise, the
            shape is `(2*hidden_size, hidden_size)`.
        cells.{k}.weight_hh : the learnable hidden-hidden weights of the :math:`k`-th
            layer, of shape `(hidden_size, hidden_size)`.
        cells.{k}.bias_ih : the learnable input-hidden biases of the :math:`k`-th
            layer, of shape `(2*hidden_size)`. Only present when ``bias=True``.
        cells.{k}.bias_hh : the learnable hidden-hidden biases of the :math:`k`-th
            layer, of shape `(hidden_size)`. Only present when ``recurrent_bias=True``.

    .. note::
        All the weights and biases are initialized according to the provided
        initializers (`kernel_init`, `recurrent_kernel_init`, etc.).

    .. note::
        ``batch_first`` argument is ignored for unbatched inputs.

    .. seealso::
        :class:`LightRUCell`

    Examples::

        >>> rnn = LightRU(8, 16, num_layers=2, dropout=0.1)
        >>> input = torch.randn(5, 3, 8)    # (seq_len, batch, input_size)
        >>> h0 = torch.zeros(2, 3, 16)      # (num_layers, batch, hidden_size)
        >>> output, hn = rnn(input, h0)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        batch_first: bool = False,
        **kwargs,
    ):
        super(LightRU, self).__init__(
            input_size, hidden_size, num_layers, dropout, batch_first
        )
        self.initialize_cells(LightRUCell, **kwargs)


class LightRUCell(BaseSingleRecurrentCell):
    r"""A Light Recurrent Unit (LightRU) cell.

    [`pub <https://www.mdpi.com/2079-9292/13/16/3204>`_]

    .. math::

        \begin{aligned}
        \tilde{\mathbf{h}}(t) &= \tanh\bigl(
            \mathbf{W}_{ih}^{h}\,\mathbf{x}(t) + \mathbf{b}_{ih}^{h}
        \bigr), \\
            \mathbf{f}(t) &= \sigma\bigl(
            \mathbf{W}_{ih}^{f}\,\mathbf{x}(t) + \mathbf{b}_{ih}^{f}
            + \mathbf{W}_{hh}^{f}\,\mathbf{h}(t-1) + \mathbf{b}_{hh}^{f}
        \bigr), \\
            \mathbf{h}(t) &= \bigl(1 - \mathbf{f}(t)\bigr)\circ \mathbf{h}(t-1)
            \;+\; \mathbf{f}(t)\circ \tilde{\mathbf{h}}(t),
        \end{aligned}

    where :math:`\sigma` is the sigmoid function and
    :math:`\circ` denotes element-wise multiplication.

    Args:
        input_size: The number of expected features in the input ``x``
        hidden_size: The number of features in the hidden state ``h``
        bias: If ``False``, the layer does not use input-side bias ``b_{ih}``.
            Default: ``True``
        recurrent_bias: If ``False``, the layer does not use recurrent bias ``b_{hh}``.
            Default: ``True``
        kernel_init: Initializer for ``W_{ih}``.
            Default: :func:`torch.nn.init.xavier_uniform_`
        recurrent_kernel_init: Initializer for ``W_{hh}``.
            Default: :func:`torch.nn.init.xavier_uniform_`
        bias_init: Initializer for ``b_{ih}`` when ``bias=True``.
            Default: :func:`torch.nn.init.zeros_`
        recurrent_bias_init: Initializer for ``b_{hh}`` when ``recurrent_bias=True``.
            Default: :func:`torch.nn.init.zeros_`
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
            of shape ``(2*hidden_size, input_size)``
        weight_hh: the learnable hidden–hidden weights,
            of shape ``(hidden_size, hidden_size)``
        bias_ih: the learnable input–hidden biases,
            of shape ``(2*hidden_size)``
        bias_hh: the learnable hidden–hidden bias,
            of shape ``(hidden_size)``

    Examples::

        >>> cell = LightRUCell(8, 16)
        >>> x = torch.randn(5, 3, 8)     # (time_steps, batch, input_size)
        >>> h = torch.zeros(3, 16)       # (batch, hidden_size)
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
        super(LightRUCell, self).__init__(
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
        gxs1, gxs2 = inp_expanded.chunk(2, 1)

        candidate_state = torch.tanh(gxs1)
        forget_gate = torch.sigmoid(gxs2 + state @ self.weight_hh.t() + self.bias_hh)
        new_state = (1.0 - forget_gate) * state + forget_gate * candidate_state

        if not is_batched:
            new_state = new_state.squeeze(0)

        return new_state

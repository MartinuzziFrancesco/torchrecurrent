import torch
from torch import nn
from torch import Tensor
from typing import Optional, Callable, Tuple, Union
from ..base import BaseSingleRecurrentLayer, BaseSingleRecurrentCell


class FastRNN(BaseSingleRecurrentLayer):
    r"""Multi-layer fast recurrent neural network.

    [`arXiv <https://arxiv.org/abs/1901.02358>`_]

    Each layer consists of a :class:`FastRNNCell`, which updates the hidden
    state according to:

    .. math::
        \begin{aligned}
            \tilde{h}_t &= \phi(W_{ih} x_t + b_{ih} + W_{hh} h_{t-1} + b_{hh}), \\
            h_t &= \sigma(\alpha) \tilde{h}_t + \sigma(\beta) h_{t-1},
        \end{aligned}

    where :math:`h_t` is the hidden state at time `t`, :math:`x_t` is the input
    at time `t`, :math:`\phi` is a pointwise nonlinearity (e.g., :math:`\tanh`),
    and :math:`\alpha`, :math:`\beta` are learnable scalar coefficients.

    In a multilayer FastRNN, the input :math:`x^{(l)}_t` of the :math:`l`-th
    layer (:math:`l \ge 2`) is the hidden state :math:`h^{(l-1)}_t` of the
    previous layer multiplied by dropout :math:`\delta^{(l-1)}_t`, where each
    :math:`\delta^{(l-1)}_t` is a Bernoulli random variable which is 0 with
    probability :attr:`dropout`.

    Args:
        input_size: The number of expected features in the input `x`.
        hidden_size: The number of features in the hidden state `h`.
        num_layers: Number of recurrent layers. E.g., setting ``num_layers=2`` would
            mean stacking two FastRNN layers, with the second receiving the outputs
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
        nonlinearity: Activation function :math:`\phi` applied to the candidate
            state. Default: :func:`torch.tanh`
        kernel_init: Initializer for `W_{ih}`. Default:
            :func:`torch.nn.init.xavier_uniform_`
        recurrent_kernel_init: Initializer for `W_{hh}`. Default:
            :func:`torch.nn.init.xavier_uniform_`
        bias_init: Initializer for `b_{ih}`. Default: :func:`torch.nn.init.zeros_`
        recurrent_bias_init: Initializer for `b_{hh}`. Default:
            :func:`torch.nn.init.zeros_`
        alpha_init: Initial value for the learnable scalar :math:`\alpha`.
            Default: 3.0
        beta_init: Initial value for the learnable scalar :math:`\beta`.
            Default: -3.0
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
          features `(h_t)` from the last layer of the FastRNN, for each `t`. If a
          :class:`torch.nn.utils.rnn.PackedSequence` has been given as the input,
          the output will also be a packed sequence.
        - **h_n**: tensor of shape :math:`(\text{num_layers}, H_{out})` for unbatched
          input or :math:`(\text{num_layers}, N, H_{out})` containing the final
          hidden state for each element in the sequence.

    Attributes:
        cells.{k}.weight_ih : the learnable input-hidden weights of the :math:`k`-th
            layer, of shape `(hidden_size, input_size)` for `k = 0`. Otherwise, the
            shape is `(hidden_size, hidden_size)`.
        cells.{k}.weight_hh : the learnable hidden-hidden weights of the :math:`k`-th
            layer, of shape `(hidden_size, hidden_size)`.
        cells.{k}.bias_ih : the learnable input-hidden bias of the :math:`k`-th
            layer, of shape `(hidden_size)`. Only present when ``bias=True``.
        cells.{k}.bias_hh : the learnable hidden-hidden bias of the :math:`k`-th
            layer, of shape `(hidden_size)`. Only present when ``recurrent_bias=True``.
        cells.{k}.alpha : the learnable scalar :math:`\alpha` of the :math:`k`-th
            layer, of shape `(1,)`.
        cells.{k}.beta : the learnable scalar :math:`\beta` of the :math:`k`-th
            layer, of shape `(1,)`.

    .. note::
        All the weights and biases are initialized according to the provided
        initializers (`kernel_init`, `recurrent_kernel_init`, etc.). Scalars
        :math:`\alpha` and :math:`\beta` are initialized from `alpha_init` and
        `beta_init`.

    .. note::
        ``batch_first`` argument is ignored for unbatched inputs.

    .. seealso::
        :class:`FastRNNCell`

    Examples::

        >>> rnn = FastRNN(10, 20, num_layers=2, dropout=0.1)
        >>> input = torch.randn(5, 3, 10)   # (seq_len, batch, input_size)
        >>> h0 = torch.zeros(2, 3, 20)      # (num_layers, batch, hidden_size)
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
        super(FastRNN, self).__init__(
            input_size, hidden_size, num_layers, dropout, batch_first
        )
        self.initialize_cells(FastRNNCell, **kwargs)


class FastRNNCell(BaseSingleRecurrentCell):
    r"""A Fast RNN cell with two scalar gates :math:`\alpha` and :math:`\beta`.

    [`arXiv <https://arxiv.org/abs/1901.02358>`_]

    .. math::

        \tilde{\mathbf{h}}(t) &= \phi\bigl(
            \mathbf{W}_{ih}\,\mathbf{x}(t)
            + \mathbf{b}_{ih}
            + \mathbf{W}_{hh}\,\mathbf{h}(t-1)
            + \mathbf{b}_{hh}
        \bigr), \\[6pt]
        \mathbf{h}(t) &= \sigma(\alpha)\,\tilde{\mathbf{h}}(t)
                        + \sigma(\beta)\,\mathbf{h}(t-1),

    where :math:`\phi` is a pointwise nonlinearity (e.g., tanh), and
    :math:`\alpha` / :math:`\beta` are learnable scalars.

    Args:
        input_size: The number of expected features in the input ``x``.
        hidden_size: The number of features in the hidden state ``h``.
        bias: If ``False``, the layer does not use input-side bias ``b_{ih}``.
            Default: ``True``.
        recurrent_bias: If ``False``, the layer does not use recurrent bias
            ``b_{hh}``. Default: ``True``.
        nonlinearity: Activation function :math:`\phi` for the candidate.
            Default: :func:`torch.tanh`.
        kernel_init: Initializer for ``W_{ih}``.
            Default: :func:`torch.nn.init.xavier_uniform_`.
        recurrent_kernel_init: Initializer for ``W_{hh}``.
            Default: :func:`torch.nn.init.xavier_uniform_`.
        bias_init: Initializer for ``b_{ih}`` when ``bias=True``.
            Default: :func:`torch.nn.init.zeros_`.
        recurrent_bias_init: Initializer for ``b_{hh}`` when
            ``recurrent_bias=True``. Default: :func:`torch.nn.init.zeros_`.
        alpha_init: Initial value for the learnable scalar :math:`\alpha`.
            Default: ``3.0``.
        beta_init: Initial value for the learnable scalar :math:`\beta`.
            Default: ``-3.0``.
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
            of shape ``(hidden_size, input_size)``.
        weight_hh: The learnable hidden–hidden weights,
            of shape ``(hidden_size, hidden_size)``.
        bias_ih: The learnable input–hidden bias,
            of shape ``(hidden_size)`` if ``bias=True``.
        bias_hh: The learnable hidden–hidden bias,
            of shape ``(hidden_size)`` if ``recurrent_bias=True``.
        alpha: The learnable scalar gating coefficient :math:`\alpha`,
            of shape ``(1,)``.
        beta: The learnable scalar gating coefficient :math:`\beta`,
            of shape ``(1,)``.

    Examples::

        >>> cell = FastRNNCell(10, 20)
        >>> x = torch.randn(5, 3, 10)      # (time_steps, batch, input_size)
        >>> h = torch.zeros(3, 20)         # (batch, hidden_size)
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
        "kernel_init",
        "recurrent_kernel_init",
        "bias_init",
        "recurrent_bias_init",
        "alpha_init",
        "beta_init",
    ]

    weight_ih: Tensor
    weight_hh: Tensor
    bias_ih: Tensor
    bias_hh: Tensor
    alpha: Tensor
    beta: Tensor

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        recurrent_bias: bool = True,
        nonlinearity: Callable = torch.tanh,
        kernel_init: Callable = nn.init.xavier_uniform_,
        recurrent_kernel_init: Callable = nn.init.xavier_uniform_,
        bias_init: Callable = nn.init.zeros_,
        recurrent_bias_init: Callable = nn.init.zeros_,
        alpha_init: float = 3.0,
        beta_init: float = -3.0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super(FastRNNCell, self).__init__(
            input_size, hidden_size, bias, device=device, dtype=dtype
        )
        self.kernel_init = kernel_init
        self.recurrent_kernel_init = recurrent_kernel_init
        self.bias_init = bias_init
        self.recurrent_bias_init = recurrent_bias_init
        self.alpha_init = alpha_init
        self.beta_init = beta_init
        self.nonlinearity = nonlinearity

        self._register_tensors(
            {
                "weight_ih": ((hidden_size, input_size), True),
                "weight_hh": ((hidden_size, hidden_size), True),
                "bias_ih": ((hidden_size,), bias),
                "bias_hh": ((hidden_size,), recurrent_bias),
                "alpha": ((1,), True),
                "beta": ((1,), True),
            }
        )
        self.init_weights()

    def init_weights(self):
        for name, param in self.named_parameters():
            if name.endswith("weight_ih"):
                self.kernel_init(param)
            elif name.endswith("weight_hh"):
                self.recurrent_kernel_init(param)
            elif name.endswith("bias_ih"):
                self.bias_init(param)
            elif name.endswith("bias_hh"):
                self.recurrent_bias_init(param)
            elif name == "alpha":
                nn.init.constant_(param, self.alpha_init)
            elif name == "beta":
                nn.init.constant_(param, self.beta_init)

    def forward(
        self, inp: Tensor, state: Optional[Union[Tensor, Tuple[Tensor, ...]]] = None
    ) -> Tensor:
        state = self._check_state(state)
        self._validate_input(inp)
        self._validate_state(state)
        inp, state, is_batched = self._preprocess_input_and_state(inp, state)

        candidate_state = self.nonlinearity(
            inp @ self.weight_ih.t()
            + self.bias_ih
            + state @ self.weight_hh.t()
            + self.bias_hh
        )
        alpha = torch.sigmoid(self.alpha)
        beta = torch.sigmoid(self.beta)
        new_state = alpha * candidate_state + beta * state

        if not is_batched:
            new_state = new_state.squeeze(0)

        return new_state


class FastGRNN(BaseSingleRecurrentLayer):
    r"""Multi-layer FastGRNN.

    [`arXiv <https://arxiv.org/abs/1901.02358>`_]

    Each layer is a :class:`FastGRNNCell`, a gated recurrent unit with two
    scalar parameters :math:`\zeta` and :math:`\nu` controlling the tradeoff
    between the candidate state and the previous hidden state:

    .. math::
        \begin{aligned}
            z(t) &= \sigma(W_{ih} x(t) + b_{ih}^z
                + W_{hh} h(t-1) + b_{hh}^z), \\
            \tilde{h}(t) &= \tanh(W_{ih} x(t) + b_{ih}^h
                + W_{hh} h(t-1) + b_{hh}^h), \\
            h(t) &= \bigl[\sigma(\zeta) (1 - z(t)) + \sigma(\nu)\bigr] \circ \tilde{h}(t)
                + z(t) \circ h(t-1),
        \end{aligned}

    where :math:`\sigma` is the sigmoid and :math:`\circ` is elementwise
    multiplication.

    Args:
        input_size: Number of expected features in the input `x`.
        hidden_size: Number of features in the hidden state.
        num_layers: Number of stacked recurrent layers. Default: 1
        dropout: If non-zero, applies dropout after each layer (except last).
            Default: 0
        batch_first: If ``True``, inputs/outputs are shaped
            `(batch, seq, feature)` instead of `(seq, batch, feature)`.
            Default: False
        bias: If ``False``, disables input biases. Default: True
        recurrent_bias: If ``False``, disables recurrent biases. Default: True
        nonlinearity: Activation for the gate :math:`z`. Default: :func:`torch.sigmoid`
        kernel_init: Initializer for `W_{ih}`.
            Default: :func:`torch.nn.init.xavier_uniform_`
        recurrent_kernel_init: Initializer for `W_{hh}`.
            Default: :func:`torch.nn.init.xavier_uniform_`
        bias_init: Initializer for `b_{ih}`.
            Default: :func:`torch.nn.init.zeros_`
        recurrent_bias_init: Initializer for `b_{hh}`.
            Default: :func:`torch.nn.init.zeros_`
        zeta_init: Initial value for scalar :math:`\zeta`. Default: 3.0
        nu_init: Initial value for scalar :math:`\nu`. Default: -3.0
        device: Desired device of parameters.
        dtype: Desired floating point type of parameters.

    Inputs: input, h_0
        - **input**: tensor of shape `(L, H_in)` for unbatched input,
          `(L, N, H_in)` when ``batch_first=False``, or `(N, L, H_in)` when
          ``batch_first=True``.
        - **h_0**: tensor of shape `(num_layers, H_out)` for unbatched input,
          or `(num_layers, N, H_out)` when batched. Defaults to zeros.

        Where:

        .. math::
            \begin{aligned}
                N &= \text{batch size} \\
                L &= \text{sequence length} \\
                H_{in} &= \text{input size} \\
                H_{out} &= \text{hidden size}
            \end{aligned}

    Outputs: output, h_n
        - **output**: tensor of shape `(L, H_out)` for unbatched input,
          `(L, N, H_out)` when ``batch_first=False``, or `(N, L, H_out)` when
          ``batch_first=True``, containing hidden states from the last layer.
        - **h_n**: tensor of shape `(num_layers, H_out)` (unbatched) or
          `(num_layers, N, H_out)` with the final hidden state.

    Attributes:
        cells.{k}.weight_ih : input–hidden weights of the :math:`k`-th layer,
            shape `(hidden_size, input_size)` for `k=0`, otherwise
            `(hidden_size, hidden_size)`.
        cells.{k}.weight_hh : hidden–hidden weights of the :math:`k`-th layer,
            shape `(hidden_size, hidden_size)`.
        cells.{k}.bias_ih : input biases of the :math:`k`-th layer,
            shape `(2*hidden_size,)` if ``bias=True``.
        cells.{k}.bias_hh : hidden biases of the :math:`k`-th layer,
            shape `(2*hidden_size,)` if ``recurrent_bias=True``.
        cells.{k}.zeta : scalar parameter :math:`\zeta`, shape `(1,)`.
        cells.{k}.nu : scalar parameter :math:`\nu`, shape `(1,)`.

    .. seealso::
        :class:`FastGRNNCell`

    Examples::

        >>> rnn = FastGRNN(10, 20, num_layers=2)
        >>> x = torch.randn(5, 3, 10)     # (seq_len, batch, input_size)
        >>> h0 = torch.zeros(2, 3, 20)
        >>> out, hn = rnn(x, h0)
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
        super(FastGRNN, self).__init__(
            input_size, hidden_size, num_layers, dropout, batch_first
        )
        self.initialize_cells(FastGRNNCell, **kwargs)


class FastGRNNCell(BaseSingleRecurrentCell):
    r"""A “Fast RNN” cell with two scalar gates.

    [`arXiv <https://arxiv.org/abs/1901.02358>`_]

        .. math::
            \begin{aligned}
                z(t) &= \sigma(W_{ih} x(t) + b_{ih}^z
                    + W_{hh} h(t-1) + b_{hh}^z), \\
                \tilde{h}(t) &= \tanh(W_{ih} x(t) + b_{ih}^h
                    + W_{hh} h(t-1) + b_{hh}^h), \\
                h(t) &= \bigl[\sigma(\zeta) (1 - z(t)) + \sigma(\nu)\bigr] \circ \tilde{h}(t)
                    + z(t) \circ h(t-1),
            \end{aligned}

        where :math:`\circ` denotes element‐wise product.

    Args:
        input_size: The number of expected features in the input ``x``
        hidden_size: The number of features in the hidden state ``h``
        bias: If ``False``, the layer does not use input-side biases.
            Default: ``True``
        recurrent_bias: If ``False``, the layer does not use recurrent biases.
            Default: ``True``
        nonlinearity: Activation for the gate :math:`\mathbf{z}`.
            Default: :func:`torch.sigmoid`
        kernel_init: Initializer for ``W_{ih}``.
            Default: :func:`torch.nn.init.xavier_uniform_`
        recurrent_kernel_init: Initializer for ``W_{hh}``.
            Default: :func:`torch.nn.init.xavier_uniform_`
        bias_init: Initializer for ``b_{ih}^{*}`` when ``bias=True``.
            Default: :func:`torch.nn.init.zeros_`
        recurrent_bias_init: Initializer for ``b_{hh}^{*}`` when
            ``recurrent_bias=True``. Default: :func:`torch.nn.init.zeros_`
        zeta_init: Initial value for scalar gate :math:`\zeta`. Default: ``3.0``
        nu_init: Initial value for scalar gate :math:`\nu`. Default: ``-3.0``
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
        weight_ih: the learnable input–hidden weights, of shape
            ``(hidden_size, input_size)``
        weight_hh: the learnable hidden–hidden weights, of shape
            ``(hidden_size, hidden_size)``
        bias_ih: the learnable input–hidden biases, of shape ``(2*hidden_size)``
            (split into z & h) if ``bias=True``
        bias_hh: the learnable hidden–hidden biases, of shape ``(2*hidden_size)``
            (split into z & h) if ``recurrent_bias=True``
        zeta: the learnable scalar gate :math:`\zeta`, shape ``(1,)``
        nu: the learnable scalar gate :math:`\nu`, shape ``(1,)``
        t_ones: a constant ones buffer, shape ``(hidden_size,)``

    Examples::

        >>> cell = FastGRNNCell(10, 20)
        >>> x = torch.randn(5, 3, 10)      # (time_steps, batch, input_size)
        >>> h = torch.zeros(3, 20)         # (batch, hidden_size)
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
        "kernel_init",
        "recurrent_kernel_init",
        "bias_init",
        "recurrent_bias_init",
        "zeta_init",
        "nu_init",
    ]

    weight_ih: Tensor
    weight_hh: Tensor
    bias_ih: Tensor
    bias_hh: Tensor
    zeta: Tensor
    nu: Tensor

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        recurrent_bias: bool = True,
        nonlinearity: Callable = torch.tanh,
        kernel_init: Callable = nn.init.xavier_uniform_,
        recurrent_kernel_init: Callable = nn.init.xavier_uniform_,
        bias_init: Callable = nn.init.zeros_,
        recurrent_bias_init: Callable = nn.init.zeros_,
        zeta_init: float = 3.0,
        nu_init: float = -3.0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super(FastGRNNCell, self).__init__(
            input_size, hidden_size, bias, device=device, dtype=dtype
        )
        self.kernel_init = kernel_init
        self.recurrent_kernel_init = recurrent_kernel_init
        self.bias_init = bias_init
        self.recurrent_bias_init = recurrent_bias_init
        self.zeta_init = zeta_init
        self.nu_init = nu_init
        self.nonlinearity = nonlinearity

        self._register_tensors(
            {
                "weight_ih": ((hidden_size, input_size), True),
                "weight_hh": ((hidden_size, hidden_size), True),
                "bias_ih": ((2 * hidden_size,), bias),
                "bias_hh": ((2 * hidden_size,), recurrent_bias),
                "zeta": ((1,), True),
                "nu": ((1,), True),
            }
        )
        self.init_weights()

    def init_weights(self):
        for name, param in self.named_parameters():
            if name.endswith("weight_ih"):
                self.kernel_init(param)
            elif name.endswith("weight_hh"):
                self.recurrent_kernel_init(param)
            elif name.endswith("bias_ih"):
                self.bias_init(param)
            elif name.endswith("bias_hh"):
                self.recurrent_bias_init(param)
            elif name == "zeta":
                nn.init.constant_(param, self.zeta_init)
            elif name == "nu":
                nn.init.constant_(param, self.nu_init)

    def forward(
        self, inp: Tensor, state: Optional[Union[Tensor, Tuple[Tensor, ...]]] = None
    ) -> Tensor:
        state = self._check_state(state)
        self._validate_input(inp)
        self._validate_state(state)
        inp, state, is_batched = self._preprocess_input_and_state(inp, state)

        bias_ih_1, bias_ih_2 = self.bias_ih.chunk(2)
        bias_hh_1, bias_hh_2 = self.bias_hh.chunk(2)

        partial_gate = inp @ self.weight_ih.t() + state @ self.weight_hh.t()
        gate = self.nonlinearity(partial_gate + bias_ih_1 + bias_hh_1)
        candidate_state = torch.tanh(partial_gate + bias_ih_2 + bias_hh_2)
        zeta = torch.sigmoid(self.zeta)
        nu = torch.sigmoid(self.nu)
        new_state = (zeta * (1.0 - gate) + nu) * candidate_state + gate * state

        if not is_batched:
            new_state = new_state.squeeze(0)

        return new_state

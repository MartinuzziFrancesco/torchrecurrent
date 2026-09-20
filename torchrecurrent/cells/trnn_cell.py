import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Tuple
from ..base import (
    SingleStateRecurrentLayerBase,
    SingleStateCellBase,
    DoubleStateRecurrentLayerBase,
    DoubleStateCellBase,
    resolve_activation,
    resolve_init_name,
)


class TRNN(SingleStateRecurrentLayerBase):
    r"""Multi-layer strongly typed recurrent neural network (T-RNN).

    [`arXiv <https://arxiv.org/abs/1602.02218>`_]

    Each layer consists of a :class:`TRNNCell`, which updates the hidden state
    according to:

    .. math::
        \begin{aligned}
        z_t &= W_{ih}^z x_t + b_{ih}^z, \\
        f_t &= \sigma(W_{ih}^f x_t + b_{ih}^f), \\
        h_t &= f_t \circ h_{t-1} + (1 - f_t) \circ z_t
        \end{aligned}

    where :math:`h_t` is the hidden state at time `t`, :math:`x_t` is the
    input at time `t`, :math:`\sigma` is the sigmoid function, and
    :math:`\circ` denotes elementwise (Hadamard) product. Unlike most
    recurrent cells, T-RNN has no hidden-to-hidden weight matrix: the type
    constraint of "strongly typed" architectures forbids a learned linear
    map acting on :math:`h_{t-1}`, so it only ever enters the update through
    the elementwise gate above.

    In a multilayer TRNN, the input :math:`x^{(l)}_t` of the :math:`l`-th
    layer (:math:`l \ge 2`) is the hidden state :math:`h^{(l-1)}_t` of the
    previous layer multiplied by dropout :math:`\delta^{(l-1)}_t`, where each
    :math:`\delta^{(l-1)}_t` is a Bernoulli random variable which is 0 with
    probability :attr:`dropout`.

    Args:
        input_size: The number of expected features in the input `x`.
        hidden_size: The number of features in the hidden state `h`.
        num_layers: Number of recurrent layers. E.g., setting ``num_layers=2`` would
            mean stacking two TRNN layers, with the second receiving the outputs of
            the first. Default: 1
        dropout: If non-zero, introduces a `Dropout` layer on the outputs of each
            layer except the last layer, with dropout probability equal to
            :attr:`dropout`. Default: 0
        batch_first: If ``True``, then the input and output tensors are provided as
            `(batch, seq, feature)` instead of `(seq, batch, feature)`. Default: False
        bias: If ``False``, then the layer does not use biases.
            Default: True
        gate_nonlinearity: Activation :math:`\sigma` for the forget gate. Default:
            :func:`torch.sigmoid`
        kernel_init: Initializer for `W_{ih}`. Default:
            :func:`torch.nn.init.xavier_uniform_`
        bias_init: Initializer for `b_{ih}`. Default:
            :func:`torch.nn.init.zeros_`
        device: The desired device of parameters.
        dtype: The desired floating point type of parameters.

    Inputs: input, h_0
        - **input**: tensor of shape :math:`(L, N, H_{in})` when
          ``batch_first=False`` or :math:`(N, L, H_{in})` when
          ``batch_first=True`` containing the features of the input sequence.
        - **h_0**: tensor of shape :math:`(\text{num_layers}, N, H_{out})`
          containing the initial hidden state for each element in the input
          sequence. Defaults to zeros if not provided.

        where:

        .. math::
            \begin{aligned}
                N ={} & \text{batch size} \\
                L ={} & \text{sequence length} \\
                H_{in} ={} & \text{input\_size} \\
                H_{out} ={} & \text{hidden\_size}
            \end{aligned}

    Outputs: output, h_n
        - **output**: tensor of shape :math:`(L, N, H_{out})` when
          ``batch_first=False`` or :math:`(N, L, H_{out})` when
          ``batch_first=True`` containing the output features `(h_t)` from
          the last layer of the TRNN, for each `t`.
        - **h_n**: tensor of shape :math:`(\text{num_layers}, N, H_{out})`
          containing the final hidden state for each element in the sequence.

    Attributes:
        cells.{k}.weight_ih : the learnable input-hidden weights of the :math:`k`-th
            layer, of shape `(2*hidden_size, input_size)` for `k = 0`. Otherwise, the
            shape is `(2*hidden_size, hidden_size)`.
        cells.{k}.bias_ih : the learnable input-hidden biases of the :math:`k`-th
            layer, of shape `(2*hidden_size)`. Only present when ``bias=True``.

    .. note::
        All the weights and biases are initialized according to the provided
        initializers (`kernel_init`, `bias_init`).

    .. seealso::
        :class:`TRNNCell`

    Examples::

        >>> rnn = TRNN(10, 20, num_layers=2, dropout=0.1)
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
        super(TRNN, self).__init__(
            input_size, hidden_size, num_layers, dropout, batch_first
        )
        self.initialize_cells(TRNNCell, **kwargs)


class TRNNCell(SingleStateCellBase):
    r"""A strongly typed recurrent neural network (T-RNN) cell.

    [`arXiv <https://arxiv.org/abs/1602.02218>`_]

    .. math::

        \mathbf{z}(t) &= \mathbf{W}_{ih}^{z}\,\mathbf{x}(t) + \mathbf{b}_{ih}^{z}, \\[4pt]
        \mathbf{f}(t) &= \sigma\bigl(
            \mathbf{W}_{ih}^{f}\,\mathbf{x}(t) + \mathbf{b}_{ih}^{f}
        \bigr), \\[4pt]
        \mathbf{h}(t) &= \mathbf{f}(t)\circ\mathbf{h}(t-1)
            + \bigl(1 - \mathbf{f}(t)\bigr)\circ\mathbf{z}(t),

    where :math:`\circ` is element-wise product. Following the "strongly
    typed" design, there is no weight matrix acting on :math:`\mathbf{h}(t-1)`:
    it enters the update only through the elementwise mix with the gate
    :math:`\mathbf{f}(t)`, which itself depends only on the current input.

    Args:
        input_size: The number of expected features in the input ``x``.
        hidden_size: The number of features in the hidden state ``h``.
        bias: If ``False``, the layer does not use biases.
            Default: ``True``.
        gate_nonlinearity: Activation :math:`\sigma` for the forget gate.
            Default: :func:`torch.sigmoid`.
        kernel_init: Initializer for ``W_{ih}``.
            Default: :func:`torch.nn.init.xavier_uniform_`.
        bias_init: Initializer for ``b_{ih}`` when ``bias=True``.
            Default: :func:`torch.nn.init.zeros_`.
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
        weight_ih: The learnable input-hidden weights,
            of shape ``(2*hidden_size, input_size)`` (``z, f`` parts).
        bias_ih: The learnable input-hidden biases,
            of shape ``(2*hidden_size)`` if ``bias=True``.

    Examples::

        >>> cell = TRNNCell(10, 20)
        >>> x = torch.randn(5, 3, 10)     # (time_steps, batch, input_size)
        >>> h = torch.zeros(3, 20)        # (batch, hidden_size)
        >>> out = []
        >>> for t in range(x.size(0)):
        ...     h = cell(x[t], h)
        ...     out.append(h)
        >>> out = torch.stack(out, dim=0) # (time_steps, batch, hidden_size)
    """

    __constants__ = ["input_size", "hidden_size", "bias"]

    weight_ih: Tensor
    bias_ih: Tensor

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        gate_nonlinearity="sigmoid",
        kernel_init=nn.init.xavier_uniform_,
        bias_init=nn.init.zeros_,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            bias=bias,
            recurrent_bias=False,
            device=device,
            dtype=dtype,
        )
        self.gate_act = resolve_activation(gate_nonlinearity)
        self.init_cfg["kernel"] = resolve_init_name(kernel_init, self.init_cfg["kernel"])
        self.init_cfg["bias"] = resolve_init_name(bias_init, self.init_cfg["bias"])

        self._register_tensors(
            {
                "weight_ih": ((2 * hidden_size, input_size), True),
                "bias_ih": ((2 * hidden_size,), self.bias),
            }
        )
        self.reset_parameters()
        self._cleanup_non_scriptable()

    def forward(self, inp: Tensor, state: Optional[Tensor] = None) -> Tensor:
        self._validate_input(inp)
        b_inp, is_batched = self._as_batched(inp)

        if state is None:
            b_state = self._zeros_state(b_inp.size(0), b_inp.device, b_inp.dtype)
        else:
            b_state = state.unsqueeze(0) if (not is_batched and state.dim() == 1) else state

        latent, forget_pre = (b_inp @ self.weight_ih.t() + self.bias_ih).chunk(2, 1)
        forget_gate = self.gate_act(forget_pre)
        new_state = forget_gate * b_state + (1.0 - forget_gate) * latent

        if not is_batched:
            new_state = new_state.squeeze(0)

        return new_state


class TGRU(DoubleStateRecurrentLayerBase):
    r"""Multi-layer strongly typed gated recurrent unit (T-GRU).

    [`arXiv <https://arxiv.org/abs/1602.02218>`_]

    Each layer consists of a :class:`TGRUCell`, which updates the hidden
    state according to:

    .. math::
        \begin{aligned}
        z_t &= W_{ih}^z x_t + b_{ih}^z + W_{hh}^z x_{t-1} + b_{hh}^z, \\
        f_t &= \sigma(W_{ih}^f x_t + b_{ih}^f + W_{hh}^f x_{t-1} + b_{hh}^f), \\
        o_t &= \tau(W_{ih}^o x_t + b_{ih}^o + W_{hh}^o x_{t-1} + b_{hh}^o), \\
        h_t &= f_t \circ h_{t-1} + z_t \circ o_t
        \end{aligned}

    where :math:`\sigma` is the sigmoid function, :math:`\tau` is a pointwise
    nonlinearity (e.g., tanh), and :math:`\circ` denotes elementwise
    (Hadamard) product. As in T-RNN, no weight matrix acts on the hidden
    state: the recurrent weights :math:`W_{hh}^{*}` act on the *previous
    input* :math:`x_{t-1}` rather than on :math:`h_{t-1}`, keeping the type
    of every linear map consistent with the "strongly typed" design.

    In a multilayer TGRU, the input :math:`x^{(l)}_t` of the :math:`l`-th
    layer (:math:`l \ge 2`) is the hidden state :math:`h^{(l-1)}_t` of the
    previous layer multiplied by dropout :math:`\delta^{(l-1)}_t`, where each
    :math:`\delta^{(l-1)}_t` is a Bernoulli random variable which is 0 with
    probability :attr:`dropout`.

    Args:
        input_size: The number of expected features in the input `x`.
        hidden_size: The number of features in the hidden state `h`.
        num_layers: Number of recurrent layers. E.g., setting ``num_layers=2`` would
            mean stacking two TGRU layers, with the second receiving the outputs of
            the first. Default: 1
        dropout: If non-zero, introduces a `Dropout` layer on the outputs of each
            layer except the last layer, with dropout probability equal to
            :attr:`dropout`. Default: 0
        batch_first: If ``True``, then the input and output tensors are provided as
            `(batch, seq, feature)` instead of `(seq, batch, feature)`. Default: False
        bias: If ``False``, then the layer does not use input-side biases.
            Default: True
        recurrent_bias: If ``False``, then the layer does not use recurrent biases.
            Default: True
        gate_nonlinearity: Activation :math:`\sigma` for the forget gate. Default:
            :func:`torch.sigmoid`
        output_nonlinearity: Activation :math:`\tau` for the output candidate.
            Default: :func:`torch.tanh`
        kernel_init: Initializer for `W_{ih}`. Default:
            :func:`torch.nn.init.xavier_uniform_`
        recurrent_kernel_init: Initializer for `W_{hh}`. Default:
            :func:`torch.nn.init.xavier_uniform_`
        bias_init: Initializer for `b_{ih}`. Default:
            :func:`torch.nn.init.zeros_`
        recurrent_bias_init: Initializer for `b_{hh}`. Default:
            :func:`torch.nn.init.zeros_`
        device: The desired device of parameters.
        dtype: The desired floating point type of parameters.

    Inputs: input, (h_0, x_{-1})
        - **input**: tensor of shape :math:`(L, N, H_{in})` when
          ``batch_first=False`` or :math:`(N, L, H_{in})` when
          ``batch_first=True`` containing the features of the input sequence.
        - **h_0**: tensor of shape :math:`(\text{num_layers}, N, H_{out})`
          containing the initial hidden state. Defaults to zeros if not
          provided.
        - **x_{-1}**: tuple of :attr:`num_layers` tensors containing the
          "previous input" fed to each layer before the sequence starts. The
          :math:`k`-th tensor has shape :math:`(N, \text{input\_size})` for
          `k = 0` and :math:`(N, H_{out})` for `k \ge 1`, since layers beyond
          the first receive the previous layer's hidden state as their
          input. A single stacked tensor of shape
          :math:`(\text{num_layers}, N, H_{in})` is also accepted whenever
          `input_size == hidden_size`. Defaults to zeros if not provided.

        where:

        .. math::
            \begin{aligned}
                N ={} & \text{batch size} \\
                L ={} & \text{sequence length} \\
                H_{in} ={} & \text{input\_size} \\
                H_{out} ={} & \text{hidden\_size}
            \end{aligned}

    Outputs: output, (h_n, x_n)
        - **output**: tensor of shape :math:`(L, N, H_{out})` when
          ``batch_first=False`` or :math:`(N, L, H_{out})` when
          ``batch_first=True`` containing the output features `(h_t)` from
          the last layer of the TGRU, for each `t`.
        - **h_n**: tensor of shape :math:`(\text{num_layers}, N, H_{out})`
          containing the final hidden state for each element in the sequence.
        - **x_n**: tuple of :attr:`num_layers` tensors, with the same
          per-layer shapes as **x_{-1}**, containing the last input seen by
          each layer. Returned as a tuple (not stacked into one tensor)
          because layer 0's input dimension may differ from every other
          layer's.

    Attributes:
        cells.{k}.weight_ih : the learnable input-hidden weights of the :math:`k`-th
            layer, of shape `(3*hidden_size, input_size)` for `k = 0`. Otherwise, the
            shape is `(3*hidden_size, hidden_size)`.
        cells.{k}.weight_hh : the learnable weights applied to the previous input of
            the :math:`k`-th layer, of the same shape as `weight_ih`.
        cells.{k}.bias_ih : the learnable input-hidden biases of the :math:`k`-th
            layer, of shape `(3*hidden_size)`. Only present when ``bias=True``.
        cells.{k}.bias_hh : the learnable biases paired with `weight_hh` of the
            :math:`k`-th layer, of shape `(3*hidden_size)`. Only present when
            ``recurrent_bias=True``.

    .. note::
        All the weights and biases are initialized according to the provided
        initializers (`kernel_init`, `recurrent_kernel_init`, etc.).

    .. note::
        Unlike every other multi-layer recurrent layer in this package,
        TGRU does not stack its per-layer state into a single tensor for the
        "previous input" component, since layer 0's input dimension can
        differ from every subsequent layer's hidden dimension. This layer
        therefore overrides the generic double-state forward loop.

    .. seealso::
        :class:`TGRUCell`

    Examples::

        >>> rnn = TGRU(10, 20, num_layers=2, dropout=0.1)
        >>> input = torch.randn(5, 3, 10)   # (seq_len, batch, input_size)
        >>> output, (hn, xn) = rnn(input)   # default zero state
        >>> xn[0].shape                     # layer 0: input_size-shaped
        torch.Size([3, 10])
        >>> xn[1].shape                     # layer 1: hidden_size-shaped
        torch.Size([3, 20])
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
        super(TGRU, self).__init__(
            input_size, hidden_size, num_layers, dropout, batch_first
        )
        self.initialize_cells(TGRUCell, **kwargs)

    def forward(
        self, inp: Tensor, state: Optional[Tuple[Tensor, Tuple[Tensor, ...]]] = None
    ) -> Tuple[Tensor, Tuple[Tensor, Tuple[Tensor, ...]]]:
        if self.batch_first:
            inp = inp.transpose(0, 1)

        seq_len, batch_size, _ = inp.size()

        if state is None:
            h = [
                torch.zeros(
                    batch_size, self.hidden_size, dtype=inp.dtype, device=inp.device
                )
                for _ in range(self.num_layers)
            ]
            x_prev = [
                torch.zeros(batch_size, cell.input_size, dtype=inp.dtype, device=inp.device)
                for cell in self.cells
            ]
        else:
            h0, x_prev0 = state
            h = [h0[layer_idx] for layer_idx in range(self.num_layers)]
            x_prev = list(x_prev0)

        outputs = []
        for t in range(seq_len):
            x = inp[t]
            new_h = []
            new_x_prev = []

            for layer_idx, cell in enumerate(self.cells):
                h_i, x_prev_i = cell(x, (h[layer_idx], x_prev[layer_idx]))
                new_h.append(h_i)
                new_x_prev.append(x_prev_i)
                x = h_i
                if self.dropout_layer is not None and layer_idx < self.num_layers - 1:
                    x = self.dropout_layer(x)

            h = new_h
            x_prev = new_x_prev
            outputs.append(x)

        out = torch.stack(outputs, dim=0)
        if self.batch_first:
            out = out.transpose(0, 1)

        h_n = torch.stack(h, dim=0)
        x_n = tuple(x_prev)
        return out, (h_n, x_n)


class TGRUCell(DoubleStateCellBase):
    r"""A strongly typed gated recurrent unit (T-GRU) cell.

    [`arXiv <https://arxiv.org/abs/1602.02218>`_]

    .. math::

        \mathbf{z}(t) &= \mathbf{W}_{ih}^{z}\,\mathbf{x}(t) + \mathbf{b}_{ih}^{z}
            + \mathbf{W}_{hh}^{z}\,\mathbf{x}(t-1) + \mathbf{b}_{hh}^{z}, \\[4pt]
        \mathbf{f}(t) &= \sigma\bigl(
            \mathbf{W}_{ih}^{f}\,\mathbf{x}(t) + \mathbf{b}_{ih}^{f}
            + \mathbf{W}_{hh}^{f}\,\mathbf{x}(t-1) + \mathbf{b}_{hh}^{f}
        \bigr), \\[4pt]
        \mathbf{o}(t) &= \tau\bigl(
            \mathbf{W}_{ih}^{o}\,\mathbf{x}(t) + \mathbf{b}_{ih}^{o}
            + \mathbf{W}_{hh}^{o}\,\mathbf{x}(t-1) + \mathbf{b}_{hh}^{o}
        \bigr), \\[4pt]
        \mathbf{h}(t) &= \mathbf{f}(t)\circ\mathbf{h}(t-1)
            + \mathbf{z}(t)\circ\mathbf{o}(t),

    where :math:`\circ` is element-wise product and :math:`\tau` is a
    pointwise nonlinearity (e.g., tanh). The weights :math:`\mathbf{W}_{hh}^{*}`
    act on the previous *input* :math:`\mathbf{x}(t-1)`, not on
    :math:`\mathbf{h}(t-1)`, per the "strongly typed" design.

    Args:
        input_size: The number of expected features in the input ``x``.
        hidden_size: The number of features in the hidden state ``h``.
        bias: If ``False``, the layer does not use input-side biases.
            Default: ``True``.
        recurrent_bias: If ``False``, the layer does not use the biases
            paired with ``W_{hh}``. Default: ``True``.
        gate_nonlinearity: Activation :math:`\sigma` for the forget gate.
            Default: :func:`torch.sigmoid`.
        output_nonlinearity: Activation :math:`\tau` for the output candidate.
            Default: :func:`torch.tanh`.
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

    Inputs: input, (h_0, x_{-1})
        - **input** of shape ``(batch, input_size)`` or ``(input_size,)``:
          Tensor containing input features.
        - **h_0** of shape ``(batch, hidden_size)`` or ``(hidden_size,)``:
          Tensor containing the initial hidden state.
        - **x_{-1}** of shape ``(batch, input_size)`` or ``(input_size,)``:
          Tensor containing the "previous input".

        If not provided, both default to zero.

    Outputs: h_1, x_1
        - **h_1** of shape ``(batch, hidden_size)`` or ``(hidden_size,)``:
          Tensor containing the next hidden state.
        - **x_1**: the current input ``x``, carried forward unchanged to
          serve as :math:`\mathbf{x}(t-1)` on the following call.

    Variables:
        weight_ih: The learnable input-hidden weights,
            of shape ``(3*hidden_size, input_size)`` (``z, f, o`` parts).
        weight_hh: The learnable weights applied to the previous input,
            of shape ``(3*hidden_size, input_size)`` (``z, f, o`` parts).
        bias_ih: The learnable input-hidden biases,
            of shape ``(3*hidden_size)`` if ``bias=True``.
        bias_hh: The learnable biases paired with ``weight_hh``,
            of shape ``(3*hidden_size)`` if ``recurrent_bias=True``.

    Examples::

        >>> cell = TGRUCell(10, 20)
        >>> x = torch.randn(5, 3, 10)     # (time_steps, batch, input_size)
        >>> h = torch.zeros(3, 20)        # (batch, hidden_size)
        >>> x_prev = torch.zeros(3, 10)   # (batch, input_size)
        >>> out = []
        >>> for t in range(x.size(0)):
        ...     h, x_prev = cell(x[t], (h, x_prev))
        ...     out.append(h)
        >>> out = torch.stack(out, dim=0) # (time_steps, batch, hidden_size)
    """

    __constants__ = ["input_size", "hidden_size", "bias", "recurrent_bias"]

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
        gate_nonlinearity="sigmoid",
        output_nonlinearity="tanh",
        kernel_init=nn.init.xavier_uniform_,
        recurrent_kernel_init=nn.init.xavier_uniform_,
        bias_init=nn.init.zeros_,
        recurrent_bias_init=nn.init.zeros_,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            bias=bias,
            recurrent_bias=recurrent_bias,
            device=device,
            dtype=dtype,
        )
        self.gate_act = resolve_activation(gate_nonlinearity)
        self.out_act = resolve_activation(output_nonlinearity)
        self.init_cfg["kernel"] = resolve_init_name(kernel_init, self.init_cfg["kernel"])
        self.init_cfg["recurrent_kernel"] = resolve_init_name(
            recurrent_kernel_init, self.init_cfg["recurrent_kernel"]
        )
        self.init_cfg["bias"] = resolve_init_name(bias_init, self.init_cfg["bias"])
        self.init_cfg["recurrent_bias"] = resolve_init_name(
            recurrent_bias_init, self.init_cfg["recurrent_bias"]
        )

        self._register_tensors(
            {
                "weight_ih": ((3 * hidden_size, input_size), True),
                "weight_hh": ((3 * hidden_size, input_size), True),
                "bias_ih": ((3 * hidden_size,), self.bias),
                "bias_hh": ((3 * hidden_size,), self.recurrent_bias),
            }
        )
        self.reset_parameters()
        self._cleanup_non_scriptable()

    def forward(
        self, inp: Tensor, state: Optional[Tuple[Tensor, Tensor]] = None
    ) -> Tuple[Tensor, Tensor]:
        self._validate_input(inp)
        b_inp, is_batched = self._as_batched(inp)

        if state is None:
            b_state = self._zeros_state(b_inp.size(0), b_inp.device, b_inp.dtype)
            b_prev_inp = torch.zeros(
                b_inp.size(0), self.input_size, device=b_inp.device, dtype=b_inp.dtype
            )
        else:
            h0, prev_inp0 = state
            b_state = h0.unsqueeze(0) if (not is_batched and h0.dim() == 1) else h0
            b_prev_inp = (
                prev_inp0.unsqueeze(0)
                if (not is_batched and prev_inp0.dim() == 1)
                else prev_inp0
            )

        gates_ih = b_inp @ self.weight_ih.t() + self.bias_ih
        gates_hh = b_prev_inp @ self.weight_hh.t() + self.bias_hh
        latent, forget_pre, out_pre = (gates_ih + gates_hh).chunk(3, 1)

        forget_gate = self.gate_act(forget_pre)
        candidate = self.out_act(out_pre)
        new_state = forget_gate * b_state + latent * candidate
        new_prev_inp = b_inp

        if not is_batched:
            new_state = new_state.squeeze(0)
            new_prev_inp = new_prev_inp.squeeze(0)

        return new_state, new_prev_inp

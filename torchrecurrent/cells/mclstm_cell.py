import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Tuple
from ..base import (
    DecoupledDoubleStateRecurrentLayerBase,
    DecoupledDoubleStateCellBase,
    resolve_activation,
    resolve_init_name,
)


class MCLSTM(DecoupledDoubleStateRecurrentLayerBase):
    r"""Multi-layer memory-controller LSTM (MC-LSTM).

    [`NIPS 2017 workshop <https://www.intel.com/content/dam/www/public/us/en/ai/documents/Sequence-Modeling-NIPS-2017.pdf>`_]

    Each layer consists of an :class:`MCLSTMCell`, which decouples the LSTM's
    "prediction" output from the recurrent signal that drives its own gates:

    .. math::
        \begin{aligned}
        f_t &= \sigma(W_{ih}^f x_t + b_{ih}^f + W_{hh}^f v_{t-1} + b_{hh}^f), \\
        i_t &= \sigma(W_{ih}^i x_t + b_{ih}^i + W_{hh}^i v_{t-1} + b_{hh}^i), \\
        o_t &= \sigma(W_{ih}^o x_t + b_{ih}^o + W_{hh}^o v_{t-1} + b_{hh}^o), \\
        m_t &= \sigma(W_{ih}^m x_t + b_{ih}^m + W_{hh}^m v_{t-1} + b_{hh}^m), \\
        n_t &= \tanh(W_{ih}^n x_t + b_{ih}^n + W_{hh}^n v_{t-1} + b_{hh}^n), \\
        c_t &= f_t \circ c_{t-1} + i_t \circ n_t, \\
        h_t &= o_t \circ \tanh(c_t), \\
        v_t &= m_t \circ \tanh(c_t)
        \end{aligned}

    where :math:`\sigma` is the sigmoid function and :math:`\circ` denotes
    elementwise (Hadamard) product. Every gate is driven by the *control
    vector* :math:`v_{t-1}`, never by :math:`h_{t-1}`: this is the paper's
    central idea, decoupling the "prediction" output :math:`h_t` from the
    memory-management role a standard LSTM's hidden state also has to play.
    :math:`h_t` is returned as the layer's output but does not itself recur;
    the recurring state is :math:`(v_t, c_t)`.

    In a multilayer MCLSTM, the input :math:`x^{(l)}_t` of the :math:`l`-th
    layer (:math:`l \ge 2`) is the *output* :math:`h^{(l-1)}_t` of the
    previous layer multiplied by dropout :math:`\delta^{(l-1)}_t`, where each
    :math:`\delta^{(l-1)}_t` is a Bernoulli random variable which is 0 with
    probability :attr:`dropout`.

    Args:
        input_size: The number of expected features in the input `x`.
        hidden_size: The number of features in the hidden and control states.
        num_layers: Number of recurrent layers. E.g., setting ``num_layers=2`` would
            mean stacking two MCLSTM layers, with the second receiving the outputs of
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
        gate_nonlinearity: Activation :math:`\sigma` for the four gates. Default:
            :func:`torch.sigmoid`
        cell_nonlinearity: Activation for the candidate content `n_t`. Default:
            :func:`torch.tanh`
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

    Inputs: input, (v_0, c_0)
        - **input**: tensor of shape :math:`(L, N, H_{in})` when
          ``batch_first=False`` or :math:`(N, L, H_{in})` when
          ``batch_first=True`` containing the features of the input sequence.
        - **v_0**: tensor of shape :math:`(\text{num_layers}, N, H_{out})`
          containing the initial control state. Defaults to zeros if not
          provided.
        - **c_0**: tensor of shape :math:`(\text{num_layers}, N, H_{out})`
          containing the initial cell state. Defaults to zeros if not
          provided.

        where:

        .. math::
            \begin{aligned}
                N ={} & \text{batch size} \\
                L ={} & \text{sequence length} \\
                H_{in} ={} & \text{input\_size} \\
                H_{out} ={} & \text{hidden\_size}
            \end{aligned}

    Outputs: output, (v_n, c_n)
        - **output**: tensor of shape :math:`(L, N, H_{out})` when
          ``batch_first=False`` or :math:`(N, L, H_{out})` when
          ``batch_first=True`` containing the output features `(h_t)` from
          the last layer of the MCLSTM, for each `t`.
        - **v_n**: tensor of shape :math:`(\text{num_layers}, N, H_{out})`
          containing the final control state for each element in the
          sequence. Note this is `v_n`, not `h_n`: the control state and the
          returned output are different tensors for every layer.
        - **c_n**: tensor of shape :math:`(\text{num_layers}, N, H_{out})`
          containing the final cell state for each element in the sequence.

    Attributes:
        cells.{k}.weight_ih : the learnable input-hidden weights of the :math:`k`-th
            layer, of shape `(5*hidden_size, input_size)` for `k = 0`. Otherwise, the
            shape is `(5*hidden_size, hidden_size)`.
        cells.{k}.weight_hh : the learnable weights applied to the control state of
            the :math:`k`-th layer, of shape `(5*hidden_size, hidden_size)`.
        cells.{k}.bias_ih : the learnable input-hidden biases of the :math:`k`-th
            layer, of shape `(5*hidden_size)`. Only present when ``bias=True``.
        cells.{k}.bias_hh : the learnable biases paired with `weight_hh` of the
            :math:`k`-th layer, of shape `(5*hidden_size)`. Only present when
            ``recurrent_bias=True``.

    .. note::
        All the weights and biases are initialized according to the provided
        initializers (`kernel_init`, `recurrent_kernel_init`, etc.).

    .. seealso::
        :class:`MCLSTMCell`

    Examples::

        >>> rnn = MCLSTM(10, 20, num_layers=2, dropout=0.1)
        >>> input = torch.randn(5, 3, 10)   # (seq_len, batch, input_size)
        >>> v0 = torch.zeros(2, 3, 20)      # (num_layers, batch, hidden_size)
        >>> c0 = torch.zeros(2, 3, 20)
        >>> output, (vn, cn) = rnn(input, (v0, c0))
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
        super(MCLSTM, self).__init__(
            input_size, hidden_size, num_layers, dropout, batch_first
        )
        self.initialize_cells(MCLSTMCell, **kwargs)


class MCLSTMCell(DecoupledDoubleStateCellBase):
    r"""A memory-controller LSTM (MC-LSTM) cell.

    [`NIPS 2017 workshop <https://www.intel.com/content/dam/www/public/us/en/ai/documents/Sequence-Modeling-NIPS-2017.pdf>`_]

    .. math::

        \mathbf{f}(t) &= \sigma\bigl(
            \mathbf{W}_{ih}^{f}\,\mathbf{x}(t) + \mathbf{b}_{ih}^{f}
            + \mathbf{W}_{hh}^{f}\,\mathbf{v}(t-1) + \mathbf{b}_{hh}^{f}
        \bigr), \\[4pt]
        \mathbf{i}(t) &= \sigma\bigl(
            \mathbf{W}_{ih}^{i}\,\mathbf{x}(t) + \mathbf{b}_{ih}^{i}
            + \mathbf{W}_{hh}^{i}\,\mathbf{v}(t-1) + \mathbf{b}_{hh}^{i}
        \bigr), \\[4pt]
        \mathbf{o}(t) &= \sigma\bigl(
            \mathbf{W}_{ih}^{o}\,\mathbf{x}(t) + \mathbf{b}_{ih}^{o}
            + \mathbf{W}_{hh}^{o}\,\mathbf{v}(t-1) + \mathbf{b}_{hh}^{o}
        \bigr), \\[4pt]
        \mathbf{m}(t) &= \sigma\bigl(
            \mathbf{W}_{ih}^{m}\,\mathbf{x}(t) + \mathbf{b}_{ih}^{m}
            + \mathbf{W}_{hh}^{m}\,\mathbf{v}(t-1) + \mathbf{b}_{hh}^{m}
        \bigr), \\[4pt]
        \mathbf{n}(t) &= \tanh\bigl(
            \mathbf{W}_{ih}^{n}\,\mathbf{x}(t) + \mathbf{b}_{ih}^{n}
            + \mathbf{W}_{hh}^{n}\,\mathbf{v}(t-1) + \mathbf{b}_{hh}^{n}
        \bigr), \\[4pt]
        \mathbf{c}(t) &= \mathbf{f}(t)\circ\mathbf{c}(t-1)
            + \mathbf{i}(t)\circ\mathbf{n}(t), \\[4pt]
        \mathbf{h}(t) &= \mathbf{o}(t)\circ\tanh\bigl(\mathbf{c}(t)\bigr), \\[4pt]
        \mathbf{v}(t) &= \mathbf{m}(t)\circ\tanh\bigl(\mathbf{c}(t)\bigr),

    where :math:`\sigma` is the sigmoid function and :math:`\circ` is
    element-wise product. Every gate depends on the *control vector*
    :math:`\mathbf{v}(t-1)`, never on :math:`\mathbf{h}(t-1)`. The returned
    output :math:`\mathbf{h}(t)` and the recurring state
    :math:`(\mathbf{v}(t), \mathbf{c}(t))` are different tensors:
    :math:`\mathbf{h}(t)` is free to specialize as a pure prediction, since
    it plays no role in the next step's gating.

    Args:
        input_size: The number of expected features in the input ``x``.
        hidden_size: The number of features in the hidden and control states.
        bias: If ``False``, the layer does not use input-side biases.
            Default: ``True``.
        recurrent_bias: If ``False``, the layer does not use the biases
            paired with ``W_{hh}``. Default: ``True``.
        gate_nonlinearity: Activation :math:`\sigma` for the four gates.
            Default: :func:`torch.sigmoid`.
        cell_nonlinearity: Activation for the candidate content ``n``.
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

    Inputs: input, (v_0, c_0)
        - **input** of shape ``(batch, input_size)`` or ``(input_size,)``:
          Tensor containing input features.
        - **v_0** of shape ``(batch, hidden_size)`` or ``(hidden_size,)``:
          Tensor containing the initial control state.
        - **c_0** of shape ``(batch, hidden_size)`` or ``(hidden_size,)``:
          Tensor containing the initial cell state.

        If not provided, both default to zero.

    Outputs: h_1, (v_1, c_1)
        - **h_1** of shape ``(batch, hidden_size)`` or ``(hidden_size,)``:
          Tensor containing the output (prediction).
        - **v_1** of shape ``(batch, hidden_size)`` or ``(hidden_size,)``:
          Tensor containing the next control state.
        - **c_1** of shape ``(batch, hidden_size)`` or ``(hidden_size,)``:
          Tensor containing the next cell state.

    Variables:
        weight_ih: The learnable input-hidden weights,
            of shape ``(5*hidden_size, input_size)`` (``f, i, o, m, n`` parts).
        weight_hh: The learnable weights applied to the control state,
            of shape ``(5*hidden_size, hidden_size)`` (``f, i, o, m, n`` parts).
        bias_ih: The learnable input-hidden biases,
            of shape ``(5*hidden_size)`` if ``bias=True``.
        bias_hh: The learnable biases paired with ``weight_hh``,
            of shape ``(5*hidden_size)`` if ``recurrent_bias=True``.

    Examples::

        >>> cell = MCLSTMCell(10, 20)
        >>> x = torch.randn(5, 3, 10)     # (time_steps, batch, input_size)
        >>> v = torch.zeros(3, 20)        # (batch, hidden_size)
        >>> c = torch.zeros(3, 20)
        >>> out = []
        >>> for t in range(x.size(0)):
        ...     h, (v, c) = cell(x[t], (v, c))
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
        cell_nonlinearity="tanh",
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
        self.cell_act = resolve_activation(cell_nonlinearity)
        self.init_cfg["kernel"] = resolve_init_name(kernel_init, self.init_cfg["kernel"])
        self.init_cfg["recurrent_kernel"] = resolve_init_name(
            recurrent_kernel_init, self.init_cfg["recurrent_kernel"]
        )
        self.init_cfg["bias"] = resolve_init_name(bias_init, self.init_cfg["bias"])
        self.init_cfg["recurrent_bias"] = resolve_init_name(
            recurrent_bias_init, self.init_cfg["recurrent_bias"]
        )

        self._default_register_tensors(ih_mult=5, hh_mult=5)
        self.reset_parameters()
        self._cleanup_non_scriptable()

    def forward(
        self, inp: Tensor, state: Optional[Tuple[Tensor, Tensor]] = None
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        self._validate_input(inp)
        b_inp, is_batched = self._as_batched(inp)

        if state is None:
            b_v = self._zeros_state(b_inp.size(0), b_inp.device, b_inp.dtype)
            b_c = self._zeros_state(b_inp.size(0), b_inp.device, b_inp.dtype)
        else:
            v0, c0 = state
            b_v = (
                self._zeros_state(b_inp.size(0), b_inp.device, b_inp.dtype)
                if v0 is None
                else (v0.unsqueeze(0) if (not is_batched and v0.dim() == 1) else v0)
            )
            b_c = (
                self._zeros_state(b_inp.size(0), b_inp.device, b_inp.dtype)
                if c0 is None
                else (c0.unsqueeze(0) if (not is_batched and c0.dim() == 1) else c0)
            )

        gates = (
            b_inp @ self.weight_ih.t()
            + self.bias_ih
            + b_v @ self.weight_hh.t()
            + self.bias_hh
        )
        forget_pre, input_pre, output_pre, control_pre, cell_pre = gates.chunk(5, 1)

        forget_gate = self.gate_act(forget_pre)
        input_gate = self.gate_act(input_pre)
        output_gate = self.gate_act(output_pre)
        control_gate = self.gate_act(control_pre)
        candidate = self.cell_act(cell_pre)

        new_c = forget_gate * b_c + input_gate * candidate
        squashed_c = torch.tanh(new_c)
        new_h = output_gate * squashed_c
        new_v = control_gate * squashed_c

        if not is_batched:
            new_h = new_h.squeeze(0)
            new_v = new_v.squeeze(0)
            new_c = new_c.squeeze(0)

        return new_h, (new_v, new_c)

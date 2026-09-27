import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Tuple
from ..base import (
    DecoupledSingleStateRecurrentLayerBase,
    DecoupledSingleStateCellBase,
    apply_init_,
    resolve_activation,
    resolve_init_name,
)


class eLSTM(DecoupledSingleStateRecurrentLayerBase):
    r"""Multi-layer LSTM with element-wise recurrence (eLSTM).

    [`arXiv <https://arxiv.org/abs/2305.19044>`_]

    Each layer consists of an :class:`eLSTMCell`, which updates the cell
    state according to:

    .. math::
        \begin{aligned}
        f_t &= \sigma(W_{ih}^f x_t + b_{ih}^f + w^f \circ c_{t-1}), \\
        z_t &= \phi(W_{ih}^z x_t + b_{ih}^z + w^z \circ c_{t-1}), \\
        c_t &= f_t \circ c_{t-1} + (1 - f_t) \circ z_t, \\
        o_t &= \sigma(W_{ih}^o x_t + b_{ih}^o + W_{hh} c_t + b_{hh}), \\
        h_t &= o_t \circ c_t
        \end{aligned}

    where :math:`c_t` is the recurrent cell state at time `t`, :math:`h_t` is
    the output at time `t`, :math:`x_t` is the input at time `t`,
    :math:`\sigma` is the sigmoid function, :math:`\phi` is a pointwise
    nonlinearity (e.g., tanh), and :math:`\circ` denotes elementwise
    multiplication. Only :math:`c_t` recurs, through the element-wise weights
    :math:`w^f` and :math:`w^z`; :math:`h_t` is passed to the next layer.

    In a multilayer eLSTM, the input :math:`x^{(l)}_t` of the :math:`l`-th layer
    (:math:`l \ge 2`) is the output :math:`h^{(l-1)}_t` of the previous
    layer multiplied by dropout :math:`\delta^{(l-1)}_t`, where each
    :math:`\delta^{(l-1)}_t` is a Bernoulli random variable which is 0 with
    probability :attr:`dropout`.

    Args:
        input_size: The number of expected features in the input `x`.
        hidden_size: The number of features in the cell state `c`.
        num_layers: Number of recurrent layers. E.g., setting ``num_layers=2`` would
            mean stacking two eLSTM layers, with the second receiving the outputs of
            the first. Default: 1
        dropout: If non-zero, introduces a `Dropout` layer on the outputs of each
            layer except the last layer, with dropout probability equal to
            :attr:`dropout`. Default: 0
        batch_first: If ``True``, then the input and output tensors are provided as
            `(batch, seq, feature)` instead of `(seq, batch, feature)`. Default: False
        bias: If ``False``, then the layer does not use input-side biases.
            Default: True
        recurrent_bias: If ``False``, then the layer does not use the bias
            :math:`b_{hh}` of the output gate. Default: True
        nonlinearity: Nonlinearity :math:`\phi` for the candidate. Default:
            :func:`torch.tanh`
        gate_nonlinearity: Activation for the forget and output gates. Default:
            :func:`torch.sigmoid`
        kernel_init: Initializer for `W_{ih}^*`. Default:
            :func:`torch.nn.init.xavier_uniform_`
        recurrent_kernel_init: Initializer for `W_{hh}`. Default:
            :func:`torch.nn.init.xavier_uniform_`
        cell_kernel_init: Initializer for the element-wise weights `w^f, w^z`.
            Default: uniform in :math:`[-0.1, 0.1]`
        bias_init: Initializer for input-side biases. Default:
            :func:`torch.nn.init.zeros_`
        recurrent_bias_init: Initializer for the output-gate bias `b_{hh}`. Default:
            :func:`torch.nn.init.zeros_`
        device: The desired device of parameters.
        dtype: The desired floating point type of parameters.

    Inputs: input, c_0
        - **input**: tensor of shape
          :math:`(L, N, H_{in})` when ``batch_first=False`` or
          :math:`(N, L, H_{in})` when ``batch_first=True`` containing the features of
          the input sequence.
        - **c_0**: tensor of shape :math:`(\text{num_layers}, N, H_{out})`
          containing the initial cell state for each element in the input
          sequence. Defaults to zeros if not provided.

        where:

        .. math::
            \begin{aligned}
                N ={} & \text{batch size} \\
                L ={} & \text{sequence length} \\
                H_{in} ={} & \text{input\_size} \\
                H_{out} ={} & \text{hidden\_size}
            \end{aligned}

    Outputs: output, c_n
        - **output**: tensor of shape
          :math:`(L, N, H_{out})` when ``batch_first=False`` or
          :math:`(N, L, H_{out})` when ``batch_first=True`` containing the output
          features `(h_t)` from the last layer of the eLSTM, for each `t`.
        - **c_n**: tensor of shape :math:`(\text{num_layers}, N, H_{out})`
          containing the final cell state for each element in the sequence.
          Note this is `c_n`, not `h_n`: the recurrent state and the returned
          output are different tensors for every layer.

    Attributes:
        cells.{k}.weight_ih : the learnable input-hidden weights of the :math:`k`-th
            layer, of shape `(3*hidden_size, input_size)` for `k = 0`. Otherwise, the
            shape is `(3*hidden_size, hidden_size)`.
        cells.{k}.weight_hh : the learnable cell-to-output-gate weights of the
            :math:`k`-th layer, of shape `(hidden_size, hidden_size)`.
        cells.{k}.weight_c : the learnable element-wise recurrent weights of the
            :math:`k`-th layer, of shape `(2*hidden_size)`.
        cells.{k}.bias_ih : the learnable input-hidden biases of the :math:`k`-th
            layer, of shape `(3*hidden_size)`. Only present when ``bias=True``.
        cells.{k}.bias_hh : the learnable output-gate bias of the :math:`k`-th
            layer, of shape `(hidden_size)`. Only present when ``recurrent_bias=True``.

    .. note::
        All the weights and biases are initialized according to the provided
        initializers (`kernel_init`, `recurrent_kernel_init`, etc.).

    .. seealso::
        :class:`eLSTMCell`

    Examples::

        >>> rnn = eLSTM(10, 20, num_layers=2, dropout=0.1)
        >>> input = torch.randn(5, 3, 10)   # (seq_len, batch, input_size)
        >>> c0 = torch.zeros(2, 3, 20)      # (num_layers, batch, hidden_size)
        >>> output, cn = rnn(input, c0)
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
        super(eLSTM, self).__init__(
            input_size, hidden_size, num_layers, dropout, batch_first
        )
        self.initialize_cells(eLSTMCell, **kwargs)


class eLSTMCell(DecoupledSingleStateCellBase):
    r"""An LSTM cell with element-wise recurrence (eLSTM).

    [`arXiv <https://arxiv.org/abs/2305.19044>`_]

    .. math::

        \mathbf{f}(t) &= \sigma\bigl(
            \mathbf{W}_{ih}^{f}\,\mathbf{x}(t) + \mathbf{b}_{ih}^{f}
            + \mathbf{w}^{f}\circ\mathbf{c}(t-1)
        \bigr), \\[6pt]
        \mathbf{z}(t) &= \phi\bigl(
            \mathbf{W}_{ih}^{z}\,\mathbf{x}(t) + \mathbf{b}_{ih}^{z}
            + \mathbf{w}^{z}\circ\mathbf{c}(t-1)
        \bigr), \\[6pt]
        \mathbf{c}(t) &= \mathbf{f}(t)\circ\mathbf{c}(t-1)
            + \bigl(1 - \mathbf{f}(t)\bigr)\circ\mathbf{z}(t), \\[6pt]
        \mathbf{o}(t) &= \sigma\bigl(
            \mathbf{W}_{ih}^{o}\,\mathbf{x}(t) + \mathbf{b}_{ih}^{o}
            + \mathbf{W}_{hh}\,\mathbf{c}(t) + \mathbf{b}_{hh}
        \bigr), \\[6pt]
        \mathbf{h}(t) &= \mathbf{o}(t)\circ\mathbf{c}(t),

    where :math:`\circ` is element‐wise product and :math:`\phi` is a
    pointwise nonlinearity (e.g., tanh). Only :math:`\mathbf{c}` recurs;
    :math:`\mathbf{h}` is the output.

    Args:
        input_size: The number of expected features in the input ``x``.
        hidden_size: The number of features in the cell state ``c``.
        bias: If ``False``, the layer does not use input-side biases.
            Default: ``True``.
        recurrent_bias: If ``False``, the layer does not use the output-gate
            bias ``b_{hh}``. Default: ``True``.
        nonlinearity: Nonlinearity :math:`\phi` for the candidate.
            Default: :func:`torch.tanh`.
        gate_nonlinearity: Activation for the forget and output gates.
            Default: :func:`torch.sigmoid`.
        kernel_init: Initializer for ``W_{ih}^*``.
            Default: :func:`torch.nn.init.xavier_uniform_`.
        recurrent_kernel_init: Initializer for ``W_{hh}``.
            Default: :func:`torch.nn.init.xavier_uniform_`.
        cell_kernel_init: Initializer for the element-wise weights ``w^f, w^z``.
            Default: uniform in :math:`[-0.1, 0.1]`.
        bias_init: Initializer for input-side biases when ``bias=True``.
            Default: :func:`torch.nn.init.zeros_`.
        recurrent_bias_init: Initializer for ``b_{hh}`` when
            ``recurrent_bias=True``. Default: :func:`torch.nn.init.zeros_`.
        device: The desired device of parameters.
        dtype: The desired floating point type of parameters.

    Inputs: input, c_0
        - **input** of shape ``(batch, input_size)`` or ``(input_size,)``:
          Tensor containing input features.
        - **c_0** of shape ``(batch, hidden_size)`` or ``(hidden_size,)``:
          Tensor containing the initial cell state.

        If **c_0** is not provided, it defaults to zero.

    Outputs: h_1, c_1
        - **h_1** of shape ``(batch, hidden_size)`` or ``(hidden_size,)``:
          Tensor containing the output, to be passed to the next layer.
        - **c_1** of shape ``(batch, hidden_size)`` or ``(hidden_size,)``:
          Tensor containing the next cell state.

    Variables:
        weight_ih: The learnable input–hidden weights,
            of shape ``(3*hidden_size, input_size)`` (``f, z, o`` parts).
        weight_hh: The learnable cell-to-output-gate weights,
            of shape ``(hidden_size, hidden_size)``.
        weight_c: The learnable element-wise recurrent weights,
            of shape ``(2*hidden_size)`` (``w^f, w^z`` parts).
        bias_ih: The learnable input–hidden biases,
            of shape ``(3*hidden_size)`` if ``bias=True``.
        bias_hh: The learnable output-gate bias,
            of shape ``(hidden_size)`` if ``recurrent_bias=True``.

    Examples::

        >>> cell = eLSTMCell(10, 20)
        >>> x = torch.randn(5, 3, 10)     # (time_steps, batch, input_size)
        >>> c = torch.zeros(3, 20)        # (batch, hidden_size)
        >>> out = []
        >>> for t in range(x.size(0)):
        ...     h, c = cell(x[t], c)
        ...     out.append(h)
        >>> out = torch.stack(out, dim=0) # (time_steps, batch, hidden_size)
    """

    __constants__ = ["input_size", "hidden_size", "bias", "recurrent_bias"]

    weight_ih: Tensor
    weight_hh: Tensor
    weight_c: Tensor
    bias_ih: Tensor
    bias_hh: Tensor

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        recurrent_bias: bool = True,
        nonlinearity="tanh",
        gate_nonlinearity="sigmoid",
        kernel_init=nn.init.xavier_uniform_,
        recurrent_kernel_init=nn.init.xavier_uniform_,
        cell_kernel_init="uniform_centered",
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
        self.act = resolve_activation(nonlinearity)
        self.gate_act = resolve_activation(gate_nonlinearity)
        self.init_cfg["kernel"] = resolve_init_name(kernel_init, self.init_cfg["kernel"])
        self.init_cfg["recurrent_kernel"] = resolve_init_name(
            recurrent_kernel_init, self.init_cfg["recurrent_kernel"]
        )
        self.init_cfg["cell_kernel"] = resolve_init_name(
            cell_kernel_init, "uniform_centered"
        )
        self.init_cfg["bias"] = resolve_init_name(bias_init, self.init_cfg["bias"])
        self.init_cfg["recurrent_bias"] = resolve_init_name(
            recurrent_bias_init, self.init_cfg["recurrent_bias"]
        )

        self._register_tensors(
            {
                "weight_ih": ((3 * hidden_size, input_size), True),
                "weight_hh": ((hidden_size, hidden_size), True),
                "weight_c": ((2 * hidden_size,), True),
                "bias_ih": ((3 * hidden_size,), bias),
                "bias_hh": ((hidden_size,), recurrent_bias),
            }
        )
        self.reset_parameters()
        self._cleanup_non_scriptable()

    def reset_parameters(self) -> None:
        apply_init_(self.weight_ih, self.init_cfg["kernel"])
        apply_init_(self.weight_hh, self.init_cfg["recurrent_kernel"])
        apply_init_(self.weight_c, self.init_cfg["cell_kernel"])
        if isinstance(self.bias_ih, nn.Parameter):
            apply_init_(self.bias_ih, self.init_cfg["bias"])
        if isinstance(self.bias_hh, nn.Parameter):
            apply_init_(self.bias_hh, self.init_cfg["recurrent_bias"])

    def forward(self, inp: Tensor, state: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        self._validate_input(inp)
        b_inp, is_batched = self._as_batched(inp)

        if state is None:
            b_state = self._zeros_state(b_inp.size(0), b_inp.device, b_inp.dtype)
        else:
            b_state = state.unsqueeze(0) if (not is_batched and state.dim() == 1) else state

        forget_pre, cand_pre, out_pre = (b_inp @ self.weight_ih.t() + self.bias_ih).chunk(
            3, 1
        )
        weight_c_f, weight_c_z = self.weight_c.chunk(2, 0)

        forget_gate = self.gate_act(forget_pre + weight_c_f * b_state)
        candidate = self.act(cand_pre + weight_c_z * b_state)
        new_state = forget_gate * b_state + (1.0 - forget_gate) * candidate
        out_gate = self.gate_act(out_pre + new_state @ self.weight_hh.t() + self.bias_hh)
        new_output = out_gate * new_state

        if not is_batched:
            new_output = new_output.squeeze(0)
            new_state = new_state.squeeze(0)

        return new_output, new_state

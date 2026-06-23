import torch
from torch import Tensor
import torch.nn as nn
from typing import Optional, Tuple
from ..base import (
    DoubleStateRecurrentLayerBase,
    DoubleStateCellBase,
    resolve_activation,
    resolve_init_name,
    apply_init_,
)


class ResLSTM(DoubleStateRecurrentLayerBase):
    r"""Multi-layer residual long short-term memory (LSTM) network.

    [`arXiv <https://arxiv.org/abs/1701.03360>`_]

    Each layer consists of a :class:`ResLSTMCell`, which adds a spatial
    shortcut path from the layer input to the projected cell output and
    reuses the output gate to control that path:

    .. math::
        \begin{aligned}
        i_t &= \sigma(W_{ih}^i x_t + b_{ih}^i + W_{hh}^i h_{t-1}
            + b_{hh}^i + p^i \odot c_{t-1}), \\
        f_t &= \sigma(W_{ih}^f x_t + b_{ih}^f + W_{hh}^f h_{t-1}
            + b_{hh}^f + p^f \odot c_{t-1}), \\
        g_t &= \phi(W_{ih}^g x_t + b_{ih}^g + W_{hh}^g h_{t-1} + b_{hh}^g), \\
        c_t &= f_t \odot c_{t-1} + i_t \odot g_t, \\
        o_t &= \sigma(W_{ih}^o x_t + b_{ih}^o + W_{hh}^o h_{t-1}
            + b_{hh}^o + p^o \odot c_t), \\
        h_t &= o_t \odot \bigl(W_p\,\phi(c_t) + W_r x_t\bigr),
        \end{aligned}

    where :math:`\sigma` is the sigmoid function, :math:`\phi` is a pointwise
    nonlinearity (e.g., tanh), and :math:`\odot` denotes elementwise
    multiplication. The projection :math:`W_p` maps the cell output back to the
    hidden size, while the residual projection :math:`W_r` maps the layer input
    :math:`x_t` to the hidden size so that the shortcut term matches dimensions.

    In a multilayer ResLSTM, the input :math:`x^{(l)}_t` of the :math:`l`-th
    layer (:math:`l \ge 2`) is the hidden state :math:`h^{(l-1)}_t` of the
    previous layer multiplied by dropout :math:`\delta^{(l-1)}_t`, where each
    :math:`\delta^{(l-1)}_t` is a Bernoulli random variable which is 0 with
    probability :attr:`dropout`.

    Args:
        input_size: The number of expected features in the input `x`.
        hidden_size: The number of features in the hidden and cell states.
        num_layers: Number of recurrent layers. E.g., setting ``num_layers=2``
            would mean stacking two ResLSTM layers, with the second receiving
            the outputs of the first. Default: 1
        dropout: If non-zero, introduces a `Dropout` layer on the outputs of
            each layer except the last layer, with dropout probability equal
            to :attr:`dropout`. Default: 0
        batch_first: If ``True``, then the input and output tensors are provided
            as `(batch, seq, feature)` instead of `(seq, batch, feature)`.
            Default: False
        bias: If ``False``, then the layer does not use input-side biases.
            Default: True
        recurrent_bias: If ``False``, then the layer does not use recurrent
            biases. Default: True
        nonlinearity: Nonlinearity :math:`\phi` for the candidate and cell
            output. Default: :func:`torch.tanh`
        gate_nonlinearity: Activation for the gates. Default:
            :func:`torch.sigmoid`
        kernel_init: Initializer for `W_{ih}^*`. Default:
            :func:`torch.nn.init.xavier_uniform_`
        recurrent_kernel_init: Initializer for `W_{hh}^*`. Default:
            :func:`torch.nn.init.xavier_uniform_`
        peephole_kernel_init: Initializer for peephole weights `p`. Default:
            :func:`torch.nn.init.normal_`
        proj_init: Initializer for the projection `W_p`. Default:
            :func:`torch.nn.init.xavier_uniform_`
        residual_init: Initializer for the residual projection `W_r`. Default:
            :func:`torch.nn.init.xavier_uniform_`
        bias_init: Initializer for input-side biases. Default:
            :func:`torch.nn.init.zeros_`
        recurrent_bias_init: Initializer for recurrent biases. Default:
            :func:`torch.nn.init.zeros_`
        device: The desired device of parameters.
        dtype: The desired floating point type of parameters.

    Inputs: input, (h_0, c_0)
        - **input**: tensor of shape :math:`(L, H_{in})` for unbatched input,
          :math:`(L, N, H_{in})` when ``batch_first=False`` or
          :math:`(N, L, H_{in})` when ``batch_first=True`` containing the
          features of the input sequence. The input can also be a packed
          variable length sequence. See
          :func:`torch.nn.utils.rnn.pack_padded_sequence` or
          :func:`torch.nn.utils.rnn.pack_sequence` for details.
        - **h_0**: tensor of shape :math:`(\text{num_layers}, H_{out})` for
          unbatched input or :math:`(\text{num_layers}, N, H_{out})` containing
          the initial hidden state. Defaults to zeros if not provided.
        - **c_0**: tensor of shape :math:`(\text{num_layers}, H_{out})` for
          unbatched input or :math:`(\text{num_layers}, N, H_{out})` containing
          the initial cell state. Defaults to zeros if not provided.

        where:

        .. math::
            \begin{aligned}
                N ={} & \text{batch size} \\
                L ={} & \text{sequence length} \\
                H_{in} ={} & \text{input\_size} \\
                H_{out} ={} & \text{hidden\_size}
            \end{aligned}

    Outputs: output, (h_n, c_n)
        - **output**: tensor of shape :math:`(L, H_{out})` for unbatched input,
          :math:`(L, N, H_{out})` when ``batch_first=False`` or
          :math:`(N, L, H_{out})` when ``batch_first=True`` containing the
          output features `(h_t)` from the last layer of the ResLSTM, for each
          `t`. If a :class:`torch.nn.utils.rnn.PackedSequence` has been given as
          the input, the output will also be a packed sequence.
        - **h_n**: tensor of shape :math:`(\text{num_layers}, H_{out})` for
          unbatched input or :math:`(\text{num_layers}, N, H_{out})` containing
          the final hidden state for each element in the sequence.
        - **c_n**: tensor of shape :math:`(\text{num_layers}, H_{out})` for
          unbatched input or :math:`(\text{num_layers}, N, H_{out})` containing
          the final cell state for each element in the sequence.

    Attributes:
        cells.{k}.weight_ih : the learnable input-hidden weights of the
            :math:`k`-th layer, of shape `(4*hidden_size, input_size)` for
            `k = 0`. Otherwise, the shape is `(4*hidden_size, hidden_size)`.
        cells.{k}.weight_hh : the learnable hidden-hidden weights of the
            :math:`k`-th layer, of shape `(4*hidden_size, hidden_size)`.
        cells.{k}.weight_proj : the learnable projection weights `W_p` of the
            :math:`k`-th layer, of shape `(hidden_size, hidden_size)`.
        cells.{k}.weight_res : the learnable residual projection weights `W_r`
            of the :math:`k`-th layer, of shape `(hidden_size, input_size)` for
            `k = 0`. Otherwise, the shape is `(hidden_size, hidden_size)`.
        cells.{k}.weight_ph : the learnable peephole weights of the
            :math:`k`-th layer, of shape `(3*hidden_size,)`.
        cells.{k}.bias_ih : the learnable input-hidden biases of the
            :math:`k`-th layer, of shape `(4*hidden_size)`. Only present when
            ``bias=True``.
        cells.{k}.bias_hh : the learnable hidden-hidden biases of the
            :math:`k`-th layer, of shape `(4*hidden_size)`. Only present when
            ``recurrent_bias=True``.

    .. note::
        All the weights and biases are initialized according to the provided
        initializers (`kernel_init`, `recurrent_kernel_init`, etc.).

    .. note::
        ``batch_first`` argument is ignored for unbatched inputs.

    .. seealso::
        :class:`ResLSTMCell`

    Examples::

        >>> rnn = ResLSTM(10, 20, num_layers=2)
        >>> input = torch.randn(5, 3, 10)   # (seq_len, batch, input_size)
        >>> h0 = torch.zeros(2, 3, 20)
        >>> c0 = torch.zeros(2, 3, 20)
        >>> output, (hn, cn) = rnn(input, (h0, c0))
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
        super(ResLSTM, self).__init__(
            input_size, hidden_size, num_layers, dropout, batch_first
        )
        self.initialize_cells(ResLSTMCell, **kwargs)


class ResLSTMCell(DoubleStateCellBase):
    r"""A residual long short-term memory (ResLSTM) cell.

    [`arXiv <https://arxiv.org/abs/1701.03360>`_]

    .. math::

        \begin{aligned}
        \mathbf{i}(t) &= \sigma\bigl(
            \mathbf{W}_{ih}^{i}\mathbf{x}(t) + \mathbf{b}_{ih}^{i}
            + \mathbf{W}_{hh}^{i}\mathbf{h}(t-1) + \mathbf{b}_{hh}^{i}
            + \mathbf{p}^{i}\odot\mathbf{c}(t-1)\bigr), \\
        \mathbf{f}(t) &= \sigma\bigl(
            \mathbf{W}_{ih}^{f}\mathbf{x}(t) + \mathbf{b}_{ih}^{f}
            + \mathbf{W}_{hh}^{f}\mathbf{h}(t-1) + \mathbf{b}_{hh}^{f}
            + \mathbf{p}^{f}\odot\mathbf{c}(t-1)\bigr), \\
        \mathbf{g}(t) &= \phi\bigl(
            \mathbf{W}_{ih}^{g}\mathbf{x}(t) + \mathbf{b}_{ih}^{g}
            + \mathbf{W}_{hh}^{g}\mathbf{h}(t-1) + \mathbf{b}_{hh}^{g}\bigr), \\
        \mathbf{c}(t) &= \mathbf{f}(t)\odot\mathbf{c}(t-1)
            + \mathbf{i}(t)\odot\mathbf{g}(t), \\
        \mathbf{o}(t) &= \sigma\bigl(
            \mathbf{W}_{ih}^{o}\mathbf{x}(t) + \mathbf{b}_{ih}^{o}
            + \mathbf{W}_{hh}^{o}\mathbf{h}(t-1) + \mathbf{b}_{hh}^{o}
            + \mathbf{p}^{o}\odot\mathbf{c}(t)\bigr), \\
        \mathbf{h}(t) &= \mathbf{o}(t)\odot\bigl(
            \mathbf{W}_{p}\,\phi(\mathbf{c}(t)) + \mathbf{W}_{r}\mathbf{x}(t)
            \bigr),
        \end{aligned}

    where :math:`\sigma` is the sigmoid function, :math:`\phi` is a pointwise
    nonlinearity (e.g., tanh), and :math:`\odot` denotes element-wise
    multiplication. The projection :math:`\mathbf{W}_{p}` maps the cell output
    back to the hidden size, and the residual projection :math:`\mathbf{W}_{r}`
    maps the input :math:`\mathbf{x}(t)` to the hidden size so that the spatial
    shortcut term matches dimensions. The output gate :math:`\mathbf{o}(t)`
    controls both the projected cell output and the shortcut path.

    Args:
        input_size: The number of expected features in the input ``x``.
        hidden_size: The number of features in the hidden and cell states.
        bias: If ``False``, the layer does not use input-side biases.
            Default: ``True``.
        recurrent_bias: If ``False``, the layer does not use recurrent biases.
            Default: ``True``.
        nonlinearity: Nonlinearity :math:`\phi` for the candidate and cell
            output. Default: :func:`torch.tanh`.
        gate_nonlinearity: Activation for the gates.
            Default: :func:`torch.sigmoid`.
        kernel_init: Initializer for ``W_{ih}^*``.
            Default: :func:`torch.nn.init.xavier_uniform_`.
        recurrent_kernel_init: Initializer for ``W_{hh}^*``.
            Default: :func:`torch.nn.init.xavier_uniform_`.
        peephole_kernel_init: Initializer for peephole weights ``p``.
            Default: :func:`torch.nn.init.normal_`.
        proj_init: Initializer for the projection ``W_p``.
            Default: :func:`torch.nn.init.xavier_uniform_`.
        residual_init: Initializer for the residual projection ``W_r``.
            Default: :func:`torch.nn.init.xavier_uniform_`.
        bias_init: Initializer for input-side biases when ``bias=True``.
            Default: :func:`torch.nn.init.zeros_`.
        recurrent_bias_init: Initializer for recurrent biases when
            ``recurrent_bias=True``. Default: :func:`torch.nn.init.zeros_`.
        device: The desired device of parameters.
        dtype: The desired floating point type of parameters.

    Inputs: input, (h_0, c_0)
        - **input** of shape ``(batch, input_size)`` or ``(input_size,)``:
          tensor containing input features.
        - **h_0** of shape ``(batch, hidden_size)`` or ``(hidden_size,)``:
          initial hidden state.
        - **c_0** of shape ``(batch, hidden_size)`` or ``(hidden_size,)``:
          initial cell state.

        If **(h_0, c_0)** is not provided, both default to zero.

    Outputs: (h_1, c_1)
        - **h_1** of shape ``(batch, hidden_size)`` or ``(hidden_size,)``:
          next hidden state.
        - **c_1** of shape ``(batch, hidden_size)`` or ``(hidden_size,)``:
          next cell state.

    Variables:
        weight_ih: the learnable input–hidden weights,
            of shape ``(4*hidden_size, input_size)``.
        weight_hh: the learnable hidden–hidden weights,
            of shape ``(4*hidden_size, hidden_size)``.
        weight_proj: the learnable projection weights ``W_p``,
            of shape ``(hidden_size, hidden_size)``.
        weight_res: the learnable residual projection weights ``W_r``,
            of shape ``(hidden_size, input_size)``.
        weight_ph: the learnable peephole weights, of shape
            ``(3*hidden_size,)``.
        bias_ih: the learnable input–hidden biases,
            of shape ``(4*hidden_size)`` if ``bias=True``.
        bias_hh: the learnable hidden–hidden biases,
            of shape ``(4*hidden_size)`` if ``recurrent_bias=True``.

    Examples::

        >>> cell = ResLSTMCell(10, 20)
        >>> x = torch.randn(5, 3, 10)   # (time_steps, batch, input_size)
        >>> h = torch.zeros(3, 20)
        >>> c = torch.zeros(3, 20)
        >>> out_h = []
        >>> for t in range(x.size(0)):
        ...     h, c = cell(x[t], (h, c))
        ...     out_h.append(h)
        >>> out_h = torch.stack(out_h, dim=0)  # (time_steps, batch, hidden_size)
    """

    __constants__ = ["input_size", "hidden_size", "bias", "recurrent_bias"]

    weight_ih: Tensor
    weight_hh: Tensor
    weight_proj: Tensor
    weight_res: Tensor
    weight_ph: Tensor
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
        peephole_kernel_init=nn.init.normal_,
        proj_init=nn.init.xavier_uniform_,
        residual_init=nn.init.xavier_uniform_,
        bias_init=nn.init.zeros_,
        recurrent_bias_init=nn.init.zeros_,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super(ResLSTMCell, self).__init__(
            input_size, hidden_size, bias, recurrent_bias, device=device, dtype=dtype
        )
        self.act = resolve_activation(nonlinearity)
        self.gate_act = resolve_activation(gate_nonlinearity)
        self.init_cfg["kernel"] = resolve_init_name(kernel_init, self.init_cfg["kernel"])
        self.init_cfg["recurrent_kernel"] = resolve_init_name(
            recurrent_kernel_init, self.init_cfg["recurrent_kernel"]
        )
        self.init_cfg["peephole_kernel"] = resolve_init_name(
            peephole_kernel_init, "normal"
        )
        self.init_cfg["proj"] = resolve_init_name(proj_init, "xavier_uniform")
        self.init_cfg["residual"] = resolve_init_name(residual_init, "xavier_uniform")
        self.init_cfg["bias"] = resolve_init_name(bias_init, self.init_cfg["bias"])
        self.init_cfg["recurrent_bias"] = resolve_init_name(
            recurrent_bias_init, self.init_cfg["recurrent_bias"]
        )

        self._register_tensors(
            {
                "weight_ih": ((4 * hidden_size, input_size), True),
                "weight_hh": ((4 * hidden_size, hidden_size), True),
                "weight_proj": ((hidden_size, hidden_size), True),
                "weight_res": ((hidden_size, input_size), True),
                "weight_ph": ((3 * hidden_size,), True),
                "bias_ih": ((4 * hidden_size,), bias),
                "bias_hh": ((4 * hidden_size,), recurrent_bias),
            }
        )
        self.reset_parameters()
        self._cleanup_non_scriptable()

    def reset_parameters(self) -> None:
        apply_init_(self.weight_ih, self.init_cfg["kernel"])
        apply_init_(self.weight_hh, self.init_cfg["recurrent_kernel"])
        apply_init_(self.weight_proj, self.init_cfg["proj"])
        apply_init_(self.weight_res, self.init_cfg["residual"])
        apply_init_(self.weight_ph, self.init_cfg["peephole_kernel"])
        if hasattr(self, "bias_ih"):
            apply_init_(self.bias_ih, self.init_cfg["bias"])
        if hasattr(self, "bias_hh"):
            apply_init_(self.bias_hh, self.init_cfg["recurrent_bias"])

    def forward(
        self, inp: Tensor, state: Optional[Tuple[Tensor, Tensor]] = None
    ) -> Tuple[Tensor, Tensor]:
        self._validate_input(inp)
        b_inp, is_batched = self._as_batched(inp)

        if state is None:
            b_h = self._zeros_state(b_inp.size(0), b_inp.device, b_inp.dtype)
            b_c = self._zeros_state(b_inp.size(0), b_inp.device, b_inp.dtype)
        else:
            h, c = state
            b_h = (
                self._zeros_state(b_inp.size(0), b_inp.device, b_inp.dtype)
                if h is None
                else (h.unsqueeze(0) if (not is_batched and h.dim() == 1) else h)
            )
            b_c = (
                self._zeros_state(b_inp.size(0), b_inp.device, b_inp.dtype)
                if c is None
                else (c.unsqueeze(0) if (not is_batched and c.dim() == 1) else c)
            )

        input_gate, forget_gate, cell_gate, output_gate = (
            b_inp @ self.weight_ih.t()
            + self.bias_ih
            + b_h @ self.weight_hh.t()
            + self.bias_hh
        ).chunk(4, 1)
        weight_ph_i, weight_ph_f, weight_ph_o = self.weight_ph.chunk(3, 0)
        input_gate = input_gate + b_c * weight_ph_i
        forget_gate = forget_gate + b_c * weight_ph_f
        new_c = self.gate_act(forget_gate) * b_c + self.gate_act(input_gate) * self.act(
            cell_gate
        )
        output_gate = output_gate + new_c * weight_ph_o
        projected = self.act(new_c) @ self.weight_proj.t()
        residual = b_inp @ self.weight_res.t()
        new_h = self.gate_act(output_gate) * (projected + residual)

        if not is_batched:
            new_h = new_h.squeeze(0)
            new_c = new_c.squeeze(0)

        return new_h, new_c

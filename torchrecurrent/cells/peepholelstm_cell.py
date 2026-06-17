import torch
from torch import Tensor
import torch.nn as nn
from typing import Optional, Tuple
from ..base import DoubleStateRecurrentLayerBase, DoubleStateCellBase, resolve_activation, resolve_init_name, apply_init_


class PeepholeLSTM(DoubleStateRecurrentLayerBase):
    r"""Multi-layer peephole long short-term memory (LSTM) network.

    [`JMLR <https://www.jmlr.org/papers/volume3/gers02a/gers02a.pdf>`_]

    Each layer consists of a :class:`PeepholeLSTMCell`, which augments the
    standard LSTM by adding learnable peephole connections from the cell
    state into the gates:

    .. math::
        \begin{aligned}
        z(t) &= \tanh\!\left(W_{ih}^z x(t) + b_{ih}^z
                 + W_{hh}^z h(t-1) + b_{hh}^z \right), \\
        i(t) &= \sigma\!\left(W_{ih}^i x(t) + b_{ih}^i
                 + W_{hh}^i h(t-1) + b_{hh}^i
                 + p^i \circ c(t-1)\right), \\
        f(t) &= \sigma\!\left(W_{ih}^f x(t) + b_{ih}^f
                 + W_{hh}^f h(t-1) + b_{hh}^f
                 + p^f \circ c(t-1)\right), \\
        c(t) &= f(t) \circ c(t-1) + i(t) \circ z(t), \\
        o(t) &= \sigma\!\left(W_{ih}^o x(t) + b_{ih}^o
                 + W_{hh}^o h(t-1) + b_{hh}^o
                 + p^o \circ c(t)\right), \\
        h(t) &= o(t) \circ \tanh(c(t)),
        \end{aligned}

    where :math:`\sigma` is the sigmoid function and :math:`\circ` denotes
    elementwise multiplication.

    Args:
        input_size: The number of expected features in the input `x`.
        hidden_size: The number of features in the hidden and cell states.
        num_layers: Number of recurrent layers. E.g., setting ``num_layers=2``
            stacks two peephole LSTM layers, with the second receiving the
            outputs of the first. Default: 1
        dropout: If non-zero, introduces a `Dropout` layer on the outputs of
            each layer except the last, with dropout probability equal to
            :attr:`dropout`. Default: 0
        batch_first: If ``True``, input and output tensors are provided as
            `(batch, seq, feature)` instead of `(seq, batch, feature)`.
            Default: False
        bias: If ``False``, the layer does not use input-side bias `b_{ih}`.
            Default: True
        recurrent_bias: If ``False``, the layer does not use recurrent bias
            `b_{hh}`. Default: True
        nonlinearity: Activation for the cell candidate `z`.
            Default: :func:`torch.tanh`
        gate_nonlinearity: Activation for the gates.
            Default: :func:`torch.sigmoid`
        kernel_init: Initializer for `W_{ih}`.
            Default: :func:`torch.nn.init.xavier_uniform_`
        recurrent_kernel_init: Initializer for `W_{hh}`.
            Default: :func:`torch.nn.init.xavier_uniform_`
        peephole_kernel_init: Initializer for peephole weights `p`.
            Default: :func:`torch.nn.init.normal_`
        bias_init: Initializer for `b_{ih}` when ``bias=True``.
            Default: :func:`torch.nn.init.zeros_`
        recurrent_bias_init: Initializer for `b_{hh}` when ``recurrent_bias=True``.
            Default: :func:`torch.nn.init.zeros_`
        device: The desired device of parameters.
        dtype: The desired floating point type of parameters.

    Inputs: input, (h_0, c_0)
        - **input**: tensor of shape :math:`(L, H_{in})` for unbatched input,
          :math:`(L, N, H_{in})` when ``batch_first=False`` or
          :math:`(N, L, H_{in})` when ``batch_first=True`` containing the
          features of the input sequence.
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
          output features from the last layer, for each timestep.
        - **h_n**: tensor of shape :math:`(\text{num_layers}, H_{out})` for
          unbatched input or :math:`(\text{num_layers}, N, H_{out})` containing
          the final hidden state for each element in the sequence.
        - **c_n**: tensor of shape :math:`(\text{num_layers}, H_{out})` for
          unbatched input or :math:`(\text{num_layers}, N, H_{out})` containing
          the final cell state for each element in the sequence.

    Attributes:
        cells.{k}.weight_ih : the learnable input–hidden weights of the
            :math:`k`-th layer, of shape `(4*hidden_size, input_size)` for
            `k=0`. Otherwise, the shape is `(4*hidden_size, hidden_size)`.
        cells.{k}.weight_hh : the learnable hidden–hidden weights of the
            :math:`k`-th layer, of shape `(4*hidden_size, hidden_size)`.
        cells.{k}.weight_ph : the learnable peephole weights of the
            :math:`k`-th layer, of shape `(3*hidden_size,)`.
        cells.{k}.bias_ih : the learnable input–hidden biases of the
            :math:`k`-th layer, of shape `(4*hidden_size)`. Only present when
            ``bias=True``.
        cells.{k}.bias_hh : the learnable hidden–hidden biases of the
            :math:`k`-th layer, of shape `(4*hidden_size)`. Only present when
            ``recurrent_bias=True``.

    .. note::
        All weights and biases are initialized according to the provided
        initializers (`kernel_init`, `recurrent_kernel_init`, etc.).

    .. seealso::
        :class:`PeepholeLSTMCell`

    Examples::

        >>> rnn = PeepholeLSTM(10, 20, num_layers=2)
        >>> x = torch.randn(5, 3, 10)   # (seq_len, batch, input_size)
        >>> h0 = torch.zeros(2, 3, 20)
        >>> c0 = torch.zeros(2, 3, 20)
        >>> output, (hn, cn) = rnn(x, (h0, c0))
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
        super(PeepholeLSTM, self).__init__(
            input_size, hidden_size, num_layers, dropout, batch_first
        )
        self.initialize_cells(PeepholeLSTMCell, **kwargs)


class PeepholeLSTMCell(DoubleStateCellBase):
    r"""A Peephole LSTM cell with learnable peephole connections.

    [`JMLR <https://www.jmlr.org/papers/volume3/gers02a/gers02a.pdf>`_]

    .. math::

        \mathbf{z}(t) &= \tanh\Bigl(
            \mathbf{W}_{ih}^{z}\,\mathbf{x}(t)
            + \mathbf{b}_{ih}^{z}
            + \mathbf{W}_{hh}^{z}\,\mathbf{h}(t-1)
            + \mathbf{b}_{hh}^{z}
        \Bigr), \\[6pt]
        \mathbf{i}(t) &= \sigma\Bigl(
            \mathbf{W}_{ih}^{i}\,\mathbf{x}(t)
            + \mathbf{b}_{ih}^{i}
            + \mathbf{W}_{hh}^{i}\,\mathbf{h}(t-1)
            + \mathbf{b}_{hh}^{i}
            + \mathbf{p}^{i}\circ\mathbf{c}(t-1)
        \Bigr), \\[6pt]
        \mathbf{f}(t) &= \sigma\Bigl(
            \mathbf{W}_{ih}^{f}\,\mathbf{x}(t)
            + \mathbf{b}_{ih}^{f}
            + \mathbf{W}_{hh}^{f}\,\mathbf{h}(t-1)
            + \mathbf{b}_{hh}^{f}
            + \mathbf{p}^{f}\circ\mathbf{c}(t-1)
        \Bigr), \\[6pt]
        \mathbf{c}(t) &= \mathbf{f}(t)\,\circ\,\mathbf{c}(t-1)
            + \mathbf{i}(t)\,\circ\,\mathbf{z}(t), \\[6pt]
        \mathbf{o}(t) &= \sigma\Bigl(
            \mathbf{W}_{ih}^{o}\,\mathbf{x}(t)
            + \mathbf{b}_{ih}^{o}
            + \mathbf{W}_{hh}^{o}\,\mathbf{h}(t-1)
            + \mathbf{b}_{hh}^{o}
            + \mathbf{p}^{o}\circ\mathbf{c}(t)
        \Bigr), \\[6pt]
        \mathbf{h}(t) &= \mathbf{o}(t)\,\circ\,\tanh\bigl(\mathbf{c}(t)\bigr)

    Args:
        input_size: The number of expected features in the input ``x``.
        hidden_size: The number of features in the hidden state ``h``.
        bias: If ``False``, the layer does not use input-side bias ``b_{ih}``.
            Default: ``True``.
        recurrent_bias: If ``False``, the layer does not use recurrent bias
            ``b_{hh}``. Default: ``True``.
        nonlinearity: Activation for the cell candidate ``z``.
            Default: :func:`torch.tanh`.
        gate_nonlinearity: Activation for input/forget/output gates.
            Default: :func:`torch.sigmoid`.
        kernel_init: Initializer for ``W_{ih}``.
            Default: :func:`torch.nn.init.xavier_uniform_`.
        recurrent_kernel_init: Initializer for ``W_{hh}``.
            Default: :func:`torch.nn.init.xavier_uniform_`.
        peephole_kernel_init: Initializer for peephole weights ``p``.
            Default: :func:`torch.nn.init.normal_`.
        bias_init: Initializer for ``b_{ih}`` when ``bias=True``.
            Default: :func:`torch.nn.init.zeros_`.
        recurrent_bias_init: Initializer for ``b_{hh}`` when
            ``recurrent_bias=True``. Default: :func:`torch.nn.init.zeros_`.
        device: The desired device of parameters.
        dtype: The desired floating point type of parameters.

    Inputs: input, (h_0, c_0)
        - **input** of shape ``(batch, input_size)`` or ``(input_size,)``:
          Tensor containing input features.
        - **h_0** of shape ``(batch, hidden_size)`` or ``(hidden_size,)``:
          Tensor containing the initial hidden state.
        - **c_0** of shape ``(batch, hidden_size)`` or ``(hidden_size,)``:
          Tensor containing the initial cell state.

        If **(h_0, c_0)** is not provided, both default to zero.

    Outputs: (h_1, c_1)
        - **h_1** of shape ``(batch, hidden_size)`` or ``(hidden_size,)``:
          Tensor containing the next hidden state.
        - **c_1** of shape ``(batch, hidden_size)`` or ``(hidden_size,)``:
          Tensor containing the next cell state.

    Variables:
        weight_ih: The learnable input–hidden weights,
            of shape ``(4*hidden_size, input_size)``.
        weight_hh: The learnable hidden–hidden weights,
            of shape ``(4*hidden_size, hidden_size)``.
        weight_ph: The learnable peephole weights (for i, f, o),
            of shape ``(3*hidden_size,)``.
        bias_ih: The learnable input–hidden biases,
            of shape ``(4*hidden_size)``.
        bias_hh: The learnable hidden–hidden biases,
            of shape ``(4*hidden_size)``.

    Examples::

        >>> cell = PeepholeLSTMCell(10, 20)
        >>> x = torch.randn(5, 3, 10)   # (time_steps, batch, input_size)
        >>> h = torch.zeros(3, 20)
        >>> c = torch.zeros(3, 20)
        >>> outs = []
        >>> for t in range(x.size(0)):
        ...     h, c = cell(x[t], (h, c))
        ...     outs.append(h)
        >>> outs = torch.stack(outs, dim=0)  # (time_steps, batch, hidden_size)
    """

    __constants__ = ["input_size", "hidden_size", "bias", "recurrent_bias"]

    weight_ih: Tensor
    weight_hh: Tensor
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
        bias_init=nn.init.zeros_,
        recurrent_bias_init=nn.init.zeros_,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super(PeepholeLSTMCell, self).__init__(
            input_size, hidden_size, bias, recurrent_bias, device=device, dtype=dtype
        )
        self.act = resolve_activation(nonlinearity)
        self.gate_act = resolve_activation(gate_nonlinearity)
        self.init_cfg["kernel"] = resolve_init_name(kernel_init, self.init_cfg["kernel"])
        self.init_cfg["recurrent_kernel"] = resolve_init_name(
            recurrent_kernel_init, self.init_cfg["recurrent_kernel"]
        )
        self.init_cfg["bias"] = resolve_init_name(bias_init, self.init_cfg["bias"])
        self.init_cfg["recurrent_bias"] = resolve_init_name(
            recurrent_bias_init, self.init_cfg["recurrent_bias"]
        )
        self.init_cfg["peephole_kernel"] = resolve_init_name(
            peephole_kernel_init, "normal"
        )

        self._register_tensors(
            {
                "weight_ih": ((4 * hidden_size, input_size), True),
                "weight_hh": ((4 * hidden_size, hidden_size), True),
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
            b_h = self._zeros_state(b_inp.size(0), b_inp.device, b_inp.dtype) if h is None else (h.unsqueeze(0) if (not is_batched and h.dim() == 1) else h)
            b_c = self._zeros_state(b_inp.size(0), b_inp.device, b_inp.dtype) if c is None else (c.unsqueeze(0) if (not is_batched and c.dim() == 1) else c)

        weight_ih_i, weight_ih_f, weight_ih_c, weight_ih_o = self.weight_ih.chunk(4, 0)
        weight_hh_i, weight_hh_f, weight_hh_c, weight_hh_o = self.weight_hh.chunk(4, 0)
        weight_ph_i, weight_ph_f, weight_ph_o = self.weight_ph.chunk(3, 0)
        bias_ih_i, bias_ih_f, bias_ih_c, bias_ih_o = self.bias_ih.chunk(4, 0)
        bias_hh_i, bias_hh_f, bias_hh_c, bias_hh_o = self.bias_hh.chunk(4, 0)

        i = (
            b_inp @ weight_ih_i.t()
            + bias_ih_i
            + b_h @ weight_hh_i.t()
            + b_c * weight_ph_i
            + bias_hh_i
        )
        input_gate = self.gate_act(i)
        f = (
            b_inp @ weight_ih_f.t()
            + bias_ih_f
            + b_h @ weight_hh_f.t()
            + bias_hh_f
            + b_c * weight_ph_f
        )
        forget_gate = self.gate_act(f)
        c_hat = b_inp @ weight_ih_c.t() + bias_ih_c + b_h @ weight_hh_c.t() + bias_hh_c
        cell_candidate = self.act(c_hat)
        new_c = forget_gate * b_c + input_gate * cell_candidate
        o = (
            b_inp @ weight_ih_o.t()
            + bias_ih_o
            + b_h @ weight_hh_o.t()
            + bias_hh_o
            + new_c * weight_ph_o
        )
        output_gate = self.gate_act(o)
        new_h = output_gate * self.act(new_c)

        if not is_batched:
            new_h = new_h.squeeze(0)
            new_c = new_c.squeeze(0)

        return new_h, new_c

import torch
from torch import Tensor
import torch.nn as nn
from typing import Optional, Callable, Union, Tuple
from ..base import BaseDoubleRecurrentLayer, BaseDoubleRecurrentCell


class MultiplicativeLSTM(BaseDoubleRecurrentLayer):
    r"""Multi-layer multiplicative long short-term memory network.

    [`arXiv <https://arxiv.org/abs/1609.07959>`_]

    Each layer consists of a :class:`MultiplicativeLSTMCell`, which updates the
    hidden and cell states according to:

    .. math::
        \begin{aligned}
        m_t &= (W_{ih}^m x_t + b_{ih}^m) \circ (W_{hh}^m h_{t-1} + b_{hh}^m), \\
        \hat{h}_t &= W_{ih}^h x_t + b_{ih}^h + W_{mh}^h m_t + b_{mh}^h, \\
        i_t &= \sigma(W_{ih}^i x_t + b_{ih}^i + W_{mh}^i m_t + b_{mh}^i), \\
        f_t &= \sigma(W_{ih}^f x_t + b_{ih}^f + W_{mh}^f m_t + b_{mh}^f), \\
        o_t &= \sigma(W_{ih}^o x_t + b_{ih}^o + W_{mh}^o m_t + b_{mh}^o), \\
        c_t &= f_t \circ c_{t-1} + i_t \circ \tanh(\hat{h}_t), \\
        h_t &= \tanh(c_t) \circ o_t
        \end{aligned}

    where :math:`h_t` is the hidden state, :math:`c_t` the cell state,
    :math:`\sigma` is the sigmoid, and :math:`\circ` the Hadamard product.

    In a multilayer multiplicative LSTM, the input :math:`x^{(l)}_t` of the
    :math:`l`-th layer (:math:`l \ge 2`) is the hidden state
    :math:`h^{(l-1)}_t` of the previous layer multiplied by dropout
    :math:`\delta^{(l-1)}_t`, where each :math:`\delta^{(l-1)}_t` is a Bernoulli
    random variable which is 0 with probability :attr:`dropout`.

    Args:
        input_size: The number of expected features in the input `x`.
        hidden_size: The number of features in the hidden and cell states `h`, `c`.
        num_layers: Number of recurrent layers. E.g., setting ``num_layers=2`` would
            mean stacking two multiplicative LSTM layers, with the second receiving
            the outputs of the first. Default: 1
        dropout: If non-zero, introduces a `Dropout` layer on the outputs of each
            layer except the last layer, with dropout probability equal to
            :attr:`dropout`. Default: 0
        batch_first: If ``True``, then the input and output tensors are provided as
            `(batch, seq, feature)` instead of `(seq, batch, feature)`. Default: False
        bias: If ``False``, then the layer does not use input-side biases.
            Default: True
        recurrent_bias: If ``False``, then the layer does not use recurrent biases.
            Default: True
        multiplicative_bias: If ``False``, then the layer does not use multiplicative
            biases. Default: True
        kernel_init: Initializer for `W_{ih}`. Default:
            :func:`torch.nn.init.xavier_uniform_`
        recurrent_kernel_init: Initializer for `W_{hh}`. Default:
            :func:`torch.nn.init.xavier_uniform_`
        multiplicative_kernel_init: Initializer for `W_{mh}`. Default:
            :func:`torch.nn.init.normal_`
        bias_init: Initializer for `b_{ih}` when ``bias=True``. Default:
            :func:`torch.nn.init.zeros_`
        recurrent_bias_init: Initializer for `b_{hh}` when ``recurrent_bias=True``.
            Default: :func:`torch.nn.init.zeros_`
        multiplicative_bias_init: Initializer for `b_{mh}` when
            ``multiplicative_bias=True``. Default: :func:`torch.nn.init.zeros_`
        device: The desired device of parameters.
        dtype: The desired floating point type of parameters.

    Inputs: input, (h_0, c_0)
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
        - **c_0**: tensor of shape :math:`(\text{num_layers}, H_{out})` for unbatched
          input or :math:`(\text{num_layers}, N, H_{out})` containing the initial
          cell state for each element in the input sequence. Defaults to zeros if
          not provided.

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
          :math:`(N, L, H_{out})` when ``batch_first=True`` containing the output
          features `(h_t)` from the last layer of the multiplicative LSTM, for each
          `t`. If a :class:`torch.nn.utils.rnn.PackedSequence` has been given as the
          input, the output will also be a packed sequence.
        - **h_n**: tensor of shape :math:`(\text{num_layers}, H_{out})` for unbatched
          input or :math:`(\text{num_layers}, N, H_{out})` containing the final
          hidden state for each element in the sequence.
        - **c_n**: tensor of shape :math:`(\text{num_layers}, H_{out})` for unbatched
          input or :math:`(\text{num_layers}, N, H_{out})` containing the final cell
          state for each element in the sequence.

    Attributes:
        cells.{k}.weight_ih : the learnable input-hidden weights of the :math:`k`-th
            layer, of shape `(5*hidden_size, input_size)` for `k = 0`. Otherwise, the
            shape is `(5*hidden_size, hidden_size)`.
        cells.{k}.weight_hh : the learnable hidden-hidden weights of the :math:`k`-th
            layer, of shape `(hidden_size, hidden_size)`.
        cells.{k}.weight_mh : the learnable multiplicative-hidden weights of the
            :math:`k`-th layer, of shape `(4*hidden_size, hidden_size)`.
        cells.{k}.bias_ih : the learnable input-hidden biases of the :math:`k`-th
            layer, of shape `(5*hidden_size)`. Only present when ``bias=True``.
        cells.{k}.bias_hh : the learnable hidden-hidden biases of the :math:`k`-th
            layer, of shape `(hidden_size)`. Only present when ``recurrent_bias=True``.
        cells.{k}.bias_mh : the learnable multiplicative biases of the :math:`k`-th
            layer, of shape `(4*hidden_size)`. Only present when
            ``multiplicative_bias=True``.

    .. note::
        All the weights and biases are initialized according to the provided
        initializers (`kernel_init`, `recurrent_kernel_init`, etc.).

    .. note::
        ``batch_first`` argument is ignored for unbatched inputs.

    .. seealso::
        :class:`MultiplicativeLSTMCell`

    Examples::

        >>> rnn = MultiplicativeLSTM(10, 20, num_layers=2, dropout=0.1)
        >>> input = torch.randn(5, 3, 10)   # (seq_len, batch, input_size)
        >>> h0 = torch.zeros(2, 3, 20)      # (num_layers, batch, hidden_size)
        >>> c0 = torch.zeros(2, 3, 20)      # (num_layers, batch, hidden_size)
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
        super(MultiplicativeLSTM, self).__init__(
            input_size, hidden_size, num_layers, dropout, batch_first
        )
        self.initialize_cells(MultiplicativeLSTMCell, **kwargs)


class MultiplicativeLSTMCell(BaseDoubleRecurrentCell):
    r"""A multiplicative LSTM cell.

    [`arXiv <https://arxiv.org/abs/1609.07959>`_]

    .. math::

        \begin{aligned}
        \mathbf{m}(t) &= \bigl(\mathbf{W}_{ih}^{m}\,\mathbf{x}(t)
            + \mathbf{b}_{ih}^{m}\bigr)\,\circ\,\bigl(\mathbf{W}_{hh}^{m}\,\mathbf{h}(t-1)
            + \mathbf{b}_{hh}^{m}\bigr), \\
            \hat{\mathbf{h}}(t) &= \mathbf{W}_{ih}^{h}\,\mathbf{x}(t)
            + \mathbf{b}_{ih}^{h}
            + \mathbf{W}_{mh}^{h}\,\mathbf{m}(t)
            + \mathbf{b}_{mh}^{h}, \\
            \mathbf{i}(t) &= \sigma\bigl(\mathbf{W}_{ih}^{i}\,\mathbf{x}(t)
            + \mathbf{b}_{ih}^{i}
            + \mathbf{W}_{mh}^{i}\,\mathbf{m}(t)
            + \mathbf{b}_{mh}^{i}\bigr), \\
            \mathbf{f}(t) &= \sigma\bigl(\mathbf{W}_{ih}^{f}\,\mathbf{x}(t)
            + \mathbf{b}_{ih}^{f}
            + \mathbf{W}_{mh}^{f}\,\mathbf{m}(t)
            + \mathbf{b}_{mh}^{f}\bigr), \\
            \mathbf{o}(t) &= \sigma\bigl(\mathbf{W}_{ih}^{o}\,\mathbf{x}(t)
            + \mathbf{b}_{ih}^{o}
            + \mathbf{W}_{mh}^{o}\,\mathbf{m}(t)
            + \mathbf{b}_{mh}^{o}\bigr), \\
            \mathbf{c}(t) &= \mathbf{f}(t)\circ\mathbf{c}(t-1)
            + \mathbf{i}(t)\circ\tanh\bigl(\hat{\mathbf{h}}(t)\bigr), \\
            \mathbf{h}(t) &= \tanh\bigl(\mathbf{c}(t)\bigr)\circ\mathbf{o}(t)
        \end{aligned}

    where :math:`\circ` is the Hadamard product and :math:`\sigma` the sigmoid.

    Args:
        input_size: The number of expected features in the input ``x``
        hidden_size: The number of features in the hidden/cell states ``h`` and ``c``
        bias: If ``False``, the layer does not use input-side bias ``b_{ih}``.
            Default: ``True``
        recurrent_bias: If ``False``, the layer does not use recurrent bias ``b_{hh}``.
            Default: ``True``
        multiplicative_bias: If ``False``, the layer does not use multiplicative bias ``b_{mh}``.
            Default: ``True``
        kernel_init: Initializer for ``W_{ih}``.
            Default: :func:`torch.nn.init.xavier_uniform_`
        recurrent_kernel_init: Initializer for ``W_{hh}``.
            Default: :func:`torch.nn.init.xavier_uniform_`
        multiplicative_kernel_init: Initializer for ``W_{mh}``.
            Default: :func:`torch.nn.init.normal_`
        bias_init: Initializer for ``b_{ih}`` when ``bias=True``.
            Default: :func:`torch.nn.init.zeros_`
        recurrent_bias_init: Initializer for ``b_{hh}`` when ``recurrent_bias=True``.
            Default: :func:`torch.nn.init.zeros_`
        multiplicative_bias_init: Initializer for ``b_{mh}`` when ``multiplicative_bias=True``.
            Default: :func:`torch.nn.init.zeros_`
        device: The desired device of parameters.
        dtype: The desired floating point type of parameters.

    Inputs: input, (h_0, c_0)
        - **input** of shape ``(batch, input_size)`` or ``(input_size,)``:
          tensor containing input features
        - **h_0** of shape ``(batch, hidden_size)`` or ``(hidden_size,)``:
          tensor containing the initial hidden state
        - **c_0** of shape ``(batch, hidden_size)`` or ``(hidden_size,)``:
          tensor containing the initial cell state

        If ``(h_0, c_0)`` is not provided, both default to zeros.

    Outputs: (h_1, c_1)
        - **h_1** of shape ``(batch, hidden_size)`` or ``(hidden_size,)``:
          tensor containing the next hidden state
        - **c_1** of shape ``(batch, hidden_size)`` or ``(hidden_size,)``:
          tensor containing the next cell state

    Variables:
        weight_ih: the learnable input–hidden weights,
            of shape ``(5*hidden_size, input_size)``
        weight_hh: the learnable hidden–hidden weights,
            of shape ``(hidden_size, hidden_size)``
        weight_mh: the learnable multiplicative–hidden weights,
            of shape ``(4*hidden_size, hidden_size)``
        bias_ih: the learnable input–hidden biases,
            of shape ``(5*hidden_size)``
        bias_hh: the learnable hidden–hidden biases,
            of shape ``(hidden_size)``
        bias_mh: the learnable multiplicative biases,
            of shape ``(4*hidden_size)``

    Examples::

        >>> cell = MultiplicativeLSTMCell(10, 20)
        >>> x = torch.randn(5, 3, 10)      # (time_steps, batch, input_size)
        >>> h, c = torch.zeros(3, 20), torch.zeros(3, 20)
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
        "multiplicative_bias",
        "kernel_init",
        "recurrent_kernel_init",
        "multiplicative_kernel_init",
        "bias_init",
        "recurrent_bias_init",
        "multiplicative_bias_init",
    ]

    weight_ih: Tensor
    weight_hh: Tensor
    weight_mh: Tensor
    bias_ih: Tensor
    bias_hh: Tensor
    bias_mh: Tensor

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        recurrent_bias: bool = True,
        multiplicative_bias: bool = True,
        kernel_init: Callable = nn.init.xavier_uniform_,
        recurrent_kernel_init: Callable = nn.init.xavier_uniform_,
        multiplicative_kernel_init: Callable = nn.init.normal_,
        bias_init: Callable = nn.init.zeros_,
        recurrent_bias_init: Callable = nn.init.zeros_,
        multiplicative_bias_init: Callable = nn.init.zeros_,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super(MultiplicativeLSTMCell, self).__init__(
            input_size, hidden_size, bias, recurrent_bias, device=device, dtype=dtype
        )
        self.kernel_init = kernel_init
        self.recurrent_kernel_init = recurrent_kernel_init
        self.multiplicative_kernel_init = multiplicative_kernel_init
        self.bias_init = bias_init
        self.recurrent_bias_init = recurrent_bias_init
        self.multiplicative_bias_init = multiplicative_bias_init

        self._register_tensors(
            {
                "weight_ih": ((5 * hidden_size, input_size), True),
                "weight_hh": ((hidden_size, hidden_size), True),
                "weight_mh": ((4 * hidden_size, hidden_size), True),
                "bias_ih": ((5 * hidden_size,), bias),
                "bias_hh": ((hidden_size,), recurrent_bias),
                "bias_mh": ((4 * hidden_size,), multiplicative_bias),
            }
        )
        self.init_weights()

    def init_weights(self):
        for name, param in self.named_parameters():
            if "weight_ih" in name:
                self.kernel_init(param)
            elif "weight_hh" in name:
                self.recurrent_kernel_init(param)
            elif "weight_mh" in name:
                self.multiplicative_kernel_init(param)
            elif "bias_ih" in name:
                self.bias_init(param)
            elif "bias_hh" in name:
                self.recurrent_bias_init(param)
            elif "bias_mh" in name:
                self.multiplicative_bias_init(param)

    def forward(
        self, inp: Tensor, state: Optional[Union[Tensor, Tuple[Tensor, ...]]] = None
    ) -> Tuple[Tensor, Tensor]:
        state, c_state = self._check_states(state)
        self._validate_input(inp)
        self._validate_states((state, c_state))
        inp, state, c_state, is_batched = self._preprocess_states(inp, (state, c_state))

        inp_expanded = inp @ self.weight_ih.t() + self.bias_ih
        gxs1, gxs2, gxs3, gxs4, gxs5 = inp_expanded.chunk(5, 1)
        multiplicative_state = gxs1 * (state @ self.weight_hh.t() + self.bias_hh)
        mult_expanded = multiplicative_state @ self.weight_mh.t() + self.bias_mh
        gms1, gms2, gms3, gms4 = mult_expanded.chunk(4, 1)
        input_gate = torch.sigmoid(gxs2 + gms1)
        forget_gate = torch.sigmoid(gxs3 + gms2)
        candidate_state = torch.tanh(gxs4 + gms3)
        output_gate = torch.sigmoid(gxs5 + gms4)
        new_cstate = forget_gate * c_state + input_gate * candidate_state
        new_state = output_gate * torch.tanh(new_cstate)

        if not is_batched:
            new_state = new_state.squeeze(0)
            new_cstate = new_cstate.squeeze(0)

        return new_state, new_cstate

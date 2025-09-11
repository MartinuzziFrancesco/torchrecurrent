import torch
from torch import Tensor
import torch.nn as nn
from typing import Optional, Callable, Union, Tuple
from ..base import BaseDoubleRecurrentLayer, BaseDoubleRecurrentCell


class OriginalLSTM(BaseDoubleRecurrentLayer):
    r"""Multi-layer original long short-term memory (LSTM) network.

    [`pub <https://ieeexplore.ieee.org/abstract/document/6795963>`_]

    Each layer consists of an :class:`OriginalLSTMCell`, which updates the hidden
    and cell states according to:

    .. math::
        \begin{aligned}
        g(t) &= W_{ih} x(t) + b_{ih} + W_{hh} h(t-1) + b_{hh}, \\
        i(t) &= \sigma(g_i(t)), \\
        \tilde{c}(t) &= \tanh(g_c(t)), \\
        o(t) &= \sigma(g_o(t)), \\
        c(t) &= c(t-1) + i(t) \odot \tilde{c}(t), \\
        h(t) &= o(t) \odot \tanh(c(t)),
        \end{aligned}

    where :math:`\sigma` is the sigmoid function and :math:`\odot` denotes
    elementwise multiplication.

    Args:
        input_size: The number of expected features in the input `x`.
        hidden_size: The number of features in the hidden and cell states.
        num_layers: Number of recurrent layers. E.g., setting ``num_layers=2``
            would mean stacking two LSTM layers, with the second receiving the
            outputs of the first. Default: 1
        dropout: If non-zero, introduces a `Dropout` layer on the outputs of
            each layer except the last layer, with dropout probability equal
            to :attr:`dropout`. Default: 0
        batch_first: If ``True``, then the input and output tensors are provided
            as `(batch, seq, feature)` instead of `(seq, batch, feature)`.
            Default: False
        bias: If ``False``, then the layer does not use input-side bias
            `b_{ih}`. Default: True
        recurrent_bias: If ``False``, then the layer does not use recurrent
            bias `b_{hh}`. Default: True
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
          output features `(h_t)` from the last layer of the LSTM, for each `t`.
          If a :class:`torch.nn.utils.rnn.PackedSequence` has been given as
          the input, the output will also be a packed sequence.
        - **h_n**: tensor of shape :math:`(\text{num_layers}, H_{out})` for
          unbatched input or :math:`(\text{num_layers}, N, H_{out})` containing
          the final hidden state for each element in the sequence.
        - **c_n**: tensor of shape :math:`(\text{num_layers}, H_{out})` for
          unbatched input or :math:`(\text{num_layers}, N, H_{out})` containing
          the final cell state for each element in the sequence.

    Attributes:
        cells.{k}.weight_ih : the learnable input-hidden weights of the
            :math:`k`-th layer, of shape `(3*hidden_size, input_size)` for
            `k = 0`. Otherwise, the shape is `(3*hidden_size, hidden_size)`.
        cells.{k}.weight_hh : the learnable hidden-hidden weights of the
            :math:`k`-th layer, of shape `(3*hidden_size, hidden_size)`.
        cells.{k}.bias_ih : the learnable input-hidden biases of the
            :math:`k`-th layer, of shape `(3*hidden_size)`. Only present when
            ``bias=True``.
        cells.{k}.bias_hh : the learnable hidden-hidden biases of the
            :math:`k`-th layer, of shape `(3*hidden_size)`. Only present when
            ``recurrent_bias=True``.

    .. note::
        All the weights and biases are initialized according to the provided
        initializers (`kernel_init`, `recurrent_kernel_init`, etc.).

    .. note::
        ``batch_first`` argument is ignored for unbatched inputs.

    .. seealso::
        :class:`OriginalLSTMCell`

    Examples::

        >>> rnn = OriginalLSTM(10, 20, num_layers=2)
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
        super(OriginalLSTM, self).__init__(
            input_size, hidden_size, num_layers, dropout, batch_first
        )
        self.initialize_cells(OriginalLSTMCell, **kwargs)


class OriginalLSTMCell(BaseDoubleRecurrentCell):
    r"""Original formulation of the LSTM cell (no forget gate).

    [`pub <https://ieeexplore.ieee.org/abstract/document/6795963>`_]

    .. math::

        \begin{aligned}
        \mathbf{g}(t) &= \mathbf{W}_{ih}\,\mathbf{x}(t) + \mathbf{b}_{ih}
            + \mathbf{W}_{hh}\,\mathbf{h}(t-1) + \mathbf{b}_{hh}, \\
            \mathbf{i}(t) &= \sigma\bigl(g_i(t)\bigr), \\
            \tilde{\mathbf{c}}(t) &= \tanh\bigl(g_c(t)\bigr), \\
            \mathbf{o}(t) &= \sigma\bigl(g_o(t)\bigr), \\
            \mathbf{c}(t) &= \mathbf{c}(t-1)
            + \mathbf{i}(t)\,\odot\,\tilde{\mathbf{c}}(t), \\
            \mathbf{h}(t) &= \mathbf{o}(t)\,\odot\,\tanh\bigl(\mathbf{c}(t)\bigr),
        \end{aligned}

    where :math:`\odot` denotes element-wise multiplication.

    Args:
        input_size: The number of expected features in the input ``x``
        hidden_size: The number of features in the hidden and cell states
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

    Inputs: input, (h_0, c_0)
        - **input** of shape ``(batch, input_size)`` or ``(input_size,)``:
          tensor containing input features
        - **h_0** of shape ``(batch, hidden_size)`` or ``(hidden_size,)``:
          initial hidden state
        - **c_0** of shape ``(batch, hidden_size)`` or ``(hidden_size,)``:
          initial cell state

        If **(h_0, c_0)** is not provided, both default to zero.

    Outputs: (h_1, c_1)
        - **h_1** of shape ``(batch, hidden_size)`` or ``(hidden_size,)``:
          next hidden state
        - **c_1** of shape ``(batch, hidden_size)`` or ``(hidden_size,)``:
          next cell state

    Variables:
        weight_ih: the learnable input–hidden weights,
            of shape ``(3*hidden_size, input_size)``
        weight_hh: the learnable hidden–hidden weights,
            of shape ``(3*hidden_size, hidden_size)``
        bias_ih: the learnable input–hidden biases,
            of shape ``(3*hidden_size)``
        bias_hh: the learnable hidden–hidden biases,
            of shape ``(3*hidden_size)``

    Examples::

        >>> cell = OriginalLSTMCell(10, 20)
        >>> x = torch.randn(5, 3, 10)   # (time_steps, batch, input_size)
        >>> h = torch.zeros(3, 20)
        >>> c = torch.zeros(3, 20)
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
        super(OriginalLSTMCell, self).__init__(
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
            hh_mult=3,
            bias=bias,
            recurrent_bias=recurrent_bias,
        )
        self.init_weights()

    def forward(
        self, inp: Tensor, state: Optional[Union[Tensor, Tuple[Tensor, ...]]] = None
    ) -> Tuple[Tensor, Tensor]:
        state, c_state = self._check_states(state)
        self._validate_input(inp)
        self._validate_states((state, c_state))
        inp, state, c_state, is_batched = self._preprocess_states(inp, (state, c_state))

        gates = (
            inp @ self.weight_ih.t()
            + self.bias_ih
            + state @ self.weight_hh.t()
            + self.bias_hh
        )
        input_gate, cell_gate, output_gate = gates.chunk(3, 1)
        new_cstate = c_state + torch.sigmoid(input_gate) * torch.tanh(cell_gate)
        new_state = torch.sigmoid(output_gate) * torch.tanh(new_cstate)

        if not is_batched:
            new_state = new_state.squeeze(0)
            new_cstate = new_cstate.squeeze(0)

        return new_state, new_cstate

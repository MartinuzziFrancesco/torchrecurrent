import torch
from torch import Tensor
import torch.nn as nn
from typing import Optional, Callable, Union, Tuple
from ..base import BaseDoubleRecurrentLayer, BaseDoubleRecurrentCell


class JANET(BaseDoubleRecurrentLayer):
    r"""Multi-layer JANET (Just Another NETwork) recurrent neural network.

    [`arXiv <https://arxiv.org/abs/1804.04849>`_]

    Each layer consists of a :class:`JANETCell`, which updates the hidden and
    cell states according to:

    .. math::
        \begin{aligned}
            s_t &= W_{ih}^{f} x_t + b_{ih}^{f}
                 + W_{hh}^{f} h_{t-1} + b_{hh}^{f}, \\
            \tilde{c}_t &= \tanh(W_{ih}^{c} x_t + b_{ih}^{c}
                 + W_{hh}^{c} h_{t-1} + b_{hh}^{c}), \\
            c_t &= \sigma(s_t) \circ c_{t-1}
                 + (1 - \sigma(s_t - \beta)) \circ \tilde{c}_t, \\
            h_t &= c_t
        \end{aligned}

    where :math:`h_t` is the hidden state at time `t`, :math:`c_t` is the cell
    state at time `t`, :math:`\sigma` is the sigmoid function, :math:`\tanh` is
    the hyperbolic tangent, and :math:`\circ` is the Hadamard product. The
    parameter :math:`\beta` shifts the threshold of the update gate.

    In a multilayer JANET, the input :math:`x^{(l)}_t` of the :math:`l`-th
    layer (:math:`l \ge 2`) is the hidden state :math:`h^{(l-1)}_t` of the
    previous layer multiplied by dropout :math:`\delta^{(l-1)}_t`, where each
    :math:`\delta^{(l-1)}_t` is a Bernoulli random variable which is 0 with
    probability :attr:`dropout`.

    Args:
        input_size: The number of expected features in the input `x`.
        hidden_size: The number of features in the hidden/cell states `h` and `c`.
        num_layers: Number of recurrent layers. E.g., setting ``num_layers=2`` would
            mean stacking two JANET layers, with the second receiving the outputs of
            the first. Default: 1
        dropout: If non-zero, introduces a `Dropout` layer on the outputs of each
            layer except the last layer, with dropout probability equal to
            :attr:`dropout`. Default: 0
        batch_first: If ``True``, then the input and output tensors are provided as
            `(batch, seq, feature)` instead of `(seq, batch, feature)`. Default: False
        bias: If ``False``, then the layer does not use input-side bias `b_ih`.
            Default: True
        recurrent_bias: If ``False``, then the layer does not use recurrent bias
            `b_{hh}`. Default: True
        kernel_init: Initializer for `W_{ih}`. Default:
            :func:`torch.nn.init.xavier_uniform_`
        recurrent_kernel_init: Initializer for `W_{hh}`. Default:
            :func:`torch.nn.init.xavier_uniform_`
        bias_init: Initializer for `b_{ih}`. Default: :func:`torch.nn.init.zeros_`
        recurrent_bias_init: Initializer for `b_{hh}`. Default:
            :func:`torch.nn.init.zeros_`
        beta: Threshold shift :math:`\beta` for the update gate. Default: 1.0
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
          cell state for each element in the input sequence. Defaults to zeros if not
          provided.

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
          features `(h_t)` from the last layer of the JANET, for each `t`. If a
          :class:`torch.nn.utils.rnn.PackedSequence` has been given as the input,
          the output will also be a packed sequence.
        - **h_n**: tensor of shape :math:`(\text{num_layers}, H_{out})` for unbatched
          input or :math:`(\text{num_layers}, N, H_{out})` containing the final
          hidden state for each element in the sequence.
        - **c_n**: tensor of shape :math:`(\text{num_layers}, H_{out})` for unbatched
          input or :math:`(\text{num_layers}, N, H_{out})` containing the final cell
          state for each element in the sequence.

    Attributes:
        cells.{k}.weight_ih : the learnable input-hidden weights of the :math:`k`-th
            layer, of shape `(2*hidden_size, input_size)` for `k = 0`. Otherwise, the
            shape is `(2*hidden_size, hidden_size)`.
        cells.{k}.weight_hh : the learnable hidden-hidden weights of the :math:`k`-th
            layer, of shape `(2*hidden_size, hidden_size)`.
        cells.{k}.bias_ih : the learnable input-hidden biases of the :math:`k`-th
            layer, of shape `(2*hidden_size)`. Only present when ``bias=True``.
        cells.{k}.bias_hh : the learnable hidden-hidden biases of the :math:`k`-th
            layer, of shape `(2*hidden_size)`. Only present when ``recurrent_bias=True``.
        cells.{k}.beta : the learnable threshold shift of the :math:`k`-th layer,
            a scalar parameter.

    .. note::
        All the weights and biases are initialized according to the provided
        initializers (`kernel_init`, `recurrent_kernel_init`, etc.). The threshold
        shift :math:`\beta` is initialized from the given value of `beta`.

    .. note::
        ``batch_first`` argument is ignored for unbatched inputs.

    .. seealso::
        :class:`JANETCell`

    Examples::

        >>> rnn = JANET(10, 20, num_layers=2, dropout=0.1, beta=0.5)
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
        super(JANET, self).__init__(
            input_size, hidden_size, num_layers, dropout, batch_first
        )
        self.initialize_cells(JANETCell, **kwargs)


class JANETCell(BaseDoubleRecurrentCell):
    r"""A JANET (Just Another NETwork) recurrent cell.

    [`arXiv <https://arxiv.org/abs/1804.04849>`_]

    .. math::

        \begin{aligned}
          \mathbf{s}(t) &= \mathbf{W}_{ih}^{f}\,\mathbf{x}(t)
             + \mathbf{b}_{ih}^{f}
             + \mathbf{W}_{hh}^{f}\,\mathbf{h}(t-1)
             + \mathbf{b}_{hh}^{f}, \\
              \tilde{\mathbf{c}}(t) &= \tanh\Bigl(
             \mathbf{W}_{ih}^{c}\,\mathbf{x}(t) + \mathbf{b}_{ih}^{c}
             + \mathbf{W}_{hh}^{c}\,\mathbf{h}(t-1)
             + \mathbf{b}_{hh}^{c}\Bigr), \\
              \mathbf{c}(t) &= \sigma\bigl(\mathbf{s}(t)\bigr)\circ \mathbf{c}(t-1)
             \;+\;\bigl(1 - \sigma\bigl(\mathbf{s}(t) - \beta\bigr)\bigr)
             \circ \tilde{\mathbf{c}}(t), \\
              \mathbf{h}(t) &= \mathbf{c}(t)
        \end{aligned}

    where :math:`\sigma` is the sigmoid function
    and :math:`\circ` is the Hadamard product.

    Args:
        input_size: The number of expected features in the input ``x``
        hidden_size: The number of features in the hidden/cell states ``h`` and ``c``
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
        beta: Threshold shift :math:`\beta` for the update gate. Default: ``1.0``
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
        weight_ih: the learnable input–hidden weights (forget + candidate),
            shape ``(2*hidden_size, input_size)``
        weight_hh: the learnable hidden–hidden weights (forget + candidate),
            shape ``(2*hidden_size, hidden_size)``
        bias_ih: the learnable input–hidden biases,
            shape ``(2*hidden_size)``
        bias_hh: the learnable hidden–hidden biases,
            shape ``(2*hidden_size)``
        beta: the learnable threshold shift
            :math:`\beta` (as a :class:`torch.nn.Parameter`)

    Examples::

        >>> cell = JANETCell(10, 20, beta=0.5)
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
        beta: float = 1.0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super(JANETCell, self).__init__(
            input_size,
            hidden_size,
            bias,
            recurrent_bias,
            beta=beta,
            device=device,
            dtype=dtype,
        )
        self.kernel_init = kernel_init
        self.recurrent_kernel_init = recurrent_kernel_init
        self.bias_init = bias_init
        self.recurrent_bias_init = recurrent_bias_init

        self._default_register_tensors(
            input_size,
            hidden_size,
            ih_mult=2,
            hh_mult=2,
            bias=bias,
            recurrent_bias=recurrent_bias,
        )
        self.beta = nn.Parameter(torch.tensor(beta))
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
        s_t, s_c = gates.chunk(2, 1)
        forget_gate = torch.sigmoid(s_t)
        candidate_state = torch.tanh(s_c)
        update_gate = torch.sigmoid(s_t - self.beta)
        new_cstate = forget_gate * c_state + (1.0 - update_gate) * candidate_state
        new_state = new_cstate

        if not is_batched:
            new_state = new_state.squeeze(0)
            new_cstate = new_cstate.squeeze(0)

        return new_state, new_cstate

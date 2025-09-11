import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Callable, Tuple, Union
from ..base import BaseDoubleRecurrentLayer, BaseDoubleRecurrentCell


class coRNN(BaseDoubleRecurrentLayer):
    r"""Multi-layer coupled oscillatory recurrent neural network.

    [`arXiv <https://arxiv.org/abs/2010.00951>`_]

    Each layer consists of a :class:`coRNNCell`, which updates the hidden and
    cell states according to:

    .. math::
        \begin{aligned}
            c_t &= c_{t-1} + \Delta t \,\tanh(
                W_{ih} x_t + b_{ih}
                + W_{hh} h_{t-1} + b_{hh}
                + W_{ch} c_{t-1} + b_{ch})
                - \Delta t \,\gamma \, h_{t-1}
                - \Delta t \,\epsilon \, c_{t-1}, \\
            h_t &= h_{t-1} + \Delta t \, c_t
        \end{aligned}

    where :math:`h_t` is the hidden state at time `t`, :math:`c_t` is the cell
    state at time `t`, :math:`\Delta t` (``dt``) is the integration step size,
    :math:`\gamma` damps the hidden state, and :math:`\epsilon` damps the cell
    state.

    In a multilayer coRNN, the input :math:`x^{(l)}_t` of the :math:`l`-th layer
    (:math:`l \ge 2`) is the hidden state :math:`h^{(l-1)}_t` of the previous
    layer multiplied by dropout :math:`\delta^{(l-1)}_t`, where each
    :math:`\delta^{(l-1)}_t` is a Bernoulli random variable which is 0 with
    probability :attr:`dropout`.

    Args:
        input_size: The number of expected features in the input `x`.
        hidden_size: The number of features in the states `h` and `c`.
        num_layers: Number of recurrent layers. E.g., setting ``num_layers=2`` would
            mean stacking two coRNN layers, with the second receiving the outputs of
            the first. Default: 1
        dropout: If non-zero, introduces a `Dropout` layer on the outputs of each
            layer except the last layer, with dropout probability equal to
            :attr:`dropout`. Default: 0
        batch_first: If ``True``, then the input and output tensors are provided as
            `(batch, seq, feature)` instead of `(seq, batch, feature)`. Default: False
        bias: If ``False``, then the layer does not use input-side bias `b_ih`.
            Default: True
        recurrent_bias: If ``False``, then the layer does not use recurrent bias
            `b_hh`. Default: True
        cell_bias: If ``False``, then the layer does not use cell bias `b_ch`.
            Default: True
        dt: Integration step size :math:`\Delta t`. Default: 1.0
        gamma: Damping coefficient on the hidden-state term. Default: 0.0
        epsilon: Damping coefficient on the cell-state term. Default: 0.0
        kernel_init: Initializer for `W_ih`. Default:
            :func:`torch.nn.init.xavier_uniform_`
        recurrent_kernel_init: Initializer for `W_{hh}`. Default:
            :func:`torch.nn.init.xavier_uniform_`
        cell_kernel_init: Initializer for `W_{ch}`. Default:
            :func:`torch.nn.init.xavier_uniform_`
        bias_init: Initializer for `b_{ih}`. Default: :func:`torch.nn.init.zeros_`
        recurrent_bias_init: Initializer for `b_{hh}`. Default:
            :func:`torch.nn.init.zeros_`
        cell_bias_init: Initializer for `b_{ch}`. Default:
            :func:`torch.nn.init.zeros_`
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
          features `(h_t)` from the last layer of the coRNN, for each `t`. If a
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
            layer, of shape `(hidden_size, input_size)` for `k = 0`. Otherwise, the
            shape is `(hidden_size, hidden_size)`.
        cells.{k}.weight_hh : the learnable hidden-hidden weights of the :math:`k`-th
            layer, of shape `(hidden_size, hidden_size)`.
        cells.{k}.weight_ch : the learnable cell-hidden weights of the :math:`k`-th
            layer, of shape `(hidden_size, hidden_size)`.
        cells.{k}.bias_ih : the learnable input-hidden bias of the :math:`k`-th layer,
            of shape `(hidden_size)`. Only present when ``bias=True``.
        cells.{k}.bias_hh : the learnable hidden-hidden bias of the :math:`k`-th
            layer, of shape `(hidden_size)`. Only present when ``recurrent_bias=True``.
        cells.{k}.bias_ch : the learnable cell-hidden bias of the :math:`k`-th layer,
            of shape `(hidden_size)`. Only present when ``cell_bias=True``.

    .. note::
        All the weights and biases are initialized according to the provided
        initializers (`kernel_init`, `recurrent_kernel_init`, etc.).

    .. note::
        ``batch_first`` argument is ignored for unbatched inputs.

    .. seealso::
        :class:`coRNNCell`

    Examples::

        >>> rnn = coRNN(10, 20, num_layers=2, dropout=0.1, dt=0.5,
        ...             gamma=0.1, epsilon=0.05)
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
        super(coRNN, self).__init__(
            input_size, hidden_size, num_layers, dropout, batch_first
        )
        self.initialize_cells(coRNNCell, **kwargs)


class coRNNCell(BaseDoubleRecurrentCell):
    r"""A Coupled Oscillatory RNN cell.

    [`arXiv <https://arxiv.org/abs/2010.00951>`_]

    .. math::

        \begin{aligned}
        \mathbf{c}(t) &= \mathbf{c}(t-1)
            + \Delta t \,\tanh\Bigl(
                \mathbf{W}_{ih}\mathbf{x}(t) + \mathbf{b}_{ih}
                + \mathbf{W}_{hh}\mathbf{h}(t-1) + \mathbf{b}_{hh}
                + \mathbf{W}_{ch}\mathbf{c}(t-1) + \mathbf{b}_{ch}
            \Bigr)
            - \Delta t\,\gamma\,\mathbf{h}(t-1)
            - \Delta t\,\epsilon\,\mathbf{c}(t), \\
        \mathbf{h}(t) &= \mathbf{h}(t-1) + \Delta t\,\mathbf{c}(t)
        \end{aligned}

    where :math:`\Delta t` (``dt``) is the integration step size,
    :math:`\gamma` damps the hidden state, and :math:`\epsilon` damps the cell state.

    Args:
        input_size: The number of expected features in the input ``x``.
        hidden_size: The number of features in the states ``h`` and ``c``.
        bias: If ``False``, the layer does not use input-side bias ``b_ih``.
            Default: ``True``.
        recurrent_bias: If ``False``, the layer does not use recurrent bias
            ``b_hh``. Default: ``True``.
        cell_bias: If ``False``, the layer does not use cell bias ``b_ch``.
            Default: ``True``.
        dt: Integration step size :math:`\Delta t`. Default: ``1.0``.
        gamma: Damping on hidden-state term. Default: ``0.0``.
        epsilon: Damping on cell-state term. Default: ``0.0``.
        kernel_init: Initializer for ``W_{ih}``.
            Default: :func:`torch.nn.init.xavier_uniform_`.
        recurrent_kernel_init: Initializer for ``W_{hh}``.
            Default: :func:`torch.nn.init.xavier_uniform_`.
        cell_kernel_init: Initializer for ``W_{ch}``.
            Default: :func:`torch.nn.init.xavier_uniform_`.
        bias_init: Initializer for ``b_{ih}`` when ``bias=True``.
            Default: :func:`torch.nn.init.zeros_`.
        recurrent_bias_init: Initializer for ``b_{hh}`` when ``recurrent_bias=True``.
            Default: :func:`torch.nn.init.zeros_`.
        cell_bias_init: Initializer for ``b_{ch}`` when ``cell_bias=True``.
            Default: :func:`torch.nn.init.zeros_`.
        device: The desired device of parameters.
        dtype: The desired floating point type of parameters.

    Inputs: input, (h_0, c_0)
        - **input** of shape ``(batch, input_size)`` or ``(input_size,)``:
          Tensor containing input features.
        - **h_0** of shape ``(batch, hidden_size)`` or ``(hidden_size,)``:
          Tensor containing the initial hidden state.
        - **c_0** of shape ``(batch, hidden_size)`` or ``(hidden_size,)``:
          Tensor containing the initial cell state.

        If ``(h_0, c_0)`` is not provided, both default to zeros.

    Outputs: (h_1, c_1)
        - **h_1** of shape ``(batch, hidden_size)`` or ``(hidden_size,)``:
          Tensor containing the next hidden state.
        - **c_1** of shape ``(batch, hidden_size)`` or ``(hidden_size,)``:
          Tensor containing the next cell state.

    Variables:
        weight_ih: The learnable input–hidden weights,
            of shape ``(hidden_size, input_size)``.
        weight_hh: The learnable hidden–hidden weights,
            of shape ``(hidden_size, hidden_size)``.
        weight_ch: The learnable cell–hidden weights,
            of shape ``(hidden_size, hidden_size)``.
        bias_ih: The learnable input–hidden bias,
            of shape ``(hidden_size)``.
        bias_hh: The learnable hidden–hidden bias,
            of shape ``(hidden_size)``.
        bias_ch: The learnable cell–hidden bias,
            of shape ``(hidden_size)``.

    .. note::
        The cell and hidden states interact like a second‐order oscillator,
        supporting rich, stable dynamics over long horizons.

    Examples::

        >>> cell = coRNNCell(10, 20, dt=0.5, gamma=0.1, epsilon=0.05)
        >>> x = torch.randn(5, 3, 10)        # (time_steps, batch, input_size)
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
        "cell_bias",
        "kernel_init",
        "recurrent_kernel_init",
        "cell_kernel_init",
        "bias_init",
        "recurrent_bias_init",
        "cell_bias_init",
        "dt",
        "gamma",
        "epsilon",
    ]

    weight_ih: Tensor
    weight_hh: Tensor
    weight_ch: Tensor
    bias_ih: Tensor
    bias_hh: Tensor
    bias_ch: Tensor

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        recurrent_bias: bool = True,
        cell_bias: bool = True,
        dt: float = 1.0,
        gamma: float = 0.0,
        epsilon: float = 0.0,
        kernel_init: Callable = nn.init.xavier_uniform_,
        recurrent_kernel_init: Callable = nn.init.xavier_uniform_,
        cell_kernel_init: Callable = nn.init.xavier_uniform_,
        bias_init: Callable = nn.init.zeros_,
        recurrent_bias_init: Callable = nn.init.zeros_,
        cell_bias_init: Callable = nn.init.zeros_,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super(coRNNCell, self).__init__(
            input_size, hidden_size, bias, device=device, dtype=dtype
        )
        self.kernel_init = kernel_init
        self.recurrent_kernel_init = recurrent_kernel_init
        self.cell_kernel_init = cell_kernel_init
        self.bias_init = bias_init
        self.recurrent_bias_init = recurrent_bias_init
        self.cell_bias_init = cell_bias_init
        self.dt = dt
        self.gamma = gamma
        self.epsilon = epsilon

        self._register_tensors(
            {
                "weight_ih": ((hidden_size, input_size), True),
                "weight_hh": ((hidden_size, hidden_size), True),
                "weight_ch": ((hidden_size, hidden_size), True),
                "bias_ih": ((hidden_size,), bias),
                "bias_hh": ((hidden_size,), recurrent_bias),
                "bias_ch": ((hidden_size,), cell_bias),
            }
        )
        self.init_weights()

    def init_weights(self):
        for name, param in self.named_parameters():
            if "weight_ih" in name:
                self.kernel_init(param)
            elif "weight_hh" in name:
                self.recurrent_kernel_init(param)
            elif "weight_ph" in name:
                self.cell_kernel_init(param)
            elif "bias_ih" in name:
                self.bias_init(param)
            elif "bias_hh" in name:
                self.recurrent_bias_init(param)
            elif "bias_ch" in name:
                self.cell_bias_init(param)

    def forward(
        self, inp: Tensor, state: Optional[Union[Tensor, Tuple[Tensor, ...]]] = None
    ) -> Tuple[Tensor, Tensor]:
        state, c_state = self._check_states(state)
        self._validate_input(inp)
        self._validate_states((state, c_state))
        inp, state, c_state, is_batched = self._preprocess_states(inp, (state, c_state))

        pre_act = (
            inp @ self.weight_ih.t()
            + self.bias_ih
            + state @ self.weight_hh.t()
            + self.bias_hh
            + c_state @ self.weight_ch.t()
            + self.bias_ch
        )
        act = torch.tanh(pre_act)
        new_cstate = (
            c_state
            + self.dt * act
            - self.dt * self.gamma * state
            - self.dt * self.epsilon * c_state
        )
        new_state = state + self.dt * new_cstate

        if not is_batched:
            new_state = new_state.squeeze(0)
            new_cstate = new_cstate.squeeze(0)

        return new_state, new_cstate

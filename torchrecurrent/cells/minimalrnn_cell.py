import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional
from ..base import (
    SingleStateRecurrentLayerBase,
    SingleStateCellBase,
    resolve_activation,
    resolve_init_name,
    apply_init_,
)


class MinimalRNN(SingleStateRecurrentLayerBase):
    r"""Multi-layer minimal recurrent neural network.

    [`arXiv <https://arxiv.org/abs/1711.06788>`_]

    Each layer consists of an :class:`MinimalRNNCell`, which updates the hidden
    state according to:

    .. math::
        \begin{aligned}
        z_t &= \phi(W_{ih} x_t + b_{ih}), \\
        u_t &= \sigma(W_{hh} h_{t-1} + W_{mm} z_t + b_{hh}), \\
        h_t &= u_t \circ h_{t-1} + (1 - u_t) \circ z_t
        \end{aligned}

    where :math:`h_t` is the hidden state at time `t`, :math:`x_t` is the
    input at time `t`, :math:`z_t` is the latent representation of the input,
    :math:`\sigma` is the sigmoid function, :math:`\phi` is a pointwise
    nonlinearity (e.g., tanh), and :math:`\circ` denotes elementwise
    multiplication.

    In a multilayer MinimalRNN, the input :math:`x^{(l)}_t` of the
    :math:`l`-th layer (:math:`l \ge 2`) is the hidden state
    :math:`h^{(l-1)}_t` of the previous layer multiplied by dropout
    :math:`\delta^{(l-1)}_t`, where each :math:`\delta^{(l-1)}_t` is a
    Bernoulli random variable which is 0 with probability :attr:`dropout`.

    Args:
        input_size: The number of expected features in the input `x`.
        hidden_size: The number of features in the hidden state `h`.
        num_layers: Number of recurrent layers. E.g., setting ``num_layers=2`` would
            mean stacking two MinimalRNN layers, with the second receiving the outputs
            of the first. Default: 1
        dropout: If non-zero, introduces a `Dropout` layer on the outputs of each
            layer except the last layer, with dropout probability equal to
            :attr:`dropout`. Default: 0
        batch_first: If ``True``, then the input and output tensors are provided as
            `(batch, seq, feature)` instead of `(seq, batch, feature)`. Default: False
        bias: If ``False``, then the layer does not use input-side biases.
            Default: True
        recurrent_bias: If ``False``, then the layer does not use recurrent biases.
            Default: True
        nonlinearity: Nonlinearity :math:`\phi` for the input encoder. Default:
            :func:`torch.tanh`
        gate_nonlinearity: Activation for the update gate. Default:
            :func:`torch.sigmoid`
        kernel_init: Initializer for `W_{ih}`. Default:
            :func:`torch.nn.init.xavier_uniform_`
        recurrent_kernel_init: Initializer for `W_{hh}`. Default:
            :func:`torch.nn.init.xavier_uniform_`
        memory_kernel_init: Initializer for `W_{mm}`. Default:
            :func:`torch.nn.init.xavier_uniform_`
        bias_init: Initializer for input-side biases. Default:
            :func:`torch.nn.init.zeros_`
        recurrent_bias_init: Initializer for recurrent biases. Default:
            :func:`torch.nn.init.zeros_`
        device: The desired device of parameters.
        dtype: The desired floating point type of parameters.

    Inputs: input, h_0
        - **input**: tensor of shape
          :math:`(L, N, H_{in})` when ``batch_first=False`` or
          :math:`(N, L, H_{in})` when ``batch_first=True`` containing the features of
          the input sequence.
        - **h_0**: tensor of shape :math:`(\text{num_layers}, N, H_{out})`
          containing the initial
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
        - **output**: tensor of shape
          :math:`(L, N, H_{out})` when ``batch_first=False`` or
          :math:`(N, L, H_{out})` when ``batch_first=True`` containing the output
          features `(h_t)` from the last layer of the MinimalRNN, for each `t`.
        - **h_n**: tensor of shape :math:`(\text{num_layers}, N, H_{out})`
          containing the final
          hidden state for each element in the sequence.

    Attributes:
        cells.{k}.weight_ih : the learnable input-hidden weights of the :math:`k`-th
            layer, of shape `(hidden_size, input_size)` for `k = 0`. Otherwise, the
            shape is `(hidden_size, hidden_size)`.
        cells.{k}.weight_hh : the learnable hidden-hidden weights of the :math:`k`-th
            layer, of shape `(hidden_size, hidden_size)`.
        cells.{k}.weight_mm : the learnable latent-hidden weights of the :math:`k`-th
            layer, of shape `(hidden_size, hidden_size)`.
        cells.{k}.bias_ih : the learnable input-hidden biases of the :math:`k`-th
            layer, of shape `(hidden_size)`. Only present when ``bias=True``.
        cells.{k}.bias_hh : the learnable hidden-hidden biases of the :math:`k`-th
            layer, of shape `(hidden_size)`. Only present when ``recurrent_bias=True``.

    .. note::
        All the weights and biases are initialized according to the provided
        initializers (`kernel_init`, `recurrent_kernel_init`, etc.).

    .. seealso::
        :class:`MinimalRNNCell`

    Examples::

        >>> rnn = MinimalRNN(10, 20, num_layers=2, dropout=0.1)
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
        super(MinimalRNN, self).__init__(
            input_size, hidden_size, num_layers, dropout, batch_first
        )
        self.initialize_cells(MinimalRNNCell, **kwargs)


class MinimalRNNCell(SingleStateCellBase):
    r"""A Minimal recurrent neural network (MinimalRNN) cell.

    [`arXiv <https://arxiv.org/abs/1711.06788>`_]

    .. math::

        \mathbf{z}(t) &= \phi\bigl(
            \mathbf{W}_{ih}\,\mathbf{x}(t) + \mathbf{b}_{ih}
        \bigr), \\[6pt]
        \mathbf{u}(t) &= \sigma\bigl(
            \mathbf{W}_{hh}\,\mathbf{h}(t-1)
            + \mathbf{W}_{mm}\,\mathbf{z}(t)
            + \mathbf{b}_{hh}
        \bigr), \\[6pt]
        \mathbf{h}(t) &= \mathbf{u}(t)\circ\mathbf{h}(t-1)
            \;+\;\bigl(1 - \mathbf{u}(t)\bigr)\circ\mathbf{z}(t),

    where :math:`\circ` is element-wise product, :math:`\phi` is a pointwise
    nonlinearity (e.g., tanh) mapping the input into the latent space of the
    hidden state, and :math:`\mathbf{z}(t)` is recomputed from the current
    input at every step (it is not part of the recurrent state).

    Args:
        input_size: The number of expected features in the input ``x``.
        hidden_size: The number of features in the hidden state ``h``.
        bias: If ``False``, the layer does not use input-side biases.
            Default: ``True``.
        recurrent_bias: If ``False``, the layer does not use recurrent biases.
            Default: ``True``.
        nonlinearity: Nonlinearity :math:`\phi` for the input encoder.
            Default: :func:`torch.tanh`.
        gate_nonlinearity: Activation for the update gate.
            Default: :func:`torch.sigmoid`.
        kernel_init: Initializer for ``W_{ih}``.
            Default: :func:`torch.nn.init.xavier_uniform_`.
        recurrent_kernel_init: Initializer for ``W_{hh}``.
            Default: :func:`torch.nn.init.xavier_uniform_`.
        memory_kernel_init: Initializer for ``W_{mm}``.
            Default: :func:`torch.nn.init.xavier_uniform_`.
        bias_init: Initializer for input-side biases when ``bias=True``.
            Default: :func:`torch.nn.init.zeros_`.
        recurrent_bias_init: Initializer for recurrent biases when
            ``recurrent_bias=True``. Default: :func:`torch.nn.init.zeros_`.
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
            of shape ``(hidden_size, input_size)``.
        weight_hh: The learnable hidden-hidden weights,
            of shape ``(hidden_size, hidden_size)``.
        weight_mm: The learnable latent-hidden weights,
            of shape ``(hidden_size, hidden_size)``.
        bias_ih: The learnable input-hidden biases,
            of shape ``(hidden_size)`` if ``bias=True``.
        bias_hh: The learnable hidden-hidden biases,
            of shape ``(hidden_size)`` if ``recurrent_bias=True``.

    Examples::

        >>> cell = MinimalRNNCell(10, 20)
        >>> x = torch.randn(5, 3, 10)     # (time_steps, batch, input_size)
        >>> h = torch.zeros(3, 20)        # (batch, hidden_size)
        >>> out = []
        >>> for t in range(x.size(0)):
        ...     h = cell(x[t], h)
        ...     out.append(h)
        >>> out = torch.stack(out, dim=0) # (time_steps, batch, hidden_size)
    """

    __constants__ = ["input_size", "hidden_size", "bias", "recurrent_bias"]

    weight_ih: Tensor
    weight_hh: Tensor
    weight_mm: Tensor
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
        memory_kernel_init=nn.init.xavier_uniform_,
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
        self.init_cfg["memory_kernel"] = resolve_init_name(
            memory_kernel_init, "xavier_uniform"
        )
        self.init_cfg["bias"] = resolve_init_name(bias_init, self.init_cfg["bias"])
        self.init_cfg["recurrent_bias"] = resolve_init_name(
            recurrent_bias_init, self.init_cfg["recurrent_bias"]
        )

        self._default_register_tensors(ih_mult=1, hh_mult=1)
        self.weight_mm = nn.Parameter(
            torch.empty(
                self.hidden_size,
                self.hidden_size,
                device=self._init_device,
                dtype=self._init_dtype,
            )
        )
        self.reset_parameters()
        self._cleanup_non_scriptable()

    def reset_parameters(self) -> None:
        super().reset_parameters()
        apply_init_(self.weight_mm, self.init_cfg["memory_kernel"])

    def forward(self, inp: Tensor, state: Optional[Tensor] = None) -> Tensor:
        self._validate_input(inp)
        b_inp, is_batched = self._as_batched(inp)

        if state is None:
            b_state = self._zeros_state(b_inp.size(0), b_inp.device, b_inp.dtype)
        else:
            b_state = state.unsqueeze(0) if (not is_batched and state.dim() == 1) else state

        latent = self.act(b_inp @ self.weight_ih.t() + self.bias_ih)
        update_gate = self.gate_act(
            b_state @ self.weight_hh.t() + latent @ self.weight_mm.t() + self.bias_hh
        )
        new_state = update_gate * b_state + (1.0 - update_gate) * latent

        if not is_batched:
            new_state = new_state.squeeze(0)

        return new_state

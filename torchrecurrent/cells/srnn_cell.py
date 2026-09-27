import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional
from ..base import (
    SingleStateRecurrentLayerBase,
    SingleStateCellBase,
    apply_init_,
    resolve_activation,
    resolve_init_name,
)


class SRNN(SingleStateRecurrentLayerBase):
    r"""Multi-layer shuffling recurrent neural network.

    [`arXiv <https://arxiv.org/abs/2007.07324>`_]

    Each layer consists of an :class:`SRNNCell`, which updates the hidden
    state according to:

    .. math::
        \begin{aligned}
        b(x_t) &= \bigl(W_{ho}\, \mathrm{ReLU}(W_{ih} x_t + b_{ih}) + b_{ho}\bigr)
                  \circ \sigma(W_{g} x_t + b_{g}), \\
        h_t &= \phi\bigl(P\, h_{t-1} + b(x_t)\bigr)
        \end{aligned}

    where :math:`h_t` is the hidden state at time `t`, :math:`x_t` is the
    input at time `t`, :math:`P` is the fixed cyclic shift permutation
    (:math:`(P h)_i = h_{i-1}`, indices modulo ``hidden_size``),
    :math:`\sigma` is the sigmoid function, :math:`\phi` is a pointwise
    nonlinearity (ReLU by default) and :math:`\circ` denotes elementwise
    multiplication. The recurrence has no learnable weights; all parameters are
    in the input network :math:`b`.

    In a multilayer SRNN, the input :math:`x^{(l)}_t` of the :math:`l`-th layer
    (:math:`l \ge 2`) is the hidden state :math:`h^{(l-1)}_t` of the previous
    layer multiplied by dropout :math:`\delta^{(l-1)}_t`, where each
    :math:`\delta^{(l-1)}_t` is a Bernoulli random variable which is 0 with
    probability :attr:`dropout`.

    Args:
        input_size: The number of expected features in the input `x`.
        hidden_size: The number of features in the hidden state `h`.
        num_layers: Number of recurrent layers. E.g., setting ``num_layers=2`` would
            mean stacking two SRNN layers, with the second receiving the outputs of
            the first. Default: 1
        dropout: If non-zero, introduces a `Dropout` layer on the outputs of each
            layer except the last layer, with dropout probability equal to
            :attr:`dropout`. Default: 0
        batch_first: If ``True``, then the input and output tensors are provided as
            `(batch, seq, feature)` instead of `(seq, batch, feature)`. Default: False
        bias: If ``False``, then the layer does not use biases. Default: True
        hyper_size: Width of the hidden layer of the input network :math:`b`.
            Default: 64
        nonlinearity: Nonlinearity :math:`\phi`. Default: :func:`torch.relu`
        kernel_init: Initializer for `W_{ih}`, `W_{ho}` and `W_{g}`. Default:
            :func:`torch.nn.init.xavier_uniform_`
        bias_init: Initializer for biases. Default: :func:`torch.nn.init.zeros_`
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
          features `(h_t)` from the last layer of the SRNN, for each `t`.
        - **h_n**: tensor of shape :math:`(\text{num_layers}, N, H_{out})`
          containing the final
          hidden state for each element in the sequence.

    Attributes:
        cells.{k}.weight_ih : the learnable first-layer weights of the input network
            of the :math:`k`-th layer, of shape `(hyper_size, input_size)` for
            `k = 0`. Otherwise, the shape is `(hyper_size, hidden_size)`.
        cells.{k}.weight_ho : the learnable second-layer weights of the input
            network, of shape `(hidden_size, hyper_size)`.
        cells.{k}.weight_g : the learnable gate weights of the input network,
            of shape `(hidden_size, input_size)` for `k = 0`. Otherwise, the shape
            is `(hidden_size, hidden_size)`.
        cells.{k}.bias_ih, cells.{k}.bias_ho, cells.{k}.bias_g : the matching
            learnable biases. Only present when ``bias=True``.

    .. note::
        All the weights and biases are initialized according to the provided
        initializers (`kernel_init`, `bias_init`).

    .. seealso::
        :class:`SRNNCell`

    Examples::

        >>> rnn = SRNN(10, 20, num_layers=2, dropout=0.1)
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
        super(SRNN, self).__init__(
            input_size, hidden_size, num_layers, dropout, batch_first
        )
        self.initialize_cells(SRNNCell, **kwargs)


class SRNNCell(SingleStateCellBase):
    r"""A shuffling recurrent neural network (SRNN) cell.

    [`arXiv <https://arxiv.org/abs/2007.07324>`_]

    .. math::

        \mathbf{b}(t) &= \bigl(
            \mathbf{W}_{ho}\,\mathrm{ReLU}\bigl(
                \mathbf{W}_{ih}\,\mathbf{x}(t) + \mathbf{b}_{ih}
            \bigr) + \mathbf{b}_{ho}
        \bigr)\circ\sigma\bigl(
            \mathbf{W}_{g}\,\mathbf{x}(t) + \mathbf{b}_{g}
        \bigr), \\[6pt]
        \mathbf{h}(t) &= \phi\bigl(
            \mathbf{P}\,\mathbf{h}(t-1) + \mathbf{b}(t)
        \bigr),

    where :math:`\mathbf{P}` is the fixed cyclic shift permutation
    (:math:`(\mathbf{P}\mathbf{h})_i = h_{i-1}`, indices modulo
    ``hidden_size``), :math:`\circ` is element‐wise product and :math:`\phi`
    is a pointwise nonlinearity (ReLU by default). The recurrence itself has no
    learnable weights.

    Args:
        input_size: The number of expected features in the input ``x``.
        hidden_size: The number of features in the hidden state ``h``.
        bias: If ``False``, the layer does not use biases.
            Default: ``True``.
        hyper_size: Width of the hidden layer of the input network.
            Default: ``64``.
        nonlinearity: Nonlinearity :math:`\phi`. Default: :func:`torch.relu`.
        kernel_init: Initializer for ``W_{ih}``, ``W_{ho}`` and ``W_{g}``.
            Default: :func:`torch.nn.init.xavier_uniform_`.
        bias_init: Initializer for biases when ``bias=True``.
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
        weight_ih: First-layer weights of the input network,
            of shape ``(hyper_size, input_size)``.
        weight_ho: Second-layer weights of the input network,
            of shape ``(hidden_size, hyper_size)``.
        weight_g: Gate weights of the input network,
            of shape ``(hidden_size, input_size)``.
        bias_ih: of shape ``(hyper_size)`` if ``bias=True``.
        bias_ho: of shape ``(hidden_size)`` if ``bias=True``.
        bias_g: of shape ``(hidden_size)`` if ``bias=True``.

    Examples::

        >>> cell = SRNNCell(10, 20)
        >>> x = torch.randn(5, 3, 10)     # (time_steps, batch, input_size)
        >>> h = torch.zeros(3, 20)        # (batch, hidden_size)
        >>> out = []
        >>> for t in range(x.size(0)):
        ...     h = cell(x[t], h)
        ...     out.append(h)
        >>> out = torch.stack(out, dim=0) # (time_steps, batch, hidden_size)
    """

    __constants__ = ["input_size", "hidden_size", "bias", "recurrent_bias", "hyper_size"]

    weight_ih: Tensor
    weight_ho: Tensor
    weight_g: Tensor
    bias_ih: Tensor
    bias_ho: Tensor
    bias_g: Tensor
    hyper_size: int

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        hyper_size: int = 64,
        nonlinearity="relu",
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
        self.hyper_size = int(hyper_size)
        self.act = resolve_activation(nonlinearity)
        self.init_cfg["kernel"] = resolve_init_name(kernel_init, self.init_cfg["kernel"])
        self.init_cfg["bias"] = resolve_init_name(bias_init, self.init_cfg["bias"])

        self._register_tensors(
            {
                "weight_ih": ((hyper_size, input_size), True),
                "weight_ho": ((hidden_size, hyper_size), True),
                "weight_g": ((hidden_size, input_size), True),
                "bias_ih": ((hyper_size,), bias),
                "bias_ho": ((hidden_size,), bias),
                "bias_g": ((hidden_size,), bias),
            }
        )
        self.reset_parameters()
        self._cleanup_non_scriptable()

    def reset_parameters(self) -> None:
        for weight in (self.weight_ih, self.weight_ho, self.weight_g):
            apply_init_(weight, self.init_cfg["kernel"])
        for b in (self.bias_ih, self.bias_ho, self.bias_g):
            if isinstance(b, nn.Parameter):
                apply_init_(b, self.init_cfg["bias"])

    def forward(self, inp: Tensor, state: Optional[Tensor] = None) -> Tensor:
        self._validate_input(inp)
        b_inp, is_batched = self._as_batched(inp)

        if state is None:
            b_state = self._zeros_state(b_inp.size(0), b_inp.device, b_inp.dtype)
        else:
            b_state = state.unsqueeze(0) if (not is_batched and state.dim() == 1) else state

        hyper = torch.relu(b_inp @ self.weight_ih.t() + self.bias_ih)
        drive = (hyper @ self.weight_ho.t() + self.bias_ho) * torch.sigmoid(
            b_inp @ self.weight_g.t() + self.bias_g
        )
        new_state = self.act(torch.roll(b_state, 1, -1) + drive)

        if not is_batched:
            new_state = new_state.squeeze(0)

        return new_state

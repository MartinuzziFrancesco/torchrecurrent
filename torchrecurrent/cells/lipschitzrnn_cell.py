import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional
from ..base import (
    SingleStateRecurrentLayerBase,
    SingleStateCellBase,
    resolve_activation,
    resolve_init_name,
)


class LipschitzRNN(SingleStateRecurrentLayerBase):
    r"""Multi-layer Lipschitz recurrent neural network.

    [`arXiv <https://arxiv.org/abs/2006.12070>`_]

    Each layer consists of a :class:`LipschitzRNNCell`, which updates the
    hidden state with a forward Euler step of the continuous-time system
    :math:`\dot{h} = A h + \phi(W h + U x + b)`:

    .. math::
        \begin{aligned}
        A &= \beta (M_A - M_A^\top) + (1 - \beta)(M_A + M_A^\top) - \gamma I, \\
        W &= \beta (M_W - M_W^\top) + (1 - \beta)(M_W + M_W^\top) - \gamma I, \\
        h_t &= h_{t-1} + \epsilon A h_{t-1}
               + \epsilon\, \phi(W h_{t-1} + b_{hh} + W_{ih} x_t + b_{ih})
        \end{aligned}

    where :math:`h_t` is the hidden state at time `t`, :math:`x_t` is the
    input at time `t`, :math:`M_A` and :math:`M_W` are learnable matrices,
    :math:`\beta` controls the skew-symmetric part, :math:`\gamma` is the
    diffusion (damping) coefficient, :math:`\epsilon` (``dt``) is the step size,
    and :math:`\phi` is a pointwise nonlinearity (e.g., tanh).

    In a multilayer LipschitzRNN, the input :math:`x^{(l)}_t` of the :math:`l`-th
    layer (:math:`l \ge 2`) is the hidden state :math:`h^{(l-1)}_t` of the
    previous layer multiplied by dropout :math:`\delta^{(l-1)}_t`, where each
    :math:`\delta^{(l-1)}_t` is a Bernoulli random variable which is 0 with
    probability :attr:`dropout`.

    Args:
        input_size: The number of expected features in the input `x`.
        hidden_size: The number of features in the hidden state `h`.
        num_layers: Number of recurrent layers. E.g., setting ``num_layers=2`` would
            mean stacking two LipschitzRNN layers, with the second receiving the
            outputs of the first. Default: 1
        dropout: If non-zero, introduces a `Dropout` layer on the outputs of each
            layer except the last layer, with dropout probability equal to
            :attr:`dropout`. Default: 0
        batch_first: If ``True``, then the input and output tensors are provided as
            `(batch, seq, feature)` instead of `(seq, batch, feature)`. Default: False
        bias: If ``False``, then the layer does not use input-side biases.
            Default: True
        recurrent_bias: If ``False``, then the layer does not use recurrent biases.
            Default: True
        dt: Euler step size :math:`\epsilon`. Default: 0.1
        beta: Skew-symmetry weight :math:`\beta`. Default: 0.7
        gamma: Diffusion coefficient :math:`\gamma`. Default: 0.001
        nonlinearity: Nonlinearity :math:`\phi`. Default: :func:`torch.tanh`
        kernel_init: Initializer for `W_{ih}`. Default:
            :func:`torch.nn.init.xavier_uniform_`
        recurrent_kernel_init: Initializer for `M_A` and `M_W`. Default:
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
          features `(h_t)` from the last layer of the LipschitzRNN, for each `t`.
        - **h_n**: tensor of shape :math:`(\text{num_layers}, N, H_{out})`
          containing the final
          hidden state for each element in the sequence.

    Attributes:
        cells.{k}.weight_ih : the learnable input-hidden weights of the :math:`k`-th
            layer, of shape `(hidden_size, input_size)` for `k = 0`. Otherwise, the
            shape is `(hidden_size, hidden_size)`.
        cells.{k}.weight_hh : the learnable matrices :math:`M_A, M_W` of the
            :math:`k`-th layer, of shape `(2*hidden_size, hidden_size)`.
        cells.{k}.bias_ih : the learnable input-hidden biases of the :math:`k`-th
            layer, of shape `(hidden_size)`. Only present when ``bias=True``.
        cells.{k}.bias_hh : the learnable hidden-hidden biases of the :math:`k`-th
            layer, of shape `(hidden_size)`. Only present when ``recurrent_bias=True``.

    .. note::
        All the weights and biases are initialized according to the provided
        initializers (`kernel_init`, `recurrent_kernel_init`, etc.).

    .. seealso::
        :class:`LipschitzRNNCell`

    Examples::

        >>> rnn = LipschitzRNN(10, 20, num_layers=2, dropout=0.1)
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
        super(LipschitzRNN, self).__init__(
            input_size, hidden_size, num_layers, dropout, batch_first
        )
        self.initialize_cells(LipschitzRNNCell, **kwargs)


class LipschitzRNNCell(SingleStateCellBase):
    r"""A Lipschitz recurrent neural network (LipschitzRNN) cell.

    [`arXiv <https://arxiv.org/abs/2006.12070>`_]

    .. math::

        \mathbf{A} &= \beta\bigl(\mathbf{M}_A - \mathbf{M}_A^\top\bigr)
            + (1 - \beta)\bigl(\mathbf{M}_A + \mathbf{M}_A^\top\bigr)
            - \gamma \mathbf{I}, \\[6pt]
        \mathbf{W} &= \beta\bigl(\mathbf{M}_W - \mathbf{M}_W^\top\bigr)
            + (1 - \beta)\bigl(\mathbf{M}_W + \mathbf{M}_W^\top\bigr)
            - \gamma \mathbf{I}, \\[6pt]
        \mathbf{h}(t) &= \mathbf{h}(t-1) + \epsilon\,\mathbf{A}\,\mathbf{h}(t-1)
            + \epsilon\,\phi\bigl(
                \mathbf{W}\,\mathbf{h}(t-1) + \mathbf{b}_{hh}
                + \mathbf{W}_{ih}\,\mathbf{x}(t) + \mathbf{b}_{ih}
            \bigr),

    where :math:`\beta` controls the skew-symmetric part, :math:`\gamma` is the
    diffusion coefficient, :math:`\epsilon` (``dt``) is the Euler step size and
    :math:`\phi` is a pointwise nonlinearity (e.g., tanh).

    Args:
        input_size: The number of expected features in the input ``x``.
        hidden_size: The number of features in the hidden state ``h``.
        bias: If ``False``, the layer does not use input-side biases.
            Default: ``True``.
        recurrent_bias: If ``False``, the layer does not use recurrent biases.
            Default: ``True``.
        dt: Euler step size :math:`\epsilon`. Default: ``0.1``.
        beta: Skew-symmetry weight :math:`\beta`. Default: ``0.7``.
        gamma: Diffusion coefficient :math:`\gamma`. Default: ``0.001``.
        nonlinearity: Nonlinearity :math:`\phi`. Default: :func:`torch.tanh`.
        kernel_init: Initializer for ``W_{ih}``.
            Default: :func:`torch.nn.init.xavier_uniform_`.
        recurrent_kernel_init: Initializer for ``M_A`` and ``M_W``.
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
        weight_ih: The learnable input–hidden weights,
            of shape ``(hidden_size, input_size)``.
        weight_hh: The learnable matrices ``M_A`` and ``M_W``,
            of shape ``(2*hidden_size, hidden_size)``.
        bias_ih: The learnable input–hidden biases,
            of shape ``(hidden_size)`` if ``bias=True``.
        bias_hh: The learnable hidden–hidden biases,
            of shape ``(hidden_size)`` if ``recurrent_bias=True``.

    Examples::

        >>> cell = LipschitzRNNCell(10, 20)
        >>> x = torch.randn(5, 3, 10)     # (time_steps, batch, input_size)
        >>> h = torch.zeros(3, 20)        # (batch, hidden_size)
        >>> out = []
        >>> for t in range(x.size(0)):
        ...     h = cell(x[t], h)
        ...     out.append(h)
        >>> out = torch.stack(out, dim=0) # (time_steps, batch, hidden_size)
    """

    __constants__ = [
        "input_size",
        "hidden_size",
        "bias",
        "recurrent_bias",
        "dt",
        "beta",
        "gamma",
    ]

    weight_ih: Tensor
    weight_hh: Tensor
    bias_ih: Tensor
    bias_hh: Tensor
    dt: float
    beta: float
    gamma: float

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        recurrent_bias: bool = True,
        dt: float = 0.1,
        beta: float = 0.7,
        gamma: float = 0.001,
        nonlinearity="tanh",
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
        self.dt = float(dt)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.act = resolve_activation(nonlinearity)
        self.init_cfg["kernel"] = resolve_init_name(kernel_init, self.init_cfg["kernel"])
        self.init_cfg["recurrent_kernel"] = resolve_init_name(
            recurrent_kernel_init, self.init_cfg["recurrent_kernel"]
        )
        self.init_cfg["bias"] = resolve_init_name(bias_init, self.init_cfg["bias"])
        self.init_cfg["recurrent_bias"] = resolve_init_name(
            recurrent_bias_init, self.init_cfg["recurrent_bias"]
        )

        self._register_tensors(
            {
                "weight_ih": ((hidden_size, input_size), True),
                "weight_hh": ((2 * hidden_size, hidden_size), True),
                "bias_ih": ((hidden_size,), bias),
                "bias_hh": ((hidden_size,), recurrent_bias),
            }
        )
        self.reset_parameters()
        self._cleanup_non_scriptable()

    def extra_repr(self) -> str:
        return (
            super().extra_repr() + f", dt={self.dt}, beta={self.beta}, gamma={self.gamma}"
        )

    def _structured(self, m: Tensor) -> Tensor:
        eye = torch.eye(self.hidden_size, device=m.device, dtype=m.dtype)
        return self.beta * (m - m.t()) + (1.0 - self.beta) * (m + m.t()) - self.gamma * eye

    def forward(self, inp: Tensor, state: Optional[Tensor] = None) -> Tensor:
        self._validate_input(inp)
        b_inp, is_batched = self._as_batched(inp)

        if state is None:
            b_state = self._zeros_state(b_inp.size(0), b_inp.device, b_inp.dtype)
        else:
            b_state = state.unsqueeze(0) if (not is_batched and state.dim() == 1) else state

        m_a, m_w = self.weight_hh.chunk(2, 0)
        a_mat = self._structured(m_a)
        w_mat = self._structured(m_w)

        pre = b_state @ w_mat.t() + self.bias_hh + b_inp @ self.weight_ih.t() + self.bias_ih
        new_state = b_state + self.dt * (b_state @ a_mat.t()) + self.dt * self.act(pre)

        if not is_batched:
            new_state = new_state.squeeze(0)

        return new_state

import math
import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Callable, Tuple, Union
from ..base import BaseSingleRecurrentLayer, BaseSingleRecurrentCell


class suRNN(BaseSingleRecurrentLayer):
    r"""Multi-layer Selective-Update RNN (suRNN).

    [`DOI <https://arxiv.org/abs/2603.02226>`_]

    Each layer consists of a :class:`suRNNCell`, which selectively updates
    the hidden state according to:

    .. math::
        \begin{aligned}
        a_{t,i} &= b_i + \sum_{k=1}^{K} \alpha_{i,k}
                   \sin(\omega_k t + \phi_{i,k}), \\
        g_{t,i} &= \mathcal{H}(a_{t,i}), \\
        \Delta h_t &= f_\theta(h_{t-1}, x_t) - h_{t-1}, \\
        h_t &= h_{t-1} + D_t \Delta h_t
        \end{aligned}

    where :math:`D_t = \mathrm{diag}(g_t)`, :math:`g_t \in \{0,1\}^H` is a
    binary gate computed via the Heaviside function :math:`\mathcal{H}`,
    :math:`f_\theta` is a standard RNN cell, and :math:`K` is the number of
    sinusoidal components.

    Args:
        input_size: The number of expected features in the input `x`.
        hidden_size: The number of features in the hidden state `h`.
        num_layers: Number of recurrent layers. Default: 1
        dropout: Dropout probability on outputs of each layer except the last.
            Default: 0
        batch_first: If ``True``, input/output tensors are ``(batch, seq,
            feature)``. Default: False
        bias: If ``False``, the layer does not use input-side biases.
            Default: True
        recurrent_bias: If ``False``, the layer does not use recurrent biases.
            Default: True
        num_harmonics: Number of sinusoidal components ``K``. Default: 1
        nonlinearity: Nonlinearity for the base RNN candidate.
            Default: :func:`torch.tanh`
        kernel_init: Initializer for ``W``.
            Default: :func:`torch.nn.init.xavier_uniform_`
        recurrent_kernel_init: Initializer for ``U``.
            Default: :func:`torch.nn.init.xavier_uniform_`
        bias_init: Initializer for input-side biases.
            Default: :func:`torch.nn.init.zeros_`
        recurrent_bias_init: Initializer for recurrent biases.
            Default: :func:`torch.nn.init.zeros_`
        device: The desired device of parameters.
        dtype: The desired floating point type of parameters.

    Inputs: input, h_0
        - **input**: tensor of shape :math:`(L, H_{in})` for unbatched input,
          :math:`(L, N, H_{in})` when ``batch_first=False`` or
          :math:`(N, L, H_{in})` when ``batch_first=True``.
        - **h_0**: tensor of shape :math:`(\text{num\_layers}, H_{out})` for
          unbatched input or :math:`(\text{num\_layers}, N, H_{out})`.
          Defaults to zeros if not provided.

    Outputs: output, h_n
        - **output**: tensor of shape :math:`(L, H_{out})` for unbatched
          input, :math:`(L, N, H_{out})` when ``batch_first=False`` or
          :math:`(N, L, H_{out})` when ``batch_first=True``.
        - **h_n**: tensor of shape :math:`(\text{num\_layers}, H_{out})` for
          unbatched input or :math:`(\text{num\_layers}, N, H_{out})`.

    Examples::

        >>> rnn = suRNN(10, 20, num_layers=2, dropout=0.1)
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
        super(suRNN, self).__init__(
            input_size, hidden_size, num_layers, dropout, batch_first
        )
        self.initialize_cells(suRNNCell, **kwargs)


class suRNNCell(BaseSingleRecurrentCell):
    r"""A Selective-Update RNN cell (suRNNCell).

    [`DOI <https://arxiv.org/abs/2603.02226>`_]

    .. math::

        a_{t,i} &= b_i + \sum_{k=1}^{K} \alpha_{i,k}
                   \sin(\omega_k\,t + \phi_{i,k}), \\[4pt]
        g_{t,i} &= \mathcal{H}(a_{t,i}), \\[4pt]
        \Delta h_t &= f_\theta(h_{t-1}, x_t) - h_{t-1}, \\[4pt]
        h_t &= h_{t-1} + D_t\,\Delta h_t,

    where :math:`D_t = \mathrm{diag}(g_t)`, :math:`g_t \in \{0,1\}^H` is a
    binary gate via the Heaviside function :math:`\mathcal{H}`,
    :math:`f_\theta(h_{t-1}, x_t) = \tanh(W x_t + b + U h_{t-1} + b_h)`,
    and :math:`K` is the number of sinusoidal components.

    Args:
        input_size: The number of expected features in the input ``x``.
        hidden_size: The number of features in the hidden state ``h``.
        bias: If ``False``, the layer does not use input-side biases.
            Default: ``True``.
        recurrent_bias: If ``False``, the layer does not use recurrent biases.
            Default: ``True``.
        num_harmonics: Number of sinusoidal components ``K``. Default: ``1``.
        nonlinearity: Nonlinearity for the base RNN candidate.
            Default: :func:`torch.tanh`.
        kernel_init: Initializer for ``W``.
            Default: :func:`torch.nn.init.xavier_uniform_`.
        recurrent_kernel_init: Initializer for ``U``.
            Default: :func:`torch.nn.init.xavier_uniform_`.
        bias_init: Initializer for input-side biases when ``bias=True``.
            Default: :func:`torch.nn.init.zeros_`.
        recurrent_bias_init: Initializer for recurrent biases when
            ``recurrent_bias=True``. Default: :func:`torch.nn.init.zeros_`.
        device: The desired device of parameters.
        dtype: The desired floating point type of parameters.

    Inputs: input, h_0
        - **input** of shape ``(batch, input_size)`` or ``(input_size,)``.
        - **h_0** of shape ``(batch, hidden_size)`` or ``(hidden_size,)``.
          Defaults to zeros if not provided.

    Outputs: h_1
        - **h_1** of shape ``(batch, hidden_size)`` or ``(hidden_size,)``.

    Variables:
        weight_ih: Learnable input–hidden weights of shape
            ``(hidden_size, input_size)``.
        weight_hh: Learnable hidden–hidden weights of shape
            ``(hidden_size, hidden_size)``.
        bias_ih: Learnable input–hidden biases of shape ``(hidden_size,)``
            if ``bias=True``.
        bias_hh: Learnable hidden–hidden biases of shape ``(hidden_size,)``
            if ``recurrent_bias=True``.
        gate_bias: Sinusoidal gate bias ``b`` of shape ``(hidden_size,)``.
        gate_alpha: Sinusoidal amplitudes ``α`` of shape
            ``(hidden_size, num_harmonics)``.
        gate_omega: Sinusoidal frequencies ``ω`` of shape
            ``(num_harmonics,)``.
        gate_phi: Sinusoidal phases ``φ`` of shape
            ``(hidden_size, num_harmonics)``.

    Examples::

        >>> cell = suRNNCell(10, 20)
        >>> x = torch.randn(5, 3, 10)   # (time_steps, batch, input_size)
        >>> h = torch.zeros(3, 20)      # (batch, hidden_size)
        >>> out = []
        >>> for t in range(x.size(0)):
        ...     h = cell(x[t], h, t=t)
        ...     out.append(h)
        >>> out = torch.stack(out, dim=0)
    """

    __constants__ = [
        "input_size",
        "hidden_size",
        "bias",
        "recurrent_bias",
        "num_harmonics",
        "nonlinearity",
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
        num_harmonics: int = 1,
        nonlinearity: Callable = torch.tanh,
        kernel_init: Callable = nn.init.xavier_uniform_,
        recurrent_kernel_init: Callable = nn.init.xavier_uniform_,
        bias_init: Callable = nn.init.zeros_,
        recurrent_bias_init: Callable = nn.init.zeros_,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super(suRNNCell, self).__init__(
            input_size, hidden_size, bias, recurrent_bias, device=device, dtype=dtype
        )
        self.num_harmonics = num_harmonics
        self.nonlinearity = nonlinearity
        self.kernel_init = kernel_init
        self.recurrent_kernel_init = recurrent_kernel_init
        self.bias_init = bias_init
        self.recurrent_bias_init = recurrent_bias_init

        self._default_register_tensors(
            input_size,
            hidden_size,
            ih_mult=1,
            hh_mult=1,
            bias=bias,
            recurrent_bias=recurrent_bias,
        )

        # Sinusoidal gating parameters
        self.gate_bias = nn.Parameter(torch.zeros(hidden_size, device=device, dtype=dtype))
        self.gate_alpha = nn.Parameter(
            torch.empty(hidden_size, num_harmonics, device=device, dtype=dtype)
        )
        self.gate_omega = nn.Parameter(
            torch.empty(num_harmonics, device=device, dtype=dtype)
        )
        self.gate_phi = nn.Parameter(
            torch.empty(hidden_size, num_harmonics, device=device, dtype=dtype)
        )

        self.init_weights()
        self._init_gate_params()

    def _init_gate_params(self) -> None:
        """Initialise sinusoidal gating parameters."""
        nn.init.ones_(self.gate_alpha)
        nn.init.zeros_(self.gate_phi)
        # Initialise frequencies as evenly spaced in (0, pi]
        with torch.no_grad():
            for k in range(self.num_harmonics):
                self.gate_omega[k] = math.pi * (k + 1) / self.num_harmonics

    def forward(
        self,
        inp: Tensor,
        state: Optional[Union[Tensor, Tuple[Tensor, ...]]] = None,
        t: int = 0,
    ) -> Tensor:
        state = self._check_state(state)
        self._validate_input(inp)
        self._validate_state(state)
        inp, state, is_batched = self._preprocess_input_and_state(inp, state)

        # Base RNN candidate: f_theta(h_{t-1}, x_t)
        candidate = self.nonlinearity(
            inp @ self.weight_ih.t()
            + self.bias_ih
            + state @ self.weight_hh.t()
            + self.bias_hh
        )

        # Sinusoidal gate signal: a_{t,i} = b_i + sum_k alpha_{i,k} * sin(omega_k * t + phi_{i,k})
        t_val = torch.tensor(float(t), device=inp.device, dtype=inp.dtype)
        # gate_omega: (K,), gate_alpha: (H, K), gate_phi: (H, K)
        sinusoids = torch.sin(self.gate_omega * t_val + self.gate_phi)  # (H, K)
        a = self.gate_bias + (self.gate_alpha * sinusoids).sum(dim=-1)   # (H,)

        # Binary gate via Heaviside: g_{t,i} = H(a_{t,i})
        gate = torch.heaviside(a, torch.zeros_like(a))  # (H,)

        # Selective update: h_t = h_{t-1} + D_t * (f_theta - h_{t-1})
        delta_h = candidate - state
        new_state = state + gate * delta_h

        if not is_batched:
            new_state = new_state.squeeze(0)

        return new_state

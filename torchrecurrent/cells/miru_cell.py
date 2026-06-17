import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Callable, Tuple, Union
from ..base import BaseSingleRecurrentLayer, BaseSingleRecurrentCell


class MiRU1(BaseSingleRecurrentLayer):
    r"""Multi-layer Minion Gated Unit 1 (MiRU1) neural network.

    [`DOI <https://doi.org/10.1016/j.neucom.2026.132847>`_]

    Each layer consists of a :class:`MiRU1Cell`, which updates the hidden
    state according to:

    .. math::
        \begin{aligned}
        r(t) &= \sigma(W_r x(t) + b_r + U_r h(t-1)), \\
        \tilde{h}(t) &= \tanh(W_h x(t) + b_h
                      + U_h (r(t) \circ h(t-1))), \\
        h(t) &= \lambda \circ h(t-1) + (1 - \lambda) \circ \tilde{h}(t)
        \end{aligned}

    where :math:`\lambda` is a hyperparameter, :math:`\sigma` is the
    sigmoid function, and :math:`\circ` denotes elementwise multiplication.

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
        update_coefficient: Mixing hyperparameter controlling the blend between the
            previous hidden state and the candidate. Default: 0.5
        nonlinearity: Nonlinearity for the candidate hidden state.
            Default: :func:`torch.tanh`
        gate_nonlinearity: Activation for the reset gate.
            Default: :func:`torch.sigmoid`
        kernel_init: Initializer for ``W_*``.
            Default: :func:`torch.nn.init.xavier_uniform_`
        recurrent_kernel_init: Initializer for ``U_*``.
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

        >>> rnn = MiRU1(10, 20, num_layers=2, dropout=0.1)
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
        super(MiRU1, self).__init__(
            input_size, hidden_size, num_layers, dropout, batch_first
        )
        self.initialize_cells(MiRU1Cell, **kwargs)


class MiRU1Cell(BaseSingleRecurrentCell):
    r"""A Minion Gated Unit 1 (MiRU1) cell.

    [`DOI <https://doi.org/10.1016/j.neucom.2026.132847>`_]

    .. math::

        r(t) &= \sigma\bigl(
            W_r\,x(t) + b_r + U_r\,h(t-1)
        \bigr), \\[6pt]
        \tilde{h}(t) &= \tanh\bigl(
            W_h\,x(t) + b_h
            + U_h\,\bigl(r(t) \circ h(t-1)\bigr)
        \bigr), \\[6pt]
        h(t) &= \lambda \circ h(t-1)
               + \bigl(1 - \lambda\bigr) \circ \tilde{h}(t),

    where :math:`\lambda` is a hyperparameter, :math:`\circ` is element-wise
    product, and :math:`\sigma` is the sigmoid function.

    Args:
        input_size: The number of expected features in the input ``x``.
        hidden_size: The number of features in the hidden state ``h``.
        bias: If ``False``, the layer does not use input-side biases.
            Default: ``True``.
        recurrent_bias: If ``False``, the layer does not use recurrent biases.
            Default: ``True``.
        update_coefficient: Mixing hyperparameter controlling the blend between the
            previous hidden state and the candidate. Default: ``0.5``.
        nonlinearity: Nonlinearity for the candidate hidden state.
            Default: :func:`torch.tanh`.
        gate_nonlinearity: Activation for the reset gate.
            Default: :func:`torch.sigmoid`.
        kernel_init: Initializer for ``W_r`` and ``W_h``.
            Default: :func:`torch.nn.init.xavier_uniform_`.
        recurrent_kernel_init: Initializer for ``U_r`` and ``U_h``.
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
            ``(2*hidden_size, input_size)`` (reset gate & candidate parts).
        weight_hh: Learnable hidden–hidden weights of shape
            ``(2*hidden_size, hidden_size)`` (reset gate & candidate parts).
        bias_ih: Learnable input–hidden biases of shape ``(2*hidden_size,)``
            if ``bias=True``.
        bias_hh: Learnable hidden–hidden biases of shape ``(2*hidden_size,)``
            if ``recurrent_bias=True``.

    Examples::

        >>> cell = MiRU1Cell(10, 20)
        >>> x = torch.randn(5, 3, 10)   # (time_steps, batch, input_size)
        >>> h = torch.zeros(3, 20)      # (batch, hidden_size)
        >>> out = []
        >>> for t in range(x.size(0)):
        ...     h = cell(x[t], h)
        ...     out.append(h)
        >>> out = torch.stack(out, dim=0)
    """

    __constants__ = [
        "input_size",
        "hidden_size",
        "bias",
        "recurrent_bias",
        "update_coefficient",
        "nonlinearity",
        "gate_nonlinearity",
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
        update_coefficient: float = 0.5,
        nonlinearity: Callable = torch.tanh,
        gate_nonlinearity: Callable = torch.sigmoid,
        kernel_init: Callable = nn.init.xavier_uniform_,
        recurrent_kernel_init: Callable = nn.init.xavier_uniform_,
        bias_init: Callable = nn.init.zeros_,
        recurrent_bias_init: Callable = nn.init.zeros_,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super(MiRU1Cell, self).__init__(
            input_size, hidden_size, bias, recurrent_bias, device=device, dtype=dtype
        )
        self.update_coefficient = update_coefficient
        self.nonlinearity = nonlinearity
        self.gate_nonlinearity = gate_nonlinearity
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
        self.init_weights()

    def forward(
        self, inp: Tensor, state: Optional[Union[Tensor, Tuple[Tensor, ...]]] = None
    ) -> Tensor:
        state = self._check_state(state)
        self._validate_input(inp)
        self._validate_state(state)
        inp, state, is_batched = self._preprocess_input_and_state(inp, state)

        weight_ih_r, weight_ih_h = self.weight_ih.chunk(2, 0)
        weight_hh_r, weight_hh_h = self.weight_hh.chunk(2, 0)
        bias_ih_r, bias_ih_h = self.bias_ih.chunk(2, 0)
        bias_hh_r, bias_hh_h = self.bias_hh.chunk(2, 0)

        # Reset gate: r(t) = sigmoid(W_r x + b_r + U_r h)
        rg = inp @ weight_ih_r.t() + bias_ih_r + state @ weight_hh_r.t() + bias_hh_r
        reset_gate = self.gate_nonlinearity(rg)

        # Candidate: h~(t) = tanh(W_h x + b_h + U_h (r(t) * h))
        ch = (
            inp @ weight_ih_h.t()
            + bias_ih_h
            + (reset_gate * state) @ weight_hh_h.t()
            + bias_hh_h
        )
        candidate = self.nonlinearity(ch)

        # Update: h(t) = lambda * h + (1 - lambda) * h~
        new_state = self.update_coefficient * state + (1.0 - self.update_coefficient) * candidate

        if not is_batched:
            new_state = new_state.squeeze(0)

        return new_state


class MiRU2(BaseSingleRecurrentLayer):
    r"""Multi-layer Minion Gated Unit 2 (MiRU2) neural network.

    [`DOI <https://doi.org/10.1016/j.neucom.2026.132847>`_]

    Each layer consists of a :class:`MiRU2Cell`, which updates the hidden
    state according to:

    .. math::
        \begin{aligned}
        \tilde{h}(t) &= \tanh(W_h x(t) + b_h
                      + U_h (\theta \circ h(t-1))), \\
        h(t) &= \lambda \circ h(t-1) + (1 - \lambda) \circ \tilde{h}(t)
        \end{aligned}

    where :math:`\lambda` and :math:`\theta` are hyperparameters, and
    :math:`\circ` denotes elementwise multiplication.

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
        update_coefficient: Mixing hyperparameter. Default: 0.5
        reset_coefficient: Hidden-state scaling hyperparameter. Default: 0.5
        nonlinearity: Nonlinearity for the candidate hidden state.
            Default: :func:`torch.tanh`
        kernel_init: Initializer for ``W_h``.
            Default: :func:`torch.nn.init.xavier_uniform_`
        recurrent_kernel_init: Initializer for ``U_h``.
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

        >>> rnn = MiRU2(10, 20, num_layers=2, dropout=0.1)
        >>> input = torch.randn(5, 3, 10)
        >>> h0 = torch.zeros(2, 3, 20)
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
        super(MiRU2, self).__init__(
            input_size, hidden_size, num_layers, dropout, batch_first
        )
        self.initialize_cells(MiRU2Cell, **kwargs)


class MiRU2Cell(BaseSingleRecurrentCell):
    r"""A Minion Gated Unit 2 (MiRU2) cell.

    [`DOI <https://doi.org/10.1016/j.neucom.2026.132847>`_]

    .. math::

        \tilde{h}(t) &= \tanh\bigl(
            W_h\,x(t) + b_h
            + U_h\,\bigl(\theta \circ h(t-1)\bigr)
        \bigr), \\[6pt]
        h(t) &= \lambda \circ h(t-1)
               + \bigl(1 - \lambda\bigr) \circ \tilde{h}(t),

    where :math:`\lambda` and :math:`\theta` are hyperparameters and
    :math:`\circ` is element-wise product.

    Args:
        input_size: The number of expected features in the input ``x``.
        hidden_size: The number of features in the hidden state ``h``.
        bias: If ``False``, the layer does not use input-side biases.
            Default: ``True``.
        recurrent_bias: If ``False``, the layer does not use recurrent biases.
            Default: ``True``.
        update_coefficient: Mixing hyperparameter. Default: ``0.5``.
        reset_coefficient: Hidden-state scaling hyperparameter. Default: ``0.5``.
        nonlinearity: Nonlinearity for the candidate hidden state.
            Default: :func:`torch.tanh`.
        kernel_init: Initializer for ``W_h``.
            Default: :func:`torch.nn.init.xavier_uniform_`.
        recurrent_kernel_init: Initializer for ``U_h``.
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

    Examples::

        >>> cell = MiRU2Cell(10, 20)
        >>> x = torch.randn(5, 3, 10)
        >>> h = torch.zeros(3, 20)
        >>> out = []
        >>> for t in range(x.size(0)):
        ...     h = cell(x[t], h)
        ...     out.append(h)
        >>> out = torch.stack(out, dim=0)
    """

    __constants__ = [
        "input_size",
        "hidden_size",
        "bias",
        "recurrent_bias",
        "update_coefficient",
        "reset_coefficient",
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
        update_coefficient: float = 0.5,
        reset_coefficient: float = 0.5,
        nonlinearity: Callable = torch.tanh,
        kernel_init: Callable = nn.init.xavier_uniform_,
        recurrent_kernel_init: Callable = nn.init.xavier_uniform_,
        bias_init: Callable = nn.init.zeros_,
        recurrent_bias_init: Callable = nn.init.zeros_,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super(MiRU2Cell, self).__init__(
            input_size, hidden_size, bias, recurrent_bias, device=device, dtype=dtype
        )
        self.update_coefficient = update_coefficient
        self.reset_coefficient = reset_coefficient
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
        self.init_weights()

    def forward(
        self, inp: Tensor, state: Optional[Union[Tensor, Tuple[Tensor, ...]]] = None
    ) -> Tensor:
        state = self._check_state(state)
        self._validate_input(inp)
        self._validate_state(state)
        inp, state, is_batched = self._preprocess_input_and_state(inp, state)

        # Candidate: h~(t) = tanh(W_h x + b_h + U_h (reset_coefficient * h))
        ch = (
            inp @ self.weight_ih.t()
            + self.bias_ih
            + (self.reset_coefficient * state) @ self.weight_hh.t()
            + self.bias_hh
        )
        candidate = self.nonlinearity(ch)

        # Update: h(t) = lambda * h + (1 - lambda) * h~
        new_state = self.update_coefficient * state + (1.0 - self.update_coefficient) * candidate

        if not is_batched:
            new_state = new_state.squeeze(0)

        return new_state

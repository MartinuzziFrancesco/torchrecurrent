import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Tuple
from ..base import (
    SingleStateRecurrentLayerBase,
    SingleStateCellBase,
    resolve_activation,
    resolve_init_name,
    apply_init_,
)


class MiRU1(SingleStateRecurrentLayerBase):
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
            Default: ``"tanh"``
        gate_nonlinearity: Activation for the reset gate.
            Default: ``"sigmoid"``
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


class MiRU1Cell(SingleStateCellBase):
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
            Default: ``"tanh"``.
        gate_nonlinearity: Activation for the reset gate.
            Default: ``"sigmoid"``.
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
        nonlinearity="tanh",
        gate_nonlinearity="sigmoid",
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
        self.update_coefficient = update_coefficient
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

        self._default_register_tensors(ih_mult=2, hh_mult=2)
        self.reset_parameters()
        self._cleanup_non_scriptable()

    def forward(self, inp: Tensor, state: Optional[Tensor] = None) -> Tensor:
        self._validate_input(inp)
        b_inp, is_batched = self._as_batched(inp)

        if state is None:
            b_state = self._zeros_state(b_inp.size(0), b_inp.device, b_inp.dtype)
        else:
            b_state = state.unsqueeze(0) if (not is_batched and state.dim() == 1) else state

        weight_ih_r, weight_ih_h = self.weight_ih.chunk(2, 0)
        weight_hh_r, weight_hh_h = self.weight_hh.chunk(2, 0)
        bias_ih_r, bias_ih_h = self.bias_ih.chunk(2, 0)
        bias_hh_r, bias_hh_h = self.bias_hh.chunk(2, 0)

        rg = b_inp @ weight_ih_r.t() + bias_ih_r + b_state @ weight_hh_r.t() + bias_hh_r
        reset_gate = self.gate_act(rg)

        ch = (
            b_inp @ weight_ih_h.t()
            + bias_ih_h
            + (reset_gate * b_state) @ weight_hh_h.t()
            + bias_hh_h
        )
        candidate = self.act(ch)

        new_state = self.update_coefficient * b_state + (1.0 - self.update_coefficient) * candidate

        if not is_batched:
            new_state = new_state.squeeze(0)

        return new_state


class MiRU2(SingleStateRecurrentLayerBase):
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
            Default: ``"tanh"``
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


class MiRU2Cell(SingleStateCellBase):
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
            Default: ``"tanh"``.
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
        self.update_coefficient = update_coefficient
        self.reset_coefficient = reset_coefficient
        self.act = resolve_activation(nonlinearity)
        self.init_cfg["kernel"] = resolve_init_name(kernel_init, self.init_cfg["kernel"])
        self.init_cfg["recurrent_kernel"] = resolve_init_name(
            recurrent_kernel_init, self.init_cfg["recurrent_kernel"]
        )
        self.init_cfg["bias"] = resolve_init_name(bias_init, self.init_cfg["bias"])
        self.init_cfg["recurrent_bias"] = resolve_init_name(
            recurrent_bias_init, self.init_cfg["recurrent_bias"]
        )

        self._default_register_tensors(ih_mult=1, hh_mult=1)
        self.reset_parameters()
        self._cleanup_non_scriptable()

    def forward(self, inp: Tensor, state: Optional[Tensor] = None) -> Tensor:
        self._validate_input(inp)
        b_inp, is_batched = self._as_batched(inp)

        if state is None:
            b_state = self._zeros_state(b_inp.size(0), b_inp.device, b_inp.dtype)
        else:
            b_state = state.unsqueeze(0) if (not is_batched and state.dim() == 1) else state

        ch = (
            b_inp @ self.weight_ih.t()
            + self.bias_ih
            + (self.reset_coefficient * b_state) @ self.weight_hh.t()
            + self.bias_hh
        )
        candidate = self.act(ch)

        new_state = self.update_coefficient * b_state + (1.0 - self.update_coefficient) * candidate

        if not is_batched:
            new_state = new_state.squeeze(0)

        return new_state

import torch
from torch import Tensor
import torch.nn as nn
from typing import Optional, Tuple
from ..base import (
    DoubleStateRecurrentLayerBase,
    DoubleStateCellBase,
    resolve_init_name,
    apply_init_,
)


class UnICORNN(DoubleStateRecurrentLayerBase):
    r"""Multi-layer Undamped Independent Controlled Oscillatory RNN (UnICORNN).

    [`arXiv <https://arxiv.org/abs/2103.05487>`_]

    Each layer consists of a :class:`UnICORNNCell`, which maintains two coupled
    state vectors, the hidden state :math:`h(t)` and the control state
    :math:`z(t)`, updated as:

    .. math::
        \begin{aligned}
            h(t) &= h(t-1) + \Delta t \, \hat{\sigma}(w_{ch}) \circ z(t), \\
            z(t) &= z(t-1) - \Delta t \, \hat{\sigma}(w_{ch}) \circ
                \Bigl[\sigma(w_{hh} \circ h(t-1) + W_{ih} x(t) + b_{ih})
                + \alpha h(t-1)\Bigr],
        \end{aligned}

    where :math:`\Delta t` is the integration step ``dt``,
    :math:`\alpha` is a leakage constant, :math:`\sigma` is the sigmoid, and
    :math:`\circ` is the elementwise product.

    Args:
        input_size: Number of expected features in the input `x`.
        hidden_size: Number of features in the hidden state `h` (and control `z`).
        num_layers: Number of stacked recurrent layers. Default: 1
        dropout: If non-zero, adds dropout after each layer (except the last).
            Default: 0
        batch_first: If ``True``, inputs and outputs are in
            `(batch, seq, feature)` format instead of `(seq, batch, feature)`.
            Default: False
        bias: If ``False``, disables input bias `b_{ih}`. Default: True
        recurrent_bias: If ``True``, enables an extra hidden bias `b_{hh}`.
            Default: False
        kernel_init: Initializer for `W_{ih}`.
            Default: :func:`torch.nn.init.xavier_uniform_`
        recurrent_kernel_init: Initializer for `w_{hh}`.
            Default: :func:`torch.nn.init.uniform_`
        control_kernel_init: Initializer for `w_{ch}`.
            Default: uniform on ``[-0.1, 0.1]``
        bias_init: Initializer for `b_{ih}` when ``bias=True``.
            Default: :func:`torch.nn.init.zeros_`
        recurrent_bias_init: Initializer for `b_{hh}` when ``recurrent_bias=True``.
            Default: :func:`torch.nn.init.zeros_`
        dt: Integration step :math:`\Delta t`. Default: 0.1
        alpha: Leakage coefficient :math:`\alpha`. Default: 0.0
        device: Desired device of parameters.
        dtype: Desired floating point type of parameters.

    Inputs: input, (h_0, z_0)
        - **input**: tensor of shape `(L, H_in)` for unbatched input,
          `(L, N, H_in)` when ``batch_first=False``, or `(N, L, H_in)` when
          ``batch_first=True`` containing input sequence features.
        - **h_0**: tensor of shape `(num_layers, H_out)` for unbatched input or
          `(num_layers, N, H_out)` containing the initial hidden state. Defaults
          to zeros if not provided.
        - **z_0**: tensor of the same shape as `h_0`, containing the initial
          control state. Defaults to zeros if not provided.

        Where:

        .. math::
            \begin{aligned}
                N &= \text{batch size} \\
                L &= \text{sequence length} \\
                H_{in} &= \text{input size} \\
                H_{out} &= \text{hidden size}
            \end{aligned}

    Outputs: output, (h_n, z_n)
        - **output**: tensor of shape `(L, H_out)` for unbatched input,
          `(L, N, H_out)` when ``batch_first=False``, or `(N, L, H_out)` when
          ``batch_first=True`` containing the hidden states from the last layer
          at each timestep.
        - **h_n**: final hidden state for each layer,
          shape `(num_layers, H_out)` (unbatched) or `(num_layers, N, H_out)`.
        - **z_n**: final control state for each layer, same shape as `h_n`.

    Attributes:
        cells.{k}.weight_ih : input–hidden weights of the :math:`k`-th layer,
            shape `(hidden_size, input_size)` for `k=0`,
            otherwise `(hidden_size, hidden_size)`.
        cells.{k}.weight_hh : elementwise hidden weights of the :math:`k`-th layer,
            shape `(hidden_size,)`.
        cells.{k}.weight_ch : control weights of the :math:`k`-th layer,
            shape `(hidden_size,)`.
        cells.{k}.bias_ih : input bias of the :math:`k`-th layer,
            shape `(hidden_size,)` if ``bias=True``.
        cells.{k}.bias_hh : optional hidden bias of the :math:`k`-th layer,
            shape `(hidden_size,)` if ``recurrent_bias=True``.

    .. seealso::
        :class:`UnICORNNCell`

    Examples::

        >>> rnn = UnICORNN(10, 20, num_layers=2, dt=0.5, alpha=0.1)
        >>> x = torch.randn(5, 3, 10)    # (seq_len, batch, input_size)
        >>> h0 = torch.zeros(2, 3, 20)
        >>> z0 = torch.zeros(2, 3, 20)
        >>> out, (hn, zn) = rnn(x, (h0, z0))
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
        super(UnICORNN, self).__init__(
            input_size, hidden_size, num_layers, dropout, batch_first
        )
        self.initialize_cells(UnICORNNCell, **kwargs)


class UnICORNNCell(DoubleStateCellBase):
    r"""An Undamped Independent Controlled Oscillatory RNN (UnICORNN) cell.

    [`arXiv <https://arxiv.org/abs/2103.05487>`_]

    The cell maintains two coupled state vectors, the hidden state
    :math:`\mathbf{h}(t)` and the control state :math:`\mathbf{z}(t)`,
    which evolve according to

    .. math::

        \begin{aligned}
            \mathbf{h}(t) &= \mathbf{h}(t-1)
                + \Delta t\,\hat{\sigma}(\mathbf{w}_{ch})
                \circ \mathbf{z}(t), \\
            \mathbf{z}(t) &= \mathbf{z}(t-1)
                - \Delta t\,\hat{\sigma}(\mathbf{w}_{ch}) \circ
                \Bigl[
                    \sigma\bigl(
                        \mathbf{w}_{hh} \circ \mathbf{h}(t-1)
                        + \mathbf{W}_{ih}\,\mathbf{x}(t)
                        + \mathbf{b}_{ih}
                    \bigr)
                    + \alpha\,\mathbf{h}(t-1)
                \Bigr],
        \end{aligned}

    where :math:`\Delta t` is the time step ``dt``, and :math:`\alpha`
    is the leakage constant.

    Args:
        input_size: The number of expected features in the input ``x``.
        hidden_size: The number of features in the hidden state ``h``.
        bias: If ``False``, the layer does not use the input bias
            ``b_{ih}``. Default: ``True``.
        recurrent_bias: If ``True``, the layer uses an extra hidden
            bias ``b_{hh}``. Default: ``False``.
        kernel_init: Initializer for ``W_{ih}``.
            Default: :func:`torch.nn.init.xavier_uniform_`.
        recurrent_kernel_init: Initializer for ``w_{hh}``.
            Default: :func:`torch.nn.init.uniform_`.
        control_kernel_init: Initializer for ``w_{ch}``.
            Default: uniform on ``[-0.1, 0.1]``.
        bias_init: Initializer for ``b_{ih}``.
            Default: :func:`torch.nn.init.zeros_`.
        recurrent_bias_init: Initializer for ``b_{hh}``.
            Default: :func:`torch.nn.init.zeros_`.
        dt: Time step :math:`\Delta t` between updates. Default: ``0.1``.
        alpha: Leakage coefficient in the control update. Default: ``0.0``.
        device: The desired device of parameters.
        dtype: The desired floating point type of parameters.

    Inputs: input, (h, z)
        - **input** of shape ``(batch, input_size)`` or ``(input_size,)``:
          Tensor containing input features.
        - **h** of shape ``(batch, hidden_size)`` or ``(hidden_size,)``:
          Previous hidden state.
        - **z** of shape ``(batch, hidden_size)`` or ``(hidden_size,)``:
          Previous control state.

        If **(h, z)** is not provided, both default to zeros.

    Outputs: (h_1, z_1)
        - **h_1** of shape ``(batch, hidden_size)`` or ``(hidden_size,)``:
          Next hidden state.
        - **z_1** of shape ``(batch, hidden_size)`` or ``(hidden_size,)``:
          Next control state.

    Variables:
        weight_ih: The learnable input–hidden weights,
            of shape ``(hidden_size, input_size)``.
        weight_hh: The learnable elementwise hidden weights,
            of shape ``(hidden_size,)``.
        weight_ch: The learnable control weights,
            of shape ``(hidden_size,)``.
        bias_ih: The learnable input bias,
            of shape ``(hidden_size,)``.
        bias_hh: The optional learnable hidden bias,
            of shape ``(hidden_size,)`` when ``recurrent_bias=True``.

    Examples::

        >>> cell = UnICORNNCell(10, 20, dt=0.5, alpha=0.1)
        >>> x = torch.randn(5, 3, 10)      # (time_steps, batch, input_size)
        >>> h = torch.zeros(3, 20)         # (batch, hidden_size)
        >>> z = torch.zeros(3, 20)         # (batch, hidden_size)
        >>> outs_h, outs_z = [], []
        >>> for t in range(x.size(0)):
        ...     h, z = cell(x[t], (h, z))
        ...     outs_h.append(h)
        ...     outs_z.append(z)
        >>> outs_h = torch.stack(outs_h, dim=0)
        >>> outs_z = torch.stack(outs_z, dim=0)
    """

    __constants__ = ["input_size", "hidden_size", "bias", "recurrent_bias", "dt", "alpha"]

    weight_ih: Tensor
    weight_hh: Tensor
    weight_ch: Tensor
    bias_ih: Tensor
    bias_hh: Tensor
    alpha: float

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        recurrent_bias: bool = False,
        kernel_init=nn.init.xavier_uniform_,
        recurrent_kernel_init=nn.init.uniform_,
        control_kernel_init="uniform_centered",
        bias_init=nn.init.zeros_,
        recurrent_bias_init=nn.init.zeros_,
        dt: float = 0.1,
        alpha: float = 0.0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super(UnICORNNCell, self).__init__(
            input_size, hidden_size, bias, recurrent_bias, device=device, dtype=dtype
        )
        self.dt = dt
        self.alpha = alpha
        self.init_cfg["kernel"] = resolve_init_name(kernel_init, self.init_cfg["kernel"])
        self.init_cfg["recurrent_kernel"] = resolve_init_name(
            recurrent_kernel_init, "uniform"
        )
        self.init_cfg["control_kernel"] = resolve_init_name(
            control_kernel_init, "uniform_centered"
        )
        self.init_cfg["bias"] = resolve_init_name(bias_init, self.init_cfg["bias"])
        self.init_cfg["recurrent_bias"] = resolve_init_name(
            recurrent_bias_init, self.init_cfg["recurrent_bias"]
        )

        self._register_tensors(
            {
                "weight_ih": ((hidden_size, input_size), True),
                "weight_hh": ((hidden_size,), True),
                "weight_ch": ((hidden_size,), True),
                "bias_ih": ((hidden_size,), bias),
                "bias_hh": ((hidden_size,), recurrent_bias),
            }
        )
        self.reset_parameters()
        self._cleanup_non_scriptable()

    def reset_parameters(self) -> None:
        apply_init_(self.weight_ih, self.init_cfg["kernel"])
        apply_init_(self.weight_hh, self.init_cfg["recurrent_kernel"])
        apply_init_(self.weight_ch, self.init_cfg["control_kernel"])
        if hasattr(self, "bias_ih"):
            apply_init_(self.bias_ih, self.init_cfg["bias"])
        if hasattr(self, "bias_hh"):
            apply_init_(self.bias_hh, self.init_cfg["recurrent_bias"])

    def forward(
        self, inp: Tensor, state: Optional[Tuple[Tensor, Tensor]] = None
    ) -> Tuple[Tensor, Tensor]:
        self._validate_input(inp)
        b_inp, is_batched = self._as_batched(inp)

        if state is None:
            b_h = self._zeros_state(b_inp.size(0), b_inp.device, b_inp.dtype)
            b_c = self._zeros_state(b_inp.size(0), b_inp.device, b_inp.dtype)
        else:
            h, c = state
            b_h = (
                self._zeros_state(b_inp.size(0), b_inp.device, b_inp.dtype)
                if h is None
                else (h.unsqueeze(0) if (not is_batched and h.dim() == 1) else h)
            )
            b_c = (
                self._zeros_state(b_inp.size(0), b_inp.device, b_inp.dtype)
                if c is None
                else (c.unsqueeze(0) if (not is_batched and c.dim() == 1) else c)
            )

        candidate_state = torch.tanh(
            b_inp @ self.weight_ih.t() + self.bias_ih + b_h * self.weight_hh + self.bias_hh
        )
        new_c = b_c - self.dt * torch.sigmoid(self.weight_ch) * (
            candidate_state + self.alpha * b_h
        )
        new_h = b_h + self.dt * torch.sigmoid(self.weight_ch) * new_c

        if not is_batched:
            new_h = new_h.squeeze(0)
            new_c = new_c.squeeze(0)

        return new_h, new_c

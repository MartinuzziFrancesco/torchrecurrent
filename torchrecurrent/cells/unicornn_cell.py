import torch
from torch import Tensor
import torch.nn as nn
from typing import Optional, Callable, Union, Tuple
from ..base import BaseDoubleRecurrentLayer, BaseDoubleRecurrentCell


class UnICORNN(BaseDoubleRecurrentLayer):
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


class UnICORNNCell(BaseDoubleRecurrentCell):
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
                        \mathbf{W}_{hh}\,\mathbf{h}(t-1)
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
        recurrent_bias: If ``False``, the layer does not use the hidden
            bias ``b_{hh}``. Default: ``True``.
        kernel_init: Initializer for ``W_{ih}``.
            Default: :func:`torch.nn.init.xavier_uniform_`.
        recurrent_kernel_init: Initializer for ``W_{hh}``.
            Default: :func:`torch.nn.init.xavier_uniform_`.
        control_kernel_init: Initializer for ``w_{ch}``.
            Default: :func:`torch.nn.init.normal_`.
        bias_init: Initializer for ``b_{ih}``.
            Default: :func:`torch.nn.init.zeros_`.
        recurrent_bias_init: Initializer for ``b_{hh}``.
            Default: :func:`torch.nn.init.zeros_`.
        dt: Time step :math:`\Delta t` between updates. Default: ``1.0``.
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
        weight_hh: The learnable hidden–hidden weights,
            of shape ``(hidden_size, hidden_size)``.
        weight_ch: The learnable control weights,
            of shape ``(hidden_size,)``.
        bias_ih: The learnable input bias,
            of shape ``(hidden_size,)``.
        bias_hh: The learnable hidden bias,
            of shape ``(hidden_size,)``.

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

    __constants__ = [
        "input_size",
        "hidden_size",
        "bias",
        "recurrent_bias",
        "kernel_init",
        "recurrent_kernel_init",
        "control_kernel_init",
        "bias_init",
        "recurrent_bias_init",
        "dt",
        "alpha",
    ]

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
        recurrent_bias: bool = True,
        kernel_init: Callable = nn.init.xavier_uniform_,
        recurrent_kernel_init: Callable = nn.init.xavier_uniform_,
        control_kernel_init: Callable = nn.init.normal_,
        bias_init: Callable = nn.init.zeros_,
        recurrent_bias_init: Callable = nn.init.zeros_,
        dt: float = 1.0,
        alpha: float = 0.0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super(UnICORNNCell, self).__init__(
            input_size, hidden_size, bias, recurrent_bias, device=device, dtype=dtype
        )
        self.kernel_init = kernel_init
        self.recurrent_kernel_init = recurrent_kernel_init
        self.control_kernel_init = control_kernel_init
        self.bias_init = bias_init
        self.recurrent_bias_init = recurrent_bias_init
        self.dt = dt
        self.alpha = alpha

        self._register_tensors(
            {
                "weight_ih": ((hidden_size, input_size), True),
                "weight_hh": ((hidden_size, hidden_size), True),
                "weight_ch": ((hidden_size,), True),
                "bias_ih": ((hidden_size,), bias),
                "bias_hh": ((hidden_size,), recurrent_bias),
            }
        )
        self.init_weights()

    def init_weights(self):
        for name, param in self.named_parameters():
            if "weight_ih" in name:
                self.kernel_init(param)
            elif "weight_hh" in name:
                self.recurrent_kernel_init(param)
            elif "weight_ch" in name:
                self.control_kernel_init(param)
            elif "bias_ih" in name:
                self.bias_init(param)
            elif "bias_hh" in name:
                self.recurrent_bias_init(param)

    def forward(
        self, inp: Tensor, state: Optional[Union[Tensor, Tuple[Tensor, ...]]] = None
    ) -> Tuple[Tensor, Tensor]:
        state, c_state = self._check_states(state)
        self._validate_input(inp)
        self._validate_states((state, c_state))
        inp, state, c_state, is_batched = self._preprocess_states(inp, (state, c_state))

        candidate_state = torch.tanh(
            inp @ self.weight_ih.t()
            + self.bias_ih
            + state @ self.weight_hh.t()
            + self.bias_hh
        )
        new_cstate = c_state - self.dt * torch.sigmoid(self.weight_ch) * (
            candidate_state + self.alpha * state
        )
        new_state = state + self.dt * torch.sigmoid(self.weight_ch) * new_cstate

        if not is_batched:
            new_state = new_state.squeeze(0)
            new_cstate = new_cstate.squeeze(0)

        return new_state, new_cstate

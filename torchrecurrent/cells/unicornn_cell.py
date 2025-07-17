import torch
from torch import Tensor
import torch.nn as nn
from typing import Optional, Union, Tuple
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
    r"""An undamped independent controlled oscillatory recurrent neural unit (UnICORNN)
    cell.

    Implements the dynamics described in arXiv:2103.05487.

    The cell maintains two coupled state vectors, the hidden state :math:`\mathbf{h}(t)`
    and the control state :math:`\mathbf{z}(t)`, which evolve according to

    .. math::
        \begin{aligned}
            \mathbf{h}(t) &= \mathbf{h}(t-1)
                + \Delta t \;\hat{\sigma}(\mathbf{w}_{ch}) \circ \mathbf{z}(t), \\
            \mathbf{z}(t) &= \mathbf{z}(t-1)
                - \Delta t \;\hat{\sigma}(\mathbf{w}_{ch}) \circ
                \Bigl[\sigma\bigl(\mathbf{W}_{hh}\,\mathbf{h}(t-1)
                + \mathbf{W}_{ih}\,\mathbf{x}(t)
                + \mathbf{b}_{ih}\bigr)
                + \alpha\,\mathbf{h}(t-1)\Bigr],
        \end{aligned}

    where :math:`\Delta t` is the time step `dt`, and :math:`\alpha` is the
    leakage constant.

    Args:
        input_size (int): Number of features in the input :math:`\mathbf{x}(t)`.
        hidden_size (int): Number of features in the hidden state :math:`\mathbf{h}(t)`.
        bias (bool, optional): If ``False``, the cell does not use bias terms
            :math:`\mathbf{b}_{ih}` and :math:`\mathbf{b}_{hh}`. Default: ``True``.
        kernel_init (Callable, optional): Initializer for input-to-hidden weights
            :math:`\mathbf{W}_{ih}` (default: ``nn.init.xavier_uniform_``).
        recurrent_kernel_init (Callable, optional): Initializer for hidden-to-hidden
            weights :math:`\mathbf{W}_{hh}` (default: ``nn.init.xavier_uniform_``).
        control_kernel_init (Callable, optional): Initializer for control weights
            :math:`\mathbf{w}_{ch}` (default: ``nn.init.normal_``).
        bias_init (Callable, optional): Initializer for input bias
            :math:`\mathbf{b}_{ih}` (default: ``nn.init.zeros_``).
        recurrent_bias_init (Callable, optional): Initializer for hidden bias
            :math:`\mathbf{b}_{hh}` (default: ``nn.init.zeros_``).
        dt (float, optional): Time step :math:`\Delta t` between updates (default: 1.0).
        alpha (float, optional): Leakage coefficient in control update (default: 0.0).
        device (torch.device, optional): Device for parameters.
        dtype (torch.dtype, optional): Data type for parameters.

    Inputs:
        - **inp** (Tensor): Input at current time step,
            shape :math:`(N, input\_size)` or :math:`(input\_size)`.
        - **state** (Tuple[Tensor, Tensor], optional): Previous states
            (:math:`\mathbf{h}(t-1)`, :math:`\mathbf{z}(t-1)`), each of shape
            :math:`(N, hidden\_size)` or :math:`(hidden\_size)`. Defaults to zero.

    Outputs:
        - **new_state** (Tensor): Updated hidden state :math:`\mathbf{h}(t)`,
            shape :math:`(N, hidden\_size)` or :math:`(hidden\_size)`.
        - **new_cstate** (Tensor): Updated control state :math:`\mathbf{z}(t)`,
            shape :math:`(N, hidden\_size)` or :math:`(hidden\_size)`.

    Attributes:
        weight_ih (Tensor): Input-to-hidden weights :math:`\mathbf{W}_{ih}`,
            shape `(hidden_size, input_size)`.
        weight_hh (Tensor): Hidden-to-hidden weights :math:`\mathbf{W}_{hh}`,
            shape `(hidden_size, hidden_size)`.
        weight_ch (Tensor): Control weights :math:`\mathbf{w}_{ch}`,
            shape `(hidden_size,)`.
        bias_ih (Tensor): Input bias :math:`\mathbf{b}_{ih}`,
            shape `(hidden_size,)`.
        bias_hh (Tensor): Hidden bias :math:`\mathbf{b}_{hh}`,
            shape `(hidden_size,)`.

    Examples::
        >>> cell = UnICORNNCell(input_size=10, hidden_size=20)
        >>> seq = torch.randn(5, 3, 10)     # seq length 5, batch size 3
        >>> h = torch.zeros(3, 20)          # initial hidden state
        >>> z = torch.zeros(3, 20)          # initial control state
        >>> outputs = []
        >>> for t in range(5):
        ...     h, z = cell(seq[t], (h, z))
        ...     outputs.append((h, z))
    """

    weight_ih: Tensor
    weight_hh: Tensor
    weight_ch: Tensor
    bias_ih: Tensor
    bias_hh: Tensor

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        kernel_init=nn.init.xavier_uniform_,
        recurrent_kernel_init=nn.init.xavier_uniform_,
        control_kernel_init=nn.init.normal_,
        bias_init=nn.init.zeros_,
        recurrent_bias_init=nn.init.zeros_,
        dt: float = 1.0,
        alpha: float = 0.0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super(UnICORNNCell, self).__init__(
            input_size, hidden_size, bias, device=device, dtype=dtype
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
                "bias_hh": ((hidden_size,), bias),
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

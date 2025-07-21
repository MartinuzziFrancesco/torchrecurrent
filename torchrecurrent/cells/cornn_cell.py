import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Callable, Tuple, Union
from ..base import BaseDoubleRecurrentLayer, BaseDoubleRecurrentCell


class coRNN(BaseDoubleRecurrentLayer):
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
    r"""A Coupled Oscillatory RNN (coRNN) cell.

    Implements the dynamics from
    “Coupled Oscillatory Recurrent Neural Units for Long‐Term Memory”
    <https://arxiv.org/abs/2010.00951>_.

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

    where :math:`\Delta t` (dt) is the integration step size,
    :math:`\gamma` is a damping coefficient on the hidden state,
    and :math:`\epsilon` damps the cell state.

    Args:
        input_size (int):          Number of features in the input `inp`.
        hidden_size (int):         Number of features in the states.
        bias (bool, optional): If ``False``, disables :math:`\mathbf{b}_{ih}`.
            Default: ``True``.
        recurrent_bias (bool, optional): If ``False``, disables :math:`\mathbf{b}_{hh}`.
            Default: ``True``.
        cell_bias (bool, optional): If ``False``, disables :math:`\mathbf{b}_{ch}`.
            Default: ``True``.
        dt (float):                Integration step size Δt. Default: 1.0.
        gamma (float):             Damping on hidden state term. Default: 0.0.
        epsilon (float):           Damping on cell state term. Default: 0.0.
        kernel_init (Callable):    Initializer for input‐to‐hidden weights
                                    (default: `nn.init.xavier_uniform_`).
        recurrent_kernel_init (Callable):
                                    Initializer for hidden‐to‐hidden weights
                                    (default: `nn.init.xavier_uniform_`).
        cell_kernel_init (Callable):
                                    Initializer for cell‐to‐hidden weights
                                    (default: `nn.init.xavier_uniform_`).
        bias_init (Callable):      Initializer for input biases
                                    (default: `nn.init.zeros_`).
        recurrent_bias_init (Callable):
                                    Initializer for hidden biases
                                    (default: `nn.init.zeros_`).
        cell_bias_init (Callable):
                                    Initializer for cell biases
                                    (default: `nn.init.zeros_`).
        device (torch.device, optional): Device for parameters.
        dtype (torch.dtype, optional):   Data type for parameters.

    Inputs:
        - **inp** (Tensor): of shape `(batch, input_size)` or `(input_size,)`.
        - **state** (Tuple[Tensor, Tensor], optional):
            previous `(h, c)` where each is shape `(batch, hidden_size)` or
            `(hidden_size,)`.
            If not provided, both default to zeros.

    Outputs:
        - **new_h** (Tensor): next hidden state, same shape as `h`.
        - **new_c** (Tensor): next cell state, same shape as `c`.

    Attributes:
        weight_ih (Tensor):  Input‐to‐hidden weights, shape `(hidden_size, input_size)`.
        weight_hh (Tensor):  Hidden‐to‐hidden weights, shape `(hidden_size, hidden_size)`.
        weight_ch (Tensor):  Cell‐to‐hidden weights,  shape `(hidden_size, hidden_size)`.
        bias_ih   (Tensor):  Input bias, shape `(hidden_size,)`.
        bias_hh   (Tensor):  Hidden bias, shape `(hidden_size,)`.
        bias_ch   (Tensor):  Cell bias,   shape `(hidden_size,)`.

    .. note::
        The cell state and hidden state interact like a second‐order oscillator,
        allowing rich, stable dynamics over long time horizons.

    Examples::
        >>> cell = coRNNCell(10, 20, dt=0.5, gamma=0.1, epsilon=0.05)
        >>> x = torch.randn(4, 10)        # batch=4
        >>> h0 = torch.zeros(4, 20)
        >>> c0 = torch.zeros(4, 20)
        >>> h1, c1 = cell(x, (h0, c0))
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

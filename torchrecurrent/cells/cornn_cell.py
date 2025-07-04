import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Callable, Tuple
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
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        dt: float = 1.0,
        gamma: float = 0.0,
        epsilon: float = 0.0,
        kernel_init: Callable = nn.init.xavier_uniform_,
        recurrent_kernel_init: Callable = nn.init.xavier_uniform_,
        bias_init: Callable = nn.init.zeros_,
    ):
        super(coRNNCell, self).__init__(input_size, hidden_size, bias)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.dt = dt
        self.gamma = gamma
        self.epsilon = epsilon

        self.Wi = nn.Parameter(torch.empty(hidden_size, input_size))
        self.Wh = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.Wz = nn.Parameter(torch.empty(hidden_size, hidden_size))
        if self.bias:
            self.bias_ih = nn.Parameter(torch.empty(hidden_size))
        else:
            self.register_buffer("bias_ih", torch.zeros(hidden_size))

        # Initialize weights
        self.reset_parameters(kernel_init, recurrent_kernel_init, bias_init)

    def reset_parameters(
        self,
        kernel_init: Callable,
        recurrent_kernel_init: Callable,
        bias_init: Callable
    ):
        kernel_init(self.Wi)
        recurrent_kernel_init(self.Wh)
        recurrent_kernel_init(self.Wz)
        if self.bias is not None:
            bias_init(self.bias_ih)

    def forward(
        self, 
        inp: Tensor, 
        states: Optional[Tuple[Tensor, Tensor]] = None
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        self._validate_input(inp)
        self._validate_states(states)
        inp, state, c_state, is_batched = self._preprocess_states(inp, states)

        a = torch.matmul(inp, self.Wi.t())
        a = a + torch.matmul(state, self.Wh.t())
        a = a + torch.matmul(c_state, self.Wz.t()) + self.bias_ih

        act = torch.tanh(a)

        new_cstate = c_state + self.dt * act - self.dt * self.gamma * state - self.dt * self.epsilon * c_state
        new_state = state + self.dt * new_cstate

        if not is_batched:
            new_state = new_state.squeeze(0)
            new_cstate = new_cstate.squeeze(0)

        return new_state, (new_state, new_cstate)
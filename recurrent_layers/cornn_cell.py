import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Callable, Tuple
from .base import BaseRecurrentLayer

class coRNNCell(nn.Module):
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
        """
        coRNNCell implements a continuous-time recurrent cell where
        the new cell state (c_state) and hidden state (state) are updated as:
        
            new_cstate = c_state + dt * tanh(Wi * inp + Wh * state + Wz * c_state + bias)
                          - dt * gamma * state - dt * epsilon * c_state
            new_state  = state + dt * new_cstate

        Args:
            input_size (int): size of the input.
            hidden_size (int): size of the hidden state.
            bias (bool): whether to include a bias term.
            dt (float): time constant (dt).
            gamma (float): coefficient for the hidden state decay.
            epsilon (float): coefficient for the cell state decay.
            kernel_init (Callable): initializer for input kernel.
            recurrent_kernel_init (Callable): initializer for recurrent kernels.
            bias_init (Callable): initializer for the bias.
        """
        super(coRNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias_flag = bias
        self.dt = dt
        self.gamma = gamma
        self.epsilon = epsilon

        # Define parameters
        self.Wi = nn.Parameter(torch.empty(hidden_size, input_size))
        self.Wh = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.Wz = nn.Parameter(torch.empty(hidden_size, hidden_size))
        if self.bias_flag:
            self.bias = nn.Parameter(torch.empty(hidden_size))
        else:
            self.register_parameter('bias', None)

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
            bias_init(self.bias)

    def forward(
        self, 
        inp: Tensor, 
        state: Optional[Tuple[Tensor, Tensor]] = None
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        """
        Args:
            inp (Tensor): input tensor of shape (batch, input_size) or (input_size,)
            state (Optional[Tuple[Tensor, Tensor]]): a tuple (state, c_state) each of shape (batch, hidden_size)
                or (hidden_size,). If None, both are initialized as zeros.
                
        Returns:
            new_state (Tensor): updated hidden state.
            (new_state, new_cstate) (Tuple[Tensor, Tensor]): tuple containing the updated hidden state and cell state.
        """
        # Determine if batched
        is_batched = (inp.dim() == 2)
        if not is_batched:
            inp = inp.unsqueeze(0)

        batch_size = inp.size(0)
        device = inp.device
        dtype = inp.dtype

        # Initialize states if not provided
        if state is None:
            state = torch.zeros(batch_size, self.hidden_size, dtype=dtype, device=device)
            c_state = torch.zeros(batch_size, self.hidden_size, dtype=dtype, device=device)
        else:
            state, c_state = state
            if state.dim() == 1:
                state = state.unsqueeze(0)
            if c_state.dim() == 1:
                c_state = c_state.unsqueeze(0)

        # inp: (batch, input_size), Wi: (hidden_size, input_size) => result: (batch, hidden_size)
        a = torch.matmul(inp, self.Wi.t())
        # state: (batch, hidden_size), Wh: (hidden_size, hidden_size)
        a = a + torch.matmul(state, self.Wh.t())
        # c_state: (batch, hidden_size), Wz: (hidden_size, hidden_size)
        a = a + torch.matmul(c_state, self.Wz.t())
        if self.bias is not None:
            a = a + self.bias

        act = torch.tanh(a)

        # Update cell state
        new_cstate = c_state + self.dt * act - self.dt * self.gamma * state - self.dt * self.epsilon * c_state
        # Update hidden state
        new_state = state + self.dt * new_cstate

        if not is_batched:
            new_state = new_state.squeeze(0)
            new_cstate = new_cstate.squeeze(0)

        return new_state, (new_state, new_cstate)
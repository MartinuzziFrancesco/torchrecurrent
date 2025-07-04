import torch
from torch import Tensor
import torch.nn as nn
from typing import Callable, Optional, Tuple
from ..base import BaseSingleRecurrentLayer, BaseSingleRecurrentCell


class IndRNN(BaseSingleRecurrentLayer):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        batch_first: bool = False,
        **kwargs,
    ):
        super(IndRNN, self).__init__(
            input_size, hidden_size, num_layers, dropout, batch_first
        )
        self.initialize_cells(IndRNN, **kwargs)


class IndRNNCell(BaseSingleRecurrentCell):
    def __init__(self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        activation_fn: Callable = torch.tanh,
        kernel_init: Callable = nn.init.xavier_uniform_,
        recurrent_kernel_init: Callable = nn.init.normal_,
        bias_init: Callable = nn.init.zeros_):

        super(IndRNNCell, self).__init__(input_size, hidden_size, bias)
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.activation_fn = activation_fn
        self.kernel_init = kernel_init
        self.recurrent_kernel_init = recurrent_kernel_init
        self.bias_init = bias_init

        self.weight_ih = nn.Parameter(torch.empty(hidden_size, input_size))
        self.vector_u = nn.Parameter(torch.empty(hidden_size))
        if self.bias:
            self.bias_ih = nn.Parameter(torch.empty(hidden_size))
        else:
            self.register_buffer("bias_ih", torch.zeros(hidden_size))

        self.init_weights()

    def init_weights(self):
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                self.kernel_init(param)
            elif 'vector_u' in name:
                self.recurrent_kernel_init(param)
            elif 'bias_ih' in name:
                self.bias_init(param)


    def forward(self,
        inp: Tensor,
        state: Optional[Tensor] = None
    ) -> Tensor:
        self._validate_input(inp)
        self._validate_state(state)
        inp, state, is_batched = self._preprocess_input_and_state(inp, state)

        new_state = torch.matmul(inp, self.weight_ih.t()) + self.vector_u * state + self.bias_ih        
        new_state = self.activation_fn(new_state)

        if not is_batched:
            new_state = new_state.squeeze(0)

        return new_state


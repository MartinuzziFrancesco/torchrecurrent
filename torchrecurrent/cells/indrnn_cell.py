import torch
from torch import Tensor
import torch.nn as nn
from typing import Callable, Optional, Tuple
from ..base import BaseRecurrentLayer


class IndRNN(BaseRecurrentLayer):
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


class IndRNNCell(nn.Module):
    def __init__(self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        activation_fn: Callable = torch.tanh,
        kernel_init: Callable = nn.init.xavier_uniform_,
        recurrent_kernel_init: Callable = nn.init.normal_,
        bias_init: Callable = nn.init.zeros_):

        super(IndRNNCell, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.activation_fn = activation_fn
        self.kernel_init = kernel_init
        self.recurrent_kernel_init = recurrent_kernel_init
        self.bias_init = bias_init

        self.weight_ih = nn.Parameter(torch.randn(hidden_size, input_size))
        self.vector_u = nn.Parameter(torch.randn(hidden_size))
        self.bias = nn.Parameter(torch.randn(hidden_size)) if bias else None

        self.init_weights()

    def init_weights(self):
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                self.kernel_init(param)
            elif 'vector_u' in name:
                self.recurrent_kernel_init(param)
            elif 'bias' in name and self.bias_ih is not None:
                self.bias_init(param)


    def forward(self,
        inp: Tensor,
        state: Optional[Tensor] = None) -> Tensor:

        #chack batching
        is_batched = inp.dim() == 2
        if not is_batched:
            inp = inp.unsqueeze(0)
        
        #define hidden state
        if state is None:
            state = torch.zeros(
                inp.size(0), self.hidden_size, dtype = inp.dtype, device = inp.device)
        else:
            state = state.unsqueeze(0) if not is_batched else state

        #actual computation
        new_state = torch.matmul(inp, self.weight_ih.t()) + self.vector_u*state

        if self.bias is not None:
            new_state += self.bias
        
        new_state = self.activation_fn(new_state)

        #return properly batched state
        if not is_batched:
            new_state = new_state.squeeze(0)

        return new_state


import torch
from torch import Tensor
import torch.nn as nn
from typing import Callable, Optional

class IndRNNCell(nn.Module):
    def __init__(self,
        input_size,
        hidden_size,
        bias = True,
        activation = torch.tanh):

        super(IndRNNCell, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.activation = activation

        self.weight_ih = nn.Parameter(torch.randn(hidden_size, input_size))
        self.vector_u = nn.Parameter(torch.randn(hidden_size))
        self.bias = nn.Parameter(torch.randn(hidden_size)) if bias else None

    def forward(self, inp, state=None):

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
        
        new_state = self.activation(new_state)

        #return properly batched state
        if not is_batched:
            new_state = new_state.squeeze(0)

        return new_state


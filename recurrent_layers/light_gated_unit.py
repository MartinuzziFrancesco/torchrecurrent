import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Callable

class LGUCell(nn.Module):
    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            bias: bool = True,
            activation_fn: Callable = torch.relu,
            gate_activation_fn: Callable = torch.sigmoid
    ):
        super(LGUCell, self).__init__()
        #assign variables
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.activation_fn = activation_fn
        self.gate_activation_fn = gate_activation_fn
        # define weights
        self.weight_ih = nn.Parameter(torch.randn(2 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.randn(2 * hidden_size, hidden_size))
        # define biases
        self.bias_ih = nn.Parameter(torch.randn(2 * hidden_size)) if bias else None
        self.bias_hh = nn.Parameter(torch.randn(2 * hidden_size)) if bias else None

    def forward(self, inp:Tensor, state: Optional[Tensor] = None) -> Tensor:

        # check batching
        is_batched = inp.dim() == 2
        if not is_batched:
            inp = inp.unsqueeze(0)

        #handle hidden state initialization
        if state is None:
            state = torch.zeros(inp.size(0), self.hidden_size, dtype = inp.dtype, device = inp.device)
        else:
            state = state if is_batched else state.unsqueeze(0)

        # created gates
        gates = torch.matmul(inp, self.weight_ih.t()) + torch.matmul(state, self.weight_hh.t())
        if self.bias:
            gates += self.bias_ih + self.bias_hh 

        # split gates
        ug, cg = gates.chunk(2, 1)

        # actual equations
        update_gate = self.gate_activation_fn(ug)
        candidate_state = self.activation_fn(cg)
        new_state = (1 - update_gate) * candidate_state + update_gate * state

        # fix batching
        if not is_batched:
            new_state = new_state.squeeze(0)

        return new_state

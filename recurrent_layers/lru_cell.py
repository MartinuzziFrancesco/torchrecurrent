import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Callable, Tuple

class LRUCell(nn.Module):
    def __init__(self,
        input_size: int,
        hidden_size: int,
        bias: bool = True):

        super(LRUCell, self).__init__()
        self.hidden_size = hidden_size
        self.weight_ih = nn.Parameter(torch.rand(2 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.rand(hidden_size, hidden_size))
        self.bias = nn.Paramter(torch.rand(2 * hidden_size)) if bias else None

    def forward(self, inp: Tensor, state: Optional[Tensor] = None) -> Tensor: 
        is_batched = inp.dim() == 2
        if not is_batched:
            inp.unsqueeze(0)
        
        if not state:
            state = torch.zeros(
                inp.size(0), self.hidden_size, dtype = inp.dtype, device = inp.device
            )
        else:
            state = state if is_batched else state.unsqueeze(0)

        weight_ih_h, weight_ih_f = self.weight_ih.chunk(2, 0)
        if self.bias is not None:
            bias_h, bias_f = self.bias.chunk(2, 0)

        candidate_state = torch.tanh(torch.matmul(inp, weight_ih_h.t()))
        fg = torch.matmul(inp, weight_ih_f.t()) + torch.matmul(state, self.weight_hh)
        if self.bias is not None:
            fg += bias_h + bias_f
        
        forget_gate = torch.sigmoid(fg)
        new_state = (1-forget_gate) * state + forget_gate * candidate_state

        return new_state


import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Callable

class minGRUCell(nn.Module):
    def __init__(self,
        input_size: int,
        hidden_size: int,
        bias: bool = True):

        super(minGRUCell, self).__init__()
        self.hidden_size = hidden_size
        self.bias = bias
        self.weights = nn.Parameter(torch.randn(2 * hidden_size, input_size))
        self.b = nn.Parameter(torch.randn(2 * hidden_size)) if bias else None

    def forward(self, inp: Tensor, state: Optional[Tensor]) -> Tensor:
        #check batch
        is_batched = inp.dim() == 2
        if not is_batched:
            inp = inp.unsqueeze(0)

        if state is None:
            state = torch.zeros(inp.size(0), self.hidden_size, dtype=inp.dtype, device=inp.device)
        else:
            state = state if is_batched else state.unsqueeze(0)

        weight_z, weight_f = self.weights.chunk(2, 0)
        if self.bias is not None:
            bias_z, bias_h = self.b.chunk(2, 0)

        zg = torch.matmul(inp, weight_z.t())
        hg = torch.matmul(inp, weight_f.t())

        if self.bias:
            zg += bias_z
            hg += bias_h

        zg = torch.sigmoid(zg)
        hg = torch.tanh(hg)

        new_state = (1 - zg) * state + zg * hg

        if not is_batched:
            new_state = new_state.squeeze(0)

        return new_state




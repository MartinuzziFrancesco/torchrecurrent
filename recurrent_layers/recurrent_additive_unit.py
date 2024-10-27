import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Callable

class RANCell(nn.Module):
    def __init__(self,
        input_size,
        hidden_size,
        bias=True):

        super(RANCell, self).__init__()
        self.hidden_size = hidden_size
        self.bias = bias
        self.weights_ih = nn.Parameter(torch.randn(3 * hidden_size, input_size))
        self.weights_hh = nn.Parameter(torch.randn(3 * hidden_size, hidden_size))

        self.bias_ih = nn.Parameter(torch.randn(2 * hidden_size)) if bias else None
        self.bias_hh = nn.Parameter(torch.randn(2 * hidden_size)) if bias else None

    def forward(self, inp, state, c_state):
        is_batched = inp.dim() == 2
        if not is_batched:
            inp.unsqueeze(0)

        if state is None:
            state = torch.zeros(inp.size(0), self.hidden_size, dtype=inp.dtype, device=inp.device)
        else:
            state = state if is_batched else state.unsqueeze(0)

        weight_ih_c, weight_ih_i, weight_ih_f = self.weight_ih.chunk(2, 0)
        weight_hh_i, weight_hh_f = self.weight_hh.chunk(2, 0)
        if self.bias:
            bias_ih_i, bias_ih_f = self.bias_ih.chunk(2, 0)
            bias_hh_i, bias_hh_f = self.bias_hh.chunk(2, 0)

        content_layer = torch.matmul(inp, weight_ih_c.t())
        ig = torch.matmul(inp, weight_ih_i.t()) + torch.matmul(state, weight_hh_i.t())
        fg = torch.matmul(inp, weight_ih_f.t()) + torch.matmul(state, weight_hh_f.t())
        if self.bias:
            ig += bias_ih_i + bias_hh_i
            fg += bias_ih_f + bias_hh_f

        input_gate = nn.Sigmoid(ig)
        forget_gate = nn.Sigmoid(fg)

        new_cstate = input_gate * content_layer + forget_gate * c_state
        new_state = nn.Tanh(new_cstate)

        if not is_batched:
            new_state, new_cstate = new_state.unsqueeze(0), new_cstate.unsqueeze(0)

        return new_state, new_cstate




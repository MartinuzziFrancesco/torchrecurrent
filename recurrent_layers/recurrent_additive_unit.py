import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Callable, Tuple
from .base import BaseRecurrentLayer


class RAN(BaseRecurrentLayer):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        batch_first: bool = False,
        **kwargs,
    ):
        super(RAN, self).__init__(
            input_size, hidden_size, num_layers, dropout, batch_first
        )
        self.initialize_cells(RAN, **kwargs)


class RANCell(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        bias=True,
        kernel_init: Callable = nn.init.xavier_uniform_,
        recurrent_kernel_init: Callable = nn.init.xavier_uniform_,
        bias_init: Callable = nn.init.zeros_,
        recurrent_bias_init: Callable = nn.init.zeros_,
    ):

        super(RANCell, self).__init__()
        self.hidden_size = hidden_size
        self.bias = bias
        self.kernel_init = kernel_init
        self.recurrent_kernel_init = recurrent_kernel_init
        self.bias_init = bias_init
        self.recurrent_bias_init = recurrent_bias_init

        self.weight_ih = nn.Parameter(torch.randn(3 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.randn(2 * hidden_size, hidden_size))

        self.bias_ih = nn.Parameter(torch.randn(2 * hidden_size)) if bias else None
        self.bias_hh = nn.Parameter(torch.randn(2 * hidden_size)) if bias else None

        self.init_weights()

    def init_weights(self):
        for name, param in self.named_parameters():
            if "weight_ih" in name:
                self.kernel_init(param)
            elif "weight_hh" in name:
                self.recurrent_kernel_init(param)
            elif "bias_ih" in name and self.bias_ih is not None:
                self.bias_init(param)
            elif "bias_hh" in name and self.bias_ih is not None:
                self.recurrent_bias_init(param)

    def _init_state(self, inp):
        state = torch.zeros(
            inp.size(0), self.hidden_size, dtype=inp.dtype, device=inp.device
        )
        return state

    def forward(
        self, inp: Tensor, states: Optional[Tuple[Tensor, Tensor]] = (None, None)
    ) -> Tuple[Tensor, Tensor]:

        state, c_state = states

        is_batched = inp.dim() == 2
        if not is_batched:
            inp = inp.unsqueeze(0)

        if state is None:
            state = self._init_state(inp)
        else:
            state = state if is_batched else state.unsqueeze(0)

        if c_state is None:
            c_state = self._init_state(inp)
        else:
            c_state = c_state if is_batched else c_state.unsqueeze(0)

        weight_ih_c, weight_ih_i, weight_ih_f = self.weight_ih.chunk(3, 0)
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

        input_gate = torch.sigmoid(ig)
        forget_gate = torch.sigmoid(fg)

        new_cstate = input_gate * content_layer + forget_gate * c_state
        new_state = torch.tanh(new_cstate)

        if not is_batched:
            new_state, new_cstate = new_state.unsqueeze(0), new_cstate.unsqueeze(0)

        return new_state, new_cstate

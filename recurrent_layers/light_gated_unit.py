import torch
from torch import nn
from torch import Tensor
from typing import Optional, Callable, Tuple
from .base import BaseRecurrentLayer


class LiGRU(BaseRecurrentLayer):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        batch_first: bool = False,
        **kwargs,
    ):
        super(LiGRU, self).__init__(
            input_size, hidden_size, num_layers, dropout, batch_first
        )
        self.initialize_cells(LiGRU, **kwargs)


class LiGRUCell(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        activation_fn: Callable = torch.relu,
        gate_activation_fn: Callable = torch.sigmoid,
        kernel_init: Callable = nn.init.xavier_uniform_,
        recurrent_kernel_init: Callable = nn.init.xavier_uniform_,
        bias_init: Callable = nn.init.zeros_,
        recurrent_bias_init: Callable = nn.init.zeros_):

        super(LiGRUCell, self).__init__()
        #assign variables
        self.hidden_size = hidden_size
        self.activation_fn = activation_fn
        self.gate_activation_fn = gate_activation_fn
        self.kernel_init = kernel_init
        self.recurrent_kernel_init = recurrent_kernel_init
        self.bias_init = bias_init
        self.recurrent_bias_init = recurrent_bias_init

        # define weights
        self.weight_ih = nn.Parameter(torch.randn(2 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.randn(2 * hidden_size, hidden_size))
        # define biases
        self.bias_ih = nn.Parameter(torch.randn(2 * hidden_size)) if bias else None
        self.bias_hh = nn.Parameter(torch.randn(2 * hidden_size)) if bias else None

        self.init_weights()

    def init_weights(self):
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                self.kernel_init(param)
            elif 'weight_hh' in name:
                self.recurrent_kernel_init(param)
            elif 'bias_ih' in name and self.bias_ih is not None:
                self.bias_init(param)
            elif 'bias_hh' in name and self.bias_ih is not None:
                self.recurrent_bias_init(param)

    def forward(self,
        inp:Tensor,
        state: Optional[Tensor] = None) -> Tensor:

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
        if self.bias_ih is not None and self.bias_hh is not None:
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

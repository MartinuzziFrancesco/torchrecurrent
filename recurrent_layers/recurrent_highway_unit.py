#https://arxiv.org/pdf/1607.03474
#https://github.com/jzilly/RecurrentHighwayNetworks/blob/master/rhn.py#L138C1-L180C60
import torch
import torch.nn as nn
from typing import Optional, Callable, Tuple
from torch import Tensor

class RHNCellLayer(nn.Module):
    def __init__(self,
        input_size: int,
        hidden_size: int,
        bias: bool = True):
        super(RHNCellLayer, self).__init__()
        self.hidden_size = hidden_size
        self.bias = bias

        self.weight = nn.Parameter(torch.Tensor(3 * hidden_size, input_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(3 * hidden_size))
        else:
            self.register_parameter('bias', None)

        #init #TODO: generalize this
        nn.init.xavier_uniform_(self.weight)
        if bias:
            nn.init.zeros_(self.bias)

    def forward(self, input: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        #compute
        pre_nonlin = torch.matmul(input, self.weight.t())
        if self.bias is not None:
            pre_nonlin += self.bias
        #split
        pre_h, pre_t, pre_c = pre_nonlin.chunk(3, dim=1)
        return pre_h, pre_t, pre_c

class RHNCell(nn.Module):
    def __init__(self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        couple_carry: bool = True, #sec 5: setup, second line
        depth: int = 3):
        super(RHNCell, self).__init__()
        self.hidden_size = hidden_size
        self.depth = depth
        self.couple_carry = couple_carry

        self.layers = nn.ModuleList()
        for layer in range(depth):
            if layer == 0:
                real_insize = input_size + hidden_size
            else:
                real_insize = hidden_size

            layer_module = RHNCellLayer(real_insize, hidden_size, bias)
            self.layers.append(layer_module)

    def forward(self, inp: Tensor, state: Optional[Tensor] = None) -> Tensor:
        is_batched = inp.dim() == 2
        if not is_batched:
            inp = inp.unsqueeze(0)

        if state is None:
            state = torch.zeros(inp.size(0), self.hidden_size, dtype=inp.dtype, device=inp.device)
        else:
            state = state if is_batched else state.unsqueeze(0)

        current_state = state
        for layer in range(self.depth):
            if layer == 0:
                inp_combined = torch.cat([inp, current_state], dim=1)
            else:
                inp_combined = current_state

            pre_h, pre_t, pre_c = self.layers[layer](inp_combined)

            #apply nonlinearities
            hidden_gate = torch.tanh(pre_h)
            transform_gate = torch.sigmoid(pre_t)
            carry_gate = torch.sigmoid(pre_c)

            #highway component
            if self.couple_carry:
                current_state = (hidden_gate - current_state) * transform_gate + current_state
            else:
                current_state = hidden_gate * transform_gate + current_state * carry_gate

        if not is_batched:
            current_state = current_state.squeeze(0)

        return current_state


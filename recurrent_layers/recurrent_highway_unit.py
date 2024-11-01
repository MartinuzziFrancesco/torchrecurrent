#https://arxiv.org/pdf/1607.03474
#https://github.com/jzilly/RecurrentHighwayNetworks/blob/master/rhn.py#L138C1-L180C60
import torch
import torch.nn as nn
from typing import Optional, Callable, Tuple
from torch import Tensor

class RHN(nn.Module):
    def __init__(self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        batch_first: bool = False,
        **kwargs):
        super(RHN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.dropout = dropout

        layers = [RHNCell(input_size, hidden_size, **kwargs)] + [
            RHNCell(hidden_size, hidden_size, **kwargs) for _ in range(1, num_layers)
        ]
        self.layers = nn.ModuleList(layers)

        if self.dropout > 0:
            self.dropout_layer = nn.Dropout(dropout)
        else:
            self.dropout_layer = None

    def forward(self,
        inp: Tensor,
        state: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:

        if self.batch_first:
            inp = inp.transpose(0, 1)

        seq_len, batch_size, _ = inp.size()

        if state is None:
            state = torch.zeros(
                self.num_layers, batch_size, self.hidden_size,
                dtype = inp.dtype,
                device = inp.device
            )
        
        output = []

        for t in range(seq_len):
            input_t = inp[t]
            new_states = []
            for cell_idx, cell in enumerate(self.layers):
                new_state = cell(input_t, state[cell_idx])

                if self.dropout_layer and cell_idx < self.num_layers - 1:
                    new_state = self.dropout_layer(new_state)

                new_states += [new_state]
                input_t = new_state

            state = torch.stack(new_states, dim = 0)
            output += [input_t]

        output = torch.stack(output, dim = 0)

        if self.batch_first:
            output = output.transpose(0, 1)

        return output, state

class RHNCell(nn.Module):
    def __init__(self,
        input_size: int,
        hidden_size: int,
        depth: int = 3,
        couple_carry: bool = True, #sec 5: setup, second line
        **kwargs
        ):
        super(RHNCell, self).__init__()
        self.hidden_size = hidden_size
        self.depth = depth
        self.couple_carry = couple_carry

        units = [RHNCellUnit(input_size + hidden_size, hidden_size, **kwargs)] + [
            RHNCellUnit(hidden_size, hidden_size, **kwargs) for _ in range(1, depth)
        ]

        self.units = nn.ModuleList(units)

    def forward(self, inp: Tensor, state: Optional[Tensor] = None) -> Tensor:
        is_batched = inp.dim() == 2
        if not is_batched:
            inp = inp.unsqueeze(0)

        if state is None:
            state = torch.zeros(inp.size(0), self.hidden_size, dtype=inp.dtype, device=inp.device)
        else:
            state = state if is_batched else state.unsqueeze(0)

        current_state = state
        for unit in self.units:
            inp_combined = torch.cat([inp, current_state], dim=1) if unit == self.units[0] else current_state
            pre_h, pre_t, pre_c = unit(inp_combined)

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

class RHNCellUnit(nn.Module):
    def __init__(self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        init_weights_fn: Callable = nn.init.xavier_uniform_,
        init_bias: Callable = nn.init.zeros_):
        super(RHNCellUnit, self).__init__()
        self.hidden_size = hidden_size
        self.bias = bias
        self.init_weights_fn = init_weights_fn
        self.init_bias = init_bias

        self.weight = nn.Parameter(torch.randn(3 * hidden_size, input_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(3 * hidden_size))
        else:
            self.register_parameter('bias', None)

        self.init_weights()

    def init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                self.init_weights_fn(param)
            elif 'bias' in name and self.bias is not None:
                self.init_bias(param)

    def forward(self, input: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        #compute
        pre_nonlin = torch.matmul(input, self.weight.t())
        if self.bias is not None:
            pre_nonlin += self.bias
        #split
        pre_h, pre_t, pre_c = pre_nonlin.chunk(3, dim=1)
        return pre_h, pre_t, pre_c
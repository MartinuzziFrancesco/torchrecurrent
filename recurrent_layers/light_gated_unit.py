import torch
from torch import nn
from torch import Tensor
from typing import Optional, Callable, Tuple

class LiGRU(nn.Module):
    def __init__(self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        batch_first: bool = True,
        dropout: float = 0.0,
        **kwargs):

        super(LiGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first

        layers = [LiGRUCell(input_size, hidden_size, **kwargs)] + [
            LiGRUCell(hidden_size, hidden_size, **kwargs) for _ in range(1, num_layers)
        ]

        self.cells = nn.ModuleList(layers)

        if dropout > 0:
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

            for cell_idx, cell in enumerate(self.cells):
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



class LiGRUCell(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        activation_fn: Callable = torch.relu,
        gate_activation_fn: Callable = torch.sigmoid,
        init_input_weights: Callable = nn.init.xavier_uniform_,
        init_recurrent_weights: Callable = nn.init.xavier_uniform_,
        init_input_bias: Callable = nn.init.zeros_,
        init_recurrent_bias: Callable = nn.init.zeros_):

        super(LiGRUCell, self).__init__()
        #assign variables
        self.hidden_size = hidden_size
        self.activation_fn = activation_fn
        self.gate_activation_fn = gate_activation_fn
        self.init_input_weights = init_input_weights
        self.init_recurrent_weights = init_recurrent_weights
        self.init_input_bias = init_input_bias
        self.init_recurrent_bias = init_recurrent_bias

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
                self.init_input_weights(param)
            elif 'weight_hh' in name:
                self.init_recurrent_weights(param)
            elif 'bias_ih' in name and self.bias_ih is not None:
                self.init_input_bias(param)
            elif 'bias_hh' in name and self.bias_ih is not None:
                self.init_recurrent_bias(param)

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

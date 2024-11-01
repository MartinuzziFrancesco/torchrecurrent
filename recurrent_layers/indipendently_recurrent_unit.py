import torch
from torch import Tensor
import torch.nn as nn
from typing import Callable, Optional, Tuple


class IndRNN(nn.Module):
    def __init__(self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        batch_first: bool = False,
        dropout: float = 0.0,
        **kwargs):
        super(IndRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.dropout = dropout

        layers = [IndRNNCell(input_size, hidden_size, **kwargs)] + [
            IndRNNCell(hidden_size, hidden_size, **kwargs) for _ in range(1, num_layers)
        ]
        self.cells = nn.ModuleList(layers)

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
            hx = [torch.zeros(
                    batch_size, self.hidden_size,
                    dtype = inp.dtype,
                    device = inp.device
                    ) for _ in range(self.num_layers)
                ]
        else:
            hx = state

        output = []

        for t in range(seq_len):
            input_t = inp[t]
            for idx_cell, cell in enumerate(self.cells):
                state_new = cell(input_t, hx[idx_cell])

                if self.dropout_layer and idx_cell < self.num_layers - 1:
                    state_new = self.dropout_layer(state_new)

                hx[idx_cell] = state_new
                input_t = state_new

            output += [input_t]

        output = torch.stack(output, dim=0)

        if self.batch_first:
            output = output.transpose(0, 1)

        final_state = torch.stack(hx, dim = 0)

        return output, final_state


class IndRNNCell(nn.Module):
    def __init__(self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        activation: Callable = torch.tanh,
        init_input_weights: Callable = nn.init.xavier_uniform_,
        init_recurrent_weights: Callable = nn.init.normal_,
        init_input_bias: Callable = nn.init.zeros_):

        super(IndRNNCell, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.activation = activation
        self.init_input_weights = init_input_weights
        self.init_recurrent_weights = init_recurrent_weights
        self.init_input_bias = init_input_bias

        self.weight_ih = nn.Parameter(torch.randn(hidden_size, input_size))
        self.vector_u = nn.Parameter(torch.randn(hidden_size))
        self.bias = nn.Parameter(torch.randn(hidden_size)) if bias else None

        self.init_weights()

    def init_weights(self):
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                self.init_input_weights(param)
            elif 'vector_u' in name:
                self.init_recurrent_weights(param)
            elif 'bias' in name and self.bias_ih is not None:
                self.init_input_bias(param)


    def forward(self,
        inp: Tensor,
        state: Optional[Tensor] = None) -> Tensor:

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


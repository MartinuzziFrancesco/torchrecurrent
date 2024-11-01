import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Callable, Tuple

__all__ = ["MGU", "MGUCell"]


    
class MGU(nn.Module):
    def __init__(self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        batch_first: bool = False,
        **kwargs):

        super(MGU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.dropout = dropout

        layers = [MGUCell(input_size, hidden_size, **kwargs)] + [
            MGUCell(hidden_size, hidden_size, **kwargs) for _ in range(1, num_layers)
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
            state = torch.zeros(
                self.num_layers, batch_size, self.hidden_size,
                dtype = inp.dtype,
                device=inp.device
            )
        
        output = []

        for t in range(seq_len):
            input_t = inp[t]
            new_states = []
            for idx_cell, cell in enumerate(self.cells):
                new_state = cell(input_t, state[idx_cell])

                if self.dropout_layer and idx_cell < self.num_layers - 1:
                    new_state = self.dropout_layer(new_state)

                new_states += [new_state]
                input_t = new_state

            state = torch.stack(new_states, dim=0)
            output += [input_t]

        output = torch.stack(output, dim = 0)

        if self.batch_first:
            output = output.transpose(0, 1)

        return output, state



class MGUCell(nn.Module):
    def __init__(self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        activation_fn: Callable = torch.tanh,
        gate_activation_fn: Callable = torch.sigmoid,
        init_input_weights = nn.init.xavier_uniform_,
        init_recurrent_weights = nn.init.xavier_uniform_,
        init_input_bias = nn.init.zeros_,
        init_recurrent_bias = nn.init.zeros_):

        super(MGUCell, self).__init__()
        self.hidden_size = hidden_size
        self.activation_fn = activation_fn
        self.gate_activation_fn = gate_activation_fn
        self.init_input_weights = init_input_weights
        self.init_recurrent_weights = init_recurrent_weights
        self.init_input_bias = init_input_bias
        self.init_recurrent_bias = init_recurrent_bias
        self.bias = bias
        self.weight_ih = nn.Parameter(torch.randn(2 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.randn(2 * hidden_size, hidden_size))

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

    def forward(self, inp: Tensor, state: Optional[Tensor] = None) -> Tensor:
        # Check input dimensions
        if inp.dim() not in (1, 2):
            raise ValueError(f"MGUCell: Expected input to be 1D or 2D, got {inp.dim()}D instead")
        if state is not None and state.dim() not in (1, 2):
            raise ValueError(f"MGUCell: Expected hidden state to be 1D or 2D, got {state.dim()}D instead")
        
        # Check batching
        is_batched = inp.dim() == 2
        if not is_batched:
            inp = inp.unsqueeze(0)

        #Check and initialize hidden state
        if state is None:
            state = torch.zeros(
                inp.size(0), self.hidden_size, dtype = inp.dtype, device = inp.device
            )
        else:
            state = state.unsqueeze(0) if not is_batched else state
        
        weight_ih_f, weight_ih_h = self.weight_ih.chunk(2, 0)
        weight_hh_f, weight_hh_h = self.weight_hh.chunk(2, 0)

        if self.bias:
            bias_ih_f, bias_ih_h = self.bias_ih.chunk(2, 0)
            bias_hh_f, bias_hh_h = self.bias_hh.chunk(2, 0)
        
        fg = torch.mm(inp, weight_ih_f.t()) + torch.mm(state, weight_hh_f.t())
        if self.bias:
            fg += bias_ih_f + bias_hh_f

        forget_gate = self.gate_activation_fn(fg)

        hidden_modulated = forget_gate * state
        ch = torch.matmul(inp, weight_ih_h.t()) + torch.matmul(hidden_modulated, weight_hh_h.t())
        if self.bias:
            ch += bias_ih_h + bias_hh_h

        candidate_hidden = self.activation_fn(ch)
        
        new_state = forget_gate * candidate_hidden + (1 - forget_gate) * state
        if not is_batched:
            new_state = new_state.squeeze(0)
        
        return new_state

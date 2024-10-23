import torch
import torch.nn as nn
from torch.nn import Module
from torch import Tensor
from typing import Optional, Callable

__all__ = ["MGU", "MGUCell"]

class MGU(Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        batch_first: bool = False,
        dropout: float = 0.0,
        bidirectional: bool = False,
        input_weight_init_fn: Callable = nn.init.xavier_uniform_,
        recurrent_weight_init_fn: Callable = nn.init.xavier_uniform_,
        input_bias_init_fn: Callable = nn.init.zeros_,
        recurrent_bias_init_fn: Callable = nn.init.zeros_,
        **kwargs):

        super(MGU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.input_weight_init_fn = input_weight_init_fn
        self.recurrent_weight_init_fn = recurrent_weight_init_fn
        self.input_bias_init_fn = input_bias_init_fn
        self.recurrent_bias_init_fn = recurrent_bias_init_fn
        num_directions = 2 if bidirectional else 1

        # Initialize cells as a ModuleList
        self.cells = nn.ModuleList()
        # Create layers
        for layer in range(num_layers):
            for direction in range(num_directions):
                input_dim = input_size if layer == 0 else hidden_size * num_directions
                self.cells.append(MGUCell(
                    input_size=input_dim,
                    hidden_size=hidden_size,
                    **kwargs
                ))
            
        if self.dropout > 0:
            self.dropout_layer = nn.Dropout(dropout)
        else:
            self.dropout_layer = None
        
    def forward(self, inp, state=None):
        # Check batch first and get needed dimensions
        # (batch_size, seq_len, input_size) instead of (seq_len, batch_size, input_size)
        if self.batch_first:
            inp = inp.transpose(0, 1)
        
        seq_len, batch_size, _ = inp.size()
        num_directions = 2 if self.bidirectional else 1

        # Define hidden state if not provided
        if state is None:
            state = self._init_hidden(batch_size, num_directions)
        
        h = state
        layer_output = inp
        final_h = []

        for layer in range(self.num_layers):
            layer_input = layer_output
            layer_output = []

            for direction in range(num_directions):
                if direction == 0:
                    # Forward direction
                    idx = layer * num_directions + direction
                    cell = self.cells[idx]
                    h_prev = h[idx]
                    output_inner = []
                    for t in range(seq_len):
                        h_prev = cell(layer_input[t], h_prev)
                        output_inner.append(h_prev)
                    output_inner = torch.stack(output_inner)
                else:
                    # Backward direction
                    idx = layer * num_directions + direction
                    cell = self.cells[idx]
                    h_prev = h[idx]
                    output_inner = []
                    for t in reversed(range(seq_len)):
                        h_prev = cell(layer_input[t], h_prev)
                        output_inner.append(h_prev)
                    output_inner.reverse()
                    output_inner = torch.stack(output_inner)
                layer_output.append(output_inner)
                final_h.append(h_prev)
            if num_directions == 1:
                layer_output = layer_output[0]
            else:
                layer_output = torch.cat(layer_output, dim=2)
            if self.dropout_layer and layer < self.num_layers - 1:
                layer_output = self.dropout_layer(layer_output)
        outputs = layer_output
        if self.batch_first:
            outputs = outputs.transpose(0, 1)
        return outputs, final_h

    def _init_hidden(self, batch_size: int, num_directions: int):
        # Initialize the hidden state for all layers
        state = [torch.zeros(
            batch_size, self.hidden_size, dtype=self.cells[0].weight_ih.dtype, device=self.cells[0].weight_ih.device
        ) for _ in range(self.num_layers * num_directions)]
        
        return state

    
    def initialize_weights(self):
    # Initialize weights and biases for each MGUCell in the layers using user-defined functions
        for cell in self.cells:
            for name, param in cell.named_parameters():
                if 'weight_ih' in name or 'weight_hh' in name:
                    self.input_weight_init_fn(param) if 'weight_ih' in name else self.recurrent_weight_init_fn(param)
                elif 'bias_ih' in name or 'bias_hh' in name:
                    self.input_bias_init_fn(param) if 'bias_ih' in name else self.recurrent_bias_init_fn(param)

    


class MGUCell(nn.Module):
    def __init__(self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        activation_fn: Callable = torch.tanh,
        gate_activation_fn: Callable = torch.sigmoid):
        super(MGUCell, self).__init__()
        self.hidden_size = hidden_size
        self.activation_fn = activation_fn
        self.gate_activation_fn = gate_activation_fn
        self.bias = bias
        self.weight_ih = nn.Parameter(torch.randn(2 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.randn(2 * hidden_size, hidden_size))

        self.bias_ih = nn.Parameter(torch.randn(2 * hidden_size)) if bias else None
        self.bias_hh = nn.Parameter(torch.randn(2 * hidden_size)) if bias else None

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
        
        ret = forget_gate * candidate_hidden + (1 - forget_gate) * state
        if not is_batched:
            ret = ret.squeeze(0)
        
        return ret

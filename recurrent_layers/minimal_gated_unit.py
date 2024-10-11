import torch
import torch.nn as nn
from torch.nn import Module
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Callable

__all__ = ["MGU", "MGUCell"]

class MGU(Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        batch_first: bool = False,
        dropout: float = 0.0,
        bidirectional: bool = False,
        device=None,
        dtype=None,
        reset_gate: bool = True,
        activation_fn: Callable = torch.tanh,
        gate_activation_fn: Callable = torch.sigmoid,
        input_weight_init_fn: Callable = nn.init.xavier_uniform_,
        recurrent_weight_init_fn: Callable = nn.init.xavier_uniform_,
        input_bias_init_fn: Callable = nn.init.zeros_,
        recurrent_bias_init_fn: Callable = nn.init.zeros_):

        super(MGU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.device = device
        self.dtype = dtype
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
                    bias=bias,
                    reset_gate=reset_gate,
                    activation_fn=activation_fn,
                    gate_activation_fn=gate_activation_fn
                ))
            
        if self.dropout > 0:
            self.dropout_layer = nn.Dropout(dropout)
        else:
            self.dropout_layer = None
        
    def forward(self, input, hx=None):
        # Check batch first and get needed dimensions
        # (batch_size, seq_len, input_size) instead of (seq_len, batch_size, input_size)
        if self.batch_first:
            input = input.transpose(0, 1)
        
        seq_len, batch_size, _ = input.size()
        num_directions = 2 if self.bidirectional else 1

        # Define hidden state if not provided
        if hx is None:
            hx = self._init_hidden(batch_size, num_directions)
        
        h = hx
        outputs = []
        # Process sequentially
        for t in range(seq_len):
            input_t = input[t]
            new_h = []

            for layer_idx in range(self.num_layers):
                hidden_states = []

                # Forward direction
                forward_cell_idx = layer_idx * num_directions
                print("input_t", input_t.shape)
                print("h[layer_idx * num_directions]", h[layer_idx * num_directions].shape)
                h_new_forward = self.cells[forward_cell_idx](input_t, h[layer_idx * num_directions])
                hidden_states.append(h_new_forward)

                # Backward direction
                if self.bidirectional:
                    backward_cell_idx = layer_idx * num_directions + 1
                    h_new_backward = self.cells[backward_cell_idx](input[seq_len - t - 1], h[layer_idx * num_directions + 1])
                    hidden_states.append(h_new_backward)
                    # Concatenate forward and backward outputs if bidirectional
                    input_t = torch.cat((h_new_forward, h_new_backward), dim=1)
                else:
                    input_t = h_new_forward

                new_h.extend(hidden_states)

                if self.dropout_layer and layer_idx < self.num_layers - 1:
                    input_t = self.dropout_layer(input_t)

            h = new_h
            outputs.append(input_t)


        
        outputs = torch.stack(outputs)
        # If batch_first, transpose the output back to (batch_size, seq_len, hidden_size)
        if self.batch_first:
            outputs = outputs.transpose(0, 1)
        
        return outputs, h

    def _init_hidden(self, batch_size: int, num_directions: int):
        # Initialize the hidden state for all layers
        h = [torch.zeros(batch_size, self.hidden_size, dtype=self.cells[0].weight_ih.dtype, device=self.cells[0].weight_ih.device)
            for _ in range(self.num_layers * num_directions)]
        
        return h

    
    def initialize_weights(self):
    # Initialize weights and biases for each MGUCell in the layers using user-defined functions
        for cell in self.cells:
            for name, param in cell.named_parameters():
                if 'weight_ih' in name or 'weight_hh' in name:
                    self.input_weight_init_fn(param) if 'weight_ih' in name else self.recurrent_weight_init_fn(param)
                elif 'bias_ih' in name or 'bias_hh' in name:
                    self.input_bias_init_fn(param) if 'bias_ih' in name else self.recurrent_bias_init_fn(param)

    


class MGUCell(Module):
    def __init__(self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        reset_gate: bool = True,
        activation_fn: Callable = torch.tanh,
        gate_activation_fn: Callable = torch.sigmoid):
        super().__init__()
        self.hidden_size = hidden_size
        self.reset_gate = reset_gate
        self.activation_fn = activation_fn
        self.gate_activation_fn = gate_activation_fn
        self.bias = bias
        self.weight_ih = nn.Parameter(torch.randn(2 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.randn(2 * hidden_size, hidden_size))

        if bias:
            self.bias_ih = nn.Parameter(torch.randn(2 * hidden_size))
            self.bias_hh = nn.Parameter(torch.randn(2 * hidden_size))

    def forward(self, input: Tensor, hx: Optional[Tensor] = None) -> Tensor:
        # Check input dimensions
        if input.dim() not in (1, 2):
            raise ValueError(f"MGUCell: Expected input to be 1D or 2D, got {input.dim()}D instead")
        if hx is not None and hx.dim() not in (1, 2):
            raise ValueError(f"MGUCell: Expected hidden state to be 1D or 2D, got {hx.dim()}D instead")
        
        # Check batching
        is_batched = input.dim() == 2
        if not is_batched:
            input = input.unsqueeze(0)

        #Check and initialize hidden state
        if hx is None:
            hx = torch.zeros(
                input.size(0), self.hidden_size, dtype = input.dtype, device = input.device
            )
        else:
            hx = hx.unsqueeze(0) if not is_batched else hx
        
        weight_ih_f, weight_ih_h = self.weight_ih.chunk(2, 0)
        weight_hh_f, weight_hh_h = self.weight_hh.chunk(2, 0)

        if self.bias:
            bias_ih_f, bias_ih_h = self.bias_ih.chunk(2, 0)
            bias_hh_f, bias_hh_h = self.bias_hh.chunk(2, 0)
        else:
            bias_ih_f = bias_ih_h = bias_hh_f = bias_hh_h = 0
        
        forget_gate = self.gate_activation_fn(
            torch.mm(input, weight_ih_f.t())
            + bias_ih_f
            + torch.mm(hx, weight_hh_f.t())
            + bias_hh_f
        )

        if self.reset_gate:
            hidden_modulated = forget_gate * hx
            candidate_hidden = self.activation_fn(
                torch.mm(input, weight_ih_h.t())
                + bias_ih_h
                + torch.mm(hidden_modulated, weight_hh_h.t())
                + bias_hh_h
            )
        else:
            candidate_hidden = self.activation_fn(
                torch.mm(input, weight_ih_h.t())
                + bias_ih_h
                + torch.mm(hx, weight_hh_h.t())
                + bias_hh_h
            )
        
        ret = forget_gate * candidate_hidden + (1 - forget_gate) * hx
        if not is_batched:
            ret = ret.squeeze(0)
        
        return ret

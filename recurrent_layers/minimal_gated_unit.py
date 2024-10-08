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
        forget_weight_init_fn: Callable = nn.init.xavier_uniform_,
        candidate_weight_init_fn: Callable = nn.init.orthogonal_,
        bias_init_fn: Callable = nn.init.zeros_):
        super(MGU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.device = device
        self.dtype = dtype
        # Initialize cells as a ModuleList
        self.cells = nn.ModuleList()
        # Create layers
        for layer in range(num_layers):
            if layer == 0:
                self.cells.append(MGUCell(input_size,
                    hidden_size,
                    bias=bias,
                    activation_fn = activation_fn,
                    gate_activation_fn = gate_activation_fn,
                    reset_gate = reset_gate)
                )
            else:
                self.cells.append(MGUCell(hidden_size,
                    hidden_size,
                    bias=bias,
                    activation_fn = activation_fn,
                    gate_activation_fn = gate_activation_fn,
                    reset_gate = reset_gate)
                )
        # Store initialization functions
        self.forget_weight_init_fn = forget_weight_init_fn
        self.candidate_weight_init_fn = candidate_weight_init_fn
        self.bias_init_fn = bias_init_fn
    def forward(self, input, hx=None):
        # Check batch first and get needed dimensions
        # (batch_size, seq_len, input_size) instead of (seq_len, batch_size, input_size)
        if self.batch_first:
            input = input.transpose(0, 1)
        seq_len, batch_size, _ = input.size()
        # Define hidden state if not provided
        if hx is None:
            hx = self._init_hidden(batch_size)
        h = hx
        outputs = []
        # Process sequentially
        for t in range(seq_len):
            # Define input
            input_t = input[t]
            # Define container for hidden states
            new_h = []
            for layer_idx in range(self.num_layers):
                h_new = self.cells[layer_idx](input_t, h[layer_idx])
                input_t = h_new
                new_h.append(h_new)
            h = new_h  # Update hidden state for all layers
            outputs.append(input_t)
        outputs = torch.stack(outputs)
        # If batch_first, transpose the output back to (batch_size, seq_len, hidden_size)
        if self.batch_first:
            outputs = outputs.transpose(0, 1)
        return outputs, h
    def _init_hidden(self, batch_size: int):
        # Initialize the hidden state for all layers
        h = [torch.zeros(batch_size, self.hidden_size, dtype=self.cells[0].linear_f.weight.dtype, device=self.cells[0].linear_f.weight.device)
             for _ in range(self.num_layers)]
        return h
    def initialize_weights(self):
        # Initialize weights and biases for each MGUCell in the layers using user-defined functions
        for cell in self.cells:
            for name, param in cell.named_parameters():
                if 'linear_f' in name or 'linear_h' in name:
                    if 'linear_f' in name:
                        self.forget_weight_init_fn(param)
                    elif 'linear_h' in name:
                        self.candidate_weight_init_fn(param)
                elif 'bias' in name:
                    self.bias_init_fn(param)
    


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
        self.linear_f = nn.Linear(input_size + hidden_size, hidden_size, bias=bias)
        self.linear_h = nn.Linear(input_size + hidden_size, hidden_size, bias=bias)
        self.reset_gate = reset_gate
        self.activation_fn = activation_fn
        self.gate_activation_fn = gate_activation_fn
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
        combined = torch.cat((input, hx), dim=1)
        forget_gate = self.gate_activation_fn(self.linear_f(combined))
        if self.reset_gate:
            hidden_modulated = torch.cat((input, forget_gate * hx), dim=1)
            candidate_hidden = self.activation_fn(self.linear_h(hidden_modulated))
        else:
            candidate_hidden = self.activation_fn(self.linear_h(hx))
        ret = forget_gate * candidate_hidden + (1 - forget_gate) * hx
        if not is_batched:
            ret = ret.squeeze(0)
        return ret

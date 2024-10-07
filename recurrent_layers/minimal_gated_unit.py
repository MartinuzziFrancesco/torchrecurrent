import torch
import torch.nn as nn
from torch.nn import RNNBase
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Callable

__all__ = ["MGU", "MGUCell"]

class MGU(RNNBase):
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
        recurrent_weight_init_fn: Callable = nn.init.orthogonal_,
        input_bias_init_fn: Callable = nn.init.zeros_,
        recurrent_bias_init_fn: Callable = nn.init.zeros_,
        **kwargs):

        # Handle kwargs in RNNBase not needed here
        if 'proj_size' in kwargs:
            raise ValueError("proj_size argument is only supported for LSTM, not MGU")
                
        # Super init from RNNBase
        super(MGU, self).__init__(mode='MGU',
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=batch_first,
            dropout=dropout,
            bidirectional=bidirectional,
            proj_size=0,
            device=device,
            dtype=dtype)

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
        self.input_weight_init_fn = input_weight_init_fn
        self.recurrent_weight_init_fn = recurrent_weight_init_fn
        self.input_bias_init_fn = input_bias_init_fn
        self.recurrent_bias_init_fn = recurrent_bias_init_fn

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
        h = [torch.zeros(batch_size, self.hidden_size, dtype=self.cells[0].weight_ih.dtype, device=self.cells[0].weight_ih.device)
             for _ in range(self.num_layers)]
        return h

    def initialize_weights(self):
        # Initialize weights and biases for each MGUCell in the layers using user-defined functions
        for cell in self.cells:
            for name, param in cell.named_parameters():
                if 'weight_ih' in name:  # Input weight matrix
                    self.input_weight_init_fn(param)
                elif 'weight_hh' in name:  # Recurrent weight matrix
                    self.recurrent_weight_init_fn(param)
                elif 'bias_ih' in name:  # Input bias
                    self.input_bias_init_fn(param)
                elif 'bias_hh' in name:  # Recurrent bias
                    self.recurrent_bias_init_fn(param)
    


class MGUCell(nn.RNNCellBase):
    def __init__(self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        use_decomposition: bool = True,
        reset_gate: bool = True,
        activation_fn: Callable = torch.tanh,
        gate_activation_fn: Callable = torch.sigmoid):
        super().__init__(input_size, hidden_size, bias, num_chunks=2)
        self.use_decomposition = use_decomposition
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
                input.size(0), self.hidden_size, dtype=input.dtype, device=input.device
            )
        else:
            hx = hx.unsqueeze(0) if not is_batched else hx
        input, hx, is_batched = self._preprocess(input, hx)
        ret = mgu_cell(
            input,
            hx,
            self.weight_ih,
            self.weight_hh,
            self.bias_ih,
            self.bias_hh,
            self.reset_gate,
            self.activation_fn,
            self.gate_activation_fn
        )
        if not is_batched:
            ret = ret.squeeze(0)
        return ret

def mgu_cell(input: Tensor,
    hidden: Tensor,
    w_ih: Tensor,
    w_hh: Tensor,
    b_ih: Tensor,
    b_hh: Tensor,
    reset_gate: bool,
    activation_fn: Callable,
    gate_activation_fn: Callable) -> Tensor:
    # Perform input-to-hidden and hidden-to-hidden transformations and add biases
    gates = torch.mm(input, w_ih.t()) + torch.mm(hidden, w_hh.t()) + b_ih + b_hh
    # Split into forget gate and candidate hidden state
    forget_gate, candidate_h = gates.chunk(2, 1)
    # Forget gate (custom activation)
    forget_gate = gate_activation_fn(forget_gate)
    # Candidate hidden state (custom activation)
    if reset_gate:
        candidate_h = activation_fn(
            torch.mm(input, w_ih.t()) + b_ih + forget_gate * (torch.mm(hidden, w_hh.t()) + b_hh)
        )
    else:
        candidate_h = activation_fn(
            torch.mm(input, w_ih.t()) + b_ih + torch.mm(hidden, w_hh.t()) + b_hh
        )
    # New hidden state
    hy = forget_gate * candidate_h + (1 - forget_gate) * hidden
    return hy

#how do I include this in the call above?
def mgu_cell_decomposed(input: Tensor,
    hidden: Tensor,
    w_ih: Tensor,
    w_hh: Tensor,
    b_ih: Optional[Tensor],
    b_hh: Optional[Tensor]) -> Tensor:
    # Decomposition-based implementation using F.linear
    chunked_igates = F.linear(input, w_ih, b_ih).chunk(2, 1)
    chunked_hgates = F.linear(hidden, w_hh, b_hh).chunk(2, 1)
    forget_gate = torch.sigmoid(chunked_igates[0] + chunked_hgates[0])
    candidate_h = torch.tanh(chunked_igates[1] + chunked_hgates[1])
    return forget_gate * candidate_h + (1 - forget_gate) * hidden

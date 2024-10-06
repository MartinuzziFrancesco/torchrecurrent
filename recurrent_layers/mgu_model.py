import torch
import torch.nn as nn
from torch.nn import RNNBase

from mgu_cell import MGUCellBasic, MGUCellFull

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
        mgucell_type = 'basic',
        **kwargs):
 
        # handle kwargs in RNNBase not needed here
        if 'proj_size' in kwargs:
            raise ValueError("proj_size argument is only supported for LSTM, not MGU")
                
        # super init from RNNbase
        super(MGU, self).__init__(mode='MGU',
            input_size = input_size,
            hidden_size = hidden_size,
            num_layers = num_layers,
            bias = bias,
            batch_first = batch_first,
            dropout = dropout,
            bidirectional = bidirectional,
            proj_size = 0,
            device = device,
            dtype = dtype)

        if mgucell_type = 'basic':
            mgucell == MGUCellBasic
        else:
            mgucell == MGUCellFull

        # Initialize cells as a ModuleList
        self.cells = nn.ModuleList()
        # create layers
        for layer in range(num_layers):
            if layer == 0:
                self.cells.append(mgucell(input_size, hidden_size, bias=bias))
            else:
                self.cells.append(mgucell(hidden_size, hidden_size, bias=bias))

    def forward(self, input, hx=None):

        # check batch first and get need dimensions 
        # (batch_size, seq_len, input_size) instead of (seq_len, batch_size, input_size)
        if self.batch_first:
            input = input.transpose(0, 1)

        seq_len, batch_size, _ = input.size()

        # define hidden state if is not provided
        if hx is None:
            hx = self._init_hidden(batch_size)

        h = hx
        outputs = []

        # process sequentially
        for t in range(seq_len):
            # define input
            input_t = input[t]
            #define container for hidden states
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
    
    

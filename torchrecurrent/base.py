import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Tuple, List


class BaseRecurrentCell(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        **kwargs
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        # store any extra named params you want to display
        self._extra_args = kwargs

    def __repr__(self) -> str:
        classname = self.__class__.__name__
        # positional args
        args = [str(self.input_size), str(self.hidden_size)]
        # only show bias if it’s False (since True is default)
        if not self.bias:
            args.append(f"bias={self.bias}")
        # any other kwargs
        for k, v in sorted(self._extra_args.items()):
            args.append(f"{k}={v}")
        return f"{classname}({', '.join(args)})"


class BaseRecurrentLayer(nn.Module):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int = 1,
                 dropout: float = 0.0,
                 batch_first: bool = False):
        super(BaseRecurrentLayer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.dropout = dropout

        if self.dropout > 0:
            self.dropout_layer = nn.Dropout(dropout)
        else:
            self.dropout_layer = None

    def __repr__(self) -> str:
        classname = self.__class__.__name__
        args = [str(self.input_size), str(self.hidden_size)]

        # only show if not default
        if self.num_layers != 1:
            args.append(f"num_layers={self.num_layers}")
        if self.dropout != 0.0:
            args.append(f"dropout={self.dropout}")
        if self.batch_first:
            args.append(f"batch_first={self.batch_first}")

        return f"{classname}({', '.join(args)})"

    def initialize_cells(self, cell_class, **kwargs):
        """ Helper method to initialize cells for the derived recurrent layer class. """
        layers = [cell_class(self.input_size, self.hidden_size, **kwargs)] + [
            cell_class(self.hidden_size, self.hidden_size, **kwargs) for _ in range(1, self.num_layers)
        ]
        self.cells = nn.ModuleList(layers)

class BaseSingleRecurrentLayer(BaseRecurrentLayer):
    """For RNN‐style cells (one hidden state per layer)."""
    def forward(
        self,
        inp: Tensor,
        state: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        if self.batch_first:
            inp = inp.transpose(0, 1)

        seq_len, batch_size, _ = inp.size()
        # init single‐state
        if state is None:
            state = torch.zeros(
                self.num_layers, batch_size, self.hidden_size,
                dtype=inp.dtype, device=inp.device
            )

        outputs = []
        for t in range(seq_len):
            x = inp[t]
            new_states = []
            for layer_idx, cell in enumerate(self.cells):
                h_prev = state[layer_idx]
                h_new  = cell(x, h_prev)
                new_states.append(h_new)
                x = h_new
                if self.dropout_layer and layer_idx < self.num_layers - 1:
                    x = self.dropout_layer(x)

            state = torch.stack(new_states, dim=0)
            outputs.append(x)

        out = torch.stack(outputs, dim=0)
        if self.batch_first:
            out = out.transpose(0, 1)
        return out, state
    
class BaseDoubleRecurrentLayer(BaseRecurrentLayer):
    """For LSTM‐style cells (hidden *and* cell state per layer)."""
    def forward(
        self,
        inp: Tensor,
        state: Optional[Tuple[Tensor, Tensor]] = None
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        if self.batch_first:
            inp = inp.transpose(0, 1)

        seq_len, batch_size, _ = inp.size()
        # init double‐state
        if state is None:
            h = torch.zeros(self.num_layers, batch_size, self.hidden_size,
                            dtype=inp.dtype, device=inp.device)
            c = torch.zeros_like(h)
            state = (h, c)

        outputs = []
        for t in range(seq_len):
            x = inp[t]
            new_h, new_c = [], []
            h_prev, c_prev = state

            for layer_idx, cell in enumerate(self.cells):
                h_i, c_i = cell(x, (h_prev[layer_idx], c_prev[layer_idx]))
                new_h.append(h_i); new_c.append(c_i)
                x = h_i
                if self.dropout_layer and layer_idx < self.num_layers - 1:
                    x = self.dropout_layer(x)

            state = (torch.stack(new_h, dim=0), torch.stack(new_c, dim=0))
            outputs.append(x)

        out = torch.stack(outputs, dim=0)
        if self.batch_first:
            out = out.transpose(0, 1)
        return out, state
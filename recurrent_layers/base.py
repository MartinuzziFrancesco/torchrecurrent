import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Tuple, List

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

    def forward(self,
                inp: Tensor,
                state: Optional[Tuple[Tensor, ...]] = None) -> Tuple[Tensor, Tuple[Tensor, ...]]:
        if self.batch_first:
            inp = inp.transpose(0, 1)

        seq_len, batch_size, _ = inp.size()

        # Initialize hidden states (double state for LSTM-type cells)
        if state is None:
            if self.uses_double_state():
                h = torch.zeros(self.num_layers, batch_size, self.hidden_size, dtype=inp.dtype, device=inp.device)
                c = torch.zeros(self.num_layers, batch_size, self.hidden_size, dtype=inp.dtype, device=inp.device)
                state = (h, c)
            else:
                state = torch.zeros(self.num_layers, batch_size, self.hidden_size, dtype=inp.dtype, device=inp.device)

        output = []

        for t in range(seq_len):
            input_t = inp[t]
            new_states = []

            for idx, cell in enumerate(self.cells):
                if self.uses_double_state():
                    # Unpack hidden and cell states if using double states
                    h, c = state
                    new_h, new_c = cell(input_t, (h[idx], c[idx]))
                    new_states.append((new_h, new_c))
                    input_t = new_h
                else:
                    new_h = cell(input_t, state[idx])
                    new_states.append(new_h)
                    input_t = new_h

                # Dropout applied only between layers
                if self.dropout_layer and idx < self.num_layers - 1:
                    input_t = self.dropout_layer(input_t)

            # Update the state with new states for the next timestep
            if self.uses_double_state():
                h, c = zip(*new_states)
                state = (torch.stack(h, dim=0), torch.stack(c, dim=0))
            else:
                state = torch.stack(new_states, dim=0)

            output.append(input_t)

        output = torch.stack(output, dim=0)

        if self.batch_first:
            output = output.transpose(0, 1)

        return output, state

    def uses_double_state(self):
        """ Returns True if the derived class requires double state (like LSTM), False otherwise. """
        # By default, return False. Override in subclasses that require double state.
        return False

    def initialize_cells(self, cell_class, **kwargs):
        """ Helper method to initialize cells for the derived recurrent layer class. """
        layers = [cell_class(self.input_size, self.hidden_size, **kwargs)] + [
            cell_class(self.hidden_size, self.hidden_size, **kwargs) for _ in range(1, self.num_layers)
        ]
        self.cells = nn.ModuleList(layers)

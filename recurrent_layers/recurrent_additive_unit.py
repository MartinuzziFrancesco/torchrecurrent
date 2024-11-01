import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Callable, Tuple


class RAN(nn.Module):
    def __init__(self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        batch_first: bool = False,
        **kwargs
        ):
        super(RAN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.dropout = dropout

        layers = [RANCell(input_size, hidden_size, **kwargs)] + [
            RANCell(hidden_size, hidden_size, **kwargs) for _ in range(1, num_layers)
        ]
        self.cells = nn.ModuleList(layers)

        if self.dropout > 0:
            self.dropout_layer = nn.Dropout(dropout)
        else:
            self.dropout_layer = None

    def forward(self,
        inp: Tensor,
        states: Optional[Tuple[Tensor, Tensor]] = (None, None)) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:

        hx, cx = states

        if self.batch_first:
            inp = inp.transpose(0, 1)

        seq_len, batch_size, _ = inp.size()

        if hx is None:
            hx = [torch.zeros(batch_size, self.hidden_size, dtype=inp.dtype, device=inp.device) for _ in range(self.num_layers)]
        
        if cx is None:
            cx = [torch.zeros(batch_size, self.hidden_size, dtype=inp.dtype, device=inp.device) for _ in range(self.num_layers)]

        output = []

        for t in range(seq_len):
            input_t = inp[t]
            for idx_cell, cell in enumerate(self.cells):
                h_new, c_new = cell(input_t, (hx[idx_cell], cx[idx_cell]))

                if self.dropout_layer and idx_cell < self.num_layers-1:
                    h_new = self.dropout_layer(h_new)
                    
                hx[idx_cell], cx[idx_cell] = h_new, c_new
                input_t = h_new

            output.append(input_t)

        output = torch.stack(output, dim=0)

        if self.batch_first:
            output = output.transpose(0, 1)

        h_n = torch.stack(hx)
        c_n = torch.stack(cx)

        return output, (h_n, c_n)


class RANCell(nn.Module):
    def __init__(self,
        input_size,
        hidden_size,
        bias=True,
        init_input_weights: Callable = nn.init.xavier_uniform_,
        init_recurrent_weights: Callable = nn.init.xavier_uniform_,
        init_input_bias: Callable = nn.init.zeros_,
        init_recurrent_bias: Callable = nn.init.zeros_):

        super(RANCell, self).__init__()
        self.hidden_size = hidden_size
        self.bias = bias
        self.init_input_weights = init_input_weights
        self.init_recurrent_weights = init_recurrent_weights
        self.init_input_bias = init_input_bias
        self.init_recurrent_bias = init_recurrent_bias
        self.weight_ih = nn.Parameter(torch.randn(3 * hidden_size, input_size))
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

    def _init_state(self, inp):
        state = torch.zeros(
                inp.size(0), self.hidden_size, dtype=inp.dtype, device=inp.device
            )
        return state

    def forward(self,
        inp: Tensor,
        states: Optional[Tuple[Tensor, Tensor]]=(None, None)) -> Tuple[Tensor, Tensor]:

        state, c_state = states

        is_batched = inp.dim() == 2
        if not is_batched:
            inp = inp.unsqueeze(0)

        if state is None:
            state = self._init_state(inp)
        else:
            state = state if is_batched else state.unsqueeze(0)

        if c_state is None:
            c_state = self._init_state(inp)
        else:
            c_state = c_state if is_batched else c_state.unsqueeze(0)

        weight_ih_c, weight_ih_i, weight_ih_f = self.weight_ih.chunk(3, 0)
        weight_hh_i, weight_hh_f = self.weight_hh.chunk(2, 0)
        if self.bias:
            bias_ih_i, bias_ih_f = self.bias_ih.chunk(2, 0)
            bias_hh_i, bias_hh_f = self.bias_hh.chunk(2, 0)

        content_layer = torch.matmul(inp, weight_ih_c.t())
        ig = torch.matmul(inp, weight_ih_i.t()) + torch.matmul(state, weight_hh_i.t())
        fg = torch.matmul(inp, weight_ih_f.t()) + torch.matmul(state, weight_hh_f.t())
        if self.bias:
            ig += bias_ih_i + bias_hh_i
            fg += bias_ih_f + bias_hh_f

        input_gate = torch.sigmoid(ig)
        forget_gate = torch.sigmoid(fg)

        new_cstate = input_gate * content_layer + forget_gate * c_state
        new_state = torch.tanh(new_cstate)

        if not is_batched:
            new_state, new_cstate = new_state.unsqueeze(0), new_cstate.unsqueeze(0)

        return new_state, new_cstate




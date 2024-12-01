import torch
import torch.nn as nn
from typing import Optional, Callable
from .base import BaseRecurrentLayer


class PeepholeLSTM(BaseRecurrentLayer):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        batch_first: bool = False,
        **kwargs,
    ):
        super(PeepholeLSTM, self).__init__(
            input_size, hidden_size, num_layers, dropout, batch_first
        )
        self.initialize_cells(PeepholeLSTMCell, **kwargs)


class PeepholeLSTMCell(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        activation_fn: Callable = torch.tanh,
        gate_activation_fn: Callable = torch.sigmoid,
        kernel_init=nn.init.xavier_uniform_,
        recurrent_kernel_init=nn.init.xavier_uniform_,
        bias_init=nn.init.zeros_,
        recurrent_bias_init=nn.init.zeros_,
    ):
        super(PeepholeLSTMCell, self).__init__()
        self.hidden_size = hidden_size
        self.activation_fn = activation_fn
        self.gate_activation_fn = gate_activation_fn
        self.kernel_init = kernel_init
        self.recurrent_kernel_init = recurrent_kernel_init
        self.bias_init = bias_init
        self.recurrent_bias_init = recurrent_bias_init
        self.bias = bias

        # Input weights
        self.weight_ih = nn.Parameter(torch.randn(4 * hidden_size, input_size))
        # Recurrent weights
        self.weight_hh = nn.Parameter(torch.randn(4 * hidden_size, hidden_size))
        # Peephole connections
        self.weight_ph = nn.Parameter(torch.randn(3 * hidden_size))

        self.bias_ih = nn.Parameter(torch.randn(4 * hidden_size)) if bias else None
        self.bias_hh = nn.Parameter(torch.randn(4 * hidden_size)) if bias else None

        self.init_weights()

    def init_weights(self):
        for name, param in self.named_parameters():
            if "weight_ih" in name:
                self.kernel_init(param)
            elif "weight_hh" in name:
                self.recurrent_kernel_init(param)
            elif "bias_ih" in name and self.bias_ih is not None:
                self.bias_init(param)
            elif "bias_hh" in name and self.bias_hh is not None:
                self.recurrent_bias_init(param)
            elif "weight_ph" in name:
                self.recurrent_kernel_init(param)

    def forward(
        self, inp: torch.Tensor, states: Optional[tuple[torch.Tensor, torch.Tensor]] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Check input dimensions
        if inp.dim() not in (1, 2):
            raise ValueError(
                f"PeepholeLSTMCell: Expected input to be 1D or 2D, got {inp.dim()}D instead"
            )
        if states is not None:
            h, c = states
            if h.dim() not in (1, 2) or c.dim() not in (1, 2):
                raise ValueError(
                    f"PeepholeLSTMCell: Expected hidden and cell states to be 1D or 2D"
                )
        else:
            h, c = None, None

        # Check batching
        is_batched = inp.dim() == 2
        if not is_batched:
            inp = inp.unsqueeze(0)

        # Initialize hidden and cell states
        batch_size = inp.size(0)
        if h is None:
            h = torch.zeros(batch_size, self.hidden_size, device=inp.device, dtype=inp.dtype)
        if c is None:
            c = torch.zeros(batch_size, self.hidden_size, device=inp.device, dtype=inp.dtype)

        # Split weights for gates
        weight_ih_i, weight_ih_f, weight_ih_c, weight_ih_o = self.weight_ih.chunk(4, 0)
        weight_hh_i, weight_hh_f, weight_hh_c, weight_hh_o = self.weight_hh.chunk(4, 0)
        weight_ph_i, weight_ph_f, weight_ph_o = self.weight_ph.chunk(3, 0)

        if self.bias:
            bias_ih_i, bias_ih_f, bias_ih_c, bias_ih_o = self.bias_ih.chunk(4, 0)
            bias_hh_i, bias_hh_f, bias_hh_c, bias_hh_o = self.bias_hh.chunk(4, 0)

        # Input and forget gates with peephole connections
        i = (
            torch.mm(inp, weight_ih_i.t())
            + torch.mm(h, weight_hh_i.t())
            + c * weight_ph_i
        )
        if self.bias:
            i += bias_ih_i + bias_hh_i
        input_gate = self.gate_activation_fn(i)

        f = (
            torch.mm(inp, weight_ih_f.t())
            + torch.mm(h, weight_hh_f.t())
            + c * weight_ph_f
        )
        if self.bias:
            f += bias_ih_f + bias_hh_f
        forget_gate = self.gate_activation_fn(f)

        # Cell candidate
        c_hat = torch.mm(inp, weight_ih_c.t()) + torch.mm(h, weight_hh_c.t())
        if self.bias:
            c_hat += bias_ih_c + bias_hh_c
        cell_candidate = self.activation_fn(c_hat)

        # Update cell state
        new_c = forget_gate * c + input_gate * cell_candidate

        # Output gate with peephole connections
        o = (
            torch.mm(inp, weight_ih_o.t())
            + torch.mm(h, weight_hh_o.t())
            + new_c * weight_ph_o
        )
        if self.bias:
            o += bias_ih_o + bias_hh_o
        output_gate = self.gate_activation_fn(o)

        # Update hidden state
        new_h = output_gate * self.activation_fn(new_c)

        if not is_batched:
            new_h = new_h.squeeze(0)
            new_c = new_c.squeeze(0)

        return new_h, new_c

import torch
from torch import Tensor
import torch.nn as nn
from typing import Optional, Callable, Union, Tuple
from ..base import BaseDoubleRecurrentLayer, BaseDoubleRecurrentCell


class PeepholeLSTM(BaseDoubleRecurrentLayer):
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


class PeepholeLSTMCell(BaseDoubleRecurrentCell):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        activation_fn: Callable = torch.tanh,
        gate_activation_fn: Callable = torch.sigmoid,
        kernel_init=nn.init.xavier_uniform_,
        recurrent_kernel_init=nn.init.xavier_uniform_,
        peephole_kernel_init=nn.init.normal_,
        bias_init=nn.init.zeros_,
        recurrent_bias_init=nn.init.zeros_,
    ):
        super(PeepholeLSTMCell, self).__init__(input_size, hidden_size, bias)
        self.hidden_size = hidden_size
        self.activation_fn = activation_fn
        self.gate_activation_fn = gate_activation_fn
        self.kernel_init = kernel_init
        self.recurrent_kernel_init = recurrent_kernel_init
        self.peephole_kernel_init = peephole_kernel_init
        self.bias_init = bias_init
        self.recurrent_bias_init = recurrent_bias_init
        self.bias = bias

        # Input weights
        self.weight_ih = nn.Parameter(torch.empty(4 * hidden_size, input_size))
        # Recurrent weights
        self.weight_hh = nn.Parameter(torch.empty(4 * hidden_size, hidden_size))
        # Peephole connections
        self.weight_ph = nn.Parameter(torch.empty(3 * hidden_size))

        if self.bias:
            self.bias_ih = nn.Parameter(torch.empty(4 * hidden_size))
            self.bias_hh = nn.Parameter(torch.empty(4 * hidden_size))
        else:
            self.register_buffer("bias_ih", torch.zeros(4 * hidden_size))
            self.register_buffer("bias_hh", torch.zeros(4 * hidden_size))

        self.init_weights()

    def init_weights(self):
        for name, param in self.named_parameters():
            if "weight_ih" in name:
                self.kernel_init(param)
            elif "weight_hh" in name:
                self.recurrent_kernel_init(param)
            elif "weight_ph" in name:
                self.peephole_kernel_init(param)
            elif "bias_ih" in name and self.bias_ih is not None:
                self.bias_init(param)
            elif "bias_hh" in name and self.bias_hh is not None:
                self.recurrent_bias_init(param)

    def forward(
        self, inp: Tensor,
        state: Optional[Union[Tensor, Tuple[Tensor, ...]]] = None
    ) -> Tuple[Tensor, Tensor]:
        state, c_state = self._check_states(state)
        self._validate_input(inp)
        self._validate_states((state, c_state))
        inp, state, c_state, is_batched = self._preprocess_states(inp, (state, c_state))

        weight_ih_i, weight_ih_f, weight_ih_c, weight_ih_o = self.weight_ih.chunk(4, 0)
        weight_hh_i, weight_hh_f, weight_hh_c, weight_hh_o = self.weight_hh.chunk(4, 0)
        weight_ph_i, weight_ph_f, weight_ph_o = self.weight_ph.chunk(3, 0)
        bias_ih_i, bias_ih_f, bias_ih_c, bias_ih_o = self.bias_ih.chunk(4, 0)
        bias_hh_i, bias_hh_f, bias_hh_c, bias_hh_o = self.bias_hh.chunk(4, 0)

        i = inp @ weight_ih_i.t() + bias_ih_i + \
            state @ weight_hh_i.t() + c_state * weight_ph_i + bias_hh_i
        input_gate = self.gate_activation_fn(i)
        f = inp @ weight_ih_f.t() + bias_ih_f + \
            state @ weight_hh_f.t() + bias_hh_f + \
            c_state * weight_ph_f
        forget_gate = self.gate_activation_fn(f)
        c_hat = inp @ weight_ih_c.t() + bias_ih_c + \
            state @ weight_hh_c.t() + bias_hh_c
        cell_candidate = self.activation_fn(c_hat)
        new_c = forget_gate * c_state + input_gate * cell_candidate
        o = inp @ weight_ih_o.t() + bias_ih_o + \
            state @ weight_hh_o.t() + bias_hh_o\
            + new_c * weight_ph_o
        output_gate = self.gate_activation_fn(o)
        new_h = output_gate * self.activation_fn(new_c)

        if not is_batched:
            new_h = new_h.squeeze(0)
            new_c = new_c.squeeze(0)

        return new_h, new_c

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Callable, Tuple, Union
from ..base import BaseDoubleRecurrentLayer, BaseDoubleRecurrentCell


class RAN(BaseDoubleRecurrentLayer):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        batch_first: bool = False,
        **kwargs,
    ):
        super(RAN, self).__init__(
            input_size, hidden_size, num_layers, dropout, batch_first
        )
        self.initialize_cells(RANCell, **kwargs)


class RANCell(BaseDoubleRecurrentCell):
    def __init__(
        self,
        input_size,
        hidden_size,
        bias=True,
        kernel_init: Callable = nn.init.xavier_uniform_,
        recurrent_kernel_init: Callable = nn.init.xavier_uniform_,
        bias_init: Callable = nn.init.zeros_,
        recurrent_bias_init: Callable = nn.init.zeros_,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,

    ):
        super(RANCell, self).__init__(
            input_size, hidden_size, bias, device = device, dtype = dtype
        )
        self.kernel_init = kernel_init
        self.recurrent_kernel_init = recurrent_kernel_init
        self.bias_init = bias_init
        self.recurrent_bias_init = recurrent_bias_init
        fk = self._factory_kwargs

        self.weight_ih = nn.Parameter(torch.empty(3 * hidden_size, input_size, **fk))
        self.weight_hh = nn.Parameter(torch.empty(2 * hidden_size, hidden_size, **fk))

        if self.bias:
            self.bias_ih = nn.Parameter(torch.empty(2 * hidden_size, **fk))
            self.bias_hh = nn.Parameter(torch.empty(2 * hidden_size, **fk))
        else:
            self.register_buffer("bias_ih", torch.zeros(2 * hidden_size, **fk))
            self.register_buffer("bias_hh", torch.zeros(2 * hidden_size, **fk))

        self.init_weights()

    def init_weights(self):
        for name, param in self.named_parameters():
            if "weight_ih" in name:
                self.kernel_init(param)
            elif "weight_hh" in name:
                self.recurrent_kernel_init(param)
            elif "bias_ih" in name:
                self.bias_init(param)
            elif "bias_hh" in name:
                self.recurrent_bias_init(param)

    def forward(self,
        inp: Tensor,
        state: Optional[Union[Tensor, Tuple[Tensor, ...]]] = None
    ) -> Tuple[Tensor, Tensor]:
        state, c_state = self._check_states(state)
        self._validate_input(inp)
        self._validate_states((state, c_state))
        inp, state, c_state, is_batched = self._preprocess_states(inp, (state, c_state))

        weight_ih_c, weight_ih_i, weight_ih_f = self.weight_ih.chunk(3, 0)
        weight_hh_i, weight_hh_f = self.weight_hh.chunk(2, 0)
        bias_ih_i, bias_ih_f = self.bias_ih.chunk(2, 0)
        bias_hh_i, bias_hh_f = self.bias_hh.chunk(2, 0)

        content_layer = inp @ weight_ih_c.t()
        ig = inp @ weight_ih_i.t() + bias_ih_i + \
            state @ weight_hh_i.t() + bias_hh_i
        fg = inp @ weight_ih_f.t() + bias_ih_f + \
            state @ weight_hh_f.t() + bias_hh_f
        input_gate = torch.sigmoid(ig)
        forget_gate = torch.sigmoid(fg)
        new_cstate = input_gate * content_layer + forget_gate * c_state
        new_state = torch.tanh(new_cstate)

        if not is_batched:
            new_state  = new_state.squeeze(0)
            new_cstate = new_cstate.squeeze(0)

        return new_state, new_cstate

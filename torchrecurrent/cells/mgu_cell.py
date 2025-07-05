import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Callable, Tuple, Union
from ..base import BaseSingleRecurrentLayer, BaseSingleRecurrentCell

class MGU(BaseSingleRecurrentLayer):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        batch_first: bool = False,
        **kwargs,
    ):
        super(MGU, self).__init__(
            input_size, hidden_size, num_layers, dropout, batch_first
        )
        self.initialize_cells(MGUCell, **kwargs)


class MGUCell(BaseSingleRecurrentCell):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        activation_fn: Callable = torch.tanh,
        gate_activation_fn: Callable = torch.sigmoid,
        kernel_init = nn.init.xavier_uniform_,
        recurrent_kernel_init = nn.init.xavier_uniform_,
        bias_init = nn.init.zeros_,
        recurrent_bias_init = nn.init.zeros_,
    ):

        super(MGUCell, self).__init__(input_size, hidden_size, bias)
        self.hidden_size = hidden_size
        self.activation_fn = activation_fn
        self.gate_activation_fn = gate_activation_fn
        self.kernel_init = kernel_init
        self.recurrent_kernel_init = recurrent_kernel_init
        self.bias_init = bias_init
        self.recurrent_bias_init = recurrent_bias_init
        self.bias = bias

        self.weight_ih = nn.Parameter(torch.empty(2 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.empty(2 * hidden_size, hidden_size))
        if self.bias:
            self.bias_ih = nn.Parameter(torch.empty(2 * hidden_size))
            self.bias_hh = nn.Parameter(torch.empty(2 * hidden_size))
        else:
            self.register_buffer("bias_ih", torch.zeros(2 * hidden_size))
            self.register_buffer("bias_hh", torch.zeros(2 * hidden_size))

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

    def forward(self,
        inp: Tensor,
        state: Optional[Union[Tensor, Tuple[Tensor, ...]]] = None
    ) -> Tensor:
        state = self._check_state(state)
        self._validate_input(inp)
        self._validate_state(state)
        inp, state, is_batched = self._preprocess_input_and_state(inp, state)

        weight_ih_f, weight_ih_h = self.weight_ih.chunk(2, 0)
        weight_hh_f, weight_hh_h = self.weight_hh.chunk(2, 0)
        bias_ih_f, bias_ih_h = self.bias_ih.chunk(2, 0)
        bias_hh_f, bias_hh_h = self.bias_hh.chunk(2, 0)

        fg = inp @ weight_ih_f.t() + bias_ih_f + \
            state @ weight_hh_f.t() + bias_hh_f
        forget_gate = self.gate_activation_fn(fg)
        hidden_modulated = forget_gate * state
        ch = inp @ weight_ih_h.t() + bias_ih_h + \
            hidden_modulated @ weight_hh_h.t() + bias_hh_h
        candidate_hidden = self.activation_fn(ch)
        new_state = forget_gate * candidate_hidden + (1 - forget_gate) * state

        if not is_batched:
            new_state = new_state.squeeze(0)

        return new_state

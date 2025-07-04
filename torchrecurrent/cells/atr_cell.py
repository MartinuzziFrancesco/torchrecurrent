import torch
from torch import Tensor
import torch.nn as nn
from typing import Callable, Optional
from ..base import BaseSingleRecurrentLayer, BaseSingleRecurrentCell


class ATR(BaseSingleRecurrentLayer):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        batch_first: bool = False,
        **kwargs,
    ):
        super(ATR, self).__init__(
            input_size, hidden_size, num_layers, dropout, batch_first
        )
        self.initialize_cells(ATRCell, **kwargs)


class ATRCell(BaseSingleRecurrentCell):
    def __init__(self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        activation_fn: Callable = torch.tanh,
        kernel_init: Callable = nn.init.xavier_uniform_,
        recurrent_kernel_init: Callable = nn.init.normal_,
        bias_init: Callable = nn.init.zeros_,
        recurrent_bias_init: Callable = nn.init.zeros_):

        super(ATRCell, self).__init__(input_size, hidden_size, bias)
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.activation_fn = activation_fn
        self.kernel_init = kernel_init
        self.recurrent_kernel_init = recurrent_kernel_init
        self.bias_init = bias_init
        self.recurrent_bias_init = recurrent_bias_init

        self._create_weights(input_size, hidden_size, ih_mult=1, hh_mult=1, bias=bias)
        self.init_weights()

    def forward(self,
        inp: Tensor,
        state: Optional[Tensor] = None
    ) -> Tensor:
        self._validate_input(inp)
        self._validate_state(state)
        inp, state, is_batched = self._preprocess_input_and_state(inp, state)

        pt = inp @ self.weight_ih.t() + self.bias_ih
        qt = state @ self.weight_hh.t() + self.bias_hh
        it = torch.sigmoid(pt + qt)
        ft = torch.sigmoid(pt - qt)
        new_state = it * pt + ft * state

        if not is_batched:
            new_state = new_state.squeeze(0)

        return new_state
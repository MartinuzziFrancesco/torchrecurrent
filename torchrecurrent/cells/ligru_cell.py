import torch
from torch import nn
from torch import Tensor
from typing import Optional, Callable, Tuple, Union
from ..base import BaseSingleRecurrentLayer, BaseSingleRecurrentCell


class LiGRU(BaseSingleRecurrentLayer):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        batch_first: bool = False,
        **kwargs,
    ):
        super(LiGRU, self).__init__(
            input_size, hidden_size, num_layers, dropout, batch_first
        )
        self.initialize_cells(LiGRUCell, **kwargs)


class LiGRUCell(BaseSingleRecurrentCell):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        activation_fn: Callable = torch.relu,
        gate_activation_fn: Callable = torch.sigmoid,
        kernel_init: Callable = nn.init.xavier_uniform_,
        recurrent_kernel_init: Callable = nn.init.xavier_uniform_,
        bias_init: Callable = nn.init.zeros_,
        recurrent_bias_init: Callable = nn.init.zeros_,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super(LiGRUCell, self).__init__(
            input_size, hidden_size, bias, device = device, dtype = dtype
        )
        self.activation_fn = activation_fn
        self.gate_activation_fn = gate_activation_fn
        self.kernel_init = kernel_init
        self.recurrent_kernel_init = recurrent_kernel_init
        self.bias_init = bias_init
        self.recurrent_bias_init = recurrent_bias_init

        self._create_weights(input_size, hidden_size, ih_mult=2, hh_mult=2, bias=bias)
        self.init_weights()

    def init_weights(self):
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                self.kernel_init(param)
            elif 'weight_hh' in name:
                self.recurrent_kernel_init(param)
            elif 'bias_ih' in name and self.bias_ih is not None:
                self.bias_init(param)
            elif 'bias_hh' in name and self.bias_ih is not None:
                self.recurrent_bias_init(param)

    def forward(self,
        inp:Tensor,
        state: Optional[Union[Tensor, Tuple[Tensor, ...]]] = None
    ) -> Tensor:
        state = self._check_state(state)
        self._validate_input(inp)
        self._validate_state(state)
        inp, state, is_batched = self._preprocess_input_and_state(inp, state)

        gates = inp @ self.weight_ih.t() + self.bias_ih + \
            state @ self.weight_hh.t() + self.bias_hh
        ug, cg = gates.chunk(2, 1)

        update_gate = self.gate_activation_fn(ug)
        candidate_state = self.activation_fn(cg)
        new_state = (1 - update_gate) * candidate_state + update_gate * state

        if not is_batched:
            new_state = new_state.squeeze(0)

        return new_state

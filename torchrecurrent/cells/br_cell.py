import torch
from torch import nn
from torch import Tensor
from typing import Optional, Callable, Tuple, Union
from ..base import BaseSingleRecurrentLayer, BaseSingleRecurrentCell


class NBR(BaseSingleRecurrentLayer):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        batch_first: bool = False,
        **kwargs,
    ):
        super(NBR, self).__init__(
            input_size, hidden_size, num_layers, dropout, batch_first
        )
        self.initialize_cells(NBRCell, **kwargs)


class NBRCell(BaseSingleRecurrentCell):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        kernel_init: Callable = nn.init.xavier_uniform_,
        recurrent_kernel_init: Callable = nn.init.xavier_uniform_,
        bias_init: Callable = nn.init.zeros_,
        recurrent_bias_init: Callable = nn.init.zeros_,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super(NBRCell, self).__init__(
            input_size, hidden_size, bias, device = device, dtype = dtype
        )
        self.kernel_init = kernel_init
        self.recurrent_kernel_init = recurrent_kernel_init
        self.bias_init = bias_init
        self.recurrent_bias_init = recurrent_bias_init

        self._default_register_tensors(input_size, hidden_size, ih_mult=3, hh_mult=2, bias=bias)
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

        input_exp = inp @ self.weight_ih.t() + self.bias_ih
        input_exp_1, input_exp_2, input_exp_3 = input_exp.chunk(3, 1)
        rec_matrix_1, rec_matrix_2 = self.weight_hh.chunk(2, 0)
        t_ones = state.new_ones(self.hidden_size)
        h1 = input_exp_1 + state @ rec_matrix_1.t()
        h2 = input_exp_2 + state @ rec_matrix_2.t()
        modulation_gate = t_ones + torch.tanh(h1)
        candidate_state = torch.sigmoid(h2)
        h3 = input_exp_3 + modulation_gate * state

        new_state = candidate_state * state + (t_ones - candidate_state) * torch.tanh(h3)

        if not is_batched:
            new_state = new_state.squeeze(0)

        return new_state

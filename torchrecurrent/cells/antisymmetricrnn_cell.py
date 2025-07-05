import torch
from torch import Tensor
import torch.nn as nn
from typing import Callable, Optional, Union, Tuple
from ..base import BaseSingleRecurrentLayer, BaseSingleRecurrentCell


class AntisymmetricRNN(BaseSingleRecurrentLayer):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        batch_first: bool = False,
        **kwargs,
    ):
        super(AntisymmetricRNN, self).__init__(
            input_size, hidden_size, num_layers, dropout, batch_first
        )
        self.initialize_cells(AntisymmetricRNNCell, **kwargs)


class AntisymmetricRNNCell(BaseSingleRecurrentCell):
    def __init__(self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        activation_fn: Callable = torch.tanh,
        kernel_init: Callable = nn.init.xavier_uniform_,
        recurrent_kernel_init: Callable = nn.init.normal_,
        bias_init: Callable = nn.init.zeros_,
        recurrent_bias_init: Callable = nn.init.zeros_,
        epsilon:float = 1.0,
        gamma:float = 0.0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):

        super(AntisymmetricRNNCell, self).__init__(
            input_size, hidden_size, bias, device = device, dtype=dtype
        )
        self.activation_fn = activation_fn
        self.kernel_init = kernel_init
        self.recurrent_kernel_init = recurrent_kernel_init
        self.bias_init = bias_init
        self.recurrent_bias_init = recurrent_bias_init
        self.epsilon = epsilon
        self.gamma = gamma

        self._create_weights(input_size, hidden_size, ih_mult=1, hh_mult=1, bias=bias)
        self.init_weights()

    def forward(self,
        inp: Tensor,
        state: Optional[Union[Tensor, Tuple[Tensor, ...]]] = None
    ) -> Tensor:
        state = self._check_state(state)
        self._validate_input(inp)
        self._validate_state(state)
        inp, state, is_batched = self._preprocess_input_and_state(inp, state)

        recurrent_matrix = _compute_asym(self.weight_hh, self.gamma)
        pre_act = inp @ self.weight_ih.t() + self.bias_ih + \
            state @ recurrent_matrix.t() + self.bias_hh
        new_state = state + self.epsilon * self.activation_fn(pre_act)

        if not is_batched:
            new_state = new_state.squeeze(0)

        return new_state


class GatedAntisymmetricRNN(BaseSingleRecurrentLayer):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        batch_first: bool = False,
        **kwargs,
    ):
        super(GatedAntisymmetricRNN, self).__init__(
            input_size, hidden_size, num_layers, dropout, batch_first
        )
        self.initialize_cells(GatedAntisymmetricRNNCell, **kwargs)


class GatedAntisymmetricRNNCell(BaseSingleRecurrentCell):
    def __init__(self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        activation_fn: Callable = torch.tanh,
        kernel_init: Callable = nn.init.xavier_uniform_,
        recurrent_kernel_init: Callable = nn.init.normal_,
        bias_init: Callable = nn.init.zeros_,
        epsilon:float = 1.0,
        gamma:float = 0.0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):

        super(GatedAntisymmetricRNNCell, self).__init__(
            input_size, hidden_size, bias, device = device, dtype = dtype
        )
        self.activation_fn = activation_fn
        self.kernel_init = kernel_init
        self.recurrent_kernel_init = recurrent_kernel_init
        self.bias_init = bias_init
        self.epsilon = epsilon
        self.gamma = gamma

        self._create_weights(input_size, hidden_size, ih_mult=2, hh_mult=1, bias=bias)
        self.init_weights()

    def forward(self,
        inp: Tensor,
        state: Optional[Union[Tensor, Tuple[Tensor, ...]]] = None
    ) -> Tensor:
        state = self._check_state(state)
        self._validate_input(inp)
        self._validate_state(state)
        inp, state, is_batched = self._preprocess_input_and_state(inp, state)

        weights_ih = inp @ self.weight_ih.t() + self.bias_ih
        weight_ih_1, weight_ih_2 = weights_ih.chunk(2, 1)
        recurrent_matrix = _compute_asym(self.weight_hh, self.gamma)
        pre_act = weight_ih_2 + state @ recurrent_matrix.t() + self.bias_hh
        input_gate = torch.sigmoid(weight_ih_1 + state @ recurrent_matrix.t())
        new_state = state + self.epsilon * input_gate * torch.tanh(pre_act)

        if not is_batched:
            new_state = new_state.squeeze(0)

        return new_state


def _compute_asym(weight_hh: Tensor, gamma: float) -> Tensor:
    if weight_hh.dim() != 2 or weight_hh.size(0) != weight_hh.size(1):
        raise ValueError(f"weight_hh must be square, got shape {weight_hh.shape}")
    I = torch.eye(weight_hh.size(0), dtype=weight_hh.dtype, device=weight_hh.device)
    return weight_hh - weight_hh.t() - gamma * I

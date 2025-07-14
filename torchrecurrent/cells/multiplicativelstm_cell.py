import torch
from torch import Tensor
import torch.nn as nn
from typing import Optional, Callable, Union, Tuple
from ..base import BaseDoubleRecurrentLayer, BaseDoubleRecurrentCell


class MultiplicativeLSTM(BaseDoubleRecurrentLayer):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        batch_first: bool = False,
        **kwargs,
    ):
        super(MultiplicativeLSTM, self).__init__(
            input_size, hidden_size, num_layers, dropout, batch_first
        )
        self.initialize_cells(MultiplicativeLSTMCell, **kwargs)


class MultiplicativeLSTMCell(BaseDoubleRecurrentCell):
    weight_ih: Tensor
    weight_hh: Tensor
    weight_mh: Tensor
    bias_ih: Tensor
    bias_hh: Tensor
    bias_mh: Tensor

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        activation_fn: Callable = torch.tanh,
        gate_activation_fn: Callable = torch.sigmoid,
        kernel_init=nn.init.xavier_uniform_,
        recurrent_kernel_init=nn.init.xavier_uniform_,
        multiplicative_kernel_init=nn.init.normal_,
        bias_init=nn.init.zeros_,
        recurrent_bias_init=nn.init.zeros_,
        multiplicative_bias_init=nn.init.zeros_,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super(MultiplicativeLSTMCell, self).__init__(
            input_size, hidden_size, bias, device=device, dtype=dtype
        )
        self.activation_fn = activation_fn
        self.gate_activation_fn = gate_activation_fn
        self.kernel_init = kernel_init
        self.recurrent_kernel_init = recurrent_kernel_init
        self.multiplicative_kernel_init = multiplicative_kernel_init
        self.bias_init = bias_init
        self.recurrent_bias_init = recurrent_bias_init
        self.multiplicative_bias_init = multiplicative_bias_init

        self._register_tensors(
            {
                "weight_ih": ((5 * hidden_size, input_size), True),
                "weight_hh": ((hidden_size, hidden_size), True),
                "weight_mh": ((4 * hidden_size, hidden_size), True),
                "bias_ih": ((5 * hidden_size,), bias),
                "bias_hh": ((hidden_size,), bias),
                "bias_mh": ((4 * hidden_size,), bias),
            }
        )
        self.init_weights()

    def init_weights(self):
        for name, param in self.named_parameters():
            if "weight_ih" in name:
                self.kernel_init(param)
            elif "weight_hh" in name:
                self.recurrent_kernel_init(param)
            elif "weight_mh" in name:
                self.multiplicative_kernel_init(param)
            elif "bias_ih" in name:
                self.bias_init(param)
            elif "bias_hh" in name:
                self.recurrent_bias_init(param)
            elif "bias_mh" in name:
                self.multiplicative_bias_init(param)

    def forward(
        self, inp: Tensor, state: Optional[Union[Tensor, Tuple[Tensor, ...]]] = None
    ) -> Tuple[Tensor, Tensor]:
        state, c_state = self._check_states(state)
        self._validate_input(inp)
        self._validate_states((state, c_state))
        inp, state, c_state, is_batched = self._preprocess_states(inp, (state, c_state))

        inp_expanded = inp @ self.weight_ih.t() + self.bias_ih
        gxs1, gxs2, gxs3, gxs4, gxs5 = inp_expanded.chunk(5, 1)
        multiplicative_state = gxs1 * (state @ self.weight_hh.t() + self.bias_hh)
        mult_expanded = multiplicative_state @ self.weight_mh.t() + self.bias_mh
        gms1, gms2, gms3, gms4 = mult_expanded.chunk(4, 1)
        input_gate = torch.sigmoid(gxs2 + gms1)
        forget_gate = torch.sigmoid(gxs3 + gms2)
        candidate_state = torch.sigmoid(gxs4 + gms3)
        output_gate = torch.sigmoid(gxs5 + gms4)

        new_cstate = forget_gate * c_state + input_gate * candidate_state
        new_state = output_gate * torch.tanh(candidate_state)

        if not is_batched:
            new_state = new_state.squeeze(0)
            new_cstate = new_cstate.squeeze(0)

        return new_state, new_cstate

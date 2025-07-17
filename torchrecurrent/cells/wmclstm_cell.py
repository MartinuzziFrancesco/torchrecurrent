import torch
from torch import Tensor
import torch.nn as nn
from typing import Optional, Union, Tuple
from ..base import BaseDoubleRecurrentLayer, BaseDoubleRecurrentCell


class WMCLSTM(BaseDoubleRecurrentLayer):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        batch_first: bool = False,
        **kwargs,
    ):
        super(WMCLSTM, self).__init__(
            input_size, hidden_size, num_layers, dropout, batch_first
        )
        self.initialize_cells(WMCLSTMCell, **kwargs)


class WMCLSTMCell(BaseDoubleRecurrentCell):
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
        kernel_init=nn.init.xavier_uniform_,
        recurrent_kernel_init=nn.init.xavier_uniform_,
        memory_kernel_init=nn.init.xavier_uniform_,
        bias_init=nn.init.zeros_,
        recurrent_bias_init=nn.init.zeros_,
        memory_bias_init=nn.init.zeros_,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super(WMCLSTMCell, self).__init__(
            input_size, hidden_size, bias, device=device, dtype=dtype
        )
        self.kernel_init = kernel_init
        self.recurrent_kernel_init = recurrent_kernel_init
        self.memory_kernel_init = memory_kernel_init
        self.bias_init = bias_init
        self.recurrent_bias_init = recurrent_bias_init
        self.memory_bias_init = memory_bias_init

        self._register_tensors(
            {
                "weight_ih": ((4 * hidden_size, input_size), True),
                "weight_hh": ((4 * hidden_size, hidden_size), True),
                "weight_mh": ((3 * hidden_size, hidden_size), True),
                "bias_ih": ((4 * hidden_size,), bias),
                "bias_hh": ((4 * hidden_size,), bias),
                "bias_mh": ((3 * hidden_size,), bias),
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
                self.memory_kernel_init(param)
            elif "bias_ih" in name:
                self.bias_init(param)
            elif "bias_hh" in name:
                self.recurrent_bias_init(param)
            elif "bias_mh" in name:
                self.memory_bias_init(param)

    def forward(
        self, inp: Tensor, state: Optional[Union[Tensor, Tuple[Tensor, ...]]] = None
    ) -> Tuple[Tensor, Tensor]:
        state, c_state = self._check_states(state)
        self._validate_input(inp)
        self._validate_states((state, c_state))
        inp, state, c_state, is_batched = self._preprocess_states(inp, (state, c_state))

        gates = (
            inp @ self.weight_ih.t()
            + self.bias_ih
            + state @ self.weight_hh.t()
            + self.bias_hh
        )
        weight_mh_1, weight_mh_2, weight_mh_3 = self.weight_mh.chunk(3, 0)
        bias_mh_1, bias_mh_2, bias_mh_3 = self.bias_mh.chunk(3, 0)
        input_gate, forget_gate, cell_gate, output_gate = gates.chunk(4, 1)

        new_input_gate = torch.sigmoid(
            input_gate + torch.tanh(c_state @ weight_mh_1.t() + bias_mh_1)
        )
        new_forget_gate = torch.sigmoid(
            forget_gate + torch.tanh(c_state @ weight_mh_2.t() + bias_mh_2)
        )
        new_cell_gate = torch.tanh(cell_gate)
        new_cstate = new_forget_gate * c_state + new_input_gate * new_cell_gate
        memory_gate = new_cstate @ weight_mh_3.t() + bias_mh_3
        new_output_gate = torch.sigmoid(output_gate + torch.tanh(memory_gate))
        new_state = new_output_gate * torch.tanh(new_cstate)

        if not is_batched:
            new_state = new_state.squeeze(0)
            new_cstate = new_cstate.squeeze(0)

        return new_state, new_cstate

# https://arxiv.org/pdf/1607.03474
# https://github.com/jzilly/RecurrentHighwayNetworks/blob/master/rhn.py#L138C1-L180C60
import torch
import torch.nn as nn
from typing import Optional, Callable, Tuple
from torch import Tensor
from ..base import BaseRecurrentLayer


class RHN(BaseRecurrentLayer):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        batch_first: bool = False,
        **kwargs,
    ):
        super(RHN, self).__init__(input_size, hidden_size, num_layers, dropout, batch_first)
        self.initialize_cells(RHNCell, **kwargs)


class RHNCell(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        depth: int = 3,
        couple_carry: bool = True,  # sec 5: setup, second line
        **kwargs,
    ):
        super(RHNCell, self).__init__()
        self.hidden_size = hidden_size
        self.depth = depth
        self.couple_carry = couple_carry

        units = [RHNCellUnit(input_size + hidden_size, hidden_size, **kwargs)] + [
            RHNCellUnit(hidden_size, hidden_size, **kwargs) for _ in range(1, depth)
        ]

        self.units = nn.ModuleList(units)

    def forward(self, inp: Tensor, state: Optional[Tensor] = None) -> Tensor:
        is_batched = inp.dim() == 2
        if not is_batched:
            inp = inp.unsqueeze(0)

        if state is None:
            state = torch.zeros(
                inp.size(0), self.hidden_size, dtype=inp.dtype, device=inp.device
            )
        else:
            state = state if is_batched else state.unsqueeze(0)

        current_state = state
        for unit in self.units:
            inp_combined = (
                torch.cat([inp, current_state], dim=1)
                if unit == self.units[0]
                else current_state
            )
            pre_h, pre_t, pre_c = unit(inp_combined)

            # apply nonlinearities
            hidden_gate = torch.tanh(pre_h)
            transform_gate = torch.sigmoid(pre_t)
            carry_gate = torch.sigmoid(pre_c)

            # highway component
            if self.couple_carry:
                current_state = (
                    hidden_gate - current_state
                ) * transform_gate + current_state
            else:
                current_state = hidden_gate * transform_gate + current_state * carry_gate

        if not is_batched:
            current_state = current_state.squeeze(0)

        return current_state


class RHNCellUnit(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        kernel_init: Callable = nn.init.xavier_uniform_,
        bias_init: Callable = nn.init.zeros_,
    ):
        super(RHNCellUnit, self).__init__()
        self.hidden_size = hidden_size
        self.bias = bias
        self.kernel_init = kernel_init
        self.bias_init = bias_init

        self.weight = nn.Parameter(torch.randn(3 * hidden_size, input_size))
        if bias:
            self.bias = nn.Parameter(torch.randn(3 * hidden_size))
        else:
            self.register_parameter("bias", None)

        self.init_weights()

    def init_weights(self):
        for name, param in self.named_parameters():
            if "weight" in name:
                self.init_weights_fn(param)
            elif "bias" in name and self.bias is not None:
                self.init_bias(param)

    def forward(self, input: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        # compute
        pre_nonlin = torch.matmul(input, self.weight.t())
        if self.bias is not None:
            pre_nonlin += self.bias
        # split
        pre_h, pre_t, pre_c = pre_nonlin.chunk(3, dim=1)
        return pre_h, pre_t, pre_c

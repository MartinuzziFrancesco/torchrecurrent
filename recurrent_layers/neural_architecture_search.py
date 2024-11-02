# This file is a reimplementation in PyTorch of the NASCell as described in:
# "Neural Architecture Search with Reinforcement Learning" (https://arxiv.org/pdf/1611.01578).
# The original implementation in TensorFlow can be found here:
# https://www.tensorflow.org/addons/api_docs/python/tfa/rnn/NASCell
# No changes were made that alter the behavior of the cell compared to the original
# implementation; differences may be due to library-specific syntax.
#
# The original implementation is licensed under the Apache License, Version 2.0.
# This reimplementation is also licensed under the Apache License, Version 2.0.

#
# Copyright 2024 Francesco Martinuzzi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
from typing import Optional, Tuple, List, Callable
from torch import Tensor
from .base import BaseRecurrentLayer


class NAS(BaseRecurrentLayer):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        batch_first: bool = False,
        **kwargs,
    ):
        super(NAS, self).__init__(
            input_size, hidden_size, num_layers, dropout, batch_first
        )
        self.initialize_cells(NAS, **kwargs)


class NASCell(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        kernel_init: Callable = nn.init.xavier_uniform_,
        recurrent_kernel_init: Callable = nn.init.xavier_uniform_,
        bias_init: Callable = nn.init.zeros_,
        recurrent_bias_init: Callable = nn.init.zeros_,
    ):

        super(NASCell, self).__init__()
        self.hidden_size = hidden_size
        self.kernel_init = kernel_init
        self.recurrent_kernel_init = recurrent_kernel_init
        self.bias_init = bias_init
        self.recurrent_bias_init = recurrent_bias_init

        self.weight_ih = nn.Parameter(torch.randn(8 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.randn(8 * hidden_size, hidden_size))
        self.bias_ih = nn.Parameter(torch.randn(8 * hidden_size)) if bias else None
        self.bias_hh = nn.Parameter(torch.randn(8 * hidden_size)) if bias else None

        self.init_weights()

    def init_weights(self):
        for name, param in self.named_parameters():
            if "weight_ih" in name:
                self.kernel_init(param)
            elif "weight_hh" in name:
                self.recurrent_kernel_init(param)
            elif "bias_ih" in name and self.bias_ih is not None:
                self.bias_init(param)
            elif "bias_hh" in name and self.bias_ih is not None:
                self.recurrent_bias_init(param)

    def _init_state(self, inp):
        state = torch.zeros(
            inp.size(0), self.hidden_size, dtype=inp.dtype, device=inp.device
        )
        return state

    def forward(
        self, inp: Tensor, states: Optional[Tuple[Tensor, Tensor]] = (None, None)
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:

        state, c_state = states
        is_batched = inp.dim() == 2
        if not is_batched:
            inp = inp.unsqueeze(0)

        if state is None:
            state = self._init_state(inp)
        else:
            state = state if is_batched else state.unsqueeze(0)

        if c_state is None:
            c_state = self._init_state(inp)
        else:
            c_state = c_state if is_batched else c_state.unsqueeze(0)

        gates = (
            torch.matmul(inp, self.weight_ih.t())
            + self.bias_ih
            + torch.matmul(state, self.weight_hh.t())
            + self.bias_hh
        )

        g0, g1, g2, g3, g4, g5, g6, g7 = gates.chunk(8, 1)

        layer1_0 = torch.sigmoid(g0)
        layer1_1 = torch.relu(g1)
        layer1_2 = torch.sigmoid(g2)
        layer1_3 = torch.relu(g3)
        layer1_4 = torch.tanh(g4)
        layer1_5 = torch.sigmoid(g5)
        layer1_6 = torch.tanh(g6)
        layer1_7 = torch.sigmoid(g7)

        l2_0 = torch.tanh(layer1_0 * layer1_1)
        l2_1 = torch.tanh(layer1_2 + layer1_3)
        l2_2 = torch.tanh(layer1_4 * layer1_5)
        l2_3 = torch.sigmoid(layer1_6 + layer1_7)

        l2_0 = torch.tanh(l2_0 + c_state)

        new_cstate = l2_0 * l2_1
        l3_1 = torch.tanh(l2_2 + l2_3)

        new_state = torch.tanh(new_cstate * l3_1)

        if not is_batched:
            new_state = new_state.squeeze(0)
            new_cstate = new_cstate.squeeze(0)

        return new_state, new_cstate

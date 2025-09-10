# This file is a reimplementation in PyTorch of the NASCell as described in:
# "Neural Architecture Search with Reinforcement Learning".
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
from typing import Optional, Tuple, Union, Callable
from torch import Tensor
from ..base import BaseDoubleRecurrentLayer, BaseDoubleRecurrentCell


class NAS(BaseDoubleRecurrentLayer):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        batch_first: bool = False,
        **kwargs,
    ):
        super(NAS, self).__init__(input_size, hidden_size, num_layers, dropout, batch_first)
        self.initialize_cells(NASCell, **kwargs)


class NASCell(BaseDoubleRecurrentCell):
    r"""A Neural Architecture Search (NAS) cell.

    [`arXiv <https://arxiv.org/abs/1611.01578>`_]

        .. math::

            \mathbf{g}(t)
                &= \mathbf{W}_{ih}\,\mathbf{x}(t) + \mathbf{b}_{ih}
                + \mathbf{W}_{hh}\,\mathbf{h}(t-1) + \mathbf{b}_{hh}, \\[6pt]
            [g_0,\dots,g_7] &= \mathrm{chunk}_8\bigl(\mathbf{g}(t)\bigr), \\[6pt]
            o_k(t) &=
                \begin{cases}
                    \sigma(g_k), & k\in\{0,2,5,7\},\\
                    \mathrm{ReLU}(g_k), & k\in\{1,3\},\\
                    \tanh(g_k), & k\in\{4,6\},
                \end{cases} \\[6pt]
            \ell_1(t) &= \tanh\bigl(o_0\,\circ\,o_1\bigr),\quad
            \ell_2(t) = \tanh\bigl(o_2 + o_3\bigr), \\[3pt]
            \ell_3(t) &= \tanh\bigl(o_4\,\circ\,o_5\bigr),\quad
            \ell_4(t) = \sigma\bigl(o_6 + o_7\bigr), \\[6pt]
            \tilde{c}(t) &= \tanh\bigl(\ell_1 + c(t-1)\bigr),\quad
            c(t) = \ell_1\,\circ\,\ell_2, \\[3pt]
            \ell_5(t) &= \tanh\bigl(\ell_3 + \ell_4\bigr),\quad
            h(t) = \tanh\bigl(c(t)\,\circ\,\ell_5(t)\bigr).

    Args:
        input_size: The number of expected features in the input ``x``
        hidden_size: The number of features in the hidden and cell states
        bias: If ``False``, the layer does not use input-side bias ``b_{ih}``.
            Default: ``True``
        recurrent_bias: If ``False``, the layer does not use recurrent bias ``b_{hh}``.
            Default: ``True``
        kernel_init: Initializer for ``W_{ih}``.
            Default: :func:`torch.nn.init.xavier_uniform_`
        recurrent_kernel_init: Initializer for ``W_{hh}``.
            Default: :func:`torch.nn.init.xavier_uniform_`
        bias_init: Initializer for ``b_{ih}`` when ``bias=True``.
            Default: :func:`torch.nn.init.zeros_`
        recurrent_bias_init: Initializer for ``b_{hh}`` when ``recurrent_bias=True``.
            Default: :func:`torch.nn.init.zeros_`
        device: The desired device of parameters.
        dtype: The desired floating point type of parameters.

    Inputs: input, (h_0, c_0)
        - **input** of shape ``(batch, input_size)`` or ``(input_size,)``:
          tensor containing input features
        - **h_0** of shape ``(batch, hidden_size)`` or ``(hidden_size,)``:
          initial hidden state
        - **c_0** of shape ``(batch, hidden_size)`` or ``(hidden_size,)``:
          initial cell state

        If **(h_0, c_0)** is not provided, both default to zero.

    Outputs: (h_1, c_1)
        - **h_1** of shape ``(batch, hidden_size)`` or ``(hidden_size,)``:
          next hidden state
        - **c_1** of shape ``(batch, hidden_size)`` or ``(hidden_size,)``:
          next cell state

    Variables:
        weight_ih: the learnable input–hidden weights,
            of shape ``(8*hidden_size, input_size)``
        weight_hh: the learnable hidden–hidden weights,
            of shape ``(8*hidden_size, hidden_size)``
        bias_ih: the learnable input–hidden biases,
            of shape ``(8*hidden_size)``
        bias_hh: the learnable hidden–hidden biases,
            of shape ``(8*hidden_size)``

    Examples::

        >>> cell = NASCell(10, 20)
        >>> x = torch.randn(5, 3, 10)   # (time_steps, batch, input_size)
        >>> h = torch.zeros(3, 20)
        >>> c = torch.zeros(3, 20)
        >>> out_h = []
        >>> for t in range(x.size(0)):
        ...     h, c = cell(x[t], (h, c))
        ...     out_h.append(h)
        >>> out_h = torch.stack(out_h, dim=0)  # (time_steps, batch, hidden_size)
    """

    __constants__ = [
        "input_size",
        "hidden_size",
        "bias",
        "recurrent_bias",
        "kernel_init",
        "recurrent_kernel_init",
        "bias_init",
        "recurrent_bias_init",
    ]

    weight_ih: Tensor
    weight_hh: Tensor
    bias_ih: Tensor
    bias_hh: Tensor

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        recurrent_bias: bool = True,
        kernel_init: Callable = nn.init.xavier_uniform_,
        recurrent_kernel_init: Callable = nn.init.xavier_uniform_,
        bias_init: Callable = nn.init.zeros_,
        recurrent_bias_init: Callable = nn.init.zeros_,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super(NASCell, self).__init__(
            input_size, hidden_size, bias, recurrent_bias, device=device, dtype=dtype
        )
        self.kernel_init = kernel_init
        self.recurrent_kernel_init = recurrent_kernel_init
        self.bias_init = bias_init
        self.recurrent_bias_init = recurrent_bias_init

        self._default_register_tensors(
            input_size,
            hidden_size,
            ih_mult=8,
            hh_mult=8,
            bias=bias,
            recurrent_bias=recurrent_bias,
        )
        self.init_weights()

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

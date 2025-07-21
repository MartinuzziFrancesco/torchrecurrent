import torch
from torch import Tensor
import torch.nn as nn
from typing import Optional, Callable, Union, Tuple
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
    r"""A long short-term memory cell with working memory connections (WMCLSTMCell).

    Based on arXiv:2109.00020.

    The cell update equations are:

    .. math::
        \begin{aligned}
            \mathbf{i}(t) &= \sigma\bigl(
                \mathbf{W}_{ih}^{i}\,\mathbf{x}(t) + \mathbf{b}_{ih}^{i} +
                \mathbf{W}_{hh}^{i}\,\mathbf{h}(t-1) + \mathbf{b}_{hh}^{i} +
                \mathbf{W}_{mh}^{i}\,\mathbf{c}(t-1) + \mathbf{b}_{mh}^{i}
            \bigr), \\
            \mathbf{f}(t) &= \sigma\bigl(
                \mathbf{W}_{ih}^{f}\,\mathbf{x}(t) + \mathbf{b}_{ih}^{f} +
                \mathbf{W}_{hh}^{f}\,\mathbf{h}(t-1) + \mathbf{b}_{hh}^{f} +
                \mathbf{W}_{mh}^{f}\,\mathbf{c}(t-1) + \mathbf{b}_{mh}^{f}
            \bigr), \\
            \mathbf{c}(t) &= \mathbf{f}(t) \circ \mathbf{c}(t-1)
                \;+\;\mathbf{i}(t)\circ\sigma_c\bigl(\mathbf{W}_{ih}^{c}\,\mathbf{x}(t)
                + \mathbf{b}_{ih}^{c}\bigr), \\
            \mathbf{o}(t) &= \sigma\bigl(
                \mathbf{W}_{ih}^{o}\,\mathbf{x}(t) + \mathbf{b}_{ih}^{o} +
                \mathbf{W}_{hh}^{o}\,\mathbf{h}(t-1) + \mathbf{b}_{hh}^{o} +
                \mathbf{W}_{mh}^{o}\,\mathbf{c}(t) + \mathbf{b}_{mh}^{o}
            \bigr), \\
            \mathbf{h}(t) &= \mathbf{o}(t)\circ\sigma_h\bigl(\mathbf{c}(t)\bigr),
        \end{aligned}

    where :math:`\sigma` is the sigmoid function, :math:`\sigma_c` and
    :math:`\sigma_h` are cell and output activations (here both are `torch.tanh`),
    and :math:`\circ` denotes elementwise multiplication.

    Args:
        input_size (int):   Number of features in the input :math:`\mathbf{x}(t)`.
        hidden_size (int):  Number of features in the hidden state :math:`\mathbf{h}(t)`.
        bias (bool, optional): If ``False``, disables :math:`\mathbf{b}_{ih}`.
            Default: ``True``.
        recurrent_bias (bool, optional): If ``False``, disables :math:`\mathbf{b}_{hh}`.
            Default: ``True``.
        memory_bias (bool, optional): If ``False``, disables :math:`\mathbf{b}_{mh}`.
            Default: ``True``.
        kernel_init (Callable, optional): Initializer for input-to-hidden weights
            :math:`\mathbf{W}_{ih}^{\{i,f,c,o\}}`. Default: ``nn.init.xavier_uniform_``.
        recurrent_kernel_init (Callable, optional):
            Initializer for hidden-to-hidden weights
            :math:`\mathbf{W}_{hh}^{\{i,f,o\}}`. Default: ``nn.init.xavier_uniform_``.
        memory_kernel_init (Callable, optional): Initializer for working-memory weights
            :math:`\mathbf{W}_{mh}^{\{i,f,o\}}`. Default: ``nn.init.xavier_uniform_``.
        bias_init (Callable, optional): Initializer for input biases
            :math:`\mathbf{b}_{ih}^{\{i,f,c,o\}}`. Default: ``nn.init.zeros_``.
        recurrent_bias_init (Callable, optional): Initializer for hidden biases
            :math:`\mathbf{b}_{hh}^{\{i,f,o\}}`. Default: ``nn.init.zeros_``.
        memory_bias_init (Callable, optional): Initializer for memory biases
            :math:`\mathbf{b}_{mh}^{\{i,f,o\}}`. Default: ``nn.init.zeros_``.
        device (torch.device, optional): Device for parameters.
        dtype (torch.dtype, optional): Data type for parameters.

    Inputs:
        - **inp** (Tensor): Current input :math:`\mathbf{x}(t)`,
            shape :math:`(N, input\_size)` or :math:`(input\_size)`.
        - **state** (Tuple[Tensor, Tensor], optional): Previous hidden and cell states
            (:math:`\mathbf{h}(t-1)`, :math:`\mathbf{c}(t-1)`), each of shape
            :math:`(N, hidden\_size)` or :math:`(hidden\_size)`. Defaults to zeros.

    Outputs:
        - **new_state** (Tensor): Updated hidden state :math:`\mathbf{h}(t)`,
            shape :math:`(N, hidden\_size)` or :math:`(hidden\_size)`.
        - **new_cstate** (Tensor): Updated cell state :math:`\mathbf{c}(t)`,
            shape :math:`(N, hidden\_size)` or :math:`(hidden\_size)`.

    Attributes:
        weight_ih (Tensor): Input-to-hidden weights, shape `(4*hidden_size, input_size)`,
            split into i, f, c, o gates along dim=0.
        weight_hh (Tensor): Hidden-to-hidden weights, shape `(4*hidden_size, hidden_size)`,
            split into i, f, c, o gates.
        weight_mh (Tensor): Working-memory weights, shape `(3*hidden_size, hidden_size)`,
            split into i, f, o contributions.
        bias_ih (Tensor): Input biases, shape `(4*hidden_size,)`.
        bias_hh (Tensor): Hidden biases, shape `(4*hidden_size,)`.
        bias_mh (Tensor): Memory biases, shape `(3*hidden_size,)`.

    Examples::
        >>> cell = WMCLSTMCell(input_size=8, hidden_size=16)
        >>> seq = torch.randn(12, 4, 8)    # seq length 12, batch size 4
        >>> h = torch.zeros(4, 16)         # initial hidden state
        >>> c = torch.zeros(4, 16)         # initial cell state
        >>> outputs = []
        >>> for t in range(12):
        ...     h, c = cell(seq[t], (h, c))
        ...     outputs.append(h)
    """

    __constants__ = [
        "input_size",
        "hidden_size",
        "bias",
        "recurrent_bias",
        "memory_bias",
        "kernel_init",
        "recurrent_kernel_init",
        "memory_kernel_init",
        "bias_init",
        "recurrent_bias_init",
        "memory_bias_init",
    ]

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
        recurrent_bias: bool = True,
        memory_bias: bool = True,
        kernel_init: Callable = nn.init.xavier_uniform_,
        recurrent_kernel_init: Callable = nn.init.xavier_uniform_,
        memory_kernel_init: Callable = nn.init.xavier_uniform_,
        bias_init: Callable = nn.init.zeros_,
        recurrent_bias_init: Callable = nn.init.zeros_,
        memory_bias_init: Callable = nn.init.zeros_,
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
                "bias_hh": ((4 * hidden_size,), recurrent_bias),
                "bias_mh": ((3 * hidden_size,), memory_bias),
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

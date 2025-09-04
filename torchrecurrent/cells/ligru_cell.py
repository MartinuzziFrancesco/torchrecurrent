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
    r"""A Light Gated Recurrent Unit (LiGRU) cell.

    This variant simplifies the GRU by using a single update gate and a
    rectified‐linear candidate, updating via:

    .. math::

        \mathbf{z}(t) &= \sigma\bigl(
            \mathbf{W}_{ih}^{z}\,\mathbf{x}(t)
            + \mathbf{b}_{ih}^{z}
            + \mathbf{W}_{hh}^{z}\,\mathbf{h}(t-1)
            + \mathbf{b}_{hh}^{z}
        \bigr), \\[6pt]
        \tilde{\mathbf{h}}(t) &= \mathrm{ReLU}\bigl(
            \mathbf{W}_{ih}^{h}\,\mathbf{x}(t)
            + \mathbf{b}_{ih}^{h}
            + \mathbf{W}_{hh}^{h}\,\mathbf{h}(t-1)
            + \mathbf{b}_{hh}^{h}
        \bigr), \\[6pt]
        \mathbf{h}(t) &= \mathbf{z}(t)\,\circ\,\mathbf{h}(t-1)
            \;+\;\bigl(1 - \mathbf{z}(t)\bigr)\,\circ\,\tilde{\mathbf{h}}(t),

    where :math:`\circ` denotes element‐wise multiplication.

    See: [“Light Gated Recurrent Unit: A Simplified GRU”](https://arxiv.org/pdf/1803.10225).

    Args:
        input_size (int):  Dimensionality of input vector :math:`\mathbf{x}(t)`.
        hidden_size (int): Number of features in hidden state :math:`\mathbf{h}(t)`.
        bias (bool, optional): If ``False``, disables :math:`\mathbf{b}_{ih}`.
            Default: ``True``.
        recurrent_bias (bool, optional): If ``False``, disables :math:`\mathbf{b}_{hh}`.
            Default: ``True``.
        nonlinearity (Callable, optional): Activation for the candidate
            :math:`\tilde{\mathbf{h}}` (default `torch.relu`).
        gate_nonlinearity (Callable, optional): Activation for the update gate
            :math:`\mathbf{z}` (default `torch.sigmoid`).
        kernel_init (Callable, optional): Initializer for input‐to‐hidden weights
            :math:`\mathbf{W}_{ih}`. Default: ``nn.init.xavier_uniform_``.
        recurrent_kernel_init (Callable, optional): Initializer for hidden‐to‐hidden
            weights :math:`\mathbf{W}_{hh}`. Default: ``nn.init.xavier_uniform_``.
        bias_init (Callable, optional): Initializer for input biases
            :math:`\mathbf{b}_{ih}` when `bias=True`. Default: ``nn.init.zeros_``.
        recurrent_bias_init (Callable, optional): Initializer for hidden biases
            :math:`\mathbf{b}_{hh}` when `bias=True`. Default: ``nn.init.zeros_``.
        device (torch.device, optional): Device of the parameters. Default: CPU.
        dtype (torch.dtype, optional): Data type of the parameters. Default: PyTorch float.

    Inputs:
        - **input** (Tensor): `(H_in,)` or `(N, H_in)`, where `H_in = input_size`.
        - **hidden** (Tensor, optional): `(H_out,)` or `(N, H_out)`,
            where `H_out = hidden_size`. Defaults to zeros if not provided.

    Outputs:
        - **h’** (Tensor): next hidden state, same shape as **hidden**.

    Shape:
        - **input**: `(N, H_in)` or `(H_in,)`
        - **hidden**: `(N, H_out)` or `(H_out,)`
        - **output**: `(N, H_out)` or `(H_out,)`

    Attributes:
        weight_ih (Tensor): input‐to‐hidden weights, shape `(2*H, I)`.
        weight_hh (Tensor): hidden‐to‐hidden weights, shape `(2*H, H)`.
        bias_ih   (Tensor): input biases, shape `(2*H,)` if `bias=True`.
        bias_hh   (Tensor): hidden biases, shape `(2*H,)` if `bias=True`.

    Examples::
        >>> cell = LiGRUCell(10, 20)
        >>> x = torch.randn(5, 10)    # sequence length 5
        >>> h0 = torch.zeros(20)
        >>> h = h0
        >>> outputs = []
        >>> for t in range(x.size(0)):
        ...     h = cell(x[t], h)
        ...     outputs.append(h)
    """

    __constants__ = [
        "input_size",
        "hidden_size",
        "bias",
        "recurrent_bias",
        "nonlinearity",
        "gate_nonlinearity",
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
        nonlinearity: Callable = torch.relu,
        gate_nonlinearity: Callable = torch.sigmoid,
        kernel_init: Callable = nn.init.xavier_uniform_,
        recurrent_kernel_init: Callable = nn.init.xavier_uniform_,
        bias_init: Callable = nn.init.zeros_,
        recurrent_bias_init: Callable = nn.init.zeros_,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super(LiGRUCell, self).__init__(
            input_size, hidden_size, bias, device=device, dtype=dtype
        )
        self.nonlinearity = nonlinearity
        self.gate_nonlinearity = gate_nonlinearity
        self.kernel_init = kernel_init
        self.recurrent_kernel_init = recurrent_kernel_init
        self.bias_init = bias_init
        self.recurrent_bias_init = recurrent_bias_init

        self._default_register_tensors(
            input_size,
            hidden_size,
            ih_mult=2,
            hh_mult=2,
            bias=bias,
            recurrent_bias=recurrent_bias,
        )
        self.init_weights()

    def forward(
        self, inp: Tensor, state: Optional[Union[Tensor, Tuple[Tensor, ...]]] = None
    ) -> Tensor:
        state = self._check_state(state)
        self._validate_input(inp)
        self._validate_state(state)
        inp, state, is_batched = self._preprocess_input_and_state(inp, state)

        gates = (
            inp @ self.weight_ih.t()
            + self.bias_ih
            + state @ self.weight_hh.t()
            + self.bias_hh
        )
        ug, cg = gates.chunk(2, 1)

        update_gate = self.gate_nonlinearity(ug)
        candidate_state = self.nonlinearity(cg)
        new_state = (1.0 - update_gate) * candidate_state + update_gate * state

        if not is_batched:
            new_state = new_state.squeeze(0)

        return new_state

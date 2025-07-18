import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Callable, Tuple, Union
from ..base import BaseSingleRecurrentLayer, BaseSingleRecurrentCell


class MGU(BaseSingleRecurrentLayer):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        batch_first: bool = False,
        **kwargs,
    ):
        super(MGU, self).__init__(input_size, hidden_size, num_layers, dropout, batch_first)
        self.initialize_cells(MGUCell, **kwargs)


class MGUCell(BaseSingleRecurrentCell):
    r"""A Minimal Gated Unit (MGU) cell.

    This cell uses a single forget‐update gate to modulate both
    carry and candidate, reducing parameters relative to an LSTM:

    .. math::

        \mathbf{f}(t) &= \sigma\bigl(
            \mathbf{W}_{ih}^{f}\,\mathbf{x}(t)
            + \mathbf{b}_{ih}^{f}
            + \mathbf{W}_{hh}^{f}\,\mathbf{h}(t-1)
            + \mathbf{b}_{hh}^{f}
        \bigr), \\[6pt]
        \tilde{\mathbf{h}}(t) &= \phi\bigl(
            \mathbf{W}_{ih}^{h}\,\mathbf{x}(t)
            + \mathbf{b}_{ih}^{h}
            + \mathbf{W}_{hh}^{h}\bigl(\mathbf{f}(t)\circ\mathbf{h}(t-1)\bigr)
            + \mathbf{b}_{hh}^{h}
        \bigr), \\[6pt]
        \mathbf{h}(t) &= \bigl(1 - \mathbf{f}(t)\bigr)\circ\mathbf{h}(t-1)
            \;+\;\mathbf{f}(t)\circ\tilde{\mathbf{h}}(t),

    where :math:`\circ` is element‐wise product and :math:`\phi` is a pointwise
    nonlinearity (default `tanh`).

    Args:
        input_size (int):   Number of features in the input :math:`\mathbf{x}(t)`.
        hidden_size (int):  Number of features in the hidden state :math:`\mathbf{h}(t)`.
        bias (bool, optional): If ``False``, disables both biases
            :math:`\mathbf{b}_{ih}` and :math:`\mathbf{b}_{hh}`. Default: ``True``.
        activation_fn (Callable, optional): Nonlinearity :math:`\phi` for the candidate.
            Default: ``torch.tanh``.
        gate_activation_fn (Callable, optional): Activation for the forget gate.
            Default: ``torch.sigmoid``.
        kernel_init (Callable, optional): Initializer for input‐to‐hidden weights
            :math:`\mathbf{W}_{ih}^*`. Default: ``nn.init.xavier_uniform_``.
        recurrent_kernel_init (Callable, optional): Initializer for hidden‐to‐hidden weights
            :math:`\mathbf{W}_{hh}^*`. Default: ``nn.init.xavier_uniform_``.
        bias_init (Callable, optional): Initializer for input biases when `bias=True`.
            Default: ``nn.init.zeros_``.
        recurrent_bias_init (Callable, optional): Initializer for hidden biases
            when `bias=True`. Default: ``nn.init.zeros_``.
        device (torch.device, optional): Device of the parameters. Default: CPU.
        dtype (torch.dtype, optional): Data type of the parameters. Default: PyTorch float.

    Inputs: input, hidden
        - **input** (Tensor): shape `(H_in,)` or `(N, H_in)`, where `H_in = input_size`.
        - **hidden** (Tensor, optional): previous hidden state of shape
            `(H_out,)` or `(N, H_out)`, where `H_out = hidden_size`.
            Defaults to zero if not provided.

    Outputs: h’
        - **h’** (Tensor): next hidden state, same shape as **hidden**.

    Shape:
        - input:  :math:`(N, H_{\mathrm{in}})` or :math:`(H_{\mathrm{in}})`.
        - hidden: :math:`(N, H_{\mathrm{out}})` or :math:`(H_{\mathrm{out}})`.
        - output: :math:`(N, H_{\mathrm{out}})` or :math:`(H_{\mathrm{out}})`.

    Attributes:
        weight_ih (Tensor): input‐to‐hidden weights, shape `(2*H, I)`,
            chunked into forget (`f`) and candidate (`h`) parts.
        weight_hh (Tensor): hidden‐to‐hidden weights, shape `(2*H, H)`,
            chunked into forget and candidate parts.
        bias_ih (Tensor): input biases, shape `(2*H,)` if `bias=True`.
        bias_hh (Tensor): hidden biases, shape `(2*H,)` if `bias=True`.

    Examples::
        >>> cell = MGUCell(10, 20)
        >>> x = torch.randn(5, 10)       # sequence length 5
        >>> h0 = torch.zeros(20)         # initial hidden state
        >>> hx = h0
        >>> outs = []
        >>> for t in range(x.size(0)):
        ...     hx = cell(x[t], hx)
        ...     outs.append(hx)
    """

    __constants__ = [
        "input_size",
        "hidden_size",
        "bias",
        "activation_fn",
        "gate_activation_fn",
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
        activation_fn: Callable = torch.tanh,
        gate_activation_fn: Callable = torch.sigmoid,
        kernel_init: Callable = nn.init.xavier_uniform_,
        recurrent_kernel_init: Callable = nn.init.xavier_uniform_,
        bias_init: Callable = nn.init.zeros_,
        recurrent_bias_init: Callable = nn.init.zeros_,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super(MGUCell, self).__init__(
            input_size, hidden_size, bias, device=device, dtype=dtype
        )
        self.activation_fn = activation_fn
        self.gate_activation_fn = gate_activation_fn
        self.kernel_init = kernel_init
        self.recurrent_kernel_init = recurrent_kernel_init
        self.bias_init = bias_init
        self.recurrent_bias_init = recurrent_bias_init

        self._default_register_tensors(
            input_size, hidden_size, ih_mult=2, hh_mult=2, bias=bias
        )
        self.init_weights()

    def forward(
        self, inp: Tensor, state: Optional[Union[Tensor, Tuple[Tensor, ...]]] = None
    ) -> Tensor:
        state = self._check_state(state)
        self._validate_input(inp)
        self._validate_state(state)
        inp, state, is_batched = self._preprocess_input_and_state(inp, state)

        weight_ih_f, weight_ih_h = self.weight_ih.chunk(2, 0)
        weight_hh_f, weight_hh_h = self.weight_hh.chunk(2, 0)
        bias_ih_f, bias_ih_h = self.bias_ih.chunk(2, 0)
        bias_hh_f, bias_hh_h = self.bias_hh.chunk(2, 0)

        fg = inp @ weight_ih_f.t() + bias_ih_f + state @ weight_hh_f.t() + bias_hh_f
        forget_gate = self.gate_activation_fn(fg)
        hidden_modulated = forget_gate * state
        ch = (
            inp @ weight_ih_h.t()
            + bias_ih_h
            + hidden_modulated @ weight_hh_h.t()
            + bias_hh_h
        )
        candidate_hidden = self.activation_fn(ch)
        new_state = forget_gate * candidate_hidden + (1.0 - forget_gate) * state

        if not is_batched:
            new_state = new_state.squeeze(0)

        return new_state

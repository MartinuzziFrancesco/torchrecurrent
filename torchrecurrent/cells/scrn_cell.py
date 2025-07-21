import torch
from torch import Tensor
import torch.nn as nn
from typing import Optional, Callable, Union, Tuple
from ..base import BaseDoubleRecurrentLayer, BaseDoubleRecurrentCell


class SCRN(BaseDoubleRecurrentLayer):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        batch_first: bool = False,
        **kwargs,
    ):
        super(SCRN, self).__init__(
            input_size, hidden_size, num_layers, dropout, batch_first
        )
        self.initialize_cells(SCRNCell, **kwargs)


class SCRNCell(BaseDoubleRecurrentCell):
    r"""A Structurally Constrained Recurrent Network (SCRN) cell.

    Implements the SCRN from
    “Structurally Constrained Recurrent Networks” <https://arxiv.org/pdf/1412.7753>_.

    .. math::

        \begin{aligned}
        \mathbf{s}(t) &= (1 - \alpha)\,\bigl(\mathbf{W}_{ih}^{s}\,\mathbf{x}(t)
            + \mathbf{b}_{ih}^{s}\bigr)
            + \alpha\,\mathbf{s}(t-1), \\
        \mathbf{h}(t) &= \sigma\Bigl(
            \mathbf{W}_{ch}^{h}\,\mathbf{s}(t)
            + \mathbf{b}_{ch}^{h}
            + \mathbf{W}_{ih}^{h}\,\mathbf{x}(t)
            + \mathbf{b}_{ih}^{h}
            + \mathbf{W}_{hh}^{h}\,\mathbf{h}(t-1)
            + \mathbf{b}_{hh}^{h}
        \Bigr), \\
        \mathbf{y}(t) &= f\Bigl(
            \mathbf{W}_{ch}^{y}\,\mathbf{s}(t)
            + \mathbf{b}_{ch}^{y}
            + \mathbf{W}_{hh}^{y}\,\mathbf{h}(t)
            + \mathbf{b}_{hh}^{y}
        \Bigr)
        \end{aligned}

    where :math:`\sigma` is the sigmoid activation and :math:`f` is an
    optional output nonlinearity.

    Args:
        input_size (int):      Number of input features.
        hidden_size (int):     Number of hidden (and context) features.
        bias (bool, optional): If ``False``, disables :math:`\mathbf{b}_{ih}`.
            Default: ``True``.
        recurrent_bias (bool, optional): If ``False``, disables :math:`\mathbf{b}_{hh}`.
            Default: ``True``.
        context_bias (bool, optional): If ``False``, disables :math:`\mathbf{b}_{ch}`.
            Default: ``True``.
        kernel_init (Callable):
                                Initializer for input‑to‑hidden kernels.
        recurrent_kernel_init (Callable):
                                Initializer for hidden‑to‑hidden kernels.
        context_kernel_init (Callable):
                                Initializer for context‑to‑hidden kernels.
        bias_init (Callable):  Initializer for input biases.
        recurrent_bias_init (Callable):
                                Initializer for hidden biases.
        context_bias_init (Callable):
                                Initializer for context biases.
        alpha (float):         Context interpolation parameter. Default: 1.0.
        device (torch.device, optional): Device for parameters.
        dtype (torch.dtype, optional):   Dtype for parameters.

    Inputs:
        - **inp** (Tensor): shape `(batch, input_size)` or `(input_size,)`.
        - **state** (Tuple[Tensor, Tensor], optional):
            previous `(h, s)` where `h` is hidden and `s` is context, each of shape
            `(batch, hidden_size)` or `(hidden_size,)`. Defaults to zeros.

    Outputs:
        - **new_h** (Tensor):   Next hidden state, same shape as `h`.
        - **new_s** (Tensor):   Next context state, same shape as `s`.

    Attributes:
        weight_ih (Tensor):    Input‑to‑hidden weights, shape `(2*H, I)`.
        weight_hh (Tensor):    Hidden‑to‑hidden weights, shape `(2*H, H)`.
        weight_ch (Tensor):    Context‑to‑hidden weights, shape `(2*H, H)`.
        bias_ih   (Tensor):    Input biases, shape `(2*H,)`.
        bias_hh   (Tensor):    Hidden biases, shape `(2*H,)`.
        bias_ch   (Tensor):    Context biases, shape `(2*H,)`.
        alpha     (Parameter): Learnable context interpolation scalar.

    Examples::
        >>> cell = SCRNCell(10, 20, alpha=0.5)
        >>> x = torch.randn(4, 10)
        >>> h0 = torch.zeros(4, 20)
        >>> s0 = torch.zeros(4, 20)
        >>> h1, s1 = cell(x, (h0, s0))
    """

    __constants__ = [
        "input_size",
        "hidden_size",
        "bias",
        "recurrent_bias",
        "context_bias",
        "kernel_init",
        "recurrent_kernel_init",
        "context_kernel_init",
        "bias_init",
        "recurrent_bias_init",
        "context_bias_init",
    ]

    weight_ih: Tensor
    weight_hh: Tensor
    weight_ch: Tensor
    bias_ih: Tensor
    bias_hh: Tensor
    bias_ch: Tensor

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        recurrent_bias: bool = True,
        context_bias: bool = True,
        kernel_init: Callable = nn.init.xavier_uniform_,
        recurrent_kernel_init: Callable = nn.init.xavier_uniform_,
        context_kernel_init: Callable = nn.init.normal_,
        bias_init: Callable = nn.init.zeros_,
        recurrent_bias_init: Callable = nn.init.zeros_,
        context_bias_init: Callable = nn.init.zeros_,
        alpha: float = 0.5,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super(SCRNCell, self).__init__(
            input_size, hidden_size, bias, device=device, dtype=dtype
        )
        self.kernel_init = kernel_init
        self.recurrent_kernel_init = recurrent_kernel_init
        self.context_kernel_init = context_kernel_init
        self.bias_init = bias_init
        self.recurrent_bias_init = recurrent_bias_init
        self.context_bias_init = context_bias_init
        self.alpha = nn.Parameter(torch.tensor(alpha))

        self._register_tensors(
            {
                "weight_ih": ((2 * hidden_size, input_size), True),
                "weight_hh": ((2 * hidden_size, hidden_size), True),
                "weight_ch": ((2 * hidden_size, hidden_size), True),
                "bias_ih": ((2 * hidden_size,), bias),
                "bias_hh": ((2 * hidden_size,), recurrent_bias),
                "bias_ch": ((2 * hidden_size,), context_bias),
            }
        )
        self.init_weights()

    def init_weights(self):
        for name, param in self.named_parameters():
            if "weight_ih" in name:
                self.kernel_init(param)
            elif "weight_hh" in name:
                self.recurrent_kernel_init(param)
            elif "weight_ch" in name:
                self.context_kernel_init(param)
            elif "bias_ih" in name:
                self.bias_init(param)
            elif "bias_hh" in name:
                self.recurrent_bias_init(param)
            elif "bias_ch" in name:
                self.context_bias_init(param)

    def forward(
        self, inp: Tensor, state: Optional[Union[Tensor, Tuple[Tensor, ...]]] = None
    ) -> Tuple[Tensor, Tensor]:
        state, c_state = self._check_states(state)
        self._validate_input(inp)
        self._validate_states((state, c_state))
        inp, state, c_state, is_batched = self._preprocess_states(inp, (state, c_state))

        inp_expanded = inp @ self.weight_ih.t() + self.bias_ih
        gxs1, gxs2 = inp_expanded.chunk(2, 1)
        weight_hh_1, weight_hh_2 = self.weight_hh.chunk(2, 0)
        bias_hh_1, bias_hh_2 = self.bias_hh.chunk(2, 0)

        new_cstate = (1 - self.alpha) * gxs1 + self.alpha * c_state
        cont_expanded = new_cstate @ self.weight_ch.t() + self.bias_ch
        gcs1, gcs2 = cont_expanded.chunk(2, 1)
        hidden_layer = torch.sigmoid(gxs2 + state @ weight_hh_1.t() + bias_hh_1 + gcs1)
        new_state = torch.tanh(hidden_layer @ weight_hh_2.t() + bias_hh_2 + gcs2)

        if not is_batched:
            new_state = new_state.squeeze(0)
            new_cstate = new_cstate.squeeze(0)

        return new_state, new_cstate

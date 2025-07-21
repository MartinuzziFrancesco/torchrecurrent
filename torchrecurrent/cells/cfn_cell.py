import torch
from torch import nn
from torch import Tensor
from typing import Optional, Callable, Tuple, Union
from ..base import BaseSingleRecurrentLayer, BaseSingleRecurrentCell


class CFN(BaseSingleRecurrentLayer):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        batch_first: bool = False,
        **kwargs,
    ):
        super(CFN, self).__init__(input_size, hidden_size, num_layers, dropout, batch_first)
        self.initialize_cells(CFNCell, **kwargs)


class CFNCell(BaseSingleRecurrentCell):
    r"""A Chaos Free Network (CFN) cell.

    This cell uses two gates—“horizontal” θ and “vertical” η—together with
    a third input projection to update the hidden state without chaotic
    dynamics:

    .. math::

        \boldsymbol{\theta}(t) &= \sigma\bigl(
            \mathbf{W}_{ih}^{\theta}\,\mathbf{x}(t)
            + \mathbf{b}_{ih}^{\theta}
            + \mathbf{W}_{hh}^{\theta}\,\mathbf{h}(t-1)
            + \mathbf{b}_{hh}^{\theta}
        \bigr), \\[6pt]
        \boldsymbol{\eta}(t) &= \sigma\bigl(
            \mathbf{W}_{ih}^{\eta}\,\mathbf{x}(t)
            + \mathbf{b}_{ih}^{\eta}
            + \mathbf{W}_{hh}^{\eta}\,\mathbf{h}(t-1)
            + \mathbf{b}_{hh}^{\eta}
        \bigr), \\[6pt]
        \mathbf{h}(t) &= \boldsymbol{\theta}(t)\,\circ\,\tanh\bigl(\mathbf{h}(t-1)\bigr)
            \;+\;\boldsymbol{\eta}(t)\,\circ\,\tanh\bigl(
                \mathbf{W}_{ih}^{h}\,\mathbf{x}(t)
                + \mathbf{b}_{ih}^{h}
            \bigr)\,.

    Args:
        input_size (int):   Number of features in the input :math:`\mathbf{x}(t)`.
        hidden_size (int):  Number of features in the hidden state :math:`\mathbf{h}(t)`.
        bias (bool, optional): If ``False``, disables :math:`\mathbf{b}_{ih}`.
            Default: ``True``.
        recurrent_bias (bool, optional): If ``False``, disables :math:`\mathbf{b}_{hh}`.
            Default: ``True``.
        kernel_init (Callable, optional): Initializer for input-to-hidden
            weights :math:`\mathbf{W}_{ih}^*`. Default: ``nn.init.xavier_uniform_``.
        recurrent_kernel_init (Callable, optional): Initializer for
            hidden-to-hidden weights :math:`\mathbf{W}_{hh}^*`.
            Default: ``nn.init.xavier_uniform_``.
        bias_init (Callable, optional): Initializer for input biases
            :math:`\mathbf{b}_{ih}^*` when `bias=True`. Default: ``nn.init.zeros_``.
        recurrent_bias_init (Callable, optional): Initializer for hidden
            biases :math:`\mathbf{b}_{hh}^*` when `bias=True`.
            Default: ``nn.init.zeros_``.
        device (torch.device, optional): Device for all parameters.
            Default: CPU.
        dtype (torch.dtype, optional): Data type for all parameters.
            Default: PyTorch default float.

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
        weight_ih (Tensor): input-to-hidden weights of shape
            `(3 * hidden_size, input_size)`, chunked into θ, η, and “h” parts.
        weight_hh (Tensor): hidden-to-hidden weights of shape
            `(2 * hidden_size, hidden_size)`, chunked into θ and η parts.
        bias_ih (Tensor): input biases of shape `(3 * hidden_size,)` if `bias=True`.
        bias_hh (Tensor): hidden biases of shape `(2 * hidden_size,)` if `bias=True`.

    Examples::
        >>> cell = CFNCell(10, 20)
        >>> x = torch.randn(5, 10)      # sequence length 5, feature size 10
        >>> h0 = torch.zeros(20)        # initial hidden state
        >>> hx = h0
        >>> outputs = []
        >>> for t in range(x.size(0)):
        ...     hx = cell(x[t], hx)
        ...     outputs.append(hx)
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
        super(CFNCell, self).__init__(
            input_size, hidden_size, bias, recurrent_bias, device=device, dtype=dtype
        )
        self.kernel_init = kernel_init
        self.recurrent_kernel_init = recurrent_kernel_init
        self.bias_init = bias_init
        self.recurrent_bias_init = recurrent_bias_init

        self._default_register_tensors(
            input_size,
            hidden_size,
            ih_mult=3,
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

        input_exp = inp @ self.weight_ih.t() + self.bias_ih
        rec_exp = state @ self.weight_hh.t() + self.bias_hh
        input_exp_1, input_exp_2, input_exp_3 = input_exp.chunk(3, 1)
        rec_exp_1, rec_exp_2 = rec_exp.chunk(2, 1)

        horizontal_gate = torch.sigmoid(input_exp_1 + rec_exp_1)
        vertical_gate = torch.sigmoid(input_exp_2 + rec_exp_2)
        new_state = horizontal_gate * torch.tanh(state) + vertical_gate * torch.tanh(
            input_exp_3
        )

        if not is_batched:
            new_state = new_state.squeeze(0)

        return new_state

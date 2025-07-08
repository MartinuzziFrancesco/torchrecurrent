import torch
from torch import nn
from torch import Tensor
from typing import Optional, Callable, Tuple, Union
from ..base import BaseSingleRecurrentLayer, BaseSingleRecurrentCell


class BR(BaseSingleRecurrentLayer):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        batch_first: bool = False,
        **kwargs,
    ):
        super(BR, self).__init__(input_size, hidden_size, num_layers, dropout, batch_first)
        self.initialize_cells(BRCell, **kwargs)


class BRCell(BaseSingleRecurrentCell):
    r"""A Bistable recurrent cell.

    This cell uses two element-wise recurrence vectors and a three-way
    split of its input projection to compute an additive modulation gate
    and a carry gate that together produce the next hidden state:

    .. math::

        \mathbf{a}(t) &= 1 + \tanh\Bigl(\mathbf{W}_{ih}^{a}\,\mathbf{x}(t)
            + \mathbf{b}_{ih}^{a}
            + \mathbf{w}_{hh}^{a} \,\circ\, \mathbf{h}(t-1)
            + \mathbf{b}_{hh}^{a}\Bigr), \\[6pt]
        \mathbf{c}(t) &= \sigma\Bigl(\mathbf{W}_{ih}^{c}\,\mathbf{x}(t)
            + \mathbf{b}_{ih}^{c}
            + \mathbf{w}_{hh}^{c} \,\circ\, \mathbf{h}(t-1)
            + \mathbf{b}_{hh}^{c}\Bigr), \\[6pt]
        \mathbf{h}(t) &= \mathbf{c}(t)\,\circ\,\mathbf{h}(t-1)
            \;+\;\bigl(1 - \mathbf{c}(t)\bigr)\,\circ\,
            \tanh\Bigl(\mathbf{W}_{ih}^{h}\,\mathbf{x}(t)
            + \mathbf{b}_{ih}^{h}
            + \mathbf{a}(t)\,\circ\,\mathbf{h}(t-1)\Bigr).

    Args:
        input_size (int): Size of each input vector :math:`\mathbf{x}(t)`.
        hidden_size (int): Size of the hidden state :math:`\mathbf{h}(t)`.
        bias (bool, optional): If ``False``, disables all biases
            :math:`\mathbf{b}_{ih}` and :math:`\mathbf{b}_{hh}`. Default: ``True``.
        kernel_init (Callable, optional): Initializer for all
            input-to-hidden weights :math:`\mathbf{W}_{ih}^*`.
            Default: ``nn.init.xavier_uniform_``.
        recurrent_kernel_init (Callable, optional): Initializer for all
            hidden-to-hidden vectors :math:`\mathbf{w}_{hh}^*`.
            Default: ``nn.init.normal_``.
        bias_init (Callable, optional): Initializer for all
            input biases :math:`\mathbf{b}_{ih}^*` when `bias=True`.
            Default: ``nn.init.zeros_``.
        recurrent_bias_init (Callable, optional): Initializer for all
            hidden biases :math:`\mathbf{b}_{hh}^*` when `bias=True`.
            Default: ``nn.init.zeros_``.
        device (torch.device, optional): Device of the parameters.
            Default: CPU.
        dtype (torch.dtype, optional): Data type of the parameters.
            Default: PyTorch default float.

    Inputs: input, hidden
        - **input** (Tensor): shape `(H_in,)` or `(N, H_in)`, where `H_in = input_size`.
        - **hidden** (Tensor, optional): previous hidden state of shape
            `(H_out,)` or `(N, H_out)`, where `H_out = hidden_size`.
            Defaults to zero if not provided.

    Outputs: h’
        - **h’** (Tensor): next hidden state, same shape as **hidden**.

    Shape:
        - input: :math:`(N, H_{\text{in}})` or :math:`(H_{\text{in}})`.
        - hidden: :math:`(N, H_{\text{out}})` or :math:`(H_{\text{out}})`.
        - output: :math:`(N, H_{\text{out}})` or :math:`(H_{\text{out}})`.

    Attributes:
        weight_ih (Tensor): input-to-hidden weights, shape
            `(3 * hidden_size, input_size)`, split into “a”, “c”, and “h” components.
        weight_hh (Tensor): hidden recurrence vectors, shape
            `(2 * hidden_size,)`, split into “a” and “c” components.
        bias_ih (Tensor): input biases, shape
            `(3 * hidden_size,)` when `bias=True`.
        bias_hh (Tensor): hidden biases, shape
            `(2 * hidden_size,)` when `bias=True`.
        t_ones (Tensor): constant ones vector, shape `(hidden_size,)`, for the “1–c” term.

    Examples::
        >>> cell = BRCell(10, 20)
        >>> x = torch.randn(5, 10)
        >>> h0 = torch.zeros(20)
        >>> hx = h0
        >>> for t in range(x.size(0)):
        ...     hx = cell(x[t], hx)
    """

    weight_ih: Tensor
    weight_hh: Tensor
    bias_ih: Tensor
    bias_hh: Tensor
    t_ones: Tensor

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        kernel_init: Callable = nn.init.xavier_uniform_,
        recurrent_kernel_init: Callable = nn.init.normal_,
        bias_init: Callable = nn.init.zeros_,
        recurrent_bias_init: Callable = nn.init.zeros_,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super(BRCell, self).__init__(
            input_size, hidden_size, bias, device=device, dtype=dtype
        )
        self.kernel_init = kernel_init
        self.recurrent_kernel_init = recurrent_kernel_init
        self.bias_init = bias_init
        self.recurrent_bias_init = recurrent_bias_init

        self._register_tensors(
            {
                "weight_ih": ((3 * hidden_size, input_size), True),
                "weight_hh": ((2 * hidden_size,), True),
                "bias_ih": ((3 * hidden_size,), bias),
                "bias_hh": ((2 * hidden_size,), bias),
                "t_ones": ((hidden_size,), False),
            }
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
        input_exp_1, input_exp_2, input_exp_3 = input_exp.chunk(3, 1)
        rec_matrix_1, rec_matrix_2 = self.weight_hh.chunk(2)

        h1 = input_exp_1 + rec_matrix_1 * state
        h2 = input_exp_2 + rec_matrix_2 * state
        modulation_gate = self.t_ones + torch.tanh(h1)
        candidate_state = torch.sigmoid(h2)
        h3 = input_exp_3 + modulation_gate * state
        new_state = candidate_state * state + (self.t_ones - candidate_state) * torch.tanh(
            h3
        )

        if not is_batched:
            new_state = new_state.squeeze(0)

        return new_state


class NBR(BaseSingleRecurrentLayer):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        batch_first: bool = False,
        **kwargs,
    ):
        super(NBR, self).__init__(input_size, hidden_size, num_layers, dropout, batch_first)
        self.initialize_cells(NBRCell, **kwargs)


class NBRCell(BaseSingleRecurrentCell):
    r"""A Neuromodulated Bistable Recurrent (NBR) cell.

    This cell computes three projections of its input and hidden state,
    uses one to generate an additive modulation gate, one to compute a
    carry gate, and a third to produce the final candidate, yielding:

    .. math::

        \mathbf{a}(t) &= 1 + \tanh\bigl(\mathbf{W}_{ih}^{a}\,\mathbf{x}(t)
            + \mathbf{b}_{ih}^{a}
            + \mathbf{W}_{hh}^{a}\,\circ\,\mathbf{h}(t-1)
            + \mathbf{b}_{hh}^{a}\bigr), \\[6pt]
        \mathbf{c}(t) &= \sigma\bigl(\mathbf{W}_{ih}^{c}\,\mathbf{x}(t)
            + \mathbf{b}_{ih}^{c}
            + \mathbf{W}_{hh}^{c}\,\circ\,\mathbf{h}(t-1)
            + \mathbf{b}_{hh}^{c}\bigr), \\[6pt]
        \mathbf{h}(t) &= \mathbf{c}(t)\,\circ\,\mathbf{h}(t-1)
            \;+\;\bigl(1 - \mathbf{c}(t)\bigr)\,\circ\,
            \tanh\bigl(\mathbf{W}_{ih}^{h}\,\mathbf{x}(t)
            + \mathbf{b}_{ih}^{h}
            + \mathbf{a}(t)\,\circ\,\mathbf{h}(t-1)\bigr).

    Args:
        input_size (int):  Number of features in the input vector :math:`\mathbf{x}(t)`.
        hidden_size (int): Number of features in the hidden state :math:`\mathbf{h}(t)`.
        bias (bool, optional): If ``False``, disables all biases
            :math:`\mathbf{b}_{ih}` and :math:`\mathbf{b}_{hh}`. Default: ``True``.
        kernel_init (Callable, optional): Initializer for all
            input‐to‐hidden weights :math:`\mathbf{W}_{ih}^*`.
            Default: ``nn.init.xavier_uniform_``.
        recurrent_kernel_init (Callable, optional): Initializer for all
            hidden‐to‐hidden weights :math:`\mathbf{W}_{hh}^*`.
            Default: ``nn.init.xavier_uniform_``.
        bias_init (Callable, optional): Initializer for all
            input biases :math:`\mathbf{b}_{ih}^*` when `bias=True`.
            Default: ``nn.init.zeros_``.
        recurrent_bias_init (Callable, optional): Initializer for all
            hidden biases :math:`\mathbf{b}_{hh}^*` when `bias=True`.
            Default: ``nn.init.zeros_``.
        device (torch.device, optional): Device of the parameters.
            Default: CPU.
        dtype (torch.dtype, optional): Data type of the parameters.
            Default: PyTorch default float.

    Inputs: input, hidden
        - **input** (Tensor): shape `(H_in,)` or `(N, H_in)`, where `H_in = input_size`.
        - **hidden** (Tensor, optional): previous hidden state of shape
            `(H_out,)` or `(N, H_out)`, where `H_out = hidden_size`.
            Defaults to zero if not provided.

    Outputs: h’
        - **h’** (Tensor): next hidden state, same shape as **hidden**.

    Shape:
        - input: :math:`(N, H_{\text{in}})` or :math:`(H_{\text{in}})`.
        - hidden: :math:`(N, H_{\text{out}})` or :math:`(H_{\text{out}})`.
        - output: :math:`(N, H_{\text{out}})` or :math:`(H_{\text{out}})`.

    Attributes:
        weight_ih (Tensor): input‐to‐hidden weights of shape
            `(3 * hidden_size, input_size)`, chunked into “a”, “c”, and “h” parts.
        weight_hh (Tensor): hidden‐to‐hidden weights of shape
            `(2 * hidden_size, hidden_size)`, chunked into “a” and “c” parts.
        bias_ih (Tensor): input biases of shape `(3 * hidden_size,)` if `bias=True`.
        bias_hh (Tensor): hidden biases of shape `(2 * hidden_size,)` if `bias=True`.
        t_ones (Tensor): constant ones vector of shape `(hidden_size,)`
        for the term `(1–c)`.

    Examples::
        >>> cell = NBRCell(10, 20)
        >>> x = torch.randn(5, 10)
        >>> h0 = torch.zeros(20)
        >>> hx = h0
        >>> for t in range(x.size(0)):
        ...     hx = cell(x[t], hx)
    """

    weight_ih: Tensor
    weight_hh: Tensor
    bias_ih: Tensor
    bias_hh: Tensor
    t_ones: Tensor

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        kernel_init: Callable = nn.init.xavier_uniform_,
        recurrent_kernel_init: Callable = nn.init.xavier_uniform_,
        bias_init: Callable = nn.init.zeros_,
        recurrent_bias_init: Callable = nn.init.zeros_,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super(NBRCell, self).__init__(
            input_size, hidden_size, bias, device=device, dtype=dtype
        )
        self.kernel_init = kernel_init
        self.recurrent_kernel_init = recurrent_kernel_init
        self.bias_init = bias_init
        self.recurrent_bias_init = recurrent_bias_init

        self._register_tensors(
            {
                "weight_ih": ((3 * hidden_size, input_size), True),
                "weight_hh": ((2 * hidden_size, hidden_size), True),
                "bias_ih": ((3 * hidden_size,), bias),
                "bias_hh": ((2 * hidden_size,), bias),
                "t_ones": ((hidden_size,), False),
            }
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
        input_exp_1, input_exp_2, input_exp_3 = input_exp.chunk(3, 1)
        rec_matrix_1, rec_matrix_2 = self.weight_hh.chunk(2, 0)
        t_ones = state.new_ones(self.hidden_size)
        h1 = input_exp_1 + state @ rec_matrix_1.t()
        h2 = input_exp_2 + state @ rec_matrix_2.t()
        modulation_gate = t_ones + torch.tanh(h1)
        candidate_state = torch.sigmoid(h2)
        h3 = input_exp_3 + modulation_gate * state

        new_state = candidate_state * state + (t_ones - candidate_state) * torch.tanh(h3)

        if not is_batched:
            new_state = new_state.squeeze(0)

        return new_state

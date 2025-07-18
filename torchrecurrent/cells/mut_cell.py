import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Callable, Tuple, Union
from ..base import BaseSingleRecurrentLayer, BaseSingleRecurrentCell


class MUT1(BaseSingleRecurrentLayer):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        batch_first: bool = False,
        **kwargs,
    ):
        super(MUT1, self).__init__(
            input_size, hidden_size, num_layers, dropout, batch_first
        )
        self.initialize_cells(MUT1Cell, **kwargs)


class MUT1Cell(BaseSingleRecurrentCell):
    r"""A Mutated Unit Type 1 (MUT1) recurrent cell.

    Implements the MUT1 update from
    “An Empirical Exploration of Recurrent Network Architectures”
    <https://proceedings.mlr.press/v37/jozefowicz15.pdf>_.

    .. math::

        \begin{aligned}
        \mathbf{z}(t) &= \sigma\bigl(\mathbf{W}_{ih}^{z}\,\mathbf{x}(t)
            + \mathbf{b}_{ih}^{z}\bigr), \\
        \mathbf{r}(t) &= \sigma\bigl(\mathbf{W}_{ih}^{r}\,\mathbf{x}(t)
            + \mathbf{b}_{ih}^{r}
            + \mathbf{W}_{hh}^{r}\,\mathbf{h}(t-1)
            + \mathbf{b}_{hh}^{r}\bigr), \\
        \mathbf{h}(t) &= \bigl[\tanh\bigl(\mathbf{W}_{hh}^{h}\,
            (\mathbf{r}(t)\circ\mathbf{h}(t-1)) + \mathbf{b}_{hh}^{h}\bigr)
            + \tanh\bigl(\mathbf{W}_{ih}^{h}\,\mathbf{x}(t)\bigr)
            + \mathbf{b}_{ih}^{h}\bigr]\circ\mathbf{z}(t)
            + \mathbf{h}(t-1)\circ\bigl(1 - \mathbf{z}(t)\bigr).
        \end{aligned}

    where :math:`\sigma` is the sigmoid function and :math:`\circ` the Hadamard product.

    Args:
        input_size (int):  Number of features in the input `inp`.
        hidden_size (int): Number of features in the hidden state.
        bias (bool):       If False, no bias terms are used. Default: True.
        kernel_init (Callable):
                            Initializer for input‑to‑hidden weights.
        recurrent_kernel_init (Callable):
                            Initializer for hidden‑to‑hidden weights.
        bias_init (Callable):
                            Initializer for input biases.
        recurrent_bias_init (Callable):
                            Initializer for hidden biases.
        device (torch.device, optional): Device on which to place parameters.
        dtype (torch.dtype, optional):   Data type for parameters.

    Inputs:
        - **inp** (Tensor): shape `(batch, input_size)` or `(input_size,)`.
        - **state** (Tensor, optional): previous hidden state of shape
            `(batch, hidden_size)` or `(hidden_size,)`. Defaults to zeros.

    Outputs:
        - **new_state** (Tensor): Updated hidden state, same shape as `state`.

    Attributes:
        weight_ih (Tensor): Input‑to‑hidden weight, shape `(3*H, I)`.
        weight_hh (Tensor): Hidden‑to‑hidden weight, shape `(H, H)`.
        bias_ih   (Tensor): Input bias, shape `(3*H,)`.
        bias_hh   (Tensor): Hidden bias, shape `(H,)`.

    Examples::
        >>> cell = MUT1Cell(10, 20)
        >>> x = torch.randn(5, 10)   # (batch=5, input_size=10)
        >>> h0 = torch.zeros(5, 20)
        >>> h1 = cell(x, h0)
    """

    __constants__ = [
        "input_size",
        "hidden_size",
        "bias",
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
        kernel_init: Callable = nn.init.xavier_uniform_,
        recurrent_kernel_init: Callable = nn.init.xavier_uniform_,
        bias_init: Callable = nn.init.zeros_,
        recurrent_bias_init: Callable = nn.init.zeros_,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super(MUT1Cell, self).__init__(
            input_size, hidden_size, bias, device=device, dtype=dtype
        )
        self.kernel_init = kernel_init
        self.recurrent_kernel_init = recurrent_kernel_init
        self.bias_init = bias_init
        self.recurrent_bias_init = recurrent_bias_init

        self._default_register_tensors(
            input_size, hidden_size, ih_mult=3, hh_mult=2, bias=bias
        )
        self.init_weights()

    def forward(
        self, inp: Tensor, state: Optional[Union[Tensor, Tuple[Tensor, ...]]] = None
    ) -> Tensor:
        state = self._check_state(state)
        self._validate_input(inp)
        self._validate_state(state)
        inp, state, is_batched = self._preprocess_input_and_state(inp, state)

        inp_expanded = inp @ self.weight_ih.t() + self.bias_ih
        gxs1, gxs2, gxs3 = inp_expanded.chunk(3, 1)
        weight_hh_1, weight_hh_2 = self.weight_hh.chunk(2, 0)
        bias_hh_1, bias_hh_2 = self.bias_hh.chunk(2, 0)

        input_gate = torch.sigmoid(gxs1)
        reset_gate = torch.sigmoid(gxs2 + state @ weight_hh_1.t() + bias_hh_1)
        candidate_state = torch.tanh(
            (reset_gate * state) @ weight_hh_2.t() + torch.tanh(gxs3) + bias_hh_2
        )
        new_state = candidate_state * input_gate + state * (1 - input_gate)

        if not is_batched:
            new_state = new_state.squeeze(0)

        return new_state


class MUT2(BaseSingleRecurrentLayer):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        batch_first: bool = False,
        **kwargs,
    ):
        super(MUT2, self).__init__(
            input_size, hidden_size, num_layers, dropout, batch_first
        )
        self.initialize_cells(MUT2Cell, **kwargs)


class MUT2Cell(BaseSingleRecurrentCell):
    r"""A Mutated Unit Type 2 (MUT2) recurrent cell.

    Implements the MUT2 update from
    “An Empirical Exploration of Recurrent Network Architectures”
    <https://proceedings.mlr.press/v37/jozefowicz15.pdf>_.

    .. math::

        \begin{aligned}
        \mathbf{z}(t) &= \sigma\bigl(
            \mathbf{W}_{ih}^{z}\,\mathbf{x}(t) + \mathbf{b}_{ih}^{z}
            + \mathbf{W}_{hh}^{z}\,\mathbf{h}(t-1) + \mathbf{b}_{hh}^{z}
        \bigr), \\
        \mathbf{r}(t) &= \sigma\bigl(
            \mathbf{W}_{ih}^{r}\,\mathbf{x}(t) + \mathbf{b}_{ih}^{r}
            + \mathbf{W}_{hh}^{r}\,\mathbf{h}(t-1) + \mathbf{b}_{hh}^{r}
        \bigr), \\
        \mathbf{h}(t) &= \tanh\Bigl(
            \mathbf{W}_{hh}^{h}\,(\mathbf{r}(t)\circ\mathbf{h}(t-1))
            + \mathbf{b}_{hh}^{h}
            + \mathbf{W}_{ih}^{h}\,\mathbf{x}(t) + \mathbf{b}_{ih}^{h}
        \Bigr)\circ\mathbf{z}(t)
        + \mathbf{h}(t-1)\circ\bigl(1 - \mathbf{z}(t)\bigr)
        \end{aligned}

    where :math:`\sigma` is the sigmoid function and :math:`\circ` is
    element‑wise multiplication.

    Args:
        input_size (int):  Number of expected features in the input `inp`.
        hidden_size (int): Number of features in the hidden state.
        bias (bool):       If False, no bias terms are used. Default: True.
        kernel_init (Callable):
                            Initializer for input‑to‑hidden weights.
        recurrent_kernel_init (Callable):
                            Initializer for hidden‑to‑hidden weights.
        bias_init (Callable):
                            Initializer for input biases.
        recurrent_bias_init (Callable):
                            Initializer for hidden biases.
        device (torch.device, optional): Device on which to place parameters.
        dtype (torch.dtype, optional):   Data type for parameters.

    Inputs:
        - **inp** (Tensor): shape `(batch, input_size)` or `(input_size,)`.
        - **state** (Tensor, optional): previous hidden state of shape
            `(batch, hidden_size)` or `(hidden_size,)`. Defaults to zero.

    Outputs:
        - **new_state** (Tensor): Updated hidden state, same shape as `state`.

    Attributes:
        weight_ih (Tensor): Input‑to‑hidden weights, shape `(3*hidden_size, input_size)`.
        weight_hh (Tensor): Hidden‑to‑hidden weights, shape `(3*hidden_size, hidden_size)`.
        bias_ih   (Tensor): Input biases, shape `(3*hidden_size,)`.
        bias_hh   (Tensor): Hidden biases, shape `(3*hidden_size,)`.

    Examples::
        >>> cell = MUT2Cell(16, 32)
        >>> x = torch.randn(5, 16)    # batch=5, input_size=16
        >>> h0 = torch.zeros(5, 32)
        >>> h1 = cell(x, h0)
    """

    __constants__ = [
        "input_size",
        "hidden_size",
        "bias",
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
        kernel_init: Callable = nn.init.xavier_uniform_,
        recurrent_kernel_init: Callable = nn.init.xavier_uniform_,
        bias_init: Callable = nn.init.zeros_,
        recurrent_bias_init: Callable = nn.init.zeros_,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super(MUT2Cell, self).__init__(
            input_size, hidden_size, bias, device=device, dtype=dtype
        )
        self.kernel_init = kernel_init
        self.recurrent_kernel_init = recurrent_kernel_init
        self.bias_init = bias_init
        self.recurrent_bias_init = recurrent_bias_init

        self._default_register_tensors(
            input_size, hidden_size, ih_mult=3, hh_mult=3, bias=bias
        )
        self.init_weights()

    def forward(
        self, inp: Tensor, state: Optional[Union[Tensor, Tuple[Tensor, ...]]] = None
    ) -> Tensor:
        state = self._check_state(state)
        self._validate_input(inp)
        self._validate_state(state)
        inp, state, is_batched = self._preprocess_input_and_state(inp, state)

        inp_expanded = inp @ self.weight_ih.t() + self.bias_ih
        gxs1, gxs2, gxs3 = inp_expanded.chunk(3, 1)
        weight_hh_1, weight_hh_2, weight_hh_3 = self.weight_hh.chunk(3, 0)
        bias_hh_1, bias_hh_2, bias_hh_3 = self.bias_hh.chunk(3, 0)

        input_gate = torch.sigmoid(gxs1 + state @ weight_hh_1.t() + bias_hh_1)
        reset_gate = torch.sigmoid(gxs2 + state @ weight_hh_2.t() + bias_hh_2)
        candidate_state = torch.tanh(
            gxs3 + (reset_gate * state) @ weight_hh_3.t() + bias_hh_3
        )
        new_state = candidate_state * input_gate + state * (1 - input_gate)

        if not is_batched:
            new_state = new_state.squeeze(0)

        return new_state


class MUT3(BaseSingleRecurrentLayer):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        batch_first: bool = False,
        **kwargs,
    ):
        super(MUT3, self).__init__(
            input_size, hidden_size, num_layers, dropout, batch_first
        )
        self.initialize_cells(MUT3Cell, **kwargs)


class MUT3Cell(BaseSingleRecurrentCell):
    r"""A Mutated Unit Type 3 (MUT3) recurrent cell.

    Implements the MUT3 update from
    “An Empirical Exploration of Recurrent Network Architectures”
    <https://proceedings.mlr.press/v37/jozefowicz15.pdf>_.

    .. math::

        \begin{aligned}
        \mathbf{z}(t) &= \sigma\Bigl(
            \mathbf{W}_{ih}^{z}\,\mathbf{x}(t) + \mathbf{b}_{ih}^{z}
            + \mathbf{W}_{hh}^{z}\,\mathbf{h}(t-1) + \mathbf{b}_{hh}^{z}
        \Bigr), \\
        \mathbf{r}(t) &= \sigma\Bigl(
            \mathbf{x}(t) + \mathbf{W}_{hh}^{r}\,\mathbf{h}(t-1)
            + \mathbf{b}_{hh}^{r}
        \Bigr), \\
        \mathbf{h}(t) &= \Bigl[\tanh\bigl(
            \mathbf{W}_{hh}^{h}\,(\mathbf{r}(t)\circ\mathbf{h}(t-1))
            + \mathbf{b}_{hh}^{h}
            + \mathbf{W}_{ih}^{h}\,\mathbf{x}(t)
            + \mathbf{b}_{ih}^{h}
        \bigr)\Bigr]\circ\mathbf{z}(t)
        + \mathbf{h}(t-1)\circ\bigl(1 - \mathbf{z}(t)\bigr)
        \end{aligned}

    where :math:`\sigma` is the sigmoid function and :math:`\circ`
    denotes element‑wise multiplication.

    Args:
        input_size (int):  Number of features in the input `inp`.
        hidden_size (int): Number of features in the hidden state.
        bias (bool):       If False, no bias terms are used. Default: True.
        kernel_init (Callable):
                            Initializer for input‑to‑hidden weights.
        recurrent_kernel_init (Callable):
                            Initializer for hidden‑to‑hidden weights.
        bias_init (Callable):
                            Initializer for input biases.
        recurrent_bias_init (Callable):
                            Initializer for hidden biases.
        device (torch.device, optional): Device on which to place parameters.
        dtype (torch.dtype, optional):   Data type for parameters.

    Inputs:
        - **inp** (Tensor): shape `(batch, input_size)` or `(input_size,)`.
        - **state** (Tensor, optional): previous hidden state of shape
            `(batch, hidden_size)` or `(hidden_size,)`. Defaults to zero.

    Outputs:
        - **new_state** (Tensor): Updated hidden state, same shape as `state`.

    Attributes:
        weight_ih (Tensor): Input‑to‑hidden weights, shape `(3*hidden_size, input_size)`.
        weight_hh (Tensor): Hidden‑to‑hidden weights, shape `(3*hidden_size, hidden_size)`.
        bias_ih   (Tensor): Input biases, shape `(3*hidden_size,)`.
        bias_hh   (Tensor): Hidden biases, shape `(3*hidden_size,)`.

    Examples::
        >>> cell = MUT3Cell(10, 20)
        >>> x = torch.randn(5, 10)   # batch=5, input_size=10
        >>> h0 = torch.zeros(5, 20)
        >>> h1 = cell(x, h0)
    """

    __constants__ = [
        "input_size",
        "hidden_size",
        "bias",
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
        kernel_init: Callable = nn.init.xavier_uniform_,
        recurrent_kernel_init: Callable = nn.init.xavier_uniform_,
        bias_init: Callable = nn.init.zeros_,
        recurrent_bias_init: Callable = nn.init.zeros_,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super(MUT3Cell, self).__init__(
            input_size, hidden_size, bias, device=device, dtype=dtype
        )
        self.kernel_init = kernel_init
        self.recurrent_kernel_init = recurrent_kernel_init
        self.bias_init = bias_init
        self.recurrent_bias_init = recurrent_bias_init

        self._default_register_tensors(
            input_size, hidden_size, ih_mult=3, hh_mult=3, bias=bias
        )
        self.init_weights()

    def forward(
        self, inp: Tensor, state: Optional[Union[Tensor, Tuple[Tensor, ...]]] = None
    ) -> Tensor:
        state = self._check_state(state)
        self._validate_input(inp)
        self._validate_state(state)
        inp, state, is_batched = self._preprocess_input_and_state(inp, state)

        inp_expanded = inp @ self.weight_ih.t() + self.bias_ih
        gxs1, gxs2, gxs3 = inp_expanded.chunk(3, 1)
        weight_hh_1, weight_hh_2, weight_hh_3 = self.weight_hh.chunk(3, 0)
        bias_hh_1, bias_hh_2, bias_hh_3 = self.bias_hh.chunk(3, 0)

        input_gate = torch.sigmoid(gxs1 + torch.tanh(state) @ weight_hh_1.t() + bias_hh_1)
        reset_gate = torch.sigmoid(gxs2 + state @ weight_hh_2.t() + bias_hh_2)
        candidate_state = torch.tanh(
            gxs3 + (reset_gate * state) @ weight_hh_3.t() + bias_hh_3
        )
        new_state = candidate_state * input_gate + state * (1 - input_gate)

        if not is_batched:
            new_state = new_state.squeeze(0)

        return new_state

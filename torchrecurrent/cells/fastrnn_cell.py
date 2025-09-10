import torch
from torch import nn
from torch import Tensor
from typing import Optional, Callable, Tuple, Union
from ..base import BaseSingleRecurrentLayer, BaseSingleRecurrentCell


class FastRNN(BaseSingleRecurrentLayer):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        batch_first: bool = False,
        **kwargs,
    ):
        super(FastRNN, self).__init__(
            input_size, hidden_size, num_layers, dropout, batch_first
        )
        self.initialize_cells(FastRNNCell, **kwargs)


class FastRNNCell(BaseSingleRecurrentCell):
    r"""A Fast RNN cell with two scalar gates :math:`\alpha` and :math:`\beta`.

    [`arXiv <https://arxiv.org/abs/1901.02358>`_]

    .. math::

        \tilde{\mathbf{h}}(t) &= \phi\bigl(
            \mathbf{W}_{ih}\,\mathbf{x}(t)
            + \mathbf{b}_{ih}
            + \mathbf{W}_{hh}\,\mathbf{h}(t-1)
            + \mathbf{b}_{hh}
        \bigr), \\[6pt]
        \mathbf{h}(t) &= \alpha\,\tilde{\mathbf{h}}(t)
                        + \beta\,\mathbf{h}(t-1),

    where :math:`\phi` is a pointwise nonlinearity (e.g., tanh), and
    :math:`\alpha` / :math:`\beta` are learnable scalars.

    Args:
        input_size: The number of expected features in the input ``x``.
        hidden_size: The number of features in the hidden state ``h``.
        bias: If ``False``, the layer does not use input-side bias ``b_{ih}``.
            Default: ``True``.
        recurrent_bias: If ``False``, the layer does not use recurrent bias
            ``b_{hh}``. Default: ``True``.
        nonlinearity: Activation function :math:`\phi` for the candidate.
            Default: :func:`torch.tanh`.
        kernel_init: Initializer for ``W_{ih}``.
            Default: :func:`torch.nn.init.xavier_uniform_`.
        recurrent_kernel_init: Initializer for ``W_{hh}``.
            Default: :func:`torch.nn.init.xavier_uniform_`.
        bias_init: Initializer for ``b_{ih}`` when ``bias=True``.
            Default: :func:`torch.nn.init.zeros_`.
        recurrent_bias_init: Initializer for ``b_{hh}`` when
            ``recurrent_bias=True``. Default: :func:`torch.nn.init.zeros_`.
        alpha_init: Initial value for the learnable scalar :math:`\alpha`.
            Default: ``3.0``.
        beta_init: Initial value for the learnable scalar :math:`\beta`.
            Default: ``-3.0``.
        device: The desired device of parameters.
        dtype: The desired floating point type of parameters.

    Inputs: input, h_0
        - **input** of shape ``(batch, input_size)`` or ``(input_size,)``:
          Tensor containing input features.
        - **h_0** of shape ``(batch, hidden_size)`` or ``(hidden_size,)``:
          Tensor containing the initial hidden state.

        If **h_0** is not provided, it defaults to zero.

    Outputs: h_1
        - **h_1** of shape ``(batch, hidden_size)`` or ``(hidden_size,)``:
          Tensor containing the next hidden state.

    Variables:
        weight_ih: The learnable input–hidden weights,
            of shape ``(hidden_size, input_size)``.
        weight_hh: The learnable hidden–hidden weights,
            of shape ``(hidden_size, hidden_size)``.
        bias_ih: The learnable input–hidden bias,
            of shape ``(hidden_size)`` if ``bias=True``.
        bias_hh: The learnable hidden–hidden bias,
            of shape ``(hidden_size)`` if ``recurrent_bias=True``.
        alpha: The learnable scalar gating coefficient :math:`\alpha`,
            of shape ``(1,)``.
        beta: The learnable scalar gating coefficient :math:`\beta`,
            of shape ``(1,)``.

    Examples::

        >>> cell = FastRNNCell(10, 20)
        >>> x = torch.randn(5, 3, 10)      # (time_steps, batch, input_size)
        >>> h = torch.zeros(3, 20)         # (batch, hidden_size)
        >>> out = []
        >>> for t in range(x.size(0)):
        ...     h = cell(x[t], h)
        ...     out.append(h)
        >>> out = torch.stack(out, dim=0)  # (time_steps, batch, hidden_size)
    """

    __constants__ = [
        "input_size",
        "hidden_size",
        "bias",
        "recurrent_bias",
        "nonlinearity",
        "kernel_init",
        "recurrent_kernel_init",
        "bias_init",
        "recurrent_bias_init",
        "alpha_init",
        "beta_init",
    ]

    weight_ih: Tensor
    weight_hh: Tensor
    bias_ih: Tensor
    bias_hh: Tensor
    alpha: Tensor
    beta: Tensor

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        recurrent_bias: bool = True,
        nonlinearity: Callable = torch.tanh,
        kernel_init: Callable = nn.init.xavier_uniform_,
        recurrent_kernel_init: Callable = nn.init.xavier_uniform_,
        bias_init: Callable = nn.init.zeros_,
        recurrent_bias_init: Callable = nn.init.zeros_,
        alpha_init: float = 3.0,
        beta_init: float = -3.0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super(FastRNNCell, self).__init__(
            input_size, hidden_size, bias, device=device, dtype=dtype
        )
        self.kernel_init = kernel_init
        self.recurrent_kernel_init = recurrent_kernel_init
        self.bias_init = bias_init
        self.recurrent_bias_init = recurrent_bias_init
        self.alpha_init = alpha_init
        self.beta_init = beta_init
        self.nonlinearity = nonlinearity

        self._register_tensors(
            {
                "weight_ih": ((hidden_size, input_size), True),
                "weight_hh": ((hidden_size, hidden_size), True),
                "bias_ih": ((hidden_size,), bias),
                "bias_hh": ((hidden_size,), recurrent_bias),
                "alpha": ((1,), True),
                "beta": ((1,), True),
            }
        )
        self.init_weights()

    def init_weights(self):
        for name, param in self.named_parameters():
            if name.endswith("weight_ih"):
                self.kernel_init(param)
            elif name.endswith("weight_hh"):
                self.recurrent_kernel_init(param)
            elif name.endswith("bias_ih"):
                self.bias_init(param)
            elif name.endswith("bias_hh"):
                self.recurrent_bias_init(param)
            elif name == "alpha":
                nn.init.constant_(param, self.alpha_init)
            elif name == "beta":
                nn.init.constant_(param, self.beta_init)

    def forward(
        self, inp: Tensor, state: Optional[Union[Tensor, Tuple[Tensor, ...]]] = None
    ) -> Tensor:
        state = self._check_state(state)
        self._validate_input(inp)
        self._validate_state(state)
        inp, state, is_batched = self._preprocess_input_and_state(inp, state)

        candidate_state = self.nonlinearity(
            inp @ self.weight_ih.t()
            + self.bias_ih
            + state @ self.weight_hh.t()
            + self.bias_hh
        )
        new_state = self.alpha * candidate_state + self.beta * state

        if not is_batched:
            new_state = new_state.squeeze(0)

        return new_state


class FastGRNN(BaseSingleRecurrentLayer):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        batch_first: bool = False,
        **kwargs,
    ):
        super(FastGRNN, self).__init__(
            input_size, hidden_size, num_layers, dropout, batch_first
        )
        self.initialize_cells(FastGRNNCell, **kwargs)


class FastGRNNCell(BaseSingleRecurrentCell):
    r"""A “Fast RNN” cell with two scalar gates α and β [`arXiv <https://arxiv.org/abs/1901.02358>`_].

    .. math::

        \mathbf{z}(t) &= \sigma\Bigl(
            \mathbf{W}_{ih}\,\mathbf{x}(t)
            + \mathbf{b}_{ih}^{z}
            + \mathbf{W}_{hh}\,\mathbf{h}(t-1)
            + \mathbf{b}_{hh}^{z}
        \Bigr), \\[6pt]
        \tilde{\mathbf{h}}(t) &= \tanh\Bigl(
            \mathbf{W}_{ih}\,\mathbf{x}(t)
            + \mathbf{b}_{ih}^{h}
            + \mathbf{W}_{hh}\,\mathbf{h}(t-1)
            + \mathbf{b}_{hh}^{h}
        \Bigr), \\[6pt]
        \mathbf{h}(t) &= \Bigl[\zeta\,\bigl(1 - \mathbf{z}(t)\bigr) + \nu\Bigr]
            \circ \tilde{\mathbf{h}}(t)
            \;+\;\mathbf{z}(t)\,\circ\,\mathbf{h}(t-1),

    where :math:`\circ` denotes element‐wise product.

    Args:
        input_size: The number of expected features in the input ``x``
        hidden_size: The number of features in the hidden state ``h``
        bias: If ``False``, the layer does not use input-side biases. Default: ``True``
        recurrent_bias: If ``False``, the layer does not use recurrent biases. Default: ``True``
        nonlinearity: Activation for the gate :math:`\mathbf{z}`. Default: :func:`torch.sigmoid`
        kernel_init: Initializer for ``W_{ih}``. Default: :func:`torch.nn.init.xavier_uniform_`
        recurrent_kernel_init: Initializer for ``W_{hh}``. Default: :func:`torch.nn.init.xavier_uniform_`
        bias_init: Initializer for ``b_{ih}^{*}`` when ``bias=True``. Default: :func:`torch.nn.init.zeros_`
        recurrent_bias_init: Initializer for ``b_{hh}^{*}`` when ``recurrent_bias=True``. Default: :func:`torch.nn.init.zeros_`
        zeta_init: Initial value for scalar gate :math:`\zeta`. Default: ``3.0``
        nu_init: Initial value for scalar gate :math:`\nu`. Default: ``-3.0``
        device: The desired device of parameters.
        dtype: The desired floating point type of parameters.

    Inputs: input, h_0
        - **input** of shape ``(batch, input_size)`` or ``(input_size,)``: tensor containing input features
        - **h_0** of shape ``(batch, hidden_size)`` or ``(hidden_size,)``: tensor containing the initial hidden state

        If **h_0** is not provided, it defaults to zero.

    Outputs: h_1
        - **h_1** of shape ``(batch, hidden_size)`` or ``(hidden_size,)``: tensor containing the next hidden state

    Variables:
        weight_ih: the learnable input–hidden weights, of shape ``(hidden_size, input_size)``
        weight_hh: the learnable hidden–hidden weights, of shape ``(hidden_size, hidden_size)``
        bias_ih: the learnable input–hidden biases, of shape ``(2*hidden_size)`` (split into z & h) if ``bias=True``
        bias_hh: the learnable hidden–hidden biases, of shape ``(2*hidden_size)`` (split into z & h) if ``recurrent_bias=True``
        zeta: the learnable scalar gate :math:`\zeta`, shape ``(1,)``
        nu: the learnable scalar gate :math:`\nu`, shape ``(1,)``
        t_ones: a constant ones buffer, shape ``(hidden_size,)``

    Examples::

        >>> cell = FastGRNNCell(10, 20)
        >>> x = torch.randn(5, 3, 10)      # (time_steps, batch, input_size)
        >>> h = torch.zeros(3, 20)         # (batch, hidden_size)
        >>> out = []
        >>> for t in range(x.size(0)):
        ...     h = cell(x[t], h)
        ...     out.append(h)
        >>> out = torch.stack(out, dim=0)  # (time_steps, batch, hidden_size)
    """

    __constants__ = [
        "input_size",
        "hidden_size",
        "bias",
        "recurrent_bias",
        "nonlinearity",
        "kernel_init",
        "recurrent_kernel_init",
        "bias_init",
        "recurrent_bias_init",
        "zeta_init",
        "nu_init",
    ]

    weight_ih: Tensor
    weight_hh: Tensor
    bias_ih: Tensor
    bias_hh: Tensor
    zeta: Tensor
    nu: Tensor

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        recurrent_bias: bool = True,
        nonlinearity: Callable = torch.tanh,
        kernel_init: Callable = nn.init.xavier_uniform_,
        recurrent_kernel_init: Callable = nn.init.xavier_uniform_,
        bias_init: Callable = nn.init.zeros_,
        recurrent_bias_init: Callable = nn.init.zeros_,
        zeta_init: float = 3.0,
        nu_init: float = -3.0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super(FastGRNNCell, self).__init__(
            input_size, hidden_size, bias, device=device, dtype=dtype
        )
        self.kernel_init = kernel_init
        self.recurrent_kernel_init = recurrent_kernel_init
        self.bias_init = bias_init
        self.recurrent_bias_init = recurrent_bias_init
        self.zeta_init = zeta_init
        self.nu_init = nu_init
        self.nonlinearity = nonlinearity

        self._register_tensors(
            {
                "weight_ih": ((hidden_size, input_size), True),
                "weight_hh": ((hidden_size, hidden_size), True),
                "bias_ih": ((2 * hidden_size,), bias),
                "bias_hh": ((2 * hidden_size,), recurrent_bias),
                "zeta": ((1,), True),
                "nu": ((1,), True),
            }
        )
        self.init_weights()

    def init_weights(self):
        for name, param in self.named_parameters():
            if name.endswith("weight_ih"):
                self.kernel_init(param)
            elif name.endswith("weight_hh"):
                self.recurrent_kernel_init(param)
            elif name.endswith("bias_ih"):
                self.bias_init(param)
            elif name.endswith("bias_hh"):
                self.recurrent_bias_init(param)
            elif name == "zeta":
                nn.init.constant_(param, self.zeta_init)
            elif name == "nu":
                nn.init.constant_(param, self.nu_init)

    def forward(
        self, inp: Tensor, state: Optional[Union[Tensor, Tuple[Tensor, ...]]] = None
    ) -> Tensor:
        state = self._check_state(state)
        self._validate_input(inp)
        self._validate_state(state)
        inp, state, is_batched = self._preprocess_input_and_state(inp, state)

        bias_ih_1, bias_ih_2 = self.bias_ih.chunk(2)
        bias_hh_1, bias_hh_2 = self.bias_hh.chunk(2)

        partial_gate = inp @ self.weight_ih.t() + state @ self.weight_hh.t()
        gate = self.nonlinearity(partial_gate + bias_ih_1 + bias_hh_1)
        candidate_state = torch.tanh(partial_gate + bias_ih_2 + bias_hh_2)
        new_state = (self.zeta * (1.0 - gate) + self.nu) * candidate_state + gate * state

        if not is_batched:
            new_state = new_state.squeeze(0)

        return new_state

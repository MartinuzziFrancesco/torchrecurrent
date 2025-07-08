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
    r"""A “Fast RNN” cell with two scalar gates α and β.

    This cell first computes a candidate hidden state via a standard
    RNN update, then linearly interpolates between the candidate and
    the previous hidden state:

    .. math::

        \tilde{\mathbf{h}}(t) &= \phi\bigl(
            \mathbf{W}_{ih}\,\mathbf{x}(t)
            + \mathbf{b}_{ih}
            + \mathbf{W}_{hh}\,\mathbf{h}(t-1)
            + \mathbf{b}_{hh}
        \bigr), \\[6pt]
        \mathbf{h}(t) &= \alpha\,\tilde{\mathbf{h}}(t)
                        + \beta\,\mathbf{h}(t-1),

    where :math:`\phi` is a pointwise nonlinearity (e.g. tanh), and
    :math:`\alpha` / :math:`\beta` are learnable scalars.

    See also: [Fast RNN Cell (Zhang et al., 2019)](https://arxiv.org/abs/1901.02358).

    Args:
        input_size (int):  Number of features in the input :math:`\mathbf{x}(t)`.
        hidden_size (int): Number of features in the hidden state :math:`\mathbf{h}(t)`.
        bias (bool, optional): If ``False``, disables both biases
            :math:`\mathbf{b}_{ih}` and :math:`\mathbf{b}_{hh}`.
            Default: ``True``.
        nonlinearity (Callable, optional): Activation function :math:`\phi` for the
            candidate. Default: ``torch.tanh``.
        kernel_init (Callable, optional): Initializer for :math:`\mathbf{W}_{ih}`.
            Default: ``nn.init.xavier_uniform_``.
        recurrent_kernel_init (Callable, optional): Initializer for :math:`\mathbf{W}_{hh}`.
            Default: ``nn.init.xavier_uniform_``.
        bias_init (Callable, optional): Initializer for :math:`\mathbf{b}_{ih}` when
            `bias=True`. Default: ``nn.init.zeros_``.
        recurrent_bias_init (Callable, optional): Initializer for :math:`\mathbf{b}_{hh}`
            when `bias=True`. Default: ``nn.init.zeros_``.
        alpha_init (float, optional): Initial value for the learnable scalar :math:`\alpha`.
            Default: ``3.0``.
        beta_init (float, optional):  Initial value for the learnable scalar :math:`\beta`.
            Default: ``-3.0``.
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
        weight_ih (Tensor): Learnable input-to-hidden weights, shape
            `(hidden_size, input_size)`.
        weight_hh (Tensor): Learnable hidden-to-hidden weights, shape
            `(hidden_size, hidden_size)`.
        bias_ih (Tensor): Learnable input bias, shape `(hidden_size,)` if `bias=True`.
        bias_hh (Tensor): Learnable hidden bias, shape `(hidden_size,)` if `bias=True`.
        alpha (Tensor):  Learnable scalar gating coefficient α, shape `(1,)`.
        beta (Tensor):   Learnable scalar gating coefficient β, shape `(1,)`.

    Examples::
        >>> cell = FastRNNCell(10, 20)
        >>> x = torch.randn(5, 10)
        >>> hx = torch.zeros(20)
        >>> outputs = []
        >>> for t in range(x.size(0)):
        ...     hx = cell(x[t], hx)
        ...     outputs.append(hx)
    """

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
                "bias_hh": ((hidden_size,), bias),
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
    r"""A Fast gated recurrent neural network cell.

    This cell computes a sigmoid gate and a tanh candidate, then
    combines them with learnable scalars ζ and ν:

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

    See also: [Fast GRNN (Zhang et al., 2019)](https://arxiv.org/abs/1901.02358).

    Args:
        input_size (int):  Number of features in the input :math:`\mathbf{x}(t)`.
        hidden_size (int): Number of features in the hidden state :math:`\mathbf{h}(t)`.
        bias (bool, optional): If ``False``, disables both bias vectors
            :math:`\mathbf{b}_{ih}` and :math:`\mathbf{b}_{hh}`. Default: ``True``.
        nonlinearity (Callable, optional): Activation for the gate :math:`\mathbf{z}`.
            Default: ``torch.tanh``.
        kernel_init (Callable, optional): Initializer for :math:`\mathbf{W}_{ih}`.
            Default: ``nn.init.xavier_uniform_``.
        recurrent_kernel_init (Callable, optional): Initializer for :math:`\mathbf{W}_{hh}`.
            Default: ``nn.init.xavier_uniform_``.
        bias_init (Callable, optional): Initializer for input biases
            :math:`\mathbf{b}_{ih}^{*}` when `bias=True`. Default: ``nn.init.zeros_``.
        recurrent_bias_init (Callable, optional): Initializer for hidden biases
            :math:`\mathbf{b}_{hh}^{*}` when `bias=True`. Default: ``nn.init.zeros_``.
        zeta_init (float, optional): Initial value for scalar gate ζ. Default: ``3.0``.
        nu_init (float, optional):  Initial value for scalar gate ν. Default: ``-3.0``.
        device (torch.device, optional): Device of the parameters. Default: CPU.
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
        - input:  :math:`(N, H_{\mathrm{in}})` or :math:`(H_{\mathrm{in}})`.
        - hidden: :math:`(N, H_{\mathrm{out}})` or :math:`(H_{\mathrm{out}})`.
        - output: :math:`(N, H_{\mathrm{out}})` or :math:`(H_{\mathrm{out}})`.

    Attributes:
        weight_ih (Tensor): input‐to‐hidden weights, shape `(hidden_size, input_size)`.
        weight_hh (Tensor): hidden‐to‐hidden weights, shape `(hidden_size, hidden_size)`.
        bias_ih (Tensor): input biases, shape `(2*hidden_size,)`
            if `bias=True` (split into z & h).
        bias_hh (Tensor): hidden biases, shape `(2*hidden_size,)`
            if `bias=True` (split into z & h).
        zeta (Tensor): scalar gate ζ, shape `(1,)`.
        nu (Tensor):    scalar gate ν, shape `(1,)`.
        t_ones (Tensor): constant ones vector, shape `(hidden_size,)`.

    Examples::
        >>> cell = FastGRNNCell(10, 20)
        >>> x = torch.randn(5, 10)
        >>> hx = torch.zeros(20)
        >>> outputs = []
        >>> for t in range(x.size(0)):
        ...     hx = cell(x[t], hx)
        ...     outputs.append(hx)
    """

    weight_ih: Tensor
    weight_hh: Tensor
    bias_ih: Tensor
    bias_hh: Tensor
    zeta: Tensor
    nu: Tensor
    t_ones: Tensor

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
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
                "bias_hh": ((2 * hidden_size,), bias),
                "zeta": ((1,), True),
                "nu": ((1,), True),
                "t_ones": ((hidden_size,), False),
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
        new_state = (
            self.zeta * (self.t_ones - gate) + self.nu
        ) * candidate_state + gate * state

        if not is_batched:
            new_state = new_state.squeeze(0)

        return new_state

import torch
from torch import Tensor
import torch.nn as nn
from typing import Callable, Optional, Union, Tuple
from ..base import BaseSingleRecurrentLayer, BaseSingleRecurrentCell


class AntisymmetricRNN(BaseSingleRecurrentLayer):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        batch_first: bool = False,
        **kwargs,
    ):
        super(AntisymmetricRNN, self).__init__(
            input_size, hidden_size, num_layers, dropout, batch_first
        )
        self.initialize_cells(AntisymmetricRNNCell, **kwargs)


class AntisymmetricRNNCell(BaseSingleRecurrentCell):
    r"""An Antisymmetric recurrent neural network (RNN) cell.

    This cell implements the update rule from
    “AntisymmetricRNN: A Dynamical System View on Recurrent Neural Networks”
    <https://arxiv.org/abs/1902.09689>_.

    .. math::

        \mathbf{h}(t) = \mathbf{h}(t-1)
            + \epsilon \cdot \tanh\Bigl(
                \mathbf{W}_{ih}\,\mathbf{x}(t) + \mathbf{b}_{ih}
                + \bigl(\mathbf{W}_{hh} - \mathbf{W}_{hh}^\top
                    - \gamma\,\mathbf{I}\bigr)\,\mathbf{h}(t-1)
                + \mathbf{b}_{hh}\Bigr)

    where :math:`\epsilon` (epsilon) controls the step size, and
    :math:`\gamma` (gamma) is a stability factor.

    Args:
        input_size (int):  Number of expected features in the input `inp`.
        hidden_size (int): Number of features in the hidden state.
        bias (bool):       If False, the layer does not use bias weights.
                            Default: True.
        nonlinearity (Callable): Activation function (default: `torch.tanh`).
        kernel_init (Callable):   Initializer for input‐to‐hidden weights
                                    (default: `nn.init.xavier_uniform_`).
        recurrent_kernel_init (Callable):
                                    Initializer for hidden‐to‐hidden weights
                                    (default: `nn.init.normal_`).
        bias_init (Callable):     Initializer for input bias
                                    (default: `nn.init.zeros_`).
        recurrent_bias_init (Callable):
                                    Initializer for hidden bias
                                    (default: `nn.init.zeros_`).
        epsilon (float):  Step‐size multiplier for the update.
                            Default: 1.0.
        gamma (float):    Damping term subtracted along the diagonal
                            for stability. Default: 0.0.
        device (torch.device, optional): Device for weights.
        dtype (torch.dtype, optional):   Data type for weights.

    Inputs:
        - **inp** (Tensor):
            shape `(batch, input_size)` or `(input_size,)`
        - **state** (Tensor or Tuple[Tensor, ...], optional):
            Previous hidden state of shape `(batch, hidden_size)` or
            `(hidden_size,)`. If not provided, initialized to zeros.

    Outputs:
        - **new_state** (Tensor):
            Next hidden state of shape `(batch, hidden_size)` or
            `(hidden_size,)`.

    Attributes:
        weight_ih (Tensor): Learnable input‐to‐hidden weights,
                                shape `(hidden_size, input_size)`.
        weight_hh (Tensor): Learnable hidden‐to‐hidden weights,
                                shape `(hidden_size, hidden_size)`.
        bias_ih   (Tensor): Learnable input bias,
                                shape `(hidden_size,)`.
        bias_hh   (Tensor): Learnable hidden bias,
                                shape `(hidden_size,)`.

    .. note::
        This cell enforces antisymmetry by subtracting the transpose of
        the recurrent weight matrix and a scaled identity term.

    Examples::
        >>> cell = AntisymmetricRNNCell(10, 20, epsilon=0.5, gamma=0.1)
        >>> x = torch.randn(5, 10)   # batch=5, input_size=10
        >>> h0 = torch.zeros(5, 20)  # batch=5, hidden_size=20
        >>> h1 = cell(x, h0)
    """

    __constants__ = [
        "input_size",
        "hidden_size",
        "bias",
        "nonlinearity",
        "kernel_init",
        "recurrent_kernel_init",
        "bias_init",
        "recurrent_bias_init",
        "epsilon",
        "gamma",
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
        nonlinearity: Callable = torch.tanh,
        kernel_init: Callable = nn.init.xavier_uniform_,
        recurrent_kernel_init: Callable = nn.init.normal_,
        bias_init: Callable = nn.init.zeros_,
        recurrent_bias_init: Callable = nn.init.zeros_,
        epsilon: float = 1.0,
        gamma: float = 0.0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super(AntisymmetricRNNCell, self).__init__(
            input_size, hidden_size, bias, device=device, dtype=dtype
        )
        self.nonlinearity = nonlinearity
        self.kernel_init = kernel_init
        self.recurrent_kernel_init = recurrent_kernel_init
        self.bias_init = bias_init
        self.recurrent_bias_init = recurrent_bias_init
        self.epsilon = epsilon
        self.gamma = gamma

        self._default_register_tensors(input_size, hidden_size, bias=bias)
        self.init_weights()

    def forward(
        self, inp: Tensor, state: Optional[Union[Tensor, Tuple[Tensor, ...]]] = None
    ) -> Tensor:
        state = self._check_state(state)
        self._validate_input(inp)
        self._validate_state(state)
        inp, state, is_batched = self._preprocess_input_and_state(inp, state)

        recurrent_matrix = _compute_asym(self.weight_hh, self.gamma)
        pre_act = (
            inp @ self.weight_ih.t()
            + self.bias_ih
            + state @ recurrent_matrix.t()
            + self.bias_hh
        )
        new_state = state + self.epsilon * self.nonlinearity(pre_act)

        if not is_batched:
            new_state = new_state.squeeze(0)

        return new_state


class GatedAntisymmetricRNN(BaseSingleRecurrentLayer):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        batch_first: bool = False,
        **kwargs,
    ):
        super(GatedAntisymmetricRNN, self).__init__(
            input_size, hidden_size, num_layers, dropout, batch_first
        )
        self.initialize_cells(GatedAntisymmetricRNNCell, **kwargs)


class GatedAntisymmetricRNNCell(BaseSingleRecurrentCell):
    r"""A gated Antisymmetric RNN cell.

    Implements the gated update from
    “AntisymmetricRNN: A Dynamical System View on Recurrent Neural Networks”
    <https://arxiv.org/abs/1902.09689>_.

    .. math::

        \begin{aligned}
            \mathbf{z}(t) &= \sigma\Bigl(
            (\mathbf{W}_{hh} - \mathbf{W}_{hh}^\top - \gamma\,\mathbf{I})
            \,\mathbf{h}(t-1) + \mathbf{b}_{hh}
            + \mathbf{W}_{ih}^z\,\mathbf{x}(t)
            + \mathbf{b}_{ih}^z \Bigr), \\
            \mathbf{h}(t) &= \mathbf{h}(t-1)
            + \epsilon \,\mathbf{z}(t)\,\circ\,
            \tanh\Bigl(
                (\mathbf{W}_{hh} - \mathbf{W}_{hh}^\top - \gamma\,\mathbf{I})
                \,\mathbf{h}(t-1) + \mathbf{b}_{hh}
                + \mathbf{W}_{ih}^x\,\mathbf{x}(t)
                + \mathbf{b}_{ih}^h \Bigr)
        \end{aligned}

    where :math:`\epsilon` controls the integration step size,
    :math:`\gamma` is a stability damping, and :math:`\circ` is element-wise product.

    Args:
        input_size (int):        Number of expected features in the input `inp`.
        hidden_size (int):       Number of features in the hidden state.
        bias (bool):             If False, no bias terms are used. Default: True.
        nonlinearity (Callable): Activation to apply (default: `torch.tanh`).
        kernel_init (Callable):   Initializer for input‐to‐hidden weights
                                    (default: `nn.init.xavier_uniform_`).
        recurrent_kernel_init (Callable):
                                    Initializer for hidden‐to‐hidden weights
                                    (default: `nn.init.normal_`).
        bias_init (Callable):     Initializer for input biases
                                    (default: `nn.init.zeros_`).
        epsilon (float):          Step‐size multiplier for state update.
                                    Default: 1.0.
        gamma (float):            Damping term for antisymmetric part.
                                    Default: 0.0.
        device (torch.device, optional): Device on which to place parameters.
        dtype (torch.dtype, optional):   Data type of parameters.

    Inputs:
        - **inp** (Tensor): Input at current time step, of shape
            `(batch, input_size)` or `(input_size,)`.
        - **state** (Tensor or Tuple[Tensor], optional): Previous hidden state,
            of shape `(batch, hidden_size)` or `(hidden_size,)`. Defaults to zeros.

    Outputs:
        - **new_state** (Tensor): Updated hidden state, same shape as `state`.

    Attributes:
        weight_ih (Tensor): Input‐to‐hidden weights, shape `(2*hidden_size, input_size)`.
        weight_hh (Tensor): Hidden‐to‐hidden weights, shape `(hidden_size, hidden_size)`.
        bias_ih   (Tensor): Input bias, shape `(2*hidden_size,)`.
        bias_hh   (Tensor): Hidden bias, shape `(hidden_size,)`.

    .. note::
        This cell splits the input projection into a gate and a candidate,
        then applies an antisymmetric recurrent transformation plus gating
        to ensure stable, expressive dynamics.

    Examples::
        >>> cell = GatedAntisymmetricRNNCell(8, 16, epsilon=0.5, gamma=0.1)
        >>> x = torch.randn(4, 8)    # batch=4, input_size=8
        >>> h0 = torch.zeros(4, 16)  # batch=4, hidden_size=16
        >>> h1 = cell(x, h0)         # new hidden state
    """

    __constants__ = [
        "input_size",
        "hidden_size",
        "bias",
        "kernel_init",
        "recurrent_kernel_init",
        "bias_init",
        "recurrent_bias_init",
        "epsilon",
        "gamma",
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
        recurrent_kernel_init: Callable = nn.init.normal_,
        bias_init: Callable = nn.init.zeros_,
        epsilon: float = 1.0,
        gamma: float = 0.0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super(GatedAntisymmetricRNNCell, self).__init__(
            input_size, hidden_size, bias, device=device, dtype=dtype
        )
        self.kernel_init = kernel_init
        self.recurrent_kernel_init = recurrent_kernel_init
        self.bias_init = bias_init
        self.epsilon = epsilon
        self.gamma = gamma

        self._default_register_tensors(
            input_size, hidden_size, ih_mult=2, hh_mult=1, bias=bias
        )
        self.init_weights()

    def forward(
        self, inp: Tensor, state: Optional[Union[Tensor, Tuple[Tensor, ...]]] = None
    ) -> Tensor:
        state = self._check_state(state)
        self._validate_input(inp)
        self._validate_state(state)
        inp, state, is_batched = self._preprocess_input_and_state(inp, state)

        weights_ih = inp @ self.weight_ih.t() + self.bias_ih
        weight_ih_1, weight_ih_2 = weights_ih.chunk(2, 1)
        recurrent_matrix = _compute_asym(self.weight_hh, self.gamma)
        pre_act = weight_ih_2 + state @ recurrent_matrix.t() + self.bias_hh
        input_gate = torch.sigmoid(weight_ih_1 + state @ recurrent_matrix.t())
        new_state = state + self.epsilon * input_gate * torch.tanh(pre_act)

        if not is_batched:
            new_state = new_state.squeeze(0)

        return new_state


def _compute_asym(weight_hh: Tensor, gamma: float) -> Tensor:
    if weight_hh.dim() != 2 or weight_hh.size(0) != weight_hh.size(1):
        raise ValueError(f"weight_hh must be square, got shape {weight_hh.shape}")
    id_mat = torch.eye(weight_hh.size(0), dtype=weight_hh.dtype, device=weight_hh.device)
    return weight_hh - weight_hh.t() - gamma * id_mat

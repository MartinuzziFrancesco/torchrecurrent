import torch
from torch import Tensor
import torch.nn as nn
from typing import Callable, Optional, Union, Tuple
from ..base import BaseSingleRecurrentLayer, BaseSingleRecurrentCell


class AntisymmetricRNN(BaseSingleRecurrentLayer):
    r"""Stacked antisymmetric RNN layer.

    Wraps :class:`AntisymmetricRNNCell` in a stacked, time-iterated layer
    compatible with the PyTorch RNN/LSTM interface.

    Args:
        input_size (int): Size of each input sample.
        hidden_size (int): Size of the hidden state in each layer.
        num_layers (int, optional): Number of stacked layers. Default: ``1``.
        dropout (float, optional): Dropout probability applied to outputs
            of all but the last layer at each time step. Default: ``0.0``.
        batch_first (bool, optional): If ``True``, input/output are
            ``(N, T, C)``. Default: ``False`` (``(T, N, C)``).
        **kwargs: Passed to each :class:`AntisymmetricRNNCell` (e.g., ``epsilon``,
            ``gamma``, initializers).

    Shape:
        See :class:`BaseSingleRecurrentLayer`.

    See Also:
        :class:`AntisymmetricRNNCell`
    """

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
    r"""An antisymmetric recurrent neural network cell [`arXiv <https://arxiv.org/abs/1902.09689>`_].

        .. math::

            \begin{array}{ll}
            \mathbf{A} = \mathbf{W}_{hh} - \mathbf{W}_{hh}^\top - \gamma \mathbf{I} \\
            h' = h + \varepsilon \,\tanh(\mathbf{W}_{ih} x + \mathbf{b}_{ih} + \mathbf{A} h + \mathbf{b}_{hh})
            \end{array}

        where :math:`\varepsilon` is a step-size scalar, :math:`\gamma \ge 0` adds diagonal damping for stability.

        Args:
            input_size: The number of expected features in the input `x`
            hidden_size: The number of features in the hidden state `h`
            bias: If ``False``, then the layer does not use input-side bias `b_ih`. Default: ``True``
            recurrent_bias: If ``False``, then the layer does not use recurrent bias `b_hh`. Default: ``True``
            nonlinearity: Elementwise nonlinearity applied to the pre-activation. Default: :func:`torch.tanh`
            kernel_init: Initializer for `weight_ih`. Default: :func:`torch.nn.init.xavier_uniform_`
            recurrent_kernel_init: Initializer for `weight_hh`. Default: :func:`torch.nn.init.normal_`
            bias_init: Initializer for `bias_ih`. Default: :func:`torch.nn.init.zeros_`
            recurrent_bias_init: Initializer for `bias_hh`. Default: :func:`torch.nn.init.zeros_`
            epsilon: Step-size multiplier :math:`\varepsilon`. Default: ``1.0``
            gamma: Damping coefficient :math:`\gamma` used in the antisymmetric transform. Default: ``0.0``
            device: The desired device of parameters.
            dtype: The desired floating point type of parameters.

        Inputs: input, h_0
            - **input** of shape `(batch, input_size)` or `(input_size,)`: tensor containing input features
            - **h_0** of shape `(batch, hidden_size)` or `(hidden_size,)`: tensor containing the initial hidden state

            If **h_0** is not provided, it defaults to zero.

        Outputs: h_1
            - **h_1** of shape `(batch, hidden_size)` or `(hidden_size,)`: tensor containing the next hidden state

        Variables:
            weight_ih: the learnable input–hidden weights, of shape `(hidden_size, input_size)`
            weight_hh: the learnable hidden–hidden weights, of shape `(hidden_size, hidden_size)`
            bias_ih: the learnable input–hidden bias, of shape `(hidden_size)`
            bias_hh: the learnable hidden–hidden bias, of shape `(hidden_size)`

        .. note::
            On each call, the antisymmetric transform
            :math:`\mathbf{A} = \mathbf{W}_{hh} - \mathbf{W}_{hh}^\top - \gamma \mathbf{I}`
            is formed and used in the update; gradients flow to :math:`\mathbf{W}_{hh}`.
            By default, parameters are initialized by calling
            ``kernel_init`` / ``recurrent_kernel_init`` for weights and
            ``bias_init`` / ``recurrent_bias_init`` for biases; override these
            or :meth:`init_weights` to customize initialization.

        Examples::

            >>> rnn = AntisymmetricRNNCell(10, 20)  # (input_size, hidden_size)
            >>> x = torch.randn(5, 3, 10)           # (time_steps, batch, input_size)
            >>> h = torch.zeros(3, 20)              # (batch, hidden_size)
            >>> out = []
            >>> for t in range(x.size(0)):
            ...     h = rnn(x[t], h)
            ...     out.append(h)
            >>> out = torch.stack(out, dim=0)       # (time_steps, batch, hidden_size)
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
        recurrent_bias: bool = True,
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
            input_size,
            hidden_size,
            bias,
            recurrent_bias,
            device=device,
            dtype=dtype,
            epsilon=epsilon,
            gamma=gamma,
        )
        self.nonlinearity = nonlinearity
        self.kernel_init = kernel_init
        self.recurrent_kernel_init = recurrent_kernel_init
        self.bias_init = bias_init
        self.recurrent_bias_init = recurrent_bias_init
        self.epsilon = epsilon
        self.gamma = gamma

        self._default_register_tensors(
            input_size, hidden_size, bias=bias, recurrent_bias=recurrent_bias
        )
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
    r"""A gated antisymmetric recurrent neural network (RNN) cell [`arXiv <https://arxiv.org/abs/1902.09689>`_].

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

        where :math:`\epsilon` controls the integration step size, :math:`\gamma` is a stability damping, and :math:`\circ` is element-wise product.

        Args:
            input_size: The number of expected features in the input `x`
            hidden_size: The number of features in the hidden state `h`
            bias: If ``False``, then the layer does not use input-side biases. Default: ``True``
            recurrent_bias: If ``False``, then the layer does not use recurrent bias. Default: ``True``
            nonlinearity: Elementwise nonlinearity applied to the candidate pre-activation. Default: :func:`torch.tanh`
            kernel_init: Initializer for `weight_ih`. Default: :func:`torch.nn.init.xavier_uniform_`
            recurrent_kernel_init: Initializer for `weight_hh`. Default: :func:`torch.nn.init.normal_`
            bias_init: Initializer for input-side biases. Default: :func:`torch.nn.init.zeros_`
            epsilon: Step-size multiplier :math:`\epsilon`. Default: ``1.0``
            gamma: Damping coefficient :math:`\gamma` used in the antisymmetric transform. Default: ``0.0``
            device: The desired device of parameters.
            dtype: The desired floating point type of parameters.

        Inputs: input, h_0
            - **input** of shape `(batch, input_size)` or `(input_size,)`: tensor containing input features
            - **h_0** of shape `(batch, hidden_size)` or `(hidden_size,)`: tensor containing the initial hidden state

            If **h_0** is not provided, it defaults to zero.

        Outputs: h_1
            - **h_1** of shape `(batch, hidden_size)` or `(hidden_size,)`: tensor containing the next hidden state

        Variables:
            weight_ih: the learnable input–hidden weights, of shape `(2*hidden_size, input_size)` (gate and candidate)
            weight_hh: the learnable hidden–hidden weights, of shape `(hidden_size, hidden_size)`
            bias_ih: the learnable input–hidden bias, of shape `(2*hidden_size)`
            bias_hh: the learnable hidden–hidden bias, of shape `(hidden_size)`

        .. note::
            This cell splits the input projection into a gate and a candidate,
            then applies an antisymmetric recurrent transformation plus gating
            to ensure stable, expressive dynamics.

        Examples::

            >>> cell = GatedAntisymmetricRNNCell(8, 16, epsilon=0.5, gamma=0.1)
            >>> x = torch.randn(5, 3, 8)     # (time_steps, batch, input_size)
            >>> h = torch.zeros(3, 16)       # (batch, hidden_size)
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
        recurrent_bias: bool = True,
        kernel_init: Callable = nn.init.xavier_uniform_,
        recurrent_kernel_init: Callable = nn.init.normal_,
        bias_init: Callable = nn.init.zeros_,
        recurrent_bias_init: Callable = nn.init.zeros_,
        epsilon: float = 1.0,
        gamma: float = 0.0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super(GatedAntisymmetricRNNCell, self).__init__(
            input_size, hidden_size, bias, recurrent_bias, device=device, dtype=dtype
        )
        self.kernel_init = kernel_init
        self.recurrent_kernel_init = recurrent_kernel_init
        self.bias_init = bias_init
        self.epsilon = epsilon
        self.gamma = gamma

        self._default_register_tensors(
            input_size,
            hidden_size,
            ih_mult=2,
            hh_mult=1,
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

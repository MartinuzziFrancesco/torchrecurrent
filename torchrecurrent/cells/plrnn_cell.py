import torch
from torch import Tensor
import torch.nn as nn
from typing import Optional, Callable, Union, Tuple
from ..base import BaseSingleRecurrentLayer, BaseSingleRecurrentCell


class PLRNN(BaseSingleRecurrentLayer):
    r"""Multi-layer Piecewise-Linear Recurrent Neural Network (PLRNN).

    [`PLOS Computational Biology <https://doi.org/10.1371/journal.pcbi.1007263>`_]

    Each layer consists of a :class:`PLRNNCell`, which updates the hidden
    state according to:

    .. math::
        \mathbf{h}(t) = \mathbf{A}\,\mathbf{h}(t-1)
            + \mathbf{W}\,\phi\!\bigl(\mathbf{h}(t-1)\bigr)
            + \mathbf{C}\,\mathbf{x}(t)
            + \mathbf{b}

    where :math:`\phi` is the ReLU activation function,
    :math:`\mathbf{A}` is a diagonal matrix of self-connections,
    :math:`\mathbf{W}` is the off-diagonal weight matrix,
    :math:`\mathbf{C}` is the input weight matrix, and
    :math:`\mathbf{b}` is the bias vector.

    Args:
        input_size: The number of expected features in the input ``x``.
        hidden_size: The number of features in the hidden state ``h``.
        num_layers: Number of recurrent layers. Default: 1
        dropout: Dropout probability on outputs of each layer except the last.
            Default: 0
        batch_first: If ``True``, input/output tensors are ``(batch, seq, feature)``.
            Default: False
        bias: If ``False``, the layer does not use bias ``b``. Default: True
        kernel_init: Initializer for ``W``. Default:
            :func:`torch.nn.init.xavier_uniform_`
        recurrent_kernel_init: Initializer for ``A`` (diagonal).
            Default: :func:`torch.nn.init.uniform_`
        bias_init: Initializer for ``b``. Default: :func:`torch.nn.init.zeros_`
        device: The desired device of parameters.
        dtype: The desired floating point type of parameters.

    Examples::

        >>> rnn = PLRNN(10, 20, num_layers=2, dropout=0.1)
        >>> input = torch.randn(5, 3, 10)
        >>> h0 = torch.zeros(2, 3, 20)
        >>> output, hn = rnn(input, h0)
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
        super(PLRNN, self).__init__(
            input_size, hidden_size, num_layers, dropout, batch_first
        )
        self.initialize_cells(PLRNNCell, **kwargs)


class PLRNNCell(BaseSingleRecurrentCell):
    r"""A Piecewise-Linear Recurrent Neural Network (PLRNN) cell.

    [`PLOS Computational Biology <https://doi.org/10.1371/journal.pcbi.1007263>`_]

    .. math::
        \mathbf{h}(t) = \mathbf{A}\,\mathbf{h}(t-1)
            + \mathbf{W}\,\phi\!\bigl(\mathbf{h}(t-1)\bigr)
            + \mathbf{C}\,\mathbf{x}(t)
            + \mathbf{b}

    where :math:`\phi` is the ReLU activation, :math:`\mathbf{A}` is a
    diagonal matrix of self-connections (stored as a vector), :math:`\mathbf{W}`
    is the off-diagonal weight matrix with zero diagonal, :math:`\mathbf{C}` is
    the input weight matrix, and :math:`\mathbf{b}` is the bias.

    Args:
        input_size: The number of expected features in the input ``x``.
        hidden_size: The number of features in the hidden state ``h``.
        bias: If ``False``, the layer does not use bias ``b``. Default: ``True``
        kernel_init: Initializer for ``W``.
            Default: :func:`torch.nn.init.xavier_uniform_`
        recurrent_kernel_init: Initializer for ``A`` diagonal.
            Default: :func:`torch.nn.init.uniform_`
        bias_init: Initializer for ``b``.
            Default: :func:`torch.nn.init.zeros_`
        device: The desired device of parameters.
        dtype: The desired floating point type of parameters.

    Inputs: input, h_0
        - **input** of shape ``(batch, input_size)`` or ``(input_size,)``
        - **h_0** of shape ``(batch, hidden_size)`` or ``(hidden_size,)``

        If ``h_0`` is not provided, it defaults to zeros.

    Outputs: h_1
        - **h_1** of shape ``(batch, hidden_size)`` or ``(hidden_size,)``

    Variables:
        weight_ih: input weight matrix ``C``,
            shape ``(hidden_size, input_size)``
        weight_hh: off-diagonal recurrent weight matrix ``W``,
            shape ``(hidden_size, hidden_size)``
        diagonal: diagonal self-connection vector ``A``,
            shape ``(hidden_size,)``
        bias_ih: bias vector ``b``,
            shape ``(hidden_size,)``

    Examples::

        >>> cell = PLRNNCell(10, 20)
        >>> x = torch.randn(3, 10)
        >>> h = torch.zeros(3, 20)
        >>> h = cell(x, h)
    """

    __constants__ = [
        "input_size",
        "hidden_size",
        "bias",
        "kernel_init",
        "recurrent_kernel_init",
        "bias_init",
    ]

    weight_ih: Tensor
    weight_hh: Tensor
    bias_ih: Tensor

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        kernel_init: Callable = nn.init.xavier_uniform_,
        recurrent_kernel_init: Callable = nn.init.uniform_,
        bias_init: Callable = nn.init.zeros_,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super(PLRNNCell, self).__init__(
            input_size,
            hidden_size,
            bias,
            device=device,
            dtype=dtype,
        )
        self.kernel_init = kernel_init
        self.recurrent_kernel_init = recurrent_kernel_init
        self.bias_init = bias_init

        factory_kwargs = {"device": device, "dtype": dtype}

        # Input weights C: (hidden_size, input_size)
        self.weight_ih = nn.Parameter(
            torch.empty(hidden_size, input_size, **factory_kwargs)
        )
        # Off-diagonal recurrent weights W: (hidden_size, hidden_size)
        self.weight_hh = nn.Parameter(
            torch.empty(hidden_size, hidden_size, **factory_kwargs)
        )
        # Diagonal self-connections A: stored as a vector (hidden_size,)
        self.diagonal = nn.Parameter(
            torch.empty(hidden_size, **factory_kwargs)
        )
        if bias:
            self.bias_ih = nn.Parameter(
                torch.empty(hidden_size, **factory_kwargs)
            )
        else:
            self.register_parameter("bias_ih", None)

        self.init_weights()

    def init_weights(self) -> None:
        """Initialise all weight parameters."""
        self.kernel_init(self.weight_ih)
        self.kernel_init(self.weight_hh)
        # Zero the diagonal of W to enforce off-diagonal structure
        with torch.no_grad():
            self.weight_hh.fill_diagonal_(0.0)
        self.recurrent_kernel_init(self.diagonal.unsqueeze(0))
        if self.bias_ih is not None:
            self.bias_init(self.bias_ih)

    def forward(
        self, inp: Tensor, state: Optional[Union[Tensor, Tuple[Tensor, ...]]] = None
    ) -> Tensor:
        state = self._check_states(state)
        self._validate_input(inp)
        self._validate_states(state)
        inp, state, is_batched = self._preprocess_states(inp, state)

        # h(t) = A * h(t-1) + W * phi(h(t-1)) + C * x(t) + b
        phi_h = torch.relu(state)
        new_state = (
            self.diagonal * state
            + phi_h @ self.weight_hh.t()
            + inp @ self.weight_ih.t()
        )
        if self.bias_ih is not None:
            new_state = new_state + self.bias_ih

        if not is_batched:
            new_state = new_state.squeeze(0)

        return new_state
import torch
from torch import nn
from torch import Tensor
from typing import Optional, Callable, Tuple, Union
from ..base import BaseSingleRecurrentLayer, BaseSingleRecurrentCell


class UGRNN(BaseSingleRecurrentLayer):
    r"""Multi-layer Update Gate Recurrent Neural Network (UGRNN).

    [`arXiv <https://arxiv.org/abs/1611.09913>`_]

    Each layer consists of a :class:`UGRNNCell`, with recurrence defined as:

    .. math::
        \begin{aligned}
            c(t) &= s\!\left(W_{hh}^c h(t-1) + b_{hh}^c
                        + W_{ih}^c x(t) + b_{ih}^c\right), \\
            g(t) &= \sigma\!\left(W_{hh}^g h(t-1) + b_{hh}^g
                        + W_{ih}^g x(t) + b_{ih}^g\right), \\
            h(t) &= g(t) \circ h(t-1) + \left(1 - g(t)\right) \circ c(t),
        \end{aligned}

    where :math:`\sigma` is the sigmoid function, :math:`s` is a pointwise
    nonlinearity (e.g. tanh), and :math:`\circ` denotes element-wise
    multiplication.

    Args:
        input_size: The number of expected features in the input `x`.
        hidden_size: The number of features in the hidden state `h`.
        num_layers: Number of recurrent layers. For example, setting
            ``num_layers=2`` stacks two UGRNN layers, with the second receiving
            the outputs of the first. Default: 1
        dropout: If non-zero, introduces a `Dropout` layer on the outputs of
            each layer except the last, with dropout probability equal to
            :attr:`dropout`. Default: 0
        batch_first: If ``True``, input and output tensors are provided as
            `(batch, seq, feature)` instead of `(seq, batch, feature)`.
            Default: False
        bias: If ``False``, the layer does not use input-side bias `b_{ih}`.
            Default: True
        recurrent_bias: If ``False``, the layer does not use recurrent bias
            `b_{hh}`. Default: True
        kernel_init: Initializer for `W_{ih}`.
            Default: :func:`torch.nn.init.xavier_uniform_`
        recurrent_kernel_init: Initializer for `W_{hh}`.
            Default: :func:`torch.nn.init.xavier_uniform_`
        bias_init: Initializer for `b_{ih}` when ``bias=True``.
            Default: :func:`torch.nn.init.zeros_`
        recurrent_bias_init: Initializer for `b_{hh}` when ``recurrent_bias=True``.
            Default: :func:`torch.nn.init.zeros_`
        device: The desired device of parameters
        dtype: The desired floating point type of parameters

    Inputs: input, h_0
        - **input**: tensor of shape :math:`(L, H_{in})` for unbatched input,
          :math:`(L, N, H_{in})` when ``batch_first=False`` or
          :math:`(N, L, H_{in})` when ``batch_first=True`` containing the
          features of the input sequence.
        - **h_0**: tensor of shape :math:`(\text{num_layers}, H_{out})` for
          unbatched input or :math:`(\text{num_layers}, N, H_{out})` containing
          the initial hidden state. Defaults to zeros if not provided.

        where:

        .. math::
            \begin{aligned}
                N &= \text{batch size} \\
                L &= \text{sequence length} \\
                H_{in} &= \text{input\_size} \\
                H_{out} &= \text{hidden\_size}
            \end{aligned}

    Outputs: output, h_n
        - **output**: tensor of shape :math:`(L, H_{out})` for unbatched input,
          :math:`(L, N, H_{out})` when ``batch_first=False`` or
          :math:`(N, L, H_{out})` when ``batch_first=True`` containing the
          output features from the last layer, for each timestep.
        - **h_n**: tensor of shape :math:`(\text{num_layers}, H_{out})` for
          unbatched input or :math:`(\text{num_layers}, N, H_{out})` containing
          the final hidden state for each element in the sequence.

    Attributes:
        cells.{k}.weight_ih : the learnable input–hidden weights of the
            :math:`k`-th layer, of shape `(2*hidden_size, input_size)` for `k=0`,
            otherwise `(2*hidden_size, hidden_size)`.
        cells.{k}.weight_hh : the learnable hidden–hidden weights of the
            :math:`k`-th layer, of shape `(2*hidden_size, hidden_size)`.
        cells.{k}.bias_ih : the learnable input–hidden bias of the :math:`k`-th
            layer, of shape `(2*hidden_size,)`. Only present when ``bias=True``.
        cells.{k}.bias_hh : the learnable hidden–hidden bias of the :math:`k`-th
            layer, of shape `(2*hidden_size,)`. Only present when ``recurrent_bias=True``.

    .. seealso::
        :class:`UGRNNCell`

    Examples::

        >>> rnn = UGRNN(8, 16, num_layers=2)
        >>> x = torch.randn(5, 3, 8)   # (seq_len, batch, input_size)
        >>> h0 = torch.zeros(2, 3, 16)
        >>> output, hn = rnn(x, h0)
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
        super(UGRNN, self).__init__(
            input_size, hidden_size, num_layers, dropout, batch_first
        )
        self.initialize_cells(UGRNNCell, **kwargs)


class UGRNNCell(BaseSingleRecurrentCell):
    r"""An Update Gate RNN (UGRNN) cell.

    [`arXiv <https://arxiv.org/abs/1611.09913>`_]

    .. math::

        \begin{aligned}
            \mathbf{c}(t) &= s\left( \mathbf{W}_{hh}^c \, \mathbf{h}(t-1) +
                \mathbf{b}_{hh}^c + \mathbf{W}_{xh}^c \, \mathbf{x}(t) +
                \mathbf{b}_{ih}^c \right), \\
                \mathbf{g}(t) &= \sigma\left( \mathbf{W}_{hh}^g \, \mathbf{h}(t-1) +
                \mathbf{b}_{hh}^g + \mathbf{W}_{xh}^g \, \mathbf{x}(t) +
                \mathbf{b}_{ih}^g \right), \\
                \mathbf{h}(t) &= \mathbf{g}(t) \circ \mathbf{h}(t-1) +
                \left( 1 - \mathbf{g}(t) \right) \circ \mathbf{c}(t).
        \end{aligned}

    where :math:`\sigma` is the sigmoid function and
    :math:`\circ` denotes element-wise multiplication.

    Args:
        input_size: Size of the input feature vector
        hidden_size: Size of the hidden state
        bias: If ``False``, disables :math:`\mathbf{b}_{ih}`.
            Default: ``True``
        recurrent_bias: If ``False``, disables :math:`\mathbf{b}_{hh}`.
            Default: ``True``
        kernel_init: Initializer for ``weight_ih``.
            Default: :func:`torch.nn.init.xavier_uniform_`
        recurrent_kernel_init: Initializer for ``weight_hh``.
            Default: :func:`torch.nn.init.xavier_uniform_`
        bias_init: Initializer for ``bias_ih`` when ``bias=True``.
            Default: :func:`torch.nn.init.zeros_`
        recurrent_bias_init: Initializer for ``bias_hh`` when ``recurrent_bias=True``.
            Default: :func:`torch.nn.init.zeros_`
        device: The desired device of parameters
        dtype: The desired floating point type of parameters

    Inputs: input, hidden
        - **input** of shape ``(batch, input_size)`` or ``(input_size,)``:
          tensor containing input features
        - **hidden** of shape ``(batch, hidden_size)`` or ``(hidden_size,)``:
          tensor containing the previous hidden state

        If **hidden** is not provided, it defaults to zero.

    Outputs: h_1
        - **h_1** of shape ``(batch, hidden_size)`` or ``(hidden_size,)``:
          tensor containing the next hidden state

    Variables:
        weight_ih: input–hidden weights, of shape ``(2*hidden_size, input_size)``
        weight_hh: hidden–hidden weights, of shape ``(2*hidden_size, hidden_size)``
        bias_ih: input biases, of shape ``(2*hidden_size,)`` if ``bias=True``
        bias_hh: hidden biases, of shape ``(2*hidden_size,)`` if ``recurrent_bias=True``

    Examples::

        >>> cell = UGRNNCell(8, 16)
        >>> x = torch.randn(5, 8)      # (time, input_size)
        >>> h = torch.zeros(16)        # (hidden_size,)
        >>> outs = []
        >>> for t in range(x.size(0)):
        ...     h = cell(x[t], h)
        ...     outs.append(h)
        >>> outs = torch.stack(outs, dim=0)  # (time, hidden_size)
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
        super(UGRNNCell, self).__init__(
            input_size, hidden_size, bias, recurrent_bias, device=device, dtype=dtype
        )
        self.kernel_init = kernel_init
        self.recurrent_kernel_init = recurrent_kernel_init
        self.bias_init = bias_init
        self.recurrent_bias_init = recurrent_bias_init

        self._default_register_tensors(
            input_size,
            hidden_size,
            ih_mult=2,
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

        gates = (
            inp @ self.weight_ih.t()
            + self.bias_ih
            + state @ self.weight_hh.t()
            + self.bias_hh
        )
        gate1, gate2 = gates.chunk(2, 1)

        candidate_state = torch.tanh(gate1)
        update_gate = torch.sigmoid(gate2)
        new_state = update_gate * state + (1.0 - update_gate) * candidate_state

        if not is_batched:
            new_state = new_state.squeeze(0)

        return new_state

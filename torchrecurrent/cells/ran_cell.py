import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Callable, Tuple, Union
from ..base import BaseDoubleRecurrentLayer, BaseDoubleRecurrentCell


class RAN(BaseDoubleRecurrentLayer):
    r"""Multi-layer Recurrent Additive Network (RAN).

    [`arXiv <https://arxiv.org/pdf/1705.07393>`_]

    Each layer consists of a :class:`RANCell`, which replaces
    the standard LSTM nonlinearities with purely additive memory
    updates gated by input and forget gates:

    .. math::
        \begin{aligned}
            \tilde{c}(t) &= W_{ih}^c x(t) + b_{ih}^c, \\
            i(t) &= \sigma(W_{ih}^i x(t) + b_{ih}^i
                    + W_{hh}^i h(t-1) + b_{hh}^i), \\
            f(t) &= \sigma(W_{ih}^f x(t) + b_{ih}^f
                    + W_{hh}^f h(t-1) + b_{hh}^f), \\
            c(t) &= i(t) \circ \tilde{c}(t) + f(t) \circ c(t-1), \\
            h(t) &= \tanh(c(t)).
        \end{aligned}

    Args:
        input_size: The number of expected features in the input `x`.
        hidden_size: The number of features in the hidden and cell states.
        num_layers: Number of recurrent layers. E.g., setting ``num_layers=2``
            stacks two RAN layers, with the second receiving the outputs
            of the first. Default: 1
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
        device: The desired device of parameters.
        dtype: The desired floating point type of parameters.

    Inputs: input, (h_0, c_0)
        - **input**: tensor of shape :math:`(L, H_{in})` for unbatched input,
          :math:`(L, N, H_{in})` when ``batch_first=False`` or
          :math:`(N, L, H_{in})` when ``batch_first=True`` containing the
          features of the input sequence.
        - **h_0**: tensor of shape :math:`(\text{num_layers}, H_{out})` for
          unbatched input or :math:`(\text{num_layers}, N, H_{out})`
          containing the initial hidden state. Defaults to zeros if not provided.
        - **c_0**: tensor of shape :math:`(\text{num_layers}, H_{out})` for
          unbatched input or :math:`(\text{num_layers}, N, H_{out})`
          containing the initial cell state. Defaults to zeros if not provided.

        where:

        .. math::
            \begin{aligned}
                N &= \text{batch size} \\
                L &= \text{sequence length} \\
                H_{in} &= \text{input\_size} \\
                H_{out} &= \text{hidden\_size}
            \end{aligned}

    Outputs: output, (h_n, c_n)
        - **output**: tensor of shape :math:`(L, H_{out})` for unbatched input,
          :math:`(L, N, H_{out})` when ``batch_first=False`` or
          :math:`(N, L, H_{out})` when ``batch_first=True`` containing the
          output features from the last layer, for each timestep.
        - **h_n**: tensor of shape :math:`(\text{num_layers}, H_{out})` for
          unbatched input or :math:`(\text{num_layers}, N, H_{out})` containing
          the final hidden state for each element in the sequence.
        - **c_n**: tensor of shape :math:`(\text{num_layers}, H_{out})` for
          unbatched input or :math:`(\text{num_layers}, N, H_{out})` containing
          the final cell state for each element in the sequence.

    Attributes:
        cells.{k}.weight_ih : the learnable input–hidden weights of the
            :math:`k`-th layer, of shape `(3*hidden_size, input_size)` for
            `k=0`, otherwise `(3*hidden_size, hidden_size)`.
        cells.{k}.weight_hh : the learnable hidden–hidden weights of the
            :math:`k`-th layer, of shape `(2*hidden_size, hidden_size)`.
        cells.{k}.bias_ih : the learnable input–hidden biases of the
            :math:`k`-th layer, of shape `(3*hidden_size)`. Only present when
            ``bias=True``.
        cells.{k}.bias_hh : the learnable hidden–hidden biases of the
            :math:`k`-th layer, of shape `(2*hidden_size)`. Only present when
            ``recurrent_bias=True``.

    .. seealso::
        :class:`RANCell`

    Examples::

        >>> rnn = RAN(16, 32, num_layers=2)
        >>> x = torch.randn(5, 3, 16)   # (seq_len, batch, input_size)
        >>> h0 = torch.zeros(2, 3, 32)
        >>> c0 = torch.zeros(2, 3, 32)
        >>> output, (hn, cn) = rnn(x, (h0, c0))
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
        super(RAN, self).__init__(input_size, hidden_size, num_layers, dropout, batch_first)
        self.initialize_cells(RANCell, **kwargs)


class RANCell(BaseDoubleRecurrentCell):
    r"""A Recurrent Additive Network (RAN) cell.

    [`arXiv <https://arxiv.org/pdf/1705.07393>`_]

    .. math::

        \begin{aligned}
            \tilde{\mathbf{c}}(t) &= \mathbf{W}_{ih}^{c}\,\mathbf{x}(t)
                + \mathbf{b}_{ih}^{c}, \\
            \mathbf{i}(t) &= \sigma\bigl(
                \mathbf{W}_{ih}^{i}\,\mathbf{x}(t) + \mathbf{b}_{ih}^{i}
                + \mathbf{W}_{hh}^{i}\,\mathbf{h}(t-1) + \mathbf{b}_{hh}^{i}
            \bigr), \\
            \mathbf{f}(t) &= \sigma\bigl(
                \mathbf{W}_{ih}^{f}\,\mathbf{x}(t) + \mathbf{b}_{ih}^{f}
                + \mathbf{W}_{hh}^{f}\,\mathbf{h}(t-1) + \mathbf{b}_{hh}^{f}
            \bigr), \\
            \mathbf{c}(t) &= \mathbf{i}(t)\circ\tilde{\mathbf{c}}(t)
                + \mathbf{f}(t)\circ\mathbf{c}(t-1), \\
            \mathbf{h}(t) &= \tanh\bigl(\mathbf{c}(t)\bigr)
        \end{aligned}

    where :math:`\circ` denotes element-wise multiplication and
    :math:`\sigma` is the sigmoid function.

    Args:
        input_size: The number of expected features in the input ``x``.
        hidden_size: The number of features in the hidden and cell states.
        bias: If ``False``, the layer does not use input-side bias ``b_{ih}``.
            Default: ``True``.
        recurrent_bias: If ``False``, the layer does not use recurrent bias
            ``b_{hh}``. Default: ``True``.
        kernel_init: Initializer for ``weight_{ih}``.
            Default: :func:`torch.nn.init.xavier_uniform_`.
        recurrent_kernel_init: Initializer for ``weight_{hh}``.
            Default: :func:`torch.nn.init.xavier_uniform_`.
        bias_init: Initializer for ``bias_{ih}`` when ``bias=True``.
            Default: :func:`torch.nn.init.zeros_`.
        recurrent_bias_init: Initializer for ``bias_{hh}`` when
            ``recurrent_bias=True``. Default: :func:`torch.nn.init.zeros_`.
        device: The desired device of parameters.
        dtype: The desired floating point type of parameters.

    Inputs: input, (h_0, c_0)
        - **input** of shape ``(batch, input_size)`` or ``(input_size,)``:
          Tensor containing input features.
        - **h_0** of shape ``(batch, hidden_size)`` or ``(hidden_size,)``:
          Tensor containing the initial hidden state.
        - **c_0** of shape ``(batch, hidden_size)`` or ``(hidden_size,)``:
          Tensor containing the initial cell state.

        If **(h_0, c_0)** is not provided, both default to zero.

    Outputs: (h_1, c_1)
        - **h_1** of shape ``(batch, hidden_size)`` or ``(hidden_size,)``:
          Tensor containing the next hidden state.
        - **c_1** of shape ``(batch, hidden_size)`` or ``(hidden_size,)``:
          Tensor containing the next cell state.

    Variables:
        weight_ih: The learnable input–hidden weights for content, input,
            and forget, of shape ``(3*hidden_size, input_size)``.
        weight_hh: The learnable hidden–hidden weights for input and forget
            gates, of shape ``(2*hidden_size, hidden_size)``.
        bias_ih: The learnable input biases for input and forget gates,
            of shape ``(2*hidden_size)`` if ``bias=True``.
        bias_hh: The learnable hidden biases for input and forget gates,
            of shape ``(2*hidden_size)`` if ``recurrent_bias=True``.

    Examples::

        >>> cell = RANCell(16, 32)
        >>> x = torch.randn(5, 16)      # (time_steps, input_size)
        >>> h = torch.zeros(32)         # (hidden_size,)
        >>> c = torch.zeros(32)         # (hidden_size,)
        >>> outs = []
        >>> for t in range(x.size(0)):
        ...     h, c = cell(x[t], (h, c))
        ...     outs.append(h)
        >>> outs = torch.stack(outs, dim=0)  # (time_steps, hidden_size)
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
        super(RANCell, self).__init__(
            input_size, hidden_size, bias, recurrent_bias, device=device, dtype=dtype
        )
        self.kernel_init = kernel_init
        self.recurrent_kernel_init = recurrent_kernel_init
        self.bias_init = bias_init
        self.recurrent_bias_init = recurrent_bias_init

        self._register_tensors(
            {
                "weight_ih": ((3 * hidden_size, input_size), True),
                "weight_hh": ((2 * hidden_size, hidden_size), True),
                "bias_ih": ((2 * hidden_size,), bias),
                "bias_hh": ((2 * hidden_size,), recurrent_bias),
            }
        )
        self.init_weights()

    def forward(
        self, inp: Tensor, state: Optional[Union[Tensor, Tuple[Tensor, ...]]] = None
    ) -> Tuple[Tensor, Tensor]:
        state, c_state = self._check_states(state)
        self._validate_input(inp)
        self._validate_states((state, c_state))
        inp, state, c_state, is_batched = self._preprocess_states(inp, (state, c_state))

        weight_ih_c, weight_ih_i, weight_ih_f = self.weight_ih.chunk(3, 0)
        weight_hh_i, weight_hh_f = self.weight_hh.chunk(2, 0)
        bias_ih_i, bias_ih_f = self.bias_ih.chunk(2, 0)
        bias_hh_i, bias_hh_f = self.bias_hh.chunk(2, 0)

        content_layer = inp @ weight_ih_c.t()
        ig = inp @ weight_ih_i.t() + bias_ih_i + state @ weight_hh_i.t() + bias_hh_i
        fg = inp @ weight_ih_f.t() + bias_ih_f + state @ weight_hh_f.t() + bias_hh_f
        input_gate = torch.sigmoid(ig)
        forget_gate = torch.sigmoid(fg)
        new_cstate = input_gate * content_layer + forget_gate * c_state
        new_state = torch.tanh(new_cstate)

        if not is_batched:
            new_state = new_state.squeeze(0)
            new_cstate = new_cstate.squeeze(0)

        return new_state, new_cstate

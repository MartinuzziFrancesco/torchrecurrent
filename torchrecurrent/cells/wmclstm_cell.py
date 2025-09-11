import torch
from torch import Tensor
import torch.nn as nn
from typing import Optional, Callable, Union, Tuple
from ..base import BaseDoubleRecurrentLayer, BaseDoubleRecurrentCell


class WMCLSTM(BaseDoubleRecurrentLayer):
    r"""Multi-layer LSTM with working-memory connections (WMCLSTM).

    [`arXiv <https://arxiv.org/abs/2109.00020>`_]

    Each layer consists of a :class:`WMCLSTMCell`, an LSTM variant where the
    input, forget, and output gates receive additional connections from the
    cell state. The updates are:

    .. math::
        \begin{aligned}
            i(t) &= \sigma\bigl(W_{ih}^i x(t) + b_{ih}^i
                + W_{hh}^i h(t-1) + b_{hh}^i
                + W_{mh}^i c(t-1) + b_{mh}^i \bigr), \\
            f(t) &= \sigma\bigl(W_{ih}^f x(t) + b_{ih}^f
                + W_{hh}^f h(t-1) + b_{hh}^f
                + W_{mh}^f c(t-1) + b_{mh}^f \bigr), \\
            c(t) &= f(t) \circ c(t-1)
                + i(t) \circ \sigma_c(W_{ih}^c x(t) + b_{ih}^c), \\
            o(t) &= \sigma\bigl(W_{ih}^o x(t) + b_{ih}^o
                + W_{hh}^o h(t-1) + b_{hh}^o
                + W_{mh}^o c(t) + b_{mh}^o \bigr), \\
            h(t) &= o(t) \circ \sigma_h(c(t)),
        \end{aligned}

    where :math:`\sigma` is the sigmoid, :math:`\sigma_c` / :math:`\sigma_h`
    are cell/output activations (usually :func:`torch.tanh`), and
    :math:`\circ` is the elementwise product.

    Args:
        input_size: Number of expected features in the input `x`.
        hidden_size: Number of features in the hidden and cell states.
        num_layers: Number of stacked recurrent layers. Default: 1
        dropout: If non-zero, adds dropout after each layer (except last).
            Default: 0
        batch_first: If ``True``, inputs and outputs are
            `(batch, seq, feature)` instead of `(seq, batch, feature)`.
            Default: False
        bias: If ``False``, disables input biases `b_{ih}`. Default: True
        recurrent_bias: If ``False``, disables recurrent biases `b_{hh}`. Default: True
        memory_bias: If ``False``, disables memory biases `b_{mh}`. Default: True
        kernel_init: Initializer for `W_{ih}`.
            Default: :func:`torch.nn.init.xavier_uniform_`
        recurrent_kernel_init: Initializer for `W_{hh}`.
            Default: :func:`torch.nn.init.xavier_uniform_`
        memory_kernel_init: Initializer for `W_{mh}`.
            Default: :func:`torch.nn.init.xavier_uniform_`
        bias_init: Initializer for `b_{ih}`.
            Default: :func:`torch.nn.init.zeros_`
        recurrent_bias_init: Initializer for `b_{hh}`.
            Default: :func:`torch.nn.init.zeros_`
        memory_bias_init: Initializer for `b_{mh}`.
            Default: :func:`torch.nn.init.zeros_`
        device: Desired device of parameters.
        dtype: Desired floating point type of parameters.

    Inputs: input, (h_0, c_0)
        - **input**: tensor of shape `(L, H_in)` for unbatched input,
          `(L, N, H_in)` when ``batch_first=False``, or `(N, L, H_in)` when
          ``batch_first=True``.
        - **h_0**: tensor of shape `(num_layers, H_out)` (unbatched) or
          `(num_layers, N, H_out)` containing initial hidden state. Defaults to
          zeros if not provided.
        - **c_0**: tensor of same shape as `h_0`, containing initial cell state.
          Defaults to zeros if not provided.

        Where:

        .. math::
            \begin{aligned}
                N &= \text{batch size} \\
                L &= \text{sequence length} \\
                H_{in} &= \text{input size} \\
                H_{out} &= \text{hidden size}
            \end{aligned}

    Outputs: output, (h_n, c_n)
        - **output**: tensor of shape `(L, H_out)` for unbatched input,
          `(L, N, H_out)` when ``batch_first=False``, or `(N, L, H_out)` when
          ``batch_first=True`` containing hidden states from the last layer at
          each timestep.
        - **h_n**: final hidden state for each layer,
          shape `(num_layers, H_out)` (unbatched) or `(num_layers, N, H_out)`.
        - **c_n**: final cell state for each layer, same shape as `h_n`.

    Attributes:
        cells.{k}.weight_ih : input–hidden weights of the :math:`k`-th layer,
            shape `(4*hidden_size, input_size)` for `k=0`, otherwise
            `(4*hidden_size, hidden_size)`.
        cells.{k}.weight_hh : hidden–hidden weights of the :math:`k`-th layer,
            shape `(4*hidden_size, hidden_size)` (i, f, o gates).
        cells.{k}.weight_mh : memory–hidden weights of the :math:`k`-th layer,
            shape `(3*hidden_size, hidden_size)` (i, f depend on `c(t-1)`, o on `c(t)`).
        cells.{k}.bias_ih : input biases of the :math:`k`-th layer,
            shape `(4*hidden_size,)` if ``bias=True``.
        cells.{k}.bias_hh : hidden biases of the :math:`k`-th layer,
            shape `(4*hidden_size,)` if ``recurrent_bias=True``.
        cells.{k}.bias_mh : memory biases of the :math:`k`-th layer,
            shape `(3*hidden_size,)` if ``memory_bias=True``.

    .. seealso::
        :class:`WMCLSTMCell`

    Examples::

        >>> rnn = WMCLSTM(8, 16, num_layers=2)
        >>> x = torch.randn(5, 3, 8)    # (seq_len, batch, input_size)
        >>> h0 = torch.zeros(2, 3, 16)
        >>> c0 = torch.zeros(2, 3, 16)
        >>> out, (hn, cn) = rnn(x, (h0, c0))
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
        super(WMCLSTM, self).__init__(
            input_size, hidden_size, num_layers, dropout, batch_first
        )
        self.initialize_cells(WMCLSTMCell, **kwargs)


class WMCLSTMCell(BaseDoubleRecurrentCell):
    r"""A Long Short-Term Memory (LSTM) cell with working-memory connections.

    [`arXiv <https://arxiv.org/abs/2109.00020>`_].

    .. math::

        \begin{aligned}
            \mathbf{i}(t) &= \sigma\Bigl(
                \mathbf{W}_{ih}^{i}\,\mathbf{x}(t) + \mathbf{b}_{ih}^{i}
                + \mathbf{W}_{hh}^{i}\,\mathbf{h}(t-1) + \mathbf{b}_{hh}^{i}
                + \mathbf{W}_{mh}^{i}\,\mathbf{c}(t-1) + \mathbf{b}_{mh}^{i}
            \Bigr), \\
            \mathbf{f}(t) &= \sigma\Bigl(
                \mathbf{W}_{ih}^{f}\,\mathbf{x}(t) + \mathbf{b}_{ih}^{f}
                + \mathbf{W}_{hh}^{f}\,\mathbf{h}(t-1) + \mathbf{b}_{hh}^{f}
                + \mathbf{W}_{mh}^{f}\,\mathbf{c}(t-1) + \mathbf{b}_{mh}^{f}
            \Bigr), \\
            \mathbf{c}(t) &= \mathbf{f}(t)\circ\mathbf{c}(t-1)
                + \mathbf{i}(t)\circ\sigma_c\Bigl(
                    \mathbf{W}_{ih}^{c}\,\mathbf{x}(t) + \mathbf{b}_{ih}^{c}
                \Bigr), \\
            \mathbf{o}(t) &= \sigma\Bigl(
                \mathbf{W}_{ih}^{o}\,\mathbf{x}(t) + \mathbf{b}_{ih}^{o}
                + \mathbf{W}_{hh}^{o}\,\mathbf{h}(t-1) + \mathbf{b}_{hh}^{o}
                + \mathbf{W}_{mh}^{o}\,\mathbf{c}(t) + \mathbf{b}_{mh}^{o}
            \Bigr), \\
            \mathbf{h}(t) &= \mathbf{o}(t)\circ\sigma_h\bigl(\mathbf{c}(t)\bigr),
        \end{aligned}

    where :math:`\sigma` is the sigmoid, :math:`\sigma_c` and :math:`\sigma_h`
    are cell/output activations (typically :func:`torch.tanh`), and
    :math:`\circ` is element‐wise (Hadamard) multiplication.

    Args:
        input_size: The number of expected features in the input ``x``.
        hidden_size: The number of features in the hidden and cell states.
        bias: If ``False``, the layer does not use input biases
            ``b_{ih}``. Default: ``True``.
        recurrent_bias: If ``False``, the layer does not use recurrent
            biases ``b_{hh}``. Default: ``True``.
        memory_bias: If ``False``, the layer does not use memory biases
            ``b_{mh}``. Default: ``True``.
        kernel_init: Initializer for
            ``W_{ih}^{\{i,f,c,o\}}``. Default:
            :func:`torch.nn.init.xavier_uniform_`.
        recurrent_kernel_init: Initializer for
            ``W_{hh}^{\{i,f,o\}}``. Default:
            :func:`torch.nn.init.xavier_uniform_`.
        memory_kernel_init: Initializer for
            ``W_{mh}^{\{i,f,o\}}``. Default:
            :func:`torch.nn.init.xavier_uniform_`.
        bias_init: Initializer for
            ``b_{ih}^{\{i,f,c,o\}}``. Default:
            :func:`torch.nn.init.zeros_`.
        recurrent_bias_init: Initializer for
            ``b_{hh}^{\{i,f,o\}}``. Default:
            :func:`torch.nn.init.zeros_`.
        memory_bias_init: Initializer for
            ``b_{mh}^{\{i,f,o\}}``. Default:
            :func:`torch.nn.init.zeros_`.
        device: The desired device of parameters.
        dtype: The desired floating point type of parameters.

    Inputs: input, (h, c)
        - **input** of shape ``(batch, input_size)`` or ``(input_size,)``:
          Tensor containing input features.
        - **h** of shape ``(batch, hidden_size)`` or ``(hidden_size,)``:
          Previous hidden state.
        - **c** of shape ``(batch, hidden_size)`` or ``(hidden_size,)``:
          Previous cell state.

        If **(h, c)** is not provided, both default to zeros.

    Outputs: (h_1, c_1)
        - **h_1** of shape ``(batch, hidden_size)`` or ``(hidden_size,)``:
          Next hidden state.
        - **c_1** of shape ``(batch, hidden_size)`` or ``(hidden_size,)``:
          Next cell state.

    Variables:
        weight_ih: The learnable input–hidden weights,
            of shape ``(4*hidden_size, input_size)``.
        weight_hh: The learnable hidden–hidden weights,
            of shape ``(4*hidden_size, hidden_size)`` (for i, f, o gates).
        weight_mh: The learnable memory–hidden weights,
            of shape ``(3*hidden_size, hidden_size)``
            (i, f use :math:`c(t-1)`, o uses :math:`c(t)`).
        bias_ih: The learnable input biases,
            of shape ``(4*hidden_size,)``.
        bias_hh: The learnable hidden biases,
            of shape ``(4*hidden_size,)``.
        bias_mh: The learnable memory biases,
            of shape ``(3*hidden_size,)``.

    Examples::

        >>> cell = WMCLSTMCell(8, 16)
        >>> x = torch.randn(12, 4, 8)     # (time_steps, batch, input_size)
        >>> h = torch.zeros(4, 16)        # (batch, hidden_size)
        >>> c = torch.zeros(4, 16)        # (batch, hidden_size)
        >>> outs = []
        >>> for t in range(x.size(0)):
        ...     h, c = cell(x[t], (h, c))
        ...     outs.append(h)
        >>> outs = torch.stack(outs, dim=0)  # (time_steps, batch, hidden_size)
    """

    __constants__ = [
        "input_size",
        "hidden_size",
        "bias",
        "recurrent_bias",
        "memory_bias",
        "kernel_init",
        "recurrent_kernel_init",
        "memory_kernel_init",
        "bias_init",
        "recurrent_bias_init",
        "memory_bias_init",
    ]

    weight_ih: Tensor
    weight_hh: Tensor
    weight_mh: Tensor
    bias_ih: Tensor
    bias_hh: Tensor
    bias_mh: Tensor

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        recurrent_bias: bool = True,
        memory_bias: bool = True,
        kernel_init: Callable = nn.init.xavier_uniform_,
        recurrent_kernel_init: Callable = nn.init.xavier_uniform_,
        memory_kernel_init: Callable = nn.init.xavier_uniform_,
        bias_init: Callable = nn.init.zeros_,
        recurrent_bias_init: Callable = nn.init.zeros_,
        memory_bias_init: Callable = nn.init.zeros_,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super(WMCLSTMCell, self).__init__(
            input_size, hidden_size, bias, recurrent_bias, device=device, dtype=dtype
        )
        self.kernel_init = kernel_init
        self.recurrent_kernel_init = recurrent_kernel_init
        self.memory_kernel_init = memory_kernel_init
        self.bias_init = bias_init
        self.recurrent_bias_init = recurrent_bias_init
        self.memory_bias_init = memory_bias_init

        self._register_tensors(
            {
                "weight_ih": ((4 * hidden_size, input_size), True),
                "weight_hh": ((4 * hidden_size, hidden_size), True),
                "weight_mh": ((3 * hidden_size, hidden_size), True),
                "bias_ih": ((4 * hidden_size,), bias),
                "bias_hh": ((4 * hidden_size,), recurrent_bias),
                "bias_mh": ((3 * hidden_size,), memory_bias),
            }
        )
        self.init_weights()

    def init_weights(self):
        for name, param in self.named_parameters():
            if "weight_ih" in name:
                self.kernel_init(param)
            elif "weight_hh" in name:
                self.recurrent_kernel_init(param)
            elif "weight_mh" in name:
                self.memory_kernel_init(param)
            elif "bias_ih" in name:
                self.bias_init(param)
            elif "bias_hh" in name:
                self.recurrent_bias_init(param)
            elif "bias_mh" in name:
                self.memory_bias_init(param)

    def forward(
        self, inp: Tensor, state: Optional[Union[Tensor, Tuple[Tensor, ...]]] = None
    ) -> Tuple[Tensor, Tensor]:
        state, c_state = self._check_states(state)
        self._validate_input(inp)
        self._validate_states((state, c_state))
        inp, state, c_state, is_batched = self._preprocess_states(inp, (state, c_state))

        gates = (
            inp @ self.weight_ih.t()
            + self.bias_ih
            + state @ self.weight_hh.t()
            + self.bias_hh
        )
        weight_mh_1, weight_mh_2, weight_mh_3 = self.weight_mh.chunk(3, 0)
        bias_mh_1, bias_mh_2, bias_mh_3 = self.bias_mh.chunk(3, 0)
        input_gate, forget_gate, cell_gate, output_gate = gates.chunk(4, 1)

        new_input_gate = torch.sigmoid(
            input_gate + torch.tanh(c_state @ weight_mh_1.t() + bias_mh_1)
        )
        new_forget_gate = torch.sigmoid(
            forget_gate + torch.tanh(c_state @ weight_mh_2.t() + bias_mh_2)
        )
        new_cell_gate = torch.tanh(cell_gate)
        new_cstate = new_forget_gate * c_state + new_input_gate * new_cell_gate
        memory_gate = new_cstate @ weight_mh_3.t() + bias_mh_3
        new_output_gate = torch.sigmoid(output_gate + torch.tanh(memory_gate))
        new_state = new_output_gate * torch.tanh(new_cstate)

        if not is_batched:
            new_state = new_state.squeeze(0)
            new_cstate = new_cstate.squeeze(0)

        return new_state, new_cstate

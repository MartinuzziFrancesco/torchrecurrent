import torch
from torch import Tensor
import torch.nn as nn
from typing import Optional, Callable, Union, Tuple
from ..base import BaseDoubleRecurrentLayer, BaseDoubleRecurrentCell


class SCRN(BaseDoubleRecurrentLayer):
    r"""Multi-layer Structurally Constrained Recurrent Network (SCRN).

    [`arXiv <https://arxiv.org/pdf/1412.7753>`_]

    Each layer consists of a :class:`SCRNCell`, which augments the recurrent
    dynamics with a slowly changing context state:

    .. math::
        \begin{aligned}
        s(t) &= (1 - \alpha) \bigl(W_{ih}^s x(t) + b_{ih}^s \bigr)
             + \alpha s(t-1), \\
        h(t) &= \sigma\bigl(
            W_{ch}^h s(t) + b_{ch}^h
          + W_{ih}^h x(t) + b_{ih}^h
          + W_{hh}^h h(t-1) + b_{hh}^h
        \bigr).
        \end{aligned}

    Here, :math:`\alpha` is a learned interpolation parameter for the
    context state.

    Args:
        input_size: The number of expected features in the input `x`.
        hidden_size: The number of features in the hidden and context states.
        num_layers: Number of recurrent layers. E.g., setting ``num_layers=2``
            stacks two SCRN layers, with the second receiving the outputs
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
        context_bias: If ``False``, the layer does not use context bias
            `b_{ch}`. Default: True
        kernel_init: Initializer for `W_{ih}`.
            Default: :func:`torch.nn.init.xavier_uniform_`
        recurrent_kernel_init: Initializer for `W_{hh}`.
            Default: :func:`torch.nn.init.xavier_uniform_`
        context_kernel_init: Initializer for `W_{ch}`.
            Default: :func:`torch.nn.init.normal_`
        bias_init: Initializer for `b_{ih}` when ``bias=True``.
            Default: :func:`torch.nn.init.zeros_`
        recurrent_bias_init: Initializer for `b_{hh}` when ``recurrent_bias=True``.
            Default: :func:`torch.nn.init.zeros_`
        context_bias_init: Initializer for `b_{ch}` when ``context_bias=True``.
            Default: :func:`torch.nn.init.zeros_`
        alpha: Initial value for the context interpolation parameter.
            Default: 0.5
        device: The desired device of parameters.
        dtype: The desired floating point type of parameters.

    Inputs: input, (h_0, s_0)
        - **input**: tensor of shape :math:`(L, H_{in})` for unbatched input,
          :math:`(L, N, H_{in})` when ``batch_first=False`` or
          :math:`(N, L, H_{in})` when ``batch_first=True`` containing the
          features of the input sequence.
        - **h_0**: tensor of shape :math:`(\text{num_layers}, H_{out})` for
          unbatched input or :math:`(\text{num_layers}, N, H_{out})`
          containing the initial hidden state. Defaults to zeros if not provided.
        - **s_0**: tensor of shape :math:`(\text{num_layers}, H_{out})` for
          unbatched input or :math:`(\text{num_layers}, N, H_{out})`
          containing the initial context state. Defaults to zeros if not provided.

        where:

        .. math::
            \begin{aligned}
                N &= \text{batch size} \\
                L &= \text{sequence length} \\
                H_{in} &= \text{input\_size} \\
                H_{out} &= \text{hidden\_size}
            \end{aligned}

    Outputs: output, (h_n, s_n)
        - **output**: tensor of shape :math:`(L, H_{out})` for unbatched input,
          :math:`(L, N, H_{out})` when ``batch_first=False`` or
          :math:`(N, L, H_{out})` when ``batch_first=True`` containing the
          output features from the last layer, for each timestep.
        - **h_n**: tensor of shape :math:`(\text{num_layers}, H_{out})` for
          unbatched input or :math:`(\text{num_layers}, N, H_{out})` containing
          the final hidden state for each element in the sequence.
        - **s_n**: tensor of shape :math:`(\text{num_layers}, H_{out})` for
          unbatched input or :math:`(\text{num_layers}, N, H_{out})` containing
          the final context state for each element in the sequence.

    Attributes:
        cells.{k}.weight_ih : the learnable input–hidden weights of the
            :math:`k`-th layer, of shape `(2*hidden_size, input_size)` for
            `k=0`, otherwise `(2*hidden_size, hidden_size)`.
        cells.{k}.weight_hh : the learnable hidden–hidden weights of the
            :math:`k`-th layer, of shape `(2*hidden_size, hidden_size)`.
        cells.{k}.weight_ch : the learnable context–hidden weights of the
            :math:`k`-th layer, of shape `(2*hidden_size, hidden_size)`.
        cells.{k}.bias_ih : the learnable input–hidden biases of the
            :math:`k`-th layer, of shape `(2*hidden_size)`. Only present when
            ``bias=True``.
        cells.{k}.bias_hh : the learnable hidden–hidden biases of the
            :math:`k`-th layer, of shape `(2*hidden_size)`. Only present when
            ``recurrent_bias=True``.
        cells.{k}.bias_ch : the learnable context–hidden biases of the
            :math:`k`-th layer, of shape `(2*hidden_size)`. Only present when
            ``context_bias=True``.
        cells.{k}.alpha : learnable scalar context interpolation parameter
            for the :math:`k`-th layer.

    .. seealso::
        :class:`SCRNCell`

    Examples::

        >>> rnn = SCRN(10, 20, num_layers=2, alpha=0.5)
        >>> x = torch.randn(5, 3, 10)    # (seq_len, batch, input_size)
        >>> h0 = torch.zeros(2, 3, 20)
        >>> s0 = torch.zeros(2, 3, 20)
        >>> output, (hn, sn) = rnn(x, (h0, s0))
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
        super(SCRN, self).__init__(
            input_size, hidden_size, num_layers, dropout, batch_first
        )
        self.initialize_cells(SCRNCell, **kwargs)


class SCRNCell(BaseDoubleRecurrentCell):
    r"""A Structurally Constrained Recurrent Network (SCRN) cell.

    [`arXiv <https://arxiv.org/pdf/1412.7753>`_]

    .. math::

        \begin{aligned}
        \mathbf{s}(t) &= (1 - \alpha)\,\bigl(\mathbf{W}_{ih}^{s}\,\mathbf{x}(t)
            + \mathbf{b}_{ih}^{s}\bigr)
            + \alpha\,\mathbf{s}(t-1), \\
            \mathbf{h}(t) &= \sigma\Bigl(
            \mathbf{W}_{ch}^{h}\,\mathbf{s}(t)
            + \mathbf{b}_{ch}^{h}
            + \mathbf{W}_{ih}^{h}\,\mathbf{x}(t)
            + \mathbf{b}_{ih}^{h}
            + \mathbf{W}_{hh}^{h}\,\mathbf{h}(t-1)
            + \mathbf{b}_{hh}^{h}
        \Bigr), \\
            \mathbf{y}(t) &= f\Bigl(
            \mathbf{W}_{ch}^{y}\,\mathbf{s}(t)
            + \mathbf{b}_{ch}^{y}
            + \mathbf{W}_{hh}^{y}\,\mathbf{h}(t)
            + \mathbf{b}_{hh}^{y}
        \Bigr)
        \end{aligned}

    where :math:`\sigma` is the sigmoid activation
    and :math:`f` is an optional output nonlinearity.

    Args:
        input_size: Number of input features
        hidden_size: Number of hidden (and context) features
        bias: If ``False``, the layer does not use input-side bias ``b_ih``.
            Default: ``True``
        recurrent_bias: If ``False``, the layer does not use recurrent bias ``b_hh``.
            Default: ``True``
        context_bias: If ``False``, the layer does not use context bias ``b_ch``.
            Default: ``True``
        kernel_init: Initializer for ``weight_ih``.
            Default: :func:`torch.nn.init.xavier_uniform_`
        recurrent_kernel_init: Initializer for ``weight_hh``.
            Default: :func:`torch.nn.init.xavier_uniform_`
        context_kernel_init: Initializer for ``weight_ch``.
            Default: :func:`torch.nn.init.normal_`
        bias_init: Initializer for ``bias_ih`` when ``bias=True``.
            Default: :func:`torch.nn.init.zeros_`
        recurrent_bias_init: Initializer for ``bias_hh`` when ``recurrent_bias=True``.
            Default: :func:`torch.nn.init.zeros_`
        context_bias_init: Initializer for ``bias_ch`` when ``context_bias=True``.
            Default: :func:`torch.nn.init.zeros_`
        alpha: Context interpolation parameter :math:`\alpha`. Default: ``0.5``
        device: The desired device of parameters
        dtype: The desired floating point type of parameters

    Inputs: input, (h_0, s_0)
        - **input** of shape ``(batch, input_size)`` or ``(input_size,)``:
          tensor containing input features
        - **h_0** of shape ``(batch, hidden_size)`` or ``(hidden_size,)``:
          initial hidden state
        - **s_0** of shape ``(batch, hidden_size)`` or ``(hidden_size,)``:
          initial context state

        If **(h_0, s_0)** is not provided, both default to zero.

    Outputs: (h_1, s_1)
        - **h_1** of shape ``(batch, hidden_size)`` or ``(hidden_size,)``:
          next hidden state
        - **s_1** of shape ``(batch, hidden_size)`` or ``(hidden_size,)``:
          next context state

    Variables:
        weight_ih: input–hidden weights,
            of shape ``(2*hidden_size, input_size)``
        weight_hh: hidden–hidden weights,
            of shape ``(2*hidden_size, hidden_size)``
        weight_ch: context–hidden weights,
            of shape ``(2*hidden_size, hidden_size)``
        bias_ih: input biases,
            of shape ``(2*hidden_size)`` if ``bias=True``
        bias_hh: hidden biases,
            of shape ``(2*hidden_size)`` if ``recurrent_bias=True``
        bias_ch: context biases,
            of shape ``(2*hidden_size)`` if ``context_bias=True``
        alpha: learnable context interpolation scalar

    Examples::

        >>> cell = SCRNCell(10, 20, alpha=0.5)
        >>> x = torch.randn(6, 3, 10)   # (time, batch, input_size)
        >>> h = torch.zeros(3, 20)      # (batch, hidden_size)
        >>> s = torch.zeros(3, 20)      # (batch, hidden_size) context
        >>> hs = []
        >>> for t in range(x.size(0)):
        ...     h, s = cell(x[t], (h, s))
        ...     hs.append(h)
        >>> hs = torch.stack(hs, dim=0)  # (time, batch, hidden_size)

    """

    __constants__ = [
        "input_size",
        "hidden_size",
        "bias",
        "recurrent_bias",
        "context_bias",
        "kernel_init",
        "recurrent_kernel_init",
        "context_kernel_init",
        "bias_init",
        "recurrent_bias_init",
        "context_bias_init",
    ]

    weight_ih: Tensor
    weight_hh: Tensor
    weight_ch: Tensor
    bias_ih: Tensor
    bias_hh: Tensor
    bias_ch: Tensor

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        recurrent_bias: bool = True,
        context_bias: bool = True,
        kernel_init: Callable = nn.init.xavier_uniform_,
        recurrent_kernel_init: Callable = nn.init.xavier_uniform_,
        context_kernel_init: Callable = nn.init.normal_,
        bias_init: Callable = nn.init.zeros_,
        recurrent_bias_init: Callable = nn.init.zeros_,
        context_bias_init: Callable = nn.init.zeros_,
        alpha: float = 0.5,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super(SCRNCell, self).__init__(
            input_size, hidden_size, bias, recurrent_bias, device=device, dtype=dtype
        )
        self.kernel_init = kernel_init
        self.recurrent_kernel_init = recurrent_kernel_init
        self.context_kernel_init = context_kernel_init
        self.bias_init = bias_init
        self.recurrent_bias_init = recurrent_bias_init
        self.context_bias_init = context_bias_init
        self.alpha = nn.Parameter(torch.tensor(alpha))

        self._register_tensors(
            {
                "weight_ih": ((2 * hidden_size, input_size), True),
                "weight_hh": ((2 * hidden_size, hidden_size), True),
                "weight_ch": ((2 * hidden_size, hidden_size), True),
                "bias_ih": ((2 * hidden_size,), bias),
                "bias_hh": ((2 * hidden_size,), recurrent_bias),
                "bias_ch": ((2 * hidden_size,), context_bias),
            }
        )
        self.init_weights()

    def init_weights(self):
        for name, param in self.named_parameters():
            if "weight_ih" in name:
                self.kernel_init(param)
            elif "weight_hh" in name:
                self.recurrent_kernel_init(param)
            elif "weight_ch" in name:
                self.context_kernel_init(param)
            elif "bias_ih" in name:
                self.bias_init(param)
            elif "bias_hh" in name:
                self.recurrent_bias_init(param)
            elif "bias_ch" in name:
                self.context_bias_init(param)

    def forward(
        self, inp: Tensor, state: Optional[Union[Tensor, Tuple[Tensor, ...]]] = None
    ) -> Tuple[Tensor, Tensor]:
        state, c_state = self._check_states(state)
        self._validate_input(inp)
        self._validate_states((state, c_state))
        inp, state, c_state, is_batched = self._preprocess_states(inp, (state, c_state))

        inp_expanded = inp @ self.weight_ih.t() + self.bias_ih
        gxs1, gxs2 = inp_expanded.chunk(2, 1)
        weight_hh_1, weight_hh_2 = self.weight_hh.chunk(2, 0)
        bias_hh_1, bias_hh_2 = self.bias_hh.chunk(2, 0)

        new_cstate = (1 - self.alpha) * gxs1 + self.alpha * c_state
        cont_expanded = new_cstate @ self.weight_ch.t() + self.bias_ch
        gcs1, gcs2 = cont_expanded.chunk(2, 1)
        hidden_layer = torch.sigmoid(gxs2 + state @ weight_hh_1.t() + bias_hh_1 + gcs1)
        new_state = torch.tanh(hidden_layer @ weight_hh_2.t() + bias_hh_2 + gcs2)

        if not is_batched:
            new_state = new_state.squeeze(0)
            new_cstate = new_cstate.squeeze(0)

        return new_state, new_cstate

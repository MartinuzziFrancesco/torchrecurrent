from typing import Optional
import torch
import torch.nn as nn
from torch import Tensor
from ..base import SingleStateCellBase, SingleStateRecurrentLayerBase, resolve_activation, resolve_init_name, apply_init_


class AntisymmetricRNN(SingleStateRecurrentLayerBase):
    r"""Multi-layer antisymmetric recurrent neural network.

    [`arXiv <https://arxiv.org/abs/1902.09689>`_]

    Each layer consists of an :class:`AntisymmetricRNNCell`, which updates the
    hidden state according to:

    .. math::
        \begin{array}{ll}
            \mathbf{A} &= \mathbf{W}_{hh} - \mathbf{W}_{hh}^\top - \gamma \mathbf{I} \\
            h_t &= h_{t-1} + \varepsilon \,\tanh(\mathbf{W}_{ih} x_t +
                \mathbf{b}_{ih} + \mathbf{A} h_{t-1} + \mathbf{b}_{hh})
        \end{array}

    where :math:`h_t` is the hidden state at time `t`, :math:`x_t` is the input at
    time `t`, :math:`\varepsilon` is a step-size parameter, and :math:`\gamma \ge 0`
    adds diagonal damping for stability. :math:`\tanh` is the default nonlinearity
    (configurable via :attr:`nonlinearity`).

    In a multilayer AntisymmetricRNN, the input :math:`x^{(l)}_t` of the
    :math:`l`-th layer (:math:`l \ge 2`) is the hidden state :math:`h^{(l-1)}_t` of
    the previous layer multiplied by dropout :math:`\delta^{(l-1)}_t`, where each
    :math:`\delta^{(l-1)}_t` is a Bernoulli random variable which is 0 with
    probability :attr:`dropout`.

    Args:
        input_size: The number of expected features in the input `x`.
        hidden_size: The number of features in the hidden state `h`.
        num_layers: Number of recurrent layers. E.g., setting ``num_layers=2`` would
            mean stacking two antisymmetric RNN layers, with the second receiving the
            outputs of the first. Default: 1
        dropout: If non-zero, introduces a `Dropout` layer on the outputs of each
            layer except the last layer, with dropout probability equal to
            :attr:`dropout`. Default: 0
        batch_first: If ``True``, then the input and output tensors are provided as
            `(batch, seq, feature)` instead of `(seq, batch, feature)`. Default: False
        bias: If ``False``, then the layer does not use input-side bias `b_ih`.
            Default: True
        recurrent_bias: If ``False``, then the layer does not use recurrent bias
            `b_hh`. Default: True
        nonlinearity: Elementwise nonlinearity applied to the pre-activation.
            Default: :func:`torch.tanh`
        kernel_init: Initializer for `weight_ih`. Default:
            :func:`torch.nn.init.xavier_uniform_`
        recurrent_kernel_init: Initializer for `weight_hh`. Default:
            :func:`torch.nn.init.normal_`
        bias_init: Initializer for `bias_ih`. Default: :func:`torch.nn.init.zeros_`
        recurrent_bias_init: Initializer for `bias_hh`. Default:
            :func:`torch.nn.init.zeros_`
        epsilon: Step-size multiplier :math:`\varepsilon`. Default: 1.0
        gamma: Damping coefficient :math:`\gamma` used in the antisymmetric transform.
            Default: 0.0
        device: The desired device of parameters.
        dtype: The desired floating point type of parameters.

    Inputs: input, h_0
        - **input**: tensor of shape :math:`(L, H_{in})` for unbatched input,
          :math:`(L, N, H_{in})` when ``batch_first=False`` or
          :math:`(N, L, H_{in})` when ``batch_first=True`` containing the features of
          the input sequence. The input can also be a packed variable length
          sequence. See :func:`torch.nn.utils.rnn.pack_padded_sequence` or
          :func:`torch.nn.utils.rnn.pack_sequence` for details.
        - **h_0**: tensor of shape :math:`(\text{num_layers}, H_{out})` for unbatched
          input or :math:`(\text{num_layers}, N, H_{out})` containing the initial
          hidden state for each element in the input sequence. Defaults to zeros if
          not provided.

        where:

        .. math::
            \begin{aligned}
                N ={} & \text{batch size} \\
                L ={} & \text{sequence length} \\
                H_{in} ={} & \text{input\_size} \\
                H_{out} ={} & \text{hidden\_size}
            \end{aligned}

    Outputs: output, h_n
        - **output**: tensor of shape :math:`(L, H_{out})` for unbatched input,
          :math:`(L, N, H_{out})` when ``batch_first=False`` or
          :math:`(N, L, H_{out})` when ``batch_first=True`` containing the output
          features `(h_t)` from the last layer of the AntisymmetricRNN, for each
          `t`. If a :class:`torch.nn.utils.rnn.PackedSequence` has been given as the
          input, the output will also be a packed sequence.
        - **h_n**: tensor of shape :math:`(\text{num_layers}, H_{out})` for unbatched
          input or :math:`(\text{num_layers}, N, H_{out})` containing the final
          hidden state for each element in the sequence.

    Attributes:
        cells.{k}.weight_ih : the learnable input-hidden weights of the :math:`k`-th
            layer, of shape `(hidden_size, input_size)` for `k = 0`. Otherwise, the
            shape is `(hidden_size, hidden_size)`.
        cells.{k}.weight_hh : the learnable hidden-hidden weights of the :math:`k`-th
            layer, of shape `(hidden_size, hidden_size)`.
        cells.{k}.bias_ih : the learnable input-hidden bias of the :math:`k`-th layer,
            of shape `(hidden_size)`. Only present when ``bias=True``.
        cells.{k}.bias_hh : the learnable hidden-hidden bias of the :math:`k`-th layer,
            of shape `(hidden_size)`. Only present when ``recurrent_bias=True``.

    .. note::
        All the weights and biases are initialized according to the provided
        initializers (`kernel_init`, `recurrent_kernel_init`, etc.).

    .. note::
        ``batch_first`` argument is ignored for unbatched inputs.

    .. seealso::
        :class:`AntisymmetricRNNCell`

    Examples::

        >>> rnn = AntisymmetricRNN(10, 20, num_layers=2, dropout=0.1)
        >>> input = torch.randn(5, 3, 10)   # (seq_len, batch, input_size)
        >>> h0 = torch.randn(2, 3, 20)      # (num_layers, batch, hidden_size)
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
        super(AntisymmetricRNN, self).__init__(
            input_size, hidden_size, num_layers, dropout, batch_first
        )
        self.initialize_cells(AntisymmetricRNNCell, **kwargs)


class AntisymmetricRNNCell(SingleStateCellBase):
    r"""An antisymmetric recurrent neural network cell.

    [`arXiv <https://arxiv.org/abs/1902.09689>`_]

    .. math::

        \begin{array}{ll}
            \mathbf{A} = \mathbf{W}_{hh} - \mathbf{W}_{hh}^\top -
                \gamma \mathbf{I} \\
            h' = h + \varepsilon \,\tanh(\mathbf{W}_{ih} x +
                \mathbf{b}_{ih} + \mathbf{A} h + \mathbf{b}_{hh})
        \end{array}

    where :math:`\varepsilon` is a step-size scalar,
    :math:`\gamma \ge 0` adds diagonal damping for stability.

    Args:
        input_size: The number of expected features in the input `x`
        hidden_size: The number of features in the hidden state `h`
        bias: If ``False``, then the layer does not use input-side bias `b_ih`.
            Default: ``True``
        recurrent_bias: If ``False``, then the layer does not use recurrent
            bias `b_hh`. Default: ``True``
        nonlinearity: Elementwise nonlinearity applied to the pre-activation.
            Default: :func:`torch.tanh`
        kernel_init: Initializer for `weight_ih`.
            Default: :func:`torch.nn.init.xavier_uniform_`
        recurrent_kernel_init: Initializer for `weight_hh`.
            Default: :func:`torch.nn.init.normal_`
        bias_init: Initializer for `bias_ih`.
            Default: :func:`torch.nn.init.zeros_`
        recurrent_bias_init: Initializer for `bias_hh`.
            Default: :func:`torch.nn.init.zeros_`
        epsilon: Step-size multiplier :math:`\varepsilon`.
            Default: ``1.0``
        gamma: Damping coefficient :math:`\gamma` used in the
            antisymmetric transform. Default: ``0.0``
        device: The desired device of parameters.
        dtype: The desired floating point type of parameters.

    Inputs: input, h_0
        - **input** of shape `(batch, input_size)` or `(input_size,)`:
          tensor containing input features
        - **h_0** of shape `(batch, hidden_size)` or `(hidden_size,)`:
          tensor containing the initial hidden state

        If **h_0** is not provided, it defaults to zero.

    Outputs: h_1
        - **h_1** of shape `(batch, hidden_size)` or `(hidden_size,)`:
          tensor containing the next hidden state

    Variables:
        weight_ih: the learnable input–hidden weights,
            of shape `(hidden_size, input_size)`
        weight_hh: the learnable hidden–hidden weights,
            of shape `(hidden_size, hidden_size)`
        bias_ih: the learnable input–hidden bias,
            of shape `(hidden_size)`
        bias_hh: the learnable hidden–hidden bias,
            of shape `(hidden_size)`

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

    __constants__ = ["input_size", "hidden_size", "bias", "recurrent_bias", "epsilon", "gamma"]

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
        nonlinearity="tanh",
        kernel_init=nn.init.xavier_uniform_,
        recurrent_kernel_init=nn.init.normal_,
        bias_init=nn.init.zeros_,
        recurrent_bias_init=nn.init.zeros_,
        epsilon: float = 1.0,
        gamma: float = 0.0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            bias=bias,
            recurrent_bias=recurrent_bias,
            device=device,
            dtype=dtype,
        )

        self.epsilon = float(epsilon)
        self.gamma = float(gamma)
        self.act = resolve_activation(nonlinearity)
        self.init_cfg["kernel"] = resolve_init_name(kernel_init, self.init_cfg["kernel"])
        self.init_cfg["recurrent_kernel"] = resolve_init_name(
            recurrent_kernel_init, self.init_cfg["recurrent_kernel"]
        )
        self.init_cfg["bias"] = resolve_init_name(bias_init, self.init_cfg["bias"])
        self.init_cfg["recurrent_bias"] = resolve_init_name(
            recurrent_bias_init, self.init_cfg["recurrent_bias"]
        )

        self._default_register_tensors()
        self.reset_parameters()
        self._cleanup_non_scriptable()

    def reset_parameters(self) -> None:
        apply_init_(self.weight_ih, self.init_cfg["kernel"])
        apply_init_(self.weight_hh, self.init_cfg["recurrent_kernel"])
        if hasattr(self, "bias_ih"):
            apply_init_(self.bias_ih, self.init_cfg["bias"])
        if hasattr(self, "bias_hh"):
            apply_init_(self.bias_hh, self.init_cfg["recurrent_bias"])

    def forward(self, inp: Tensor, state: Optional[Tensor] = None) -> Tensor:
        self._validate_input(inp)
        b_inp, is_batched = self._as_batched(inp)

        if state is None:
            b_state = self._zeros_state(b_inp.size(0), b_inp.device, b_inp.dtype)
        else:
            b_state = state.unsqueeze(0) if (not is_batched and state.dim() == 1) else state

        recurrent_matrix = _compute_asym(self.weight_hh, self.gamma)
        pre_act = b_inp @ self.weight_ih.t() + self.bias_ih + b_state @ recurrent_matrix.t() + self.bias_hh
        h_new = b_state + self.epsilon * self.act(pre_act)

        if not is_batched:
            h_new = h_new.squeeze(0)
        return h_new


class GatedAntisymmetricRNN(SingleStateRecurrentLayerBase):
    r"""Multi-layer gated antisymmetric recurrent neural network.

    [`arXiv <https://arxiv.org/abs/1902.09689>`_]

    Each layer consists of a :class:`GatedAntisymmetricRNNCell`, which updates the
    hidden state according to:

    .. math::
        \begin{aligned}
            \mathbf{z}_t &= \sigma\Bigl(
                (\mathbf{W}_{hh} - \mathbf{W}_{hh}^\top - \gamma \mathbf{I}) h_{t-1}
                + \mathbf{b}_{hh} + \mathbf{W}_{ih}^z x_t + \mathbf{b}_{ih}^z
            \Bigr), \\
            h_t &= h_{t-1} + \epsilon \,\mathbf{z}_t \circ \tanh\Bigl(
                (\mathbf{W}_{hh} - \mathbf{W}_{hh}^\top - \gamma \mathbf{I}) h_{t-1}
                + \mathbf{b}_{hh} + \mathbf{W}_{ih}^x x_t + \mathbf{b}_{ih}^h
            \Bigr)
        \end{aligned}

    where :math:`h_t` is the hidden state at time `t`, :math:`x_t` is the input at
    time `t`, :math:`\epsilon` controls the integration step size, :math:`\gamma \ge 0`
    adds diagonal damping for stability, and :math:`\circ` is elementwise product.

    In a multilayer GatedAntisymmetricRNN, the input :math:`x^{(l)}_t` of the
    :math:`l`-th layer (:math:`l \ge 2`) is the hidden state :math:`h^{(l-1)}_t` of
    the previous layer multiplied by dropout :math:`\delta^{(l-1)}_t`, where each
    :math:`\delta^{(l-1)}_t` is a Bernoulli random variable which is 0 with
    probability :attr:`dropout`.

    Args:
        input_size: The number of expected features in the input `x`.
        hidden_size: The number of features in the hidden state `h`.
        num_layers: Number of recurrent layers. E.g., setting ``num_layers=2`` would
            mean stacking two gated antisymmetric RNN layers, with the second
            receiving the outputs of the first. Default: 1
        dropout: If non-zero, introduces a `Dropout` layer on the outputs of each
            layer except the last layer, with dropout probability equal to
            :attr:`dropout`. Default: 0
        batch_first: If ``True``, then the input and output tensors are provided as
            `(batch, seq, feature)` instead of `(seq, batch, feature)`. Default: False
        bias: If ``False``, then the layer does not use input-side biases. Default: True
        recurrent_bias: If ``False``, then the layer does not use recurrent bias.
            Default: True
        nonlinearity: Elementwise nonlinearity applied to the candidate
            pre-activation. Default: :func:`torch.tanh`
        kernel_init: Initializer for `weight_ih`. Default:
            :func:`torch.nn.init.xavier_uniform_`
        recurrent_kernel_init: Initializer for `weight_hh`. Default:
            :func:`torch.nn.init.normal_`
        bias_init: Initializer for input-side biases. Default:
            :func:`torch.nn.init.zeros_`
        recurrent_bias_init: Initializer for recurrent biases. Default:
            :func:`torch.nn.init.zeros_`
        epsilon: Step-size multiplier :math:`\epsilon`. Default: 1.0
        gamma: Damping coefficient :math:`\gamma` used in the antisymmetric transform.
            Default: 0.0
        device: The desired device of parameters.
        dtype: The desired floating point type of parameters.

    Inputs: input, h_0
        - **input**: tensor of shape :math:`(L, H_{in})` for unbatched input,
          :math:`(L, N, H_{in})` when ``batch_first=False`` or
          :math:`(N, L, H_{in})` when ``batch_first=True`` containing the features of
          the input sequence. The input can also be a packed variable length
          sequence. See :func:`torch.nn.utils.rnn.pack_padded_sequence` or
          :func:`torch.nn.utils.rnn.pack_sequence` for details.
        - **h_0**: tensor of shape :math:`(\text{num_layers}, H_{out})` for unbatched
          input or :math:`(\text{num_layers}, N, H_{out})` containing the initial
          hidden state for each element in the input sequence. Defaults to zeros if
          not provided.

        where:

        .. math::
            \begin{aligned}
                N ={} & \text{batch size} \\
                L ={} & \text{sequence length} \\
                H_{in} ={} & \text{input\_size} \\
                H_{out} ={} & \text{hidden\_size}
            \end{aligned}

    Outputs: output, h_n
        - **output**: tensor of shape :math:`(L, H_{out})` for unbatched input,
          :math:`(L, N, H_{out})` when ``batch_first=False`` or
          :math:`(N, L, H_{out})` when ``batch_first=True`` containing the output
          features `(h_t)` from the last layer of the GatedAntisymmetricRNN, for each
          `t`. If a :class:`torch.nn.utils.rnn.PackedSequence` has been given as the
          input, the output will also be a packed sequence.
        - **h_n**: tensor of shape :math:`(\text{num_layers}, H_{out})` for unbatched
          input or :math:`(\text{num_layers}, N, H_{out})` containing the final
          hidden state for each element in the sequence.

    Attributes:
        cells.{k}.weight_ih : the learnable input-hidden weights of the :math:`k`-th
            layer, of shape `(2*hidden_size, input_size)` for `k = 0`. Otherwise, the
            shape is `(2*hidden_size, hidden_size)`.
        cells.{k}.weight_hh : the learnable hidden-hidden weights of the :math:`k`-th
            layer, of shape `(hidden_size, hidden_size)`.
        cells.{k}.bias_ih : the learnable input-hidden bias of the :math:`k`-th
            layer, of shape `(2*hidden_size)`. Only present when ``bias=True``.
        cells.{k}.bias_hh : the learnable hidden-hidden bias of the :math:`k`-th
            layer, of shape `(hidden_size)`. Only present when ``recurrent_bias=True``.

    .. note::
        All the weights and biases are initialized according to the provided
        initializers (`kernel_init`, `recurrent_kernel_init`, etc.).

    .. note::
        ``batch_first`` argument is ignored for unbatched inputs.

    .. seealso::
        :class:`GatedAntisymmetricRNNCell`

    Examples::

        >>> rnn = GatedAntisymmetricRNN(8, 16, num_layers=2, dropout=0.1,
        ...                             epsilon=0.5, gamma=0.1)
        >>> input = torch.randn(5, 3, 8)   # (seq_len, batch, input_size)
        >>> h0 = torch.zeros(2, 3, 16)     # (num_layers, batch, hidden_size)
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
        super(GatedAntisymmetricRNN, self).__init__(
            input_size, hidden_size, num_layers, dropout, batch_first
        )
        self.initialize_cells(GatedAntisymmetricRNNCell, **kwargs)


class GatedAntisymmetricRNNCell(SingleStateCellBase):
    r"""A gated antisymmetric recurrent neural network (RNN) cell.

    [`arXiv <https://arxiv.org/abs/1902.09689>`_]

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

    where :math:`\epsilon` controls the integration step size, :math:`\gamma`
    is a stability damping, and :math:`\circ` is element-wise product.

    Args:
        input_size: The number of expected features in the input `x`
        hidden_size: The number of features in the hidden state `h`
        bias: If ``False``, then the layer does not use input-side biases.
            Default: ``True``
        recurrent_bias: If ``False``, then the layer does not use recurrent bias.
            Default: ``True``
        nonlinearity: Elementwise nonlinearity applied to the
            candidate pre-activation. Default: :func:`torch.tanh`
        kernel_init: Initializer for `weight_ih`.
            Default: :func:`torch.nn.init.xavier_uniform_`
        recurrent_kernel_init: Initializer for `weight_hh`.
            Default: :func:`torch.nn.init.normal_`
        bias_init: Initializer for input-side biases.
            Default: :func:`torch.nn.init.zeros_`
        epsilon: Step-size multiplier :math:`\epsilon`.
            Default: ``1.0``
        gamma: Damping coefficient :math:`\gamma` used in the
            antisymmetric transform. Default: ``0.0``
        device: The desired device of parameters.
        dtype: The desired floating point type of parameters.

    Inputs: input, h_0
        - **input** of shape `(batch, input_size)` or `(input_size,)`:
          tensor containing input features
        - **h_0** of shape `(batch, hidden_size)` or `(hidden_size,)`:
          tensor containing the initial hidden state

        If **h_0** is not provided, it defaults to zero.

    Outputs: h_1
        - **h_1** of shape `(batch, hidden_size)` or `(hidden_size,)`:
          tensor containing the next hidden state

    Variables:
        weight_ih: the learnable input–hidden weights,
            of shape `(2*hidden_size, input_size)` (gate and candidate)
        weight_hh: the learnable hidden–hidden weights,
            of shape `(hidden_size, hidden_size)`
        bias_ih: the learnable input–hidden bias,
            of shape `(2*hidden_size)`
        bias_hh: the learnable hidden–hidden bias,
            of shape `(hidden_size)`

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

    __constants__ = ["input_size", "hidden_size", "bias", "recurrent_bias", "epsilon", "gamma"]

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
        kernel_init=nn.init.xavier_uniform_,
        recurrent_kernel_init=nn.init.normal_,
        bias_init=nn.init.zeros_,
        recurrent_bias_init=nn.init.zeros_,
        epsilon: float = 1.0,
        gamma: float = 0.0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            bias=bias,
            recurrent_bias=recurrent_bias,
            device=device,
            dtype=dtype,
        )

        self.epsilon = float(epsilon)
        self.gamma = float(gamma)
        self.init_cfg["kernel"] = resolve_init_name(kernel_init, self.init_cfg["kernel"])
        self.init_cfg["recurrent_kernel"] = resolve_init_name(
            recurrent_kernel_init, self.init_cfg["recurrent_kernel"]
        )
        self.init_cfg["bias"] = resolve_init_name(bias_init, self.init_cfg["bias"])
        self.init_cfg["recurrent_bias"] = resolve_init_name(
            recurrent_bias_init, self.init_cfg["recurrent_bias"]
        )

        self._default_register_tensors(ih_mult=2, hh_mult=1)
        self.reset_parameters()
        self._cleanup_non_scriptable()

    def reset_parameters(self) -> None:
        apply_init_(self.weight_ih, self.init_cfg["kernel"])
        apply_init_(self.weight_hh, self.init_cfg["recurrent_kernel"])
        if hasattr(self, "bias_ih"):
            apply_init_(self.bias_ih, self.init_cfg["bias"])
        if hasattr(self, "bias_hh"):
            apply_init_(self.bias_hh, self.init_cfg["recurrent_bias"])

    def forward(self, inp: Tensor, state: Optional[Tensor] = None) -> Tensor:
        self._validate_input(inp)
        b_inp, is_batched = self._as_batched(inp)

        if state is None:
            b_state = self._zeros_state(b_inp.size(0), b_inp.device, b_inp.dtype)
        else:
            b_state = state.unsqueeze(0) if (not is_batched and state.dim() == 1) else state

        weights_ih = b_inp @ self.weight_ih.t() + self.bias_ih
        weight_ih_1, weight_ih_2 = weights_ih.chunk(2, 1)
        recurrent_matrix = _compute_asym(self.weight_hh, self.gamma)
        pre_act = weight_ih_2 + b_state @ recurrent_matrix.t() + self.bias_hh
        input_gate = torch.sigmoid(weight_ih_1 + b_state @ recurrent_matrix.t())
        h_new = b_state + self.epsilon * input_gate * torch.tanh(pre_act)

        if not is_batched:
            h_new = h_new.squeeze(0)

        return h_new


def _compute_asym(W_hh: Tensor, gamma: float) -> Tensor:
    n = W_hh.size(0)
    I = torch.eye(n, device=W_hh.device, dtype=W_hh.dtype)
    return W_hh - W_hh.transpose(0, 1) - gamma * I

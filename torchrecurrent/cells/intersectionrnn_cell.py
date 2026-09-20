import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Tuple
from ..base import (
    DecoupledSingleStateRecurrentLayerBase,
    DecoupledSingleStateCellBase,
    resolve_activation,
    resolve_init_name,
)


class IntersectionRNN(DecoupledSingleStateRecurrentLayerBase):
    r"""Multi-layer Intersection RNN (+RNN).

    [`arXiv <https://arxiv.org/abs/1611.09913>`_]

    Each layer consists of an :class:`IntersectionRNNCell`, which couples a
    "depth" gate (between layers) and a "recurrent" gate (through time):

    .. math::
        \begin{aligned}
        y^{in}_t &= \phi(W_{ih}^{y} x_t + b_{ih}^{y} + W_{hh}^{y} h_{t-1} + b_{hh}^{y}), \\
        h^{in}_t &= \tau(W_{ih}^{h} x_t + b_{ih}^{h} + W_{hh}^{h} h_{t-1} + b_{hh}^{h}), \\
        g^{y}_t &= \sigma(W_{ih}^{g^y} x_t + b_{ih}^{g^y} + W_{hh}^{g^y} h_{t-1} + b_{hh}^{g^y}), \\
        g^{h}_t &= \sigma(W_{ih}^{g^h} x_t + b_{ih}^{g^h} + W_{hh}^{g^h} h_{t-1} + b_{hh}^{g^h}), \\
        y_t &= g^{y}_t \circ x_t + (1 - g^{y}_t) \circ y^{in}_t, \\
        h_t &= g^{h}_t \circ h_{t-1} + (1 - g^{h}_t) \circ h^{in}_t
        \end{aligned}

    where :math:`\sigma` is the sigmoid function, :math:`\phi` is ReLU,
    :math:`\tau` is a pointwise nonlinearity (e.g., tanh), and :math:`\circ`
    denotes elementwise (Hadamard) product. Unlike every other layer in this
    package, the per-timestep output :math:`y_t` fed to the next layer is
    *not* the same tensor as the recurrent state :math:`h_t` carried to the
    next timestep: :math:`y_t` is a "depth" gate mixing the raw input
    :math:`x_t` with a candidate, while :math:`h_t` is an ordinary recurrent
    gate mix. This requires ``input_size == hidden_size``, since :math:`y_t`
    mixes :math:`x_t` and :math:`y^{in}_t` elementwise.

    The original paper studies this architecture for ``num_layers >= 2`` and
    explicitly excludes single-layer (``num_layers=1``) configurations from
    its experiments, since the depth gate has no second layer to feed. The
    per-timestep computation is still well-defined for ``num_layers=1``
    (:math:`y_t` is simply returned as the layer's output), but this
    configuration was not studied by the authors.

    In a multilayer IntersectionRNN, the input :math:`x^{(l)}_t` of the
    :math:`l`-th layer (:math:`l \ge 2`) is the *output* :math:`y^{(l-1)}_t`
    of the previous layer multiplied by dropout :math:`\delta^{(l-1)}_t`,
    where each :math:`\delta^{(l-1)}_t` is a Bernoulli random variable which
    is 0 with probability :attr:`dropout`.

    Args:
        input_size: The number of expected features in the input `x`. Must
            equal `hidden_size`.
        hidden_size: The number of features in the hidden state `h`. Must
            equal `input_size`.
        num_layers: Number of recurrent layers. E.g., setting ``num_layers=2`` would
            mean stacking two IntersectionRNN layers, with the second receiving the
            outputs of the first. Default: 1
        dropout: If non-zero, introduces a `Dropout` layer on the outputs of each
            layer except the last layer, with dropout probability equal to
            :attr:`dropout`. Default: 0
        batch_first: If ``True``, then the input and output tensors are provided as
            `(batch, seq, feature)` instead of `(seq, batch, feature)`. Default: False
        bias: If ``False``, then the layer does not use input-side biases.
            Default: True
        recurrent_bias: If ``False``, then the layer does not use recurrent biases.
            Default: True
        depth_nonlinearity: Nonlinearity :math:`\phi` for the depth candidate
            :math:`y^{in}`. Default: :func:`torch.relu`
        recurrent_nonlinearity: Nonlinearity :math:`\tau` for the recurrent
            candidate :math:`h^{in}`. Default: :func:`torch.tanh`
        gate_nonlinearity: Activation :math:`\sigma` for both gates. Default:
            :func:`torch.sigmoid`
        kernel_init: Initializer for `W_{ih}`. Default:
            :func:`torch.nn.init.xavier_uniform_`
        recurrent_kernel_init: Initializer for `W_{hh}`. Default:
            :func:`torch.nn.init.xavier_uniform_`
        bias_init: Initializer for `b_{ih}`. Default:
            :func:`torch.nn.init.zeros_`
        recurrent_bias_init: Initializer for `b_{hh}`. Default:
            :func:`torch.nn.init.zeros_`
        device: The desired device of parameters.
        dtype: The desired floating point type of parameters.

    Inputs: input, h_0
        - **input**: tensor of shape :math:`(L, N, H)` when ``batch_first=False``
          or :math:`(N, L, H)` when ``batch_first=True`` containing the features of
          the input sequence, where :math:`H` = `input_size` = `hidden_size`.
        - **h_0**: tensor of shape :math:`(\text{num_layers}, N, H)` containing the
          initial hidden state for each element in the input sequence. Defaults to
          zeros if not provided.

        where:

        .. math::
            \begin{aligned}
                N ={} & \text{batch size} \\
                L ={} & \text{sequence length} \\
                H ={} & \text{input\_size} = \text{hidden\_size}
            \end{aligned}

    Outputs: output, h_n
        - **output**: tensor of shape :math:`(L, N, H)` when ``batch_first=False``
          or :math:`(N, L, H)` when ``batch_first=True`` containing the output
          features `(y_t)` from the last layer of the IntersectionRNN, for each `t`.
        - **h_n**: tensor of shape :math:`(\text{num_layers}, N, H)` containing the
          final recurrent state for each element in the sequence. Note this is
          `h_n`, not `y_n`: the recurrent state and the returned output are
          different tensors for every layer.

    Attributes:
        cells.{k}.weight_ih : the learnable input-hidden weights of the :math:`k`-th
            layer, of shape `(4*hidden_size, input_size)`.
        cells.{k}.weight_hh : the learnable hidden-hidden weights of the :math:`k`-th
            layer, of shape `(4*hidden_size, hidden_size)`.
        cells.{k}.bias_ih : the learnable input-hidden biases of the :math:`k`-th
            layer, of shape `(4*hidden_size)`. Only present when ``bias=True``.
        cells.{k}.bias_hh : the learnable hidden-hidden biases of the :math:`k`-th
            layer, of shape `(4*hidden_size)`. Only present when ``recurrent_bias=True``.

    .. note::
        All the weights and biases are initialized according to the provided
        initializers (`kernel_init`, `recurrent_kernel_init`, etc.).

    .. seealso::
        :class:`IntersectionRNNCell`

    Examples::

        >>> rnn = IntersectionRNN(10, 10, num_layers=2, dropout=0.1)
        >>> input = torch.randn(5, 3, 10)   # (seq_len, batch, input_size)
        >>> h0 = torch.zeros(2, 3, 10)      # (num_layers, batch, hidden_size)
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
        super(IntersectionRNN, self).__init__(
            input_size, hidden_size, num_layers, dropout, batch_first
        )
        self.initialize_cells(IntersectionRNNCell, **kwargs)


class IntersectionRNNCell(DecoupledSingleStateCellBase):
    r"""An Intersection RNN (+RNN) cell.

    [`arXiv <https://arxiv.org/abs/1611.09913>`_]

    .. math::

        \mathbf{y}^{in}(t) &= \phi\bigl(
            \mathbf{W}_{ih}^{y}\,\mathbf{x}(t) + \mathbf{b}_{ih}^{y}
            + \mathbf{W}_{hh}^{y}\,\mathbf{h}(t-1) + \mathbf{b}_{hh}^{y}
        \bigr), \\[4pt]
        \mathbf{h}^{in}(t) &= \tau\bigl(
            \mathbf{W}_{ih}^{h}\,\mathbf{x}(t) + \mathbf{b}_{ih}^{h}
            + \mathbf{W}_{hh}^{h}\,\mathbf{h}(t-1) + \mathbf{b}_{hh}^{h}
        \bigr), \\[4pt]
        \mathbf{g}^{y}(t) &= \sigma\bigl(
            \mathbf{W}_{ih}^{g^y}\,\mathbf{x}(t) + \mathbf{b}_{ih}^{g^y}
            + \mathbf{W}_{hh}^{g^y}\,\mathbf{h}(t-1) + \mathbf{b}_{hh}^{g^y}
        \bigr), \\[4pt]
        \mathbf{g}^{h}(t) &= \sigma\bigl(
            \mathbf{W}_{ih}^{g^h}\,\mathbf{x}(t) + \mathbf{b}_{ih}^{g^h}
            + \mathbf{W}_{hh}^{g^h}\,\mathbf{h}(t-1) + \mathbf{b}_{hh}^{g^h}
        \bigr), \\[4pt]
        \mathbf{y}(t) &= \mathbf{g}^{y}(t)\circ\mathbf{x}(t)
            + \bigl(1 - \mathbf{g}^{y}(t)\bigr)\circ\mathbf{y}^{in}(t), \\[4pt]
        \mathbf{h}(t) &= \mathbf{g}^{h}(t)\circ\mathbf{h}(t-1)
            + \bigl(1 - \mathbf{g}^{h}(t)\bigr)\circ\mathbf{h}^{in}(t),

    where :math:`\circ` is element-wise product, :math:`\phi` is ReLU, and
    :math:`\tau` is a pointwise nonlinearity (e.g., tanh). The returned
    output :math:`\mathbf{y}(t)` and the recurrent state
    :math:`\mathbf{h}(t)` are different tensors: :math:`\mathbf{y}(t)` gates
    a mix of the raw input and a candidate, while :math:`\mathbf{h}(t)`
    gates a mix of the previous state and a (different) candidate. Requires
    ``input_size == hidden_size``, since :math:`\mathbf{y}(t)` mixes
    :math:`\mathbf{x}(t)` and :math:`\mathbf{y}^{in}(t)` elementwise.

    Args:
        input_size: The number of expected features in the input ``x``. Must
            equal ``hidden_size``.
        hidden_size: The number of features in the hidden state ``h``. Must
            equal ``input_size``.
        bias: If ``False``, the layer does not use input-side biases.
            Default: ``True``.
        recurrent_bias: If ``False``, the layer does not use recurrent biases.
            Default: ``True``.
        depth_nonlinearity: Nonlinearity :math:`\phi` for the depth candidate.
            Default: :func:`torch.relu`.
        recurrent_nonlinearity: Nonlinearity :math:`\tau` for the recurrent
            candidate. Default: :func:`torch.tanh`.
        gate_nonlinearity: Activation :math:`\sigma` for both gates.
            Default: :func:`torch.sigmoid`.
        kernel_init: Initializer for ``W_{ih}``.
            Default: :func:`torch.nn.init.xavier_uniform_`.
        recurrent_kernel_init: Initializer for ``W_{hh}``.
            Default: :func:`torch.nn.init.xavier_uniform_`.
        bias_init: Initializer for ``b_{ih}`` when ``bias=True``.
            Default: :func:`torch.nn.init.zeros_`.
        recurrent_bias_init: Initializer for ``b_{hh}`` when
            ``recurrent_bias=True``. Default: :func:`torch.nn.init.zeros_`.
        device: The desired device of parameters.
        dtype: The desired floating point type of parameters.

    Inputs: input, h_0
        - **input** of shape ``(batch, input_size)`` or ``(input_size,)``:
          Tensor containing input features.
        - **h_0** of shape ``(batch, hidden_size)`` or ``(hidden_size,)``:
          Tensor containing the initial hidden state.

        If **h_0** is not provided, it defaults to zero.

    Outputs: y_1, h_1
        - **y_1** of shape ``(batch, hidden_size)`` or ``(hidden_size,)``:
          Tensor containing the depth output, to be passed to the next layer.
        - **h_1** of shape ``(batch, hidden_size)`` or ``(hidden_size,)``:
          Tensor containing the next recurrent state.

    Variables:
        weight_ih: The learnable input-hidden weights,
            of shape ``(4*hidden_size, input_size)`` (``y, h, g^y, g^h`` parts).
        weight_hh: The learnable hidden-hidden weights,
            of shape ``(4*hidden_size, hidden_size)`` (``y, h, g^y, g^h`` parts).
        bias_ih: The learnable input-hidden biases,
            of shape ``(4*hidden_size)`` if ``bias=True``.
        bias_hh: The learnable hidden-hidden biases,
            of shape ``(4*hidden_size)`` if ``recurrent_bias=True``.

    Examples::

        >>> cell = IntersectionRNNCell(10, 10)
        >>> x = torch.randn(5, 3, 10)     # (time_steps, batch, input_size)
        >>> h = torch.zeros(3, 10)        # (batch, hidden_size)
        >>> out = []
        >>> for t in range(x.size(0)):
        ...     y, h = cell(x[t], h)
        ...     out.append(y)
        >>> out = torch.stack(out, dim=0) # (time_steps, batch, hidden_size)
    """

    __constants__ = ["input_size", "hidden_size", "bias", "recurrent_bias"]

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
        depth_nonlinearity="relu",
        recurrent_nonlinearity="tanh",
        gate_nonlinearity="sigmoid",
        kernel_init=nn.init.xavier_uniform_,
        recurrent_kernel_init=nn.init.xavier_uniform_,
        bias_init=nn.init.zeros_,
        recurrent_bias_init=nn.init.zeros_,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        if input_size != hidden_size:
            raise ValueError(
                "IntersectionRNNCell requires input_size == hidden_size (got "
                f"input_size={input_size}, hidden_size={hidden_size}), since the "
                "output is a gated mix of x(t) and a candidate of dimension "
                "hidden_size."
            )
        super().__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            bias=bias,
            recurrent_bias=recurrent_bias,
            device=device,
            dtype=dtype,
        )
        self.depth_act = resolve_activation(depth_nonlinearity)
        self.recurrent_act = resolve_activation(recurrent_nonlinearity)
        self.gate_act = resolve_activation(gate_nonlinearity)
        self.init_cfg["kernel"] = resolve_init_name(kernel_init, self.init_cfg["kernel"])
        self.init_cfg["recurrent_kernel"] = resolve_init_name(
            recurrent_kernel_init, self.init_cfg["recurrent_kernel"]
        )
        self.init_cfg["bias"] = resolve_init_name(bias_init, self.init_cfg["bias"])
        self.init_cfg["recurrent_bias"] = resolve_init_name(
            recurrent_bias_init, self.init_cfg["recurrent_bias"]
        )

        self._default_register_tensors(ih_mult=4, hh_mult=4)
        self.reset_parameters()
        self._cleanup_non_scriptable()

    def forward(self, inp: Tensor, state: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        self._validate_input(inp)
        b_inp, is_batched = self._as_batched(inp)

        if state is None:
            b_state = self._zeros_state(b_inp.size(0), b_inp.device, b_inp.dtype)
        else:
            b_state = state.unsqueeze(0) if (not is_batched and state.dim() == 1) else state

        gates = (
            b_inp @ self.weight_ih.t()
            + self.bias_ih
            + b_state @ self.weight_hh.t()
            + self.bias_hh
        )
        depth_pre, recurrent_pre, gate_y_pre, gate_h_pre = gates.chunk(4, 1)

        depth_candidate = self.depth_act(depth_pre)
        recurrent_candidate = self.recurrent_act(recurrent_pre)
        gate_y = self.gate_act(gate_y_pre)
        gate_h = self.gate_act(gate_h_pre)

        new_output = gate_y * b_inp + (1.0 - gate_y) * depth_candidate
        new_state = gate_h * b_state + (1.0 - gate_h) * recurrent_candidate

        if not is_batched:
            new_output = new_output.squeeze(0)
            new_state = new_state.squeeze(0)

        return new_output, new_state

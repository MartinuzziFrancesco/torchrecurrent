import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional
from ..base import (
    SingleStateRecurrentLayerBase,
    SingleStateCellBase,
    apply_init_,
    resolve_activation,
    resolve_init_name,
)


class DSGU(SingleStateRecurrentLayerBase):
    r"""Multi-layer Deep Simple Gated Unit neural network.

    [`PMLR <https://proceedings.mlr.press/v63/gao30.html>`_]

    Each layer consists of a :class:`DSGUCell`, which updates the hidden state
    according to:

    .. math::
        \begin{aligned}
        x_g(t) &= W_{ih}^{g}x(t) + b_{ih}^{g}, \\
        z_g(t) &= \phi\left(W_{hh}^{g}\left(x_g(t)\circ h(t-1)\right)
                  + b_{hh}^{g}\right), \\
        z_{out}(t) &= \psi\left(W_{ho}\left(z_g(t)\circ h(t-1)\right)\right), \\
        z_t &= \sigma_h\left(W_{ih}^{z}x(t) + b_{ih}^{z}
               + W_{hh}^{z}h(t-1) + b_{hh}^{z}\right), \\
        h(t) &= \left(1 - z_t\right)\circ h(t-1) + z_t\circ z_{out}(t).
        \end{aligned}

    where :math:`\sigma_h` is hard sigmoid, :math:`\phi` defaults to tanh,
    :math:`\psi` defaults to sigmoid, and :math:`\circ` denotes elementwise
    multiplication.

    Args:
        input_size: The number of expected features in the input `x`.
        hidden_size: The number of features in the hidden state `h`.
        num_layers: Number of recurrent layers. Default: 1
        dropout: If non-zero, introduces a `Dropout` layer on the outputs of each
            layer except the last layer, with dropout probability equal to
            :attr:`dropout`. Default: 0
        batch_first: If ``True``, then the input and output tensors are provided as
            `(batch, seq, feature)` instead of `(seq, batch, feature)`. Default: False
        bias: If ``False``, then the layer does not use input-side biases.
            Default: True
        recurrent_bias: If ``False``, then the layer does not use recurrent biases.
            Default: True
        nonlinearity: Nonlinearity :math:`\phi` for the modulation path.
            Default: :func:`torch.tanh`
        output_nonlinearity: Nonlinearity :math:`\psi` for the gated output.
            Default: :func:`torch.sigmoid`
        gate_nonlinearity: Activation for the update gate. Default:
            :class:`torch.nn.Hardsigmoid`
        kernel_init: Initializer for `W_{ih}^*`. Default:
            :func:`torch.nn.init.xavier_uniform_`
        recurrent_kernel_init: Initializer for recurrent weights. Default:
            :func:`torch.nn.init.xavier_uniform_`
        bias_init: Initializer for input-side biases. Default:
            :func:`torch.nn.init.zeros_`
        recurrent_bias_init: Initializer for recurrent biases. Default:
            :func:`torch.nn.init.zeros_`
        device: The desired device of parameters.
        dtype: The desired floating point type of parameters.

    Inputs: input, h_0
        - **input**: tensor of shape :math:`(L, H_{in})` for unbatched input,
          :math:`(L, N, H_{in})` when ``batch_first=False`` or
          :math:`(N, L, H_{in})` when ``batch_first=True`` containing the features of
          the input sequence.
        - **h_0**: tensor of shape :math:`(\text{num_layers}, H_{out})` for
          unbatched input or :math:`(\text{num_layers}, N, H_{out})` containing the
          initial hidden state for each element in the input sequence. Defaults to
          zeros if not provided.

    Outputs: output, h_n
        - **output**: tensor of shape :math:`(L, H_{out})` for unbatched input,
          :math:`(L, N, H_{out})` when ``batch_first=False`` or
          :math:`(N, L, H_{out})` when ``batch_first=True`` containing the output
          features `(h_t)` from the last layer of the DSGU, for each `t`.
        - **h_n**: tensor of shape :math:`(\text{num_layers}, H_{out})` for
          unbatched input or :math:`(\text{num_layers}, N, H_{out})` containing the
          final hidden state for each element in the sequence.

    .. seealso::
        :class:`DSGUCell`

    Examples::

        >>> rnn = DSGU(10, 20, num_layers=2, dropout=0.1)
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
        super(DSGU, self).__init__(
            input_size, hidden_size, num_layers, dropout, batch_first
        )
        self.initialize_cells(DSGUCell, **kwargs)


class SGU(SingleStateRecurrentLayerBase):
    r"""Multi-layer Simple Gated Unit neural network.

    [`arXiv <https://arxiv.org/abs/1604.02910>`_]

    Each layer consists of an :class:`SGUCell`, which updates the hidden state
    according to:

    .. math::
        \begin{aligned}
        x_g(t) &= W_{ih}^{g}x(t) + b_{ih}^{g}, \\
        z_g(t) &= \phi\left(W_{hh}^{g}\left(x_g(t)\circ h(t-1)\right)
                  + b_{hh}^{g}\right), \\
        z_{out}(t) &= \psi\left(z_g(t)\circ h(t-1)\right), \\
        z_t &= \sigma_h\left(W_{ih}^{z}x(t) + b_{ih}^{z}
               + W_{hh}^{z}h(t-1) + b_{hh}^{z}\right), \\
        h(t) &= \left(1 - z_t\right)\circ h(t-1) + z_t\circ z_{out}(t).
        \end{aligned}

    where :math:`\sigma_h` is hard sigmoid, :math:`\phi` defaults to tanh,
    :math:`\psi` defaults to softplus, and :math:`\circ` denotes elementwise
    multiplication.

    Args:
        input_size: The number of expected features in the input `x`.
        hidden_size: The number of features in the hidden state `h`.
        num_layers: Number of recurrent layers. Default: 1
        dropout: If non-zero, introduces a `Dropout` layer on the outputs of each
            layer except the last layer, with dropout probability equal to
            :attr:`dropout`. Default: 0
        batch_first: If ``True``, then the input and output tensors are provided as
            `(batch, seq, feature)` instead of `(seq, batch, feature)`. Default: False
        bias: If ``False``, then the layer does not use input-side biases.
            Default: True
        recurrent_bias: If ``False``, then the layer does not use recurrent biases.
            Default: True
        nonlinearity: Nonlinearity :math:`\phi` for the modulation path.
            Default: :func:`torch.tanh`
        output_nonlinearity: Nonlinearity :math:`\psi` for the gated output.
            Default: :class:`torch.nn.Softplus`
        gate_nonlinearity: Activation for the update gate. Default:
            :class:`torch.nn.Hardsigmoid`
        kernel_init: Initializer for `W_{ih}^*`. Default:
            :func:`torch.nn.init.xavier_uniform_`
        recurrent_kernel_init: Initializer for `W_{hh}^*`. Default:
            :func:`torch.nn.init.xavier_uniform_`
        bias_init: Initializer for input-side biases. Default:
            :func:`torch.nn.init.zeros_`
        recurrent_bias_init: Initializer for recurrent biases. Default:
            :func:`torch.nn.init.zeros_`
        device: The desired device of parameters.
        dtype: The desired floating point type of parameters.

    Inputs: input, h_0
        - **input**: tensor of shape :math:`(L, H_{in})` for unbatched input,
          :math:`(L, N, H_{in})` when ``batch_first=False`` or
          :math:`(N, L, H_{in})` when ``batch_first=True`` containing the features of
          the input sequence.
        - **h_0**: tensor of shape :math:`(\text{num_layers}, H_{out})` for
          unbatched input or :math:`(\text{num_layers}, N, H_{out})` containing the
          initial hidden state for each element in the input sequence. Defaults to
          zeros if not provided.

    Outputs: output, h_n
        - **output**: tensor of shape :math:`(L, H_{out})` for unbatched input,
          :math:`(L, N, H_{out})` when ``batch_first=False`` or
          :math:`(N, L, H_{out})` when ``batch_first=True`` containing the output
          features `(h_t)` from the last layer of the SGU, for each `t`.
        - **h_n**: tensor of shape :math:`(\text{num_layers}, H_{out})` for
          unbatched input or :math:`(\text{num_layers}, N, H_{out})` containing the
          final hidden state for each element in the sequence.

    .. seealso::
        :class:`SGUCell`

    Examples::

        >>> rnn = SGU(10, 20, num_layers=2, dropout=0.1)
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
        super(SGU, self).__init__(input_size, hidden_size, num_layers, dropout, batch_first)
        self.initialize_cells(SGUCell, **kwargs)


class SGUCell(SingleStateCellBase):
    r"""A Simple Gated Unit (SGU) cell.

    [`arXiv <https://arxiv.org/abs/1604.02910>`_]

    .. math::

        \begin{aligned}
        \mathbf{x}_g(t) &= \mathbf{W}_{ih}^{g}\mathbf{x}(t)
            + \mathbf{b}_{ih}^{g}, \\
        \mathbf{z}_g(t) &= \phi\left(
            \mathbf{W}_{hh}^{g}\left(\mathbf{x}_g(t)\circ\mathbf{h}(t-1)\right)
            + \mathbf{b}_{hh}^{g}\right), \\
        \mathbf{z}_{out}(t) &= \psi\left(\mathbf{z}_g(t)\circ\mathbf{h}(t-1)\right), \\
        \mathbf{z}_t &= \sigma_h\left(
            \mathbf{W}_{ih}^{z}\mathbf{x}(t) + \mathbf{b}_{ih}^{z}
            + \mathbf{W}_{hh}^{z}\mathbf{h}(t-1) + \mathbf{b}_{hh}^{z}
        \right), \\
        \mathbf{h}(t) &= \left(1 - \mathbf{z}_t\right)\circ\mathbf{h}(t-1)
            + \mathbf{z}_t\circ\mathbf{z}_{out}(t).
        \end{aligned}

    where :math:`\sigma_h` is hard sigmoid, :math:`\phi` defaults to tanh,
    :math:`\psi` defaults to softplus, and :math:`\circ` denotes elementwise
    multiplication.

    Args:
        input_size: The number of expected features in the input ``x``.
        hidden_size: The number of features in the hidden state ``h``.
        bias: If ``False``, the layer does not use input-side biases.
            Default: ``True``.
        recurrent_bias: If ``False``, the layer does not use recurrent biases.
            Default: ``True``.
        nonlinearity: Nonlinearity :math:`\phi` for the modulation path.
            Default: :func:`torch.tanh`.
        output_nonlinearity: Nonlinearity :math:`\psi` for the gated output.
            Default: :class:`torch.nn.Softplus`.
        gate_nonlinearity: Activation for the update gate.
            Default: :class:`torch.nn.Hardsigmoid`.
        kernel_init: Initializer for ``W_{ih}^*``.
            Default: :func:`torch.nn.init.xavier_uniform_`.
        recurrent_kernel_init: Initializer for ``W_{hh}^*``.
            Default: :func:`torch.nn.init.xavier_uniform_`.
        bias_init: Initializer for input-side biases when ``bias=True``.
            Default: :func:`torch.nn.init.zeros_`.
        recurrent_bias_init: Initializer for recurrent biases when
            ``recurrent_bias=True``. Default: :func:`torch.nn.init.zeros_`.
        device: The desired device of parameters.
        dtype: The desired floating point type of parameters.

    Inputs: input, h_0
        - **input** of shape ``(batch, input_size)`` or ``(input_size,)``:
          Tensor containing input features.
        - **h_0** of shape ``(batch, hidden_size)`` or ``(hidden_size,)``:
          Tensor containing the initial hidden state.

        If **h_0** is not provided, it defaults to zero.

    Outputs: h_1
        - **h_1** of shape ``(batch, hidden_size)`` or ``(hidden_size,)``:
          Tensor containing the next hidden state.

    Variables:
        weight_ih: The learnable input-hidden weights, of shape
            ``(2*hidden_size, input_size)``.
        weight_hh: The learnable hidden-hidden weights, of shape
            ``(2*hidden_size, hidden_size)``.
        bias_ih: The learnable input-hidden biases, of shape
            ``(2*hidden_size)`` if ``bias=True``.
        bias_hh: The learnable hidden-hidden biases, of shape
            ``(2*hidden_size)`` if ``recurrent_bias=True``.

    Examples::

        >>> cell = SGUCell(10, 20)
        >>> x = torch.randn(5, 3, 10)
        >>> h = torch.zeros(3, 20)
        >>> out = []
        >>> for t in range(x.size(0)):
        ...     h = cell(x[t], h)
        ...     out.append(h)
        >>> out = torch.stack(out, dim=0)
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
        nonlinearity="tanh",
        output_nonlinearity="softplus",
        gate_nonlinearity="hard_sigmoid",
        kernel_init=nn.init.xavier_uniform_,
        recurrent_kernel_init=nn.init.xavier_uniform_,
        bias_init=nn.init.zeros_,
        recurrent_bias_init=nn.init.zeros_,
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
        self.act = resolve_activation(nonlinearity)
        self.output_act = resolve_activation(output_nonlinearity)
        self.gate_act = resolve_activation(gate_nonlinearity)
        self.init_cfg["kernel"] = resolve_init_name(kernel_init, self.init_cfg["kernel"])
        self.init_cfg["recurrent_kernel"] = resolve_init_name(
            recurrent_kernel_init, self.init_cfg["recurrent_kernel"]
        )
        self.init_cfg["bias"] = resolve_init_name(bias_init, self.init_cfg["bias"])
        self.init_cfg["recurrent_bias"] = resolve_init_name(
            recurrent_bias_init, self.init_cfg["recurrent_bias"]
        )

        self._default_register_tensors(ih_mult=2, hh_mult=2)
        self.reset_parameters()
        self._cleanup_non_scriptable()

    def forward(self, inp: Tensor, state: Optional[Tensor] = None) -> Tensor:
        self._validate_input(inp)
        b_inp, is_batched = self._as_batched(inp)

        if state is None:
            b_state = self._zeros_state(b_inp.size(0), b_inp.device, b_inp.dtype)
        else:
            b_state = state.unsqueeze(0) if (not is_batched and state.dim() == 1) else state

        weight_hh_g, weight_hh_z = self.weight_hh.chunk(2, 0)
        bias_hh_g, bias_hh_z = self.bias_hh.chunk(2, 0)

        proj_ih = b_inp @ self.weight_ih.t() + self.bias_ih
        x_g, x_z = proj_ih.chunk(2, 1)

        z_g = self.act((x_g * b_state) @ weight_hh_g.t() + bias_hh_g)
        z_out = self.output_act(z_g * b_state)
        z_t = self.gate_act(x_z + b_state @ weight_hh_z.t() + bias_hh_z)
        new_state = (1.0 - z_t) * b_state + z_t * z_out

        if not is_batched:
            new_state = new_state.squeeze(0)

        return new_state


class DSGUCell(SGUCell):
    r"""A Deep Simple Gated Unit (DSGU) cell.

    [`PMLR <https://proceedings.mlr.press/v63/gao30.html>`_]

    .. math::

        \begin{aligned}
        \mathbf{x}_g(t) &= \mathbf{W}_{ih}^{g}\mathbf{x}(t)
            + \mathbf{b}_{ih}^{g}, \\
        \mathbf{z}_g(t) &= \phi\left(
            \mathbf{W}_{hh}^{g}\left(\mathbf{x}_g(t)\circ\mathbf{h}(t-1)\right)
            + \mathbf{b}_{hh}^{g}\right), \\
        \mathbf{z}_{out}(t) &= \psi\left(
            \mathbf{W}_{ho}\left(\mathbf{z}_g(t)\circ\mathbf{h}(t-1)\right)
        \right), \\
        \mathbf{z}_t &= \sigma_h\left(
            \mathbf{W}_{ih}^{z}\mathbf{x}(t) + \mathbf{b}_{ih}^{z}
            + \mathbf{W}_{hh}^{z}\mathbf{h}(t-1) + \mathbf{b}_{hh}^{z}
        \right), \\
        \mathbf{h}(t) &= \left(1 - \mathbf{z}_t\right)\circ\mathbf{h}(t-1)
            + \mathbf{z}_t\circ\mathbf{z}_{out}(t).
        \end{aligned}

    DSGU differs from :class:`SGUCell` by adding the learned projection
    :math:`W_{ho}` before the output activation. The default output nonlinearity is
    sigmoid, matching the DSGU graph in the paper.

    Args:
        input_size: The number of expected features in the input ``x``.
        hidden_size: The number of features in the hidden state ``h``.
        bias: If ``False``, the layer does not use input-side biases.
            Default: ``True``.
        recurrent_bias: If ``False``, the layer does not use recurrent biases.
            Default: ``True``.
        nonlinearity: Nonlinearity :math:`\phi` for the modulation path.
            Default: :func:`torch.tanh`.
        output_nonlinearity: Nonlinearity :math:`\psi` for the gated output.
            Default: :func:`torch.sigmoid`.
        gate_nonlinearity: Activation for the update gate.
            Default: :class:`torch.nn.Hardsigmoid`.
        kernel_init: Initializer for ``W_{ih}^*``.
            Default: :func:`torch.nn.init.xavier_uniform_`.
        recurrent_kernel_init: Initializer for recurrent weights.
            Default: :func:`torch.nn.init.xavier_uniform_`.
        bias_init: Initializer for input-side biases when ``bias=True``.
            Default: :func:`torch.nn.init.zeros_`.
        recurrent_bias_init: Initializer for recurrent biases when
            ``recurrent_bias=True``. Default: :func:`torch.nn.init.zeros_`.
        device: The desired device of parameters.
        dtype: The desired floating point type of parameters.

    Inputs: input, h_0
        - **input** of shape ``(batch, input_size)`` or ``(input_size,)``:
          Tensor containing input features.
        - **h_0** of shape ``(batch, hidden_size)`` or ``(hidden_size,)``:
          Tensor containing the initial hidden state.

        If **h_0** is not provided, it defaults to zero.

    Outputs: h_1
        - **h_1** of shape ``(batch, hidden_size)`` or ``(hidden_size,)``:
          Tensor containing the next hidden state.

    Variables:
        weight_ih: The learnable input-hidden weights, of shape
            ``(2*hidden_size, input_size)``.
        weight_hh: The learnable hidden-hidden weights, of shape
            ``(2*hidden_size, hidden_size)``.
        weight_ho: The learnable output projection weights, of shape
            ``(hidden_size, hidden_size)``.
        bias_ih: The learnable input-hidden biases, of shape
            ``(2*hidden_size)`` if ``bias=True``.
        bias_hh: The learnable hidden-hidden biases, of shape
            ``(2*hidden_size)`` if ``recurrent_bias=True``.

    Examples::

        >>> cell = DSGUCell(10, 20)
        >>> x = torch.randn(5, 3, 10)
        >>> h = torch.zeros(3, 20)
        >>> out = []
        >>> for t in range(x.size(0)):
        ...     h = cell(x[t], h)
        ...     out.append(h)
        >>> out = torch.stack(out, dim=0)
    """

    weight_ho: Tensor

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        recurrent_bias: bool = True,
        nonlinearity="tanh",
        output_nonlinearity="sigmoid",
        gate_nonlinearity="hard_sigmoid",
        kernel_init=nn.init.xavier_uniform_,
        recurrent_kernel_init=nn.init.xavier_uniform_,
        bias_init=nn.init.zeros_,
        recurrent_bias_init=nn.init.zeros_,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super(SGUCell, self).__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            bias=bias,
            recurrent_bias=recurrent_bias,
            device=device,
            dtype=dtype,
        )
        self.act = resolve_activation(nonlinearity)
        self.output_act = resolve_activation(output_nonlinearity)
        self.gate_act = resolve_activation(gate_nonlinearity)
        self.init_cfg["kernel"] = resolve_init_name(kernel_init, self.init_cfg["kernel"])
        self.init_cfg["recurrent_kernel"] = resolve_init_name(
            recurrent_kernel_init, self.init_cfg["recurrent_kernel"]
        )
        self.init_cfg["bias"] = resolve_init_name(bias_init, self.init_cfg["bias"])
        self.init_cfg["recurrent_bias"] = resolve_init_name(
            recurrent_bias_init, self.init_cfg["recurrent_bias"]
        )

        self._default_register_tensors(ih_mult=2, hh_mult=2)
        self.weight_ho = nn.Parameter(
            torch.empty(
                self.hidden_size,
                self.hidden_size,
                device=self._init_device,
                dtype=self._init_dtype,
            )
        )
        self.reset_parameters()
        self._cleanup_non_scriptable()

    def reset_parameters(self) -> None:
        super().reset_parameters()
        apply_init_(self.weight_ho, self.init_cfg["recurrent_kernel"])

    def forward(self, inp: Tensor, state: Optional[Tensor] = None) -> Tensor:
        self._validate_input(inp)
        b_inp, is_batched = self._as_batched(inp)

        if state is None:
            b_state = self._zeros_state(b_inp.size(0), b_inp.device, b_inp.dtype)
        else:
            b_state = state.unsqueeze(0) if (not is_batched and state.dim() == 1) else state

        weight_hh_g, weight_hh_z = self.weight_hh.chunk(2, 0)
        bias_hh_g, bias_hh_z = self.bias_hh.chunk(2, 0)

        proj_ih = b_inp @ self.weight_ih.t() + self.bias_ih
        x_g, x_z = proj_ih.chunk(2, 1)

        z_g = self.act((x_g * b_state) @ weight_hh_g.t() + bias_hh_g)
        z_out = self.output_act((z_g * b_state) @ self.weight_ho.t())
        z_t = self.gate_act(x_z + b_state @ weight_hh_z.t() + bias_hh_z)
        new_state = (1.0 - z_t) * b_state + z_t * z_out

        if not is_batched:
            new_state = new_state.squeeze(0)

        return new_state

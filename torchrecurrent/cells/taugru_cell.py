from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from ..base import (
    SingleStateCellBase,
    SingleStateRecurrentLayerBase,
    resolve_activation,
    resolve_init_name,
)


class tauGRU(SingleStateRecurrentLayerBase):
    r"""Multi-layer tau-GRU neural network.

    [`arXiv <https://arxiv.org/abs/2212.00228>`_]

    Each layer consists of a :class:`tauGRUCell`, which updates the hidden state
    with a weighted time-delay feedback term:

    .. math::
        \begin{aligned}
        u_n &= \tanh(W_1 h_n + U_1 x_n), \\
        z_n &= \tanh(W_2 h_{n-d} + U_2 x_n), \\
        g_n &= \sigma(W_3 h_n + U_3 x_n), \\
        a_n &= \sigma(W_4 h_n + U_4 x_n), \\
        h_{n+1} &= (1 - g_n) \circ h_n + g_n \circ (u_n + a_n \circ z_n),
        \end{aligned}

    where :math:`d` is the integer delay in recurrent steps. Delayed states
    before the beginning of the sequence are initialized to zero.

    Args:
        input_size: The number of expected features in the input `x`.
        hidden_size: The number of features in the hidden state `h`.
        num_layers: Number of recurrent layers. Default: 1
        dropout: If non-zero, introduces a `Dropout` layer on the outputs of
            each layer except the last. Default: 0
        batch_first: If ``True``, input and output tensors are provided as
            `(batch, seq, feature)` instead of `(seq, batch, feature)`.
            Default: False
        delay: Integer delay :math:`d` in recurrent steps. Default: 1
        bias: If ``False``, the layer does not use input-side biases.
            Default: True
        recurrent_bias: If ``False``, the layer does not use recurrent biases.
            Default: True
        nonlinearity: Nonlinearity for :math:`u_n` and :math:`z_n`.
            Default: :func:`torch.tanh`
        gate_nonlinearity: Activation for :math:`g_n` and :math:`a_n`.
            Default: :func:`torch.sigmoid`
        kernel_init: Initializer for `U_i`. Default:
            :func:`torch.nn.init.xavier_uniform_`
        recurrent_kernel_init: Initializer for `W_i`. Default:
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
          :math:`(N, L, H_{in})` when ``batch_first=True``.
        - **h_0**: tensor of shape :math:`(\text{num_layers}, H_{out})` for
          unbatched input or :math:`(\text{num_layers}, N, H_{out})`.
          Defaults to zeros if not provided.

    Outputs: output, h_n
        - **output**: tensor containing the output features from the last layer,
          for each timestep.
        - **h_n**: tensor containing the final hidden state for each layer.

    Attributes:
        cells.{k}.weight_ih : input-hidden weights of shape
            `(4*hidden_size, input_size)` for `k = 0`, otherwise
            `(4*hidden_size, hidden_size)`.
        cells.{k}.weight_hh : hidden-hidden weights of shape
            `(4*hidden_size, hidden_size)`.
        cells.{k}.bias_ih : input-hidden biases of shape `(4*hidden_size)`.
            Only present when ``bias=True``.
        cells.{k}.bias_hh : hidden-hidden biases of shape `(4*hidden_size)`.
            Only present when ``recurrent_bias=True``.

    .. seealso::
        :class:`tauGRUCell`
    """

    __constants__ = ["delay"]

    delay: int

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        batch_first: bool = False,
        delay: int = 1,
        **kwargs,
    ):
        if delay < 1:
            raise ValueError("delay must be a positive integer.")
        super(tauGRU, self).__init__(
            input_size, hidden_size, num_layers, dropout, batch_first
        )
        self.delay = int(delay)
        self.initialize_cells(tauGRUCell, **kwargs)

    def extra_repr(self) -> str:
        parts = [super().extra_repr()]
        if self.delay != 1:
            parts.append(f"delay={self.delay}")
        return ", ".join(parts)

    def forward(self, inp: Tensor, state: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        if self.batch_first:
            inp = inp.transpose(0, 1)

        seq_len, batch_size, _ = inp.size()

        if state is None:
            state = torch.zeros(
                self.num_layers,
                batch_size,
                self.hidden_size,
                dtype=inp.dtype,
                device=inp.device,
            )

        histories = torch.jit.annotate(List[Tensor], [])
        for _ in range(self.delay):
            histories.append(torch.zeros_like(state))

        outputs = torch.jit.annotate(List[Tensor], [])

        for t in range(seq_len):
            x = inp[t]
            new_states = torch.jit.annotate(List[Tensor], [])
            delayed_state = histories[0]

            for layer_idx, cell in enumerate(self.cells):
                h_prev = state[layer_idx]
                h_delay = delayed_state[layer_idx]
                h_new = cell(x, h_prev, h_delay)
                new_states.append(h_new)
                x = h_new
                if self.dropout_layer is not None and layer_idx < self.num_layers - 1:
                    x = self.dropout_layer(x)

            histories = histories[1:] + [state]
            state = torch.stack(new_states, dim=0)
            outputs.append(x)

        out = torch.stack(outputs, dim=0)
        if self.batch_first:
            out = out.transpose(0, 1)
        return out, state


class tauGRUCell(SingleStateCellBase):
    r"""A tau-GRU cell with weighted time-delay feedback.

    [`arXiv <https://arxiv.org/abs/2212.00228>`_]

    .. math::

        \begin{aligned}
        \mathbf{u}_n &= \phi(W_1 \mathbf{h}_n + U_1 \mathbf{x}_n), \\
        \mathbf{z}_n &= \phi(W_2 \mathbf{h}_{n-d} + U_2 \mathbf{x}_n), \\
        \mathbf{g}_n &= \sigma(W_3 \mathbf{h}_n + U_3 \mathbf{x}_n), \\
        \mathbf{a}_n &= \sigma(W_4 \mathbf{h}_n + U_4 \mathbf{x}_n), \\
        \mathbf{h}_{n+1} &= (1 - \mathbf{g}_n) \circ \mathbf{h}_n
            + \mathbf{g}_n \circ
              (\mathbf{u}_n + \mathbf{a}_n \circ \mathbf{z}_n).
        \end{aligned}

    Args:
        input_size: The number of expected features in the input ``x``.
        hidden_size: The number of features in the hidden state ``h``.
        bias: If ``False``, disables input-side biases. Default: ``True``.
        recurrent_bias: If ``False``, disables recurrent biases. Default: ``True``.
        nonlinearity: Nonlinearity for the instantaneous and delayed candidates.
            Default: :func:`torch.tanh`.
        gate_nonlinearity: Activation for the update and feedback gates.
            Default: :func:`torch.sigmoid`.
        kernel_init: Initializer for ``weight_ih``. Default:
            :func:`torch.nn.init.xavier_uniform_`.
        recurrent_kernel_init: Initializer for ``weight_hh``. Default:
            :func:`torch.nn.init.xavier_uniform_`.
        bias_init: Initializer for input-side biases when ``bias=True``.
            Default: :func:`torch.nn.init.zeros_`.
        recurrent_bias_init: Initializer for recurrent biases when
            ``recurrent_bias=True``. Default: :func:`torch.nn.init.zeros_`.
        device: The desired device of parameters.
        dtype: The desired floating point type of parameters.

    Inputs: input, hidden, delayed_hidden
        - **input** of shape ``(batch, input_size)`` or ``(input_size,)``:
          tensor containing input features.
        - **hidden** of shape ``(batch, hidden_size)`` or ``(hidden_size,)``:
          tensor containing the current hidden state. Defaults to zero.
        - **delayed_hidden** of shape ``(batch, hidden_size)`` or
          ``(hidden_size,)``: tensor containing the delayed hidden state.
          Defaults to zero.

    Outputs: h_1
        - **h_1** of shape ``(batch, hidden_size)`` or ``(hidden_size,)``:
          tensor containing the next hidden state.

    Variables:
        weight_ih: input-hidden weights, of shape ``(4*hidden_size, input_size)``
        weight_hh: hidden-hidden weights, of shape ``(4*hidden_size, hidden_size)``
        bias_ih: input biases, of shape ``(4*hidden_size,)`` if ``bias=True``
        bias_hh: hidden biases, of shape ``(4*hidden_size,)`` if
            ``recurrent_bias=True``
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
        gate_nonlinearity="sigmoid",
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

    def forward(
        self,
        inp: Tensor,
        state: Optional[Tensor] = None,
        delayed_state: Optional[Tensor] = None,
    ) -> Tensor:
        self._validate_input(inp)
        b_inp, is_batched = self._as_batched(inp)

        if state is None:
            b_state = self._zeros_state(b_inp.size(0), b_inp.device, b_inp.dtype)
        else:
            b_state = state.unsqueeze(0) if (not is_batched and state.dim() == 1) else state

        if delayed_state is None:
            b_delayed = self._zeros_state(b_inp.size(0), b_inp.device, b_inp.dtype)
        else:
            b_delayed = (
                delayed_state.unsqueeze(0)
                if (not is_batched and delayed_state.dim() == 1)
                else delayed_state
            )

        weight_ih_u, weight_ih_z, weight_ih_g, weight_ih_a = self.weight_ih.chunk(4, 0)
        weight_hh_u, weight_hh_z, weight_hh_g, weight_hh_a = self.weight_hh.chunk(4, 0)
        bias_ih_u, bias_ih_z, bias_ih_g, bias_ih_a = self.bias_ih.chunk(4, 0)
        bias_hh_u, bias_hh_z, bias_hh_g, bias_hh_a = self.bias_hh.chunk(4, 0)

        u = self.act(
            b_inp @ weight_ih_u.t() + bias_ih_u + b_state @ weight_hh_u.t() + bias_hh_u
        )
        z = self.act(
            b_inp @ weight_ih_z.t() + bias_ih_z + b_delayed @ weight_hh_z.t() + bias_hh_z
        )
        g = self.gate_act(
            b_inp @ weight_ih_g.t() + bias_ih_g + b_state @ weight_hh_g.t() + bias_hh_g
        )
        a = self.gate_act(
            b_inp @ weight_ih_a.t() + bias_ih_a + b_state @ weight_hh_a.t() + bias_hh_a
        )

        new_state = (1.0 - g) * b_state + g * (u + a * z)

        if not is_batched:
            new_state = new_state.squeeze(0)

        return new_state

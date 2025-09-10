import torch
from torch import nn
from torch import Tensor
from typing import Optional, Callable, Tuple, Union
from ..base import BaseSingleRecurrentLayer, BaseSingleRecurrentCell


class BR(BaseSingleRecurrentLayer):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        batch_first: bool = False,
        **kwargs,
    ):
        super(BR, self).__init__(input_size, hidden_size, num_layers, dropout, batch_first)
        self.initialize_cells(BRCell, **kwargs)


class BRCell(BaseSingleRecurrentCell):
    r"""A Bistable recurrent cell.

    [`pub <https://doi.org/10.1371/journal.pone.0252676>`_]

    .. math::

        \mathbf{a}(t) &= 1 + \tanh\Bigl(\mathbf{W}_{ih}^{a}\,\mathbf{x}(t)
            + \mathbf{b}_{ih}^{a}
            + \mathbf{w}_{hh}^{a} \,\circ\, \mathbf{h}(t-1)
            + \mathbf{b}_{hh}^{a}\Bigr), \\
        \mathbf{c}(t) &= \sigma\Bigl(\mathbf{W}_{ih}^{c}\,\mathbf{x}(t)
            + \mathbf{b}_{ih}^{c}
            + \mathbf{w}_{hh}^{c} \,\circ\, \mathbf{h}(t-1)
            + \mathbf{b}_{hh}^{c}\Bigr), \\
        \mathbf{h}(t) &= \mathbf{c}(t)\,\circ\,\mathbf{h}(t-1)
            \;+\;\bigl(1 - \mathbf{c}(t)\bigr)\,\circ\,
            \tanh\Bigl(\mathbf{W}_{ih}^{h}\,\mathbf{x}(t)
            + \mathbf{b}_{ih}^{h}
            + \mathbf{a}(t)\,\circ\,\mathbf{h}(t-1)\Bigr).

    Args:
        input_size: The number of expected features in the input ``x``.
        hidden_size: The number of features in the hidden state ``h``.
        bias: If ``False``, the layer does not use input-side biases.
            Default: ``True``.
        recurrent_bias: If ``False``, the layer does not use recurrent biases.
            Default: ``True``.
        kernel_init: Initializer for input–hidden weights ``W_ih^*``.
            Default: :func:`torch.nn.init.xavier_uniform_`.
        recurrent_kernel_init: Initializer for hidden recurrence vectors
            ``w_{hh}^*``. Default: :func:`torch.nn.init.normal_`.
        bias_init: Initializer for input-side biases ``b_{ih}^*`` when
            ``bias=True``. Default: :func:`torch.nn.init.zeros_`.
        recurrent_bias_init: Initializer for hidden biases ``b_{hh}^*`` when
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
        weight_ih: The learnable input–hidden weights, of shape
            ``(3*hidden_size, input_size)`` (split into a/c/h parts).
        weight_hh: The learnable hidden recurrence vectors, of shape
            ``(2*hidden_size,)`` (split into a/c parts).
        bias_ih: The learnable input–hidden biases,
            of shape ``(3*hidden_size)``.
        bias_hh: The learnable hidden biases,
            of shape ``(2*hidden_size)``.

    Examples::

        >>> cell = BRCell(10, 20)
        >>> x = torch.randn(5, 3, 10)     # (time_steps, batch, input_size)
        >>> h = torch.zeros(3, 20)        # (batch, hidden_size)
        >>> out = []
        >>> for t in range(x.size(0)):
        ...     h = cell(x[t], h)
        ...     out.append(h)
        >>> out = torch.stack(out, dim=0) # (time_steps, batch, hidden_size)
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
        recurrent_kernel_init: Callable = nn.init.normal_,
        bias_init: Callable = nn.init.zeros_,
        recurrent_bias_init: Callable = nn.init.zeros_,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super(BRCell, self).__init__(
            input_size, hidden_size, bias, recurrent_bias, device=device, dtype=dtype
        )
        self.kernel_init = kernel_init
        self.recurrent_kernel_init = recurrent_kernel_init
        self.bias_init = bias_init
        self.recurrent_bias_init = recurrent_bias_init

        self._register_tensors(
            {
                "weight_ih": ((3 * hidden_size, input_size), True),
                "weight_hh": ((2 * hidden_size,), True),
                "bias_ih": ((3 * hidden_size,), bias),
                "bias_hh": ((2 * hidden_size,), recurrent_bias),
            }
        )
        self.init_weights()

    def forward(
        self, inp: Tensor, state: Optional[Union[Tensor, Tuple[Tensor, ...]]] = None
    ) -> Tensor:
        state = self._check_state(state)
        self._validate_input(inp)
        self._validate_state(state)
        inp, state, is_batched = self._preprocess_input_and_state(inp, state)

        input_exp = inp @ self.weight_ih.t() + self.bias_ih
        input_exp_1, input_exp_2, input_exp_3 = input_exp.chunk(3, 1)
        rec_matrix_1, rec_matrix_2 = self.weight_hh.chunk(2)
        bias_hh_1, bias_hh_2 = self.bias_hh.chunk(2, 0)

        h1 = input_exp_1 + rec_matrix_1 * state + bias_hh_1
        h2 = input_exp_2 + rec_matrix_2 * state + bias_hh_2
        modulation_gate = 1.0 + torch.tanh(h1)
        candidate_state = torch.sigmoid(h2)
        h3 = input_exp_3 + modulation_gate * state
        new_state = candidate_state * state + (1.0 - candidate_state) * torch.tanh(h3)

        if not is_batched:
            new_state = new_state.squeeze(0)

        return new_state


class NBR(BaseSingleRecurrentLayer):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        batch_first: bool = False,
        **kwargs,
    ):
        super(NBR, self).__init__(input_size, hidden_size, num_layers, dropout, batch_first)
        self.initialize_cells(NBRCell, **kwargs)


class NBRCell(BaseSingleRecurrentCell):
    r"""A Neuromodulated Bistable Recurrent cell.

    [`pub <https://doi.org/10.1371/journal.pone.0252676>`_]

    .. math::

        \mathbf{a}(t) &= 1 + \tanh\bigl(\mathbf{W}_{ih}^{a}\,\mathbf{x}(t)
            + \mathbf{b}_{ih}^{a}
            + \mathbf{W}_{hh}^{a}\,\circ\,\mathbf{h}(t-1)
            + \mathbf{b}_{hh}^{a}\bigr), \\[6pt]
        \mathbf{c}(t) &= \sigma\bigl(\mathbf{W}_{ih}^{c}\,\mathbf{x}(t)
            + \mathbf{b}_{ih}^{c}
            + \mathbf{W}_{hh}^{c}\,\circ\,\mathbf{h}(t-1)
            + \mathbf{b}_{hh}^{c}\bigr), \\[6pt]
        \mathbf{h}(t) &= \mathbf{c}(t)\,\circ\,\mathbf{h}(t-1)
            \;+\;\bigl(1 - \mathbf{c}(t)\bigr)\,\circ\,
            \tanh\bigl(\mathbf{W}_{ih}^{h}\,\mathbf{x}(t)
            + \mathbf{b}_{ih}^{h}
            + \mathbf{a}(t)\,\circ\,\mathbf{h}(t-1)\bigr).

    Args:
        input_size: The number of expected features in the input ``x``
        hidden_size: The number of features in the hidden state ``h``
        bias: If ``False``, the layer does not use input-side biases.
            Default: ``True``
        recurrent_bias: If ``False``, the layer does not use recurrent biases.
            Default: ``True``
        kernel_init: Initializer for input–hidden weights ``W_ih^*``.
            Default: :func:`torch.nn.init.xavier_uniform_`
        recurrent_kernel_init: Initializer for hidden–hidden weights ``W_hh^*``.
            Default: :func:`torch.nn.init.xavier_uniform_`
        bias_init: Initializer for input-side biases ``b_{ih}^*`` when ``bias=True``.
            Default: :func:`torch.nn.init.zeros_`
        recurrent_bias_init: Initializer for hidden biases ``b_{hh}^*``
            when ``recurrent_bias=True``. Default: :func:`torch.nn.init.zeros_`
        device: The desired device of parameters.
        dtype: The desired floating point type of parameters.

    Inputs: input, h_0
        - **input** of shape ``(batch, input_size)`` or ``(input_size,)``:
          tensor containing input features
        - **h_0** of shape ``(batch, hidden_size)`` or ``(hidden_size,)``:
          tensor containing the initial hidden state

        If **h_0** is not provided, it defaults to zero.

    Outputs: h_1
        - **h_1** of shape ``(batch, hidden_size)`` or ``(hidden_size,)``:
          tensor containing the next hidden state

    Variables:
        weight_ih: the learnable input–hidden weights,
            of shape ``(3*hidden_size, input_size)`` (split into a/c/h parts)
        weight_hh: the learnable hidden–hidden weights,
            of shape ``(2*hidden_size, hidden_size)`` (split into a/c parts)
        bias_ih: the learnable input–hidden biases,
            of shape ``(3*hidden_size)``
        bias_hh: the learnable hidden–hidden biases,
            of shape ``(2*hidden_size)``

    Examples::

        >>> cell = NBRCell(10, 20)
        >>> x = torch.randn(5, 3, 10)     # (time_steps, batch, input_size)
        >>> h = torch.zeros(3, 20)        # (batch, hidden_size)
        >>> out = []
        >>> for t in range(x.size(0)):
        ...     h = cell(x[t], h)
        ...     out.append(h)
        >>> out = torch.stack(out, dim=0) # (time_steps, batch, hidden_size)
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
        super(NBRCell, self).__init__(
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
                "bias_ih": ((3 * hidden_size,), bias),
                "bias_hh": ((2 * hidden_size,), recurrent_bias),
            }
        )
        self.init_weights()

    def forward(
        self, inp: Tensor, state: Optional[Union[Tensor, Tuple[Tensor, ...]]] = None
    ) -> Tensor:
        state = self._check_state(state)
        self._validate_input(inp)
        self._validate_state(state)
        inp, state, is_batched = self._preprocess_input_and_state(inp, state)

        input_exp = inp @ self.weight_ih.t() + self.bias_ih
        input_exp_1, input_exp_2, input_exp_3 = input_exp.chunk(3, 1)
        rec_matrix_1, rec_matrix_2 = self.weight_hh.chunk(2, 0)
        bias_hh_1, bias_hh_2 = self.bias_hh.chunk(2, 0)

        h1 = input_exp_1 + state @ rec_matrix_1.t() + bias_hh_1
        h2 = input_exp_2 + state @ rec_matrix_2.t() + bias_hh_2
        modulation_gate = 1.0 + torch.tanh(h1)
        candidate_state = torch.sigmoid(h2)
        h3 = input_exp_3 + modulation_gate * state

        new_state = candidate_state * state + (1.0 - candidate_state) * torch.tanh(h3)

        if not is_batched:
            new_state = new_state.squeeze(0)

        return new_state

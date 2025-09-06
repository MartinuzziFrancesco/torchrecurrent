import torch
from torch import Tensor
import torch.nn as nn
from typing import Callable, Optional, Tuple, Union
from ..base import BaseSingleRecurrentLayer, BaseSingleRecurrentCell


class IndRNN(BaseSingleRecurrentLayer):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        batch_first: bool = False,
        **kwargs,
    ):
        super(IndRNN, self).__init__(
            input_size, hidden_size, num_layers, dropout, batch_first
        )
        self.initialize_cells(IndRNNCell, **kwargs)


class IndRNNCell(BaseSingleRecurrentCell):
    r"""An Independently Recurrent Neural Network (IndRNN) cell [`arXiv <https://arxiv.org/abs/1803.04831>`_].

        .. math::

            \mathbf{h}(t) = \phi\bigl(\mathbf{W}_{ih}\,\mathbf{x}(t)
                + \mathbf{b}_{ih}
                + \mathbf{w}_{hh}\,\circ\,\mathbf{h}(t-1)\bigr)

        where :math:`\circ` denotes element‐wise (Hadamard) product and
        :math:`\phi` is a pointwise nonlinearity (e.g., tanh).

        Args:
            input_size: The number of expected features in the input ``x``
            hidden_size: The number of features in the hidden state ``h``
            bias: If ``False``, the layer does not use input-side bias ``b_{ih}``. Default: ``True``
            recurrent_bias: If ``False``, the layer does not use recurrent bias ``b_{hh}``. Default: ``True``
            nonlinearity: Activation function :math:`\phi`. Default: :func:`torch.tanh`
            kernel_init: Initializer for ``W_{ih}``. Default: :func:`torch.nn.init.xavier_uniform_`
            recurrent_kernel_init: Initializer for the recurrent vector ``w_{hh}``. Default: :func:`torch.nn.init.normal_`
            bias_init: Initializer for ``b_{ih}`` when ``bias=True``. Default: :func:`torch.nn.init.zeros_`
            device: The desired device of parameters.
            dtype: The desired floating point type of parameters.

        Inputs: input, h_0
            - **input** of shape ``(batch, input_size)`` or ``(input_size,)``: tensor containing input features
            - **h_0** of shape ``(batch, hidden_size)`` or ``(hidden_size,)``: tensor containing the initial hidden state

            If **h_0** is not provided, it defaults to zero.

        Outputs: h_1
            - **h_1** of shape ``(batch, hidden_size)`` or ``(hidden_size,)``: tensor containing the next hidden state

        Variables:
            weight_ih: the learnable input–hidden weights, of shape ``(hidden_size, input_size)``
            vector_u: the learnable recurrent vector ``w_{hh}``, shape ``(hidden_size,)``
            bias_ih: the learnable input–hidden bias, of shape ``(hidden_size)`` if ``bias=True``

        Examples::

            >>> cell = IndRNNCell(10, 20)
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
        "nonlinearity",
        "kernel_init",
        "recurrent_kernel_init",
        "bias_init",
    ]

    weight_ih: Tensor
    vector_u: Tensor
    bias_ih: Tensor
    bias_hh: Tensor

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        recurrent_bias: bool = True,
        nonlinearity: Callable = torch.tanh,
        kernel_init: Callable = nn.init.xavier_uniform_,
        recurrent_kernel_init: Callable = nn.init.normal_,
        bias_init: Callable = nn.init.zeros_,
        recurrent_bias_init: Callable = nn.init.zeros_,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super(IndRNNCell, self).__init__(
            input_size, hidden_size, bias, device=device, dtype=dtype
        )
        self.nonlinearity = nonlinearity
        self.kernel_init = kernel_init
        self.recurrent_kernel_init = recurrent_kernel_init
        self.bias_init = bias_init
        self.recurrent_bias_init = recurrent_bias_init

        self._register_tensors(
            {
                "weight_ih": ((hidden_size, input_size), True),
                "vector_u": ((hidden_size,), True),
                "bias_ih": ((hidden_size,), bias),
                "bias_hh": ((hidden_size,), recurrent_bias),
            }
        )
        self.init_weights()

    def init_weights(self):
        for name, param in self.named_parameters():
            if "weight_ih" in name:
                self.kernel_init(param)
            elif "vector_u" in name:
                self.recurrent_kernel_init(param)
            elif "bias_ih" in name:
                self.bias_init(param)
            elif "bias_hh" in name:
                self.recurrent_bias_init(param)

    def forward(
        self, inp: Tensor, state: Optional[Union[Tensor, Tuple[Tensor, ...]]] = None
    ) -> Tensor:
        state = self._check_state(state)
        self._validate_input(inp)
        self._validate_state(state)
        inp, state, is_batched = self._preprocess_input_and_state(inp, state)

        new_state = (
            inp @ self.weight_ih.t() + self.bias_ih + self.vector_u * state + self.bias_hh
        )
        new_state = self.nonlinearity(new_state)

        if not is_batched:
            new_state = new_state.squeeze(0)

        return new_state

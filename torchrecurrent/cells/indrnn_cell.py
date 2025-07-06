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
    r"""An Independently Recurrent Neural Network (IndRNN) cell.

    In an IndRNN, each hidden unit has its own scalar recurrent weight,
    enabling deep stacking without gradient vanishing/exploding.

    .. math::

        \mathbf{h}(t) = \phi\bigl(\mathbf{W}_{ih}\,\mathbf{x}(t)
            + \mathbf{b}_{ih}
            + \mathbf{w}_{hh}\,\circ\,\mathbf{h}(t-1)\bigr)

    where :math:`\circ` denotes element‐wise (Hadamard) product and
    :math:`\phi` is a pointwise nonlinearity (e.g. `tanh`).

    Args:
        input_size (int): size of each input vector :math:`\mathbf{x}(t)`.
        hidden_size (int): size of the hidden state :math:`\mathbf{h}(t)`.
        bias (bool, optional): if ``False``, disables bias :math:`\mathbf{b}_{ih}`.
            Default: ``True``.
        activation_fn (Callable, optional): activation function :math:`\phi`.
            Default: ``torch.tanh``.
        kernel_init (Callable, optional): initializer for :math:`\mathbf{W}_{ih}`.
            Default: ``nn.init.xavier_uniform_``.
        recurrent_kernel_init (Callable, optional): initializer for the vector
            :math:`\mathbf{w}_{hh}`. Default: ``nn.init.normal_``.
        bias_init (Callable, optional): initializer for :math:`\mathbf{b}_{ih}`
            when `bias=True`. Default: ``nn.init.zeros_``.
        device (torch.device, optional): device of the parameters.
            Default: CPU.
        dtype (torch.dtype, optional): data type of the parameters.
            Default: PyTorch default float.

    Inputs: input, hidden
        - **input** (Tensor): shape `(H_in,)` or `(N, H_in)`, where `H_in = input_size`.
        - **hidden** (Tensor, optional): previous hidden state of shape
            `(H_out,)` or `(N, H_out)`, where `H_out = hidden_size`.
            Defaults to zero if not provided.

    Outputs: h’
        - **h’** (Tensor): next hidden state, same shape as **hidden**.

    Shape:
        - input: :math:`(N, H_{\text{in}})` or :math:`(H_{\text{in}})`.
        - hidden: :math:`(N, H_{\text{out}})` or :math:`(H_{\text{out}})`.
        - output: :math:`(N, H_{\text{out}})` or :math:`(H_{\text{out}})`.

    Attributes:
        weight_ih (Tensor): input‐to‐hidden weights, shape `(hidden_size, input_size)`.
        vector_u (Tensor): hidden recurrence vector, shape `(hidden_size,)`.
        bias_ih (Tensor): input bias, shape `(hidden_size,)` if `bias=True`.

    Examples::
        >>> cell = IndRNNCell(10, 20)
        >>> x = torch.randn(5, 10)
        >>> h0 = torch.zeros(20)
        >>> hx = h0
        >>> outputs = []
        >>> for t in range(x.size(0)):
        ...     hx = cell(x[t], hx)
        ...     outputs.append(hx)
    """
    weight_ih: Tensor
    vector_u: Tensor
    bias_ih: Tensor

    def __init__(self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        activation_fn: Callable = torch.tanh,
        kernel_init: Callable = nn.init.xavier_uniform_,
        recurrent_kernel_init: Callable = nn.init.normal_,
        bias_init: Callable = nn.init.zeros_,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super(IndRNNCell, self).__init__(
            input_size, hidden_size, bias, device = device, dtype = dtype
        )
        self.activation_fn = activation_fn
        self.kernel_init = kernel_init
        self.recurrent_kernel_init = recurrent_kernel_init
        self.bias_init = bias_init

        self._register_tensors({
            "weight_ih": ((hidden_size, input_size), True),
            "vector_u": ((hidden_size, ), True),
            "bias_ih": ((hidden_size, ), bias),
        })
        self.init_weights()

    def init_weights(self):
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                self.kernel_init(param)
            elif 'vector_u' in name:
                self.recurrent_kernel_init(param)
            elif 'bias_ih' in name:
                self.bias_init(param)


    def forward(self,
        inp: Tensor,
        state: Optional[Union[Tensor, Tuple[Tensor, ...]]] = None
    ) -> Tensor:
        state = self._check_state(state)
        self._validate_input(inp)
        self._validate_state(state)
        inp, state, is_batched = self._preprocess_input_and_state(inp, state)

        new_state = inp @ self.weight_ih.t() + self.vector_u * state + self.bias_ih
        new_state = self.activation_fn(new_state)

        if not is_batched:
            new_state = new_state.squeeze(0)

        return new_state

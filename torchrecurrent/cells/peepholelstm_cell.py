import torch
from torch import Tensor
import torch.nn as nn
from typing import Optional, Callable, Union, Tuple
from ..base import BaseDoubleRecurrentLayer, BaseDoubleRecurrentCell


class PeepholeLSTM(BaseDoubleRecurrentLayer):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        batch_first: bool = False,
        **kwargs,
    ):
        super(PeepholeLSTM, self).__init__(
            input_size, hidden_size, num_layers, dropout, batch_first
        )
        self.initialize_cells(PeepholeLSTMCell, **kwargs)


class PeepholeLSTMCell(BaseDoubleRecurrentCell):
    r"""A Peephole LSTM cell with learnable peephole connections.

    This LSTM variant adds element-wise “peephole” terms from the cell state
    into the input, forget, and output gates.

    .. math::

        \mathbf{z}(t) &= \tanh\Bigl(
            \mathbf{W}_{ih}^{z}\,\mathbf{x}(t)
            + \mathbf{b}_{ih}^{z}
            + \mathbf{W}_{hh}^{z}\,\mathbf{h}(t-1)
            + \mathbf{b}_{hh}^{z}
        \Bigr), \\[6pt]
        \mathbf{i}(t) &= \sigma\Bigl(
            \mathbf{W}_{ih}^{i}\,\mathbf{x}(t)
            + \mathbf{b}_{ih}^{i}
            + \mathbf{W}_{hh}^{i}\,\mathbf{h}(t-1)
            + \mathbf{b}_{hh}^{i}
            + \mathbf{p}^{i}\circ\mathbf{c}(t-1)
        \Bigr), \\[6pt]
        \mathbf{f}(t) &= \sigma\Bigl(
            \mathbf{W}_{ih}^{f}\,\mathbf{x}(t)
            + \mathbf{b}_{ih}^{f}
            + \mathbf{W}_{hh}^{f}\,\mathbf{h}(t-1)
            + \mathbf{b}_{hh}^{f}
            + \mathbf{p}^{f}\circ\mathbf{c}(t-1)
        \Bigr), \\[6pt]
        \mathbf{c}(t) &= \mathbf{f}(t)\,\circ\,\mathbf{c}(t-1)
            \;+\;\mathbf{i}(t)\,\circ\,\mathbf{z}(t), \\[6pt]
        \mathbf{o}(t) &= \sigma\Bigl(
            \mathbf{W}_{ih}^{o}\,\mathbf{x}(t)
            + \mathbf{b}_{ih}^{o}
            + \mathbf{W}_{hh}^{o}\,\mathbf{h}(t-1)
            + \mathbf{b}_{hh}^{o}
            + \mathbf{p}^{o}\circ\mathbf{c}(t)
        \Bigr), \\[6pt]
        \mathbf{h}(t) &= \mathbf{o}(t)\,\circ\,\tanh\bigl(\mathbf{c}(t)\bigr)


    Args:
        input_size (int):   Number of input features :math:`\dim(\mathbf{x}(t))`.
        hidden_size (int):  Number of hidden units :math:`\dim(\mathbf{h}(t))`.
        bias (bool, optional): If ``False``, disables all biases
            :math:`\mathbf{b}_{ih}` and :math:`\mathbf{b}_{hh}`. Default: ``True``.
        activation_fn (Callable, optional): Activation for the cell candidate
            :math:`\mathbf{z}`. Default: ``torch.tanh``.
        gate_activation_fn (Callable, optional): Activation for input/forget/output gates.
            Default: ``torch.sigmoid``.
        kernel_init (Callable, optional): Initializer for input-to-hidden weights
            :math:`\mathbf{W}_{ih}`. Default: ``nn.init.xavier_uniform_``.
        recurrent_kernel_init (Callable, optional): Initializer for hidden-to-hidden weights
            :math:`\mathbf{W}_{hh}`. Default: ``nn.init.xavier_uniform_``.
        peephole_kernel_init (Callable, optional): Initializer for peephole weights
            :math:`\mathbf{p}`. Default: ``nn.init.normal_``.
        bias_init (Callable, optional): Initializer for input biases
            :math:`\mathbf{b}_{ih}` when `bias=True`. Default: ``nn.init.zeros_``.
        recurrent_bias_init (Callable, optional): Initializer for hidden biases
            :math:`\mathbf{b}_{hh}` when `bias=True`. Default: ``nn.init.zeros_``.
        device (torch.device, optional): Device of the parameters. Default: CPU.
        dtype (torch.dtype, optional): Data type of the parameters. Default: PyTorch float.

    Inputs: input, (h, c)
        - **input** (Tensor): shape `(H_in,)` or `(N, H_in)`, where `H_in = input_size`.
        - **h**, **c** (Tensor): previous hidden and cell states of shape
            `(H_out,)` or `(N, H_out)`, where `H_out = hidden_size`. If not provided,
            both default to zero.

    Outputs: h’, c’
        - **h’**, **c’** (Tensor): next hidden and cell states, same shapes as inputs.

    Shape:
        - input: :math:`(N, H_{\mathrm{in}})` or :math:`(H_{\mathrm{in}})`.
        - h, c:   :math:`(N, H_{\mathrm{out}})` or :math:`(H_{\mathrm{out}})`.
        - output: same as **h** and **c**.

    Attributes:
        weight_ih (Tensor): input‐to‐hidden weights, shape `(4*H, I)`.
        weight_hh (Tensor): hidden‐to‐hidden weights, shape `(4*H, H)`.
        weight_ph (Tensor): peephole weights, shape `(3*H,)` for i, f, o.
        bias_ih (Tensor): input biases, shape `(4*H,)` if `bias=True`.
        bias_hh (Tensor): hidden biases, shape `(4*H,)` if `bias=True`.

    Examples::
        >>> cell = PeepholeLSTMCell(10, 20)
        >>> x = torch.randn(5, 10)
        >>> h, c = torch.zeros(20), torch.zeros(20)
        >>> for t in range(x.size(0)):
        ...     h, c = cell(x[t], (h, c))
    """

    weight_ih: Tensor
    weight_hh: Tensor
    weight_ph: Tensor
    bias_ih: Tensor
    bias_hh: Tensor

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        activation_fn: Callable = torch.tanh,
        gate_activation_fn: Callable = torch.sigmoid,
        kernel_init: Callable = nn.init.xavier_uniform_,
        recurrent_kernel_init: Callable = nn.init.xavier_uniform_,
        peephole_kernel_init: Callable = nn.init.normal_,
        bias_init: Callable = nn.init.zeros_,
        recurrent_bias_init: Callable = nn.init.zeros_,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super(PeepholeLSTMCell, self).__init__(
            input_size, hidden_size, bias, device=device, dtype=dtype
        )
        self.activation_fn = activation_fn
        self.gate_activation_fn = gate_activation_fn
        self.kernel_init = kernel_init
        self.recurrent_kernel_init = recurrent_kernel_init
        self.peephole_kernel_init = peephole_kernel_init
        self.bias_init = bias_init
        self.recurrent_bias_init = recurrent_bias_init

        self._register_tensors(
            {
                "weight_ih": ((4 * hidden_size, input_size), True),
                "weight_hh": ((4 * hidden_size, hidden_size), True),
                "weight_ph": ((3 * hidden_size,), True),
                "bias_ih": ((4 * hidden_size,), bias),
                "bias_hh": ((4 * hidden_size,), bias),
            }
        )
        self.init_weights()

    def init_weights(self):
        for name, param in self.named_parameters():
            if "weight_ih" in name:
                self.kernel_init(param)
            elif "weight_hh" in name:
                self.recurrent_kernel_init(param)
            elif "weight_ph" in name:
                self.peephole_kernel_init(param)
            elif "bias_ih" in name:
                self.bias_init(param)
            elif "bias_hh" in name:
                self.recurrent_bias_init(param)

    def forward(
        self, inp: Tensor, state: Optional[Union[Tensor, Tuple[Tensor, ...]]] = None
    ) -> Tuple[Tensor, Tensor]:
        state, c_state = self._check_states(state)
        self._validate_input(inp)
        self._validate_states((state, c_state))
        inp, state, c_state, is_batched = self._preprocess_states(inp, (state, c_state))

        weight_ih_i, weight_ih_f, weight_ih_c, weight_ih_o = self.weight_ih.chunk(4, 0)
        weight_hh_i, weight_hh_f, weight_hh_c, weight_hh_o = self.weight_hh.chunk(4, 0)
        weight_ph_i, weight_ph_f, weight_ph_o = self.weight_ph.chunk(3, 0)
        bias_ih_i, bias_ih_f, bias_ih_c, bias_ih_o = self.bias_ih.chunk(4, 0)
        bias_hh_i, bias_hh_f, bias_hh_c, bias_hh_o = self.bias_hh.chunk(4, 0)

        i = (
            inp @ weight_ih_i.t()
            + bias_ih_i
            + state @ weight_hh_i.t()
            + c_state * weight_ph_i
            + bias_hh_i
        )
        input_gate = self.gate_activation_fn(i)
        f = (
            inp @ weight_ih_f.t()
            + bias_ih_f
            + state @ weight_hh_f.t()
            + bias_hh_f
            + c_state * weight_ph_f
        )
        forget_gate = self.gate_activation_fn(f)
        c_hat = inp @ weight_ih_c.t() + bias_ih_c + state @ weight_hh_c.t() + bias_hh_c
        cell_candidate = self.activation_fn(c_hat)
        new_c = forget_gate * c_state + input_gate * cell_candidate
        o = (
            inp @ weight_ih_o.t()
            + bias_ih_o
            + state @ weight_hh_o.t()
            + bias_hh_o
            + new_c * weight_ph_o
        )
        output_gate = self.gate_activation_fn(o)
        new_h = output_gate * self.activation_fn(new_c)

        if not is_batched:
            new_h = new_h.squeeze(0)
            new_c = new_c.squeeze(0)

        return new_h, new_c

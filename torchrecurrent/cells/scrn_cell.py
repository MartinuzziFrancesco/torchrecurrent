import torch
import torch.nn as nn
from typing import Callable


class SCRNCell(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        activation_fn: Callable = torch.tanh,
        kernel_init=nn.init.xavier_uniform_,
        recurrent_kernel_init=nn.init.xavier_uniform_,
        bias_init=nn.init.zeros_,
        recurrent_bias_init=nn.init.zeros_,
    ):
        super(SCRNCell, self).__init__()
        self.hidden_size = hidden_size
        self.activation_fn = activation_fn
        self.kernel_init = kernel_init
        self.recurrent_kernel_init = recurrent_kernel_init
        self.bias_init = bias_init
        self.recurrent_bias_init = recurrent_bias_init
        self.bias = bias

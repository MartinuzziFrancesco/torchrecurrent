import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional


class MGUCell(nn.RNNCellBase):
    def __init__(self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        use_decomposition: bool = True):
        super().__init__(input_size, hidden_size, bias, num_chunks=2)
        self.use_decomposition = use_decomposition
    def forward(self, input: Tensor, hx: Optional[Tensor] = None) -> Tensor:
        # Check input dimensions
        #if input.dim() not in (1, 2):
        #    raise ValueError(f"MGUCell: Expected input to be 1D or 2D, got {input.dim()}D instead")
        #if hx is not None and hx.dim() not in (1, 2):
        #    raise ValueError(f"MGUCell: Expected hidden state to be 1D or 2D, got {hx.dim()}D instead")
        # Check batching
        #is_batched = input.dim() == 2
        #if not is_batched:
        #    input = input.unsqueeze(0)
        #Check and initialize hidden state
        #if hx is None:
        #    hx = torch.zeros(
        #        input.size(0), self.hidden_size, dtype=input.dtype, device=input.device
        #    )
        #else:
        #    hx = hx.unsqueeze(0) if not is_batched else hx
        input, hx, is_batched = self._preprocess(input, hx)
        ret = mgu_cell_decomposed(
            input,
            hx,
            self.weight_ih,
            self.weight_hh,
            self.bias_ih,
            self.bias_hh
        )
        #if not is_batched:
        #    ret = ret.squeeze(0)
        return ret

def mgu_cell(input: Tensor,
    hidden: Tensor,
    w_ih: Tensor,
    w_hh: Tensor,
    b_ih: Tensor,
    b_hh: Tensor) -> Tensor:
    # Perform input-to-hidden and hidden-to-hidden transformations and add biases
    gates = torch.mm(input, w_ih.t()) + torch.mm(hidden, w_hh.t()) + b_ih + b_hh
    # Split into forget gate and candidate hidden state
    forget_gate, candidate_h = gates.chunk(2, 1)
    # Forget gate (sigmoid activation)
    forget_gate = torch.sigmoid(forget_gate)
    # Candidate hidden state (tanh activation)
    candidate_h = torch.tanh(candidate_h)
    # New hidden state
    hy = forget_gate * candidate_h + (1 - forget_gate) * hidden
    return hy

def mgu_cell_decomposed(input: Tensor,
    hidden: Tensor,
    w_ih: Tensor,
    w_hh: Tensor,
    b_ih: Optional[Tensor],
    b_hh: Optional[Tensor]) -> Tensor:
    # Decomposition-based implementation using F.linear
    chunked_igates = F.linear(input, w_ih, b_ih).chunk(2, 1)
    chunked_hgates = F.linear(hidden, w_hh, b_hh).chunk(2, 1)
    forget_gate = torch.sigmoid(chunked_igates[0] + chunked_hgates[0])
    candidate_h = torch.tanh(chunked_igates[1] + chunked_hgates[1])
    return forget_gate * candidate_h + (1 - forget_gate) * hidden

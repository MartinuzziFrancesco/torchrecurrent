# torchrecurrent/base.py (proposed rewrite)

from __future__ import annotations

import warnings
from typing import Dict, Optional, Tuple, List, Any

import torch
import torch.nn as nn
from torch import Tensor

_ACTIVATIONS: Dict[str, nn.Module] = {
    "identity": nn.Identity(),
    "tanh": nn.Tanh(),
    "relu": nn.ReLU(),
    "sigmoid": nn.Sigmoid(),
    "gelu": nn.GELU(),
    "silu": nn.SiLU(),
    "elu": nn.ELU(),
    "leaky_relu": nn.LeakyReLU(),
}


def resolve_activation(act: Any) -> nn.Module:
    """
    Resolve an activation specification into an nn.Module, without storing python callables.

    Supported:
      - string keys in _ACTIVATIONS
      - nn.Module instances
      - a few well-known torch callables mapped to modules (torch.tanh, torch.relu, torch.sigmoid)
    """
    if isinstance(act, nn.Module):
        return act

    if isinstance(act, str):
        key = act.lower()
        if key not in _ACTIVATIONS:
            raise ValueError(f"Unknown activation '{act}'. Supported: {sorted(_ACTIVATIONS.keys())}")
        return type(_ACTIVATIONS[key])()

    if act is torch.tanh:
        return nn.Tanh()
    if act is torch.relu:
        return nn.ReLU()
    if act is torch.sigmoid:
        return nn.Sigmoid()

    raise TypeError(
        "Activation must be a string (e.g. 'tanh') or an nn.Module. "
        "Storing arbitrary python callables is not TorchScript-friendly."
    )


def resolve_init_name(init: Any, default: str) -> str:
    """
    Convert an initializer spec to a string name.
    - If init is a known nn.init function, map by identity/name.
    - If init is a string, use it.
    - Otherwise fall back to default.

    This avoids storing python callables on the module.
    """
    if init is None:
        return default
    if isinstance(init, str):
        return init
    if init is nn.init.zeros_:
        return "zeros"
    if init is nn.init.ones_:
        return "ones"
    if init is nn.init.normal_:
        return "normal"
    if init is nn.init.uniform_:
        return "uniform"
    if init is nn.init.xavier_uniform_:
        return "xavier_uniform"
    if init is nn.init.xavier_normal_:
        return "xavier_normal"
    if init is nn.init.kaiming_uniform_:
        return "kaiming_uniform"
    if init is nn.init.kaiming_normal_:
        return "kaiming_normal"
    if init is nn.init.orthogonal_:
        return "orthogonal"
    name = getattr(init, "__name__", "")
    name = name.rstrip("_")
    known = {
        "zeros", "ones", "normal", "uniform",
        "xavier_uniform", "xavier_normal",
        "kaiming_uniform", "kaiming_normal",
        "orthogonal",
    }
    if name in known:
        return name
    return default


def apply_init_(t: Tensor, name: str) -> None:
    """
    Apply an initializer by name in a TorchScript-safe way (names are strings).
    """
    key = name.lower()
    if key == "zeros":
        nn.init.zeros_(t)
    elif key == "ones":
        nn.init.ones_(t)
    elif key == "normal":
        nn.init.normal_(t)
    elif key == "uniform":
        nn.init.uniform_(t)
    elif key == "xavier_uniform":
        nn.init.xavier_uniform_(t)
    elif key == "xavier_normal":
        nn.init.xavier_normal_(t)
    elif key == "kaiming_uniform":
        nn.init.kaiming_uniform_(t)
    elif key == "kaiming_normal":
        nn.init.kaiming_normal_(t)
    elif key == "orthogonal":
        nn.init.orthogonal_(t)
    else:
        raise ValueError(f"Unknown initializer '{name}'.")

class RecurrentCellBase(nn.Module):
    __constants__ = ["input_size", "hidden_size", "bias", "recurrent_bias"]

    input_size: int
    hidden_size: int
    bias: bool
    recurrent_bias: bool
    _param_device: torch.device
    _param_dtype: torch.dtype
    init_cfg: Dict[str, str]

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        recurrent_bias: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.recurrent_bias = recurrent_bias

        self._param_device = device if device is not None else torch.device("cpu")
        self._param_dtype = dtype if dtype is not None else torch.get_default_dtype()

        self.init_cfg = {
            "kernel": "xavier_uniform",
            "recurrent_kernel": "xavier_uniform",
            "bias": "zeros",
            "recurrent_bias": "zeros",
        }

    def extra_repr(self) -> str:
        return (
            f"input_size={self.input_size}, hidden_size={self.hidden_size}, "
            f"bias={self.bias}, recurrent_bias={self.recurrent_bias}"
        )

    def _validate_input(self, inp: Tensor) -> None:
        if inp.dim() not in (1, 2):
            raise ValueError(
                f"{self.__class__.__name__}: expected input 1D or 2D, got {inp.dim()}D."
            )

    def _as_batched(self, inp: Tensor) -> Tuple[Tensor, bool]:
        if inp.dim() == 1:
            return inp.unsqueeze(0), False
        return inp, True

    def _zeros_state(self, batch: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        return torch.zeros(batch, self.hidden_size, device=device, dtype=dtype)

    def _register_tensors(self, specs: Dict[str, Tuple[Tuple[int, ...], bool]]) -> None:
        """
        specs: name -> (shape, trainable)
        trainable=False registers a buffer of zeros, trainable=True registers a Parameter.
        """
        for name, (shape, trainable) in specs.items():
            if trainable:
                p = nn.Parameter(torch.empty(*shape, device=self._param_device, dtype=self._param_dtype))
                self.register_parameter(name, p)
            else:
                b = torch.zeros(*shape, device=self._param_device, dtype=self._param_dtype)
                self.register_buffer(name, b)

    def _default_register_tensors(
        self,
        ih_mult: int = 1,
        hh_mult: int = 1,
        prefix_ih: str = "weight_ih",
        prefix_hh: str = "weight_hh",
        prefix_bih: str = "bias_ih",
        prefix_bhh: str = "bias_hh",
    ) -> None:
        self._register_tensors(
            {
                prefix_ih: ((ih_mult * self.hidden_size, self.input_size), True),
                prefix_hh: ((hh_mult * self.hidden_size, self.hidden_size), True),
                prefix_bih: ((ih_mult * self.hidden_size,), self.bias),
                prefix_bhh: ((hh_mult * self.hidden_size,), self.recurrent_bias),
            }
        )

    def reset_parameters(self) -> None:
        """
        Default initializer behavior by naming convention.
        Cells with nonstandard parameter names should override this and use apply_init_().
        """
        for name, p in self.named_parameters():
            if "weight_ih" in name:
                apply_init_(p, self.init_cfg["kernel"])
            elif "weight_hh" in name:
                apply_init_(p, self.init_cfg["recurrent_kernel"])
            elif "bias_ih" in name:
                apply_init_(p, self.init_cfg["bias"])
            elif "bias_hh" in name:
                apply_init_(p, self.init_cfg["recurrent_bias"])


class SingleStateCellBase(RecurrentCellBase):
    """
    TorchScript-friendly single-state interface:
      forward(inp, h) -> h_new
    """
    def forward(self, inp: Tensor, state: Optional[Tensor] = None) -> Tensor:
        raise NotImplementedError


class DoubleStateCellBase(RecurrentCellBase):
    """
    TorchScript-friendly double-state interface:
      forward(inp, (h, c)) -> (h_new, c_new)
    """
    def forward(
        self, inp: Tensor, state: Optional[Tuple[Tensor, Tensor]] = None
    ) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError

# Recurrent layers
class RecurrentLayerBase(nn.Module):
    __constants__ = ["input_size", "hidden_size", "num_layers", "batch_first", "dropout"]

    input_size: int
    hidden_size: int
    num_layers: int
    batch_first: bool
    dropout: float

    cells: nn.ModuleList

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        batch_first: bool = False,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.dropout = float(dropout)

        self.dropout_layer = nn.Dropout(self.dropout) if self.dropout > 0.0 else None

    def initialize_cells(self, cell_class, **kwargs) -> None:
        layers = [cell_class(self.input_size, self.hidden_size, **kwargs)] + [
            cell_class(self.hidden_size, self.hidden_size, **kwargs)
            for _ in range(1, self.num_layers)
        ]
        self.cells = nn.ModuleList(layers)


class SingleStateRecurrentLayerBase(RecurrentLayerBase):
    def forward(self, inp: Tensor, state: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        if self.batch_first:
            inp = inp.transpose(0, 1)

        seq_len, batch_size, _ = inp.size()

        if state is None:
            state = torch.zeros(
                self.num_layers, batch_size, self.hidden_size, dtype=inp.dtype, device=inp.device
            )

        outputs = torch.jit.annotate(List[Tensor], [])

        for t in range(seq_len):
            x = inp[t]
            new_states = torch.jit.annotate(List[Tensor], [])

            for layer_idx, cell in enumerate(self.cells):
                h_prev = state[layer_idx]
                h_new = cell(x, h_prev)
                new_states.append(h_new)
                x = h_new
                if self.dropout_layer is not None and layer_idx < self.num_layers - 1:
                    x = self.dropout_layer(x)

            state = torch.stack(new_states, dim=0)
            outputs.append(x)

        out = torch.stack(outputs, dim=0)
        if self.batch_first:
            out = out.transpose(0, 1)
        return out, state


class DoubleStateRecurrentLayerBase(RecurrentLayerBase):
    def forward(
        self, inp: Tensor, state: Optional[Tuple[Tensor, Tensor]] = None
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        if self.batch_first:
            inp = inp.transpose(0, 1)

        seq_len, batch_size, _ = inp.size()

        if state is None:
            h = torch.zeros(
                self.num_layers, batch_size, self.hidden_size, dtype=inp.dtype, device=inp.device
            )
            c = torch.zeros_like(h)
            state = (h, c)

        outputs = torch.jit.annotate(List[Tensor], [])

        for t in range(seq_len):
            x = inp[t]
            new_h = torch.jit.annotate(List[Tensor], [])
            new_c = torch.jit.annotate(List[Tensor], [])
            h_prev, c_prev = state

            for layer_idx, cell in enumerate(self.cells):
                h_i, c_i = cell(x, (h_prev[layer_idx], c_prev[layer_idx]))
                new_h.append(h_i)
                new_c.append(c_i)
                x = h_i
                if self.dropout_layer is not None and layer_idx < self.num_layers - 1:
                    x = self.dropout_layer(x)

            state = (torch.stack(new_h, dim=0), torch.stack(new_c, dim=0))
            outputs.append(x)

        out = torch.stack(outputs, dim=0)
        if self.batch_first:
            out = out.transpose(0, 1)
        return out, state

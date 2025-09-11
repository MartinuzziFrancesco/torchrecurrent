from abc import ABC, abstractmethod
import warnings
import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Tuple, Union, Dict


class BaseRecurrentCell(nn.Module, ABC):
    r"""Abstract base class for recurrent *cells*.

    Subclasses implement :meth:`uses_double_state` and :meth:`forward`.
    This base provides common utilities for input/state validation, state
    initialization, weight/bias registration, and weight initialization hooks.

    Args:
        input_size (int): Size of each input sample.
        hidden_size (int): Size of the hidden state.
        bias (bool, optional): If ``False``, input-side bias tensors are
            not used. Default: ``True``.
        recurrent_bias (bool, optional): If ``False``, recurrent-side bias
            tensors are not used. Default: ``True``.
        device (torch.device, optional): Device to place parameters on.
        dtype (torch.dtype, optional): Data type of parameters.
        **kwargs: Stored as extra args for representation; subclasses may
            consume these.

    Attributes:
        input_size (int): As above.
        hidden_size (int): As above.
        bias (bool): As above.
        recurrent_bias (bool): As above.

    Shape:
        - Input: ``(N, input_size)`` or ``(input_size,)``
        - State: subclass-defined (single or double state)
        - Output: subclass-defined

    Notes:
        - :meth:`init_weights` expects the subclass to define attributes
            ``kernel_init``, ``recurrent_kernel_init``, ``bias_init``, and
            (optionally) ``recurrent_bias_init``. Override :meth:`init_weights`
            if you prefer a different scheme.
        - :meth:`_register_tensors` and :meth:`_default_register_tensors`
            are helpers to create parameters/buffers with consistent shapes.
    """

    __constants__ = ["input_size", "hidden_size", "bias", "recurrent_bias"]

    input_size: int
    hidden_size: int
    bias: bool
    recurrent_bias: bool

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        recurrent_bias: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        **kwargs,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.recurrent_bias = recurrent_bias
        self._factory_kwargs = {
            k: v for k, v in {"device": device, "dtype": dtype}.items() if v is not None
        }
        self._extra_args = kwargs

    def __repr__(self) -> str:
        classname = self.__class__.__name__
        args = [str(self.input_size), str(self.hidden_size)]
        if not self.bias:
            args.append(f"bias={self.bias}")
        for k, v in sorted(self._extra_args.items()):
            args.append(f"{k}={v}")
        return f"{classname}({', '.join(args)})"

    @abstractmethod
    def uses_double_state(self) -> bool:
        """Return True if forward returns (h, c), else just h."""
        raise NotImplementedError

    @abstractmethod
    def forward(
        self, inp: Tensor, state: Optional[Union[Tensor, Tuple[Tensor, ...]]] = None
    ) -> Union[Tensor, Tuple[Tensor, ...]]:
        """Run one step of the recurrent cell.

        Args:
            inp (Tensor):
                - shape (input_size,) for a single time-step
                - or (batch_size, input_size) when batched
            state (Optional[Union[Tensor, Tuple[Tensor, ...]]]):
                Previous hidden state(s).
                - For single-state cells: Tuple containing one Tensor of shape
                  (hidden_size,) or (batch_size, hidden_size).
                - For double-state cells: Tuple of two such Tensors (h, c).

        Returns:
            Union[Tensor, Tuple[Tensor, ...]]:
                - For single-state cells: a 1-tuple `(h_new,)`.
                - For double-state cells: a 2-tuple `(h_new, c_new)`.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__}.forward() must be implemented by subclass"
        )

    def _init_state(self, inp: Tensor) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """Create the zero‐state for a fresh sequence.

        Args:
            inp:  Tensor of shape (batch_size, input_size) or (input_size,)
                  so you can infer batch_size, dtype, and device.

        Returns:
            - If uses_double_state() is False: a Tensor of shape
              (batch_size, hidden_size) or (hidden_size,).
            - If uses_double_state() is True: a 2‐tuple of Tensors
              (h0, c0), each of shape (batch_size, hidden_size)
              or (hidden_size,).
        """
        batch = inp.shape[0] if inp.dim() == 2 else 1
        return torch.zeros(batch, self.hidden_size, device=inp.device, dtype=inp.dtype)

    def _validate_input(self, inp: Tensor):
        if inp.dim() not in (1, 2):
            cls = self.__class__.__name__
            raise ValueError(
                f"{cls}: Expected input to be 1D or 2D, got {inp.dim()}D instead"
            )

    def _register_tensors(self, specs: Dict[str, Tuple[Tuple[int, ...], bool]]):
        """Create a Parameter (if trainable_flag=True) or a zero‐buffer otherwise.

        Example specs:
            {
            "weight_ih": ((3*H,  I), True),
            "weight_hh": ((2*H,  H), True),
            "bias_ih":   ((3*H,   ), bias_flag),
            "bias_hh":   ((2*H,   ), bias_flag),
            }
        """
        for name, (shape, trainable) in specs.items():
            if trainable:
                data = torch.empty(*shape, **self._factory_kwargs)
                param = nn.Parameter(data)
                self.register_parameter(name, param)
            else:
                buf = torch.zeros(*shape, **self._factory_kwargs)
                self.register_buffer(name, buf)

    def _default_register_tensors(
        self,
        input_size: int,
        hidden_size: int,
        ih_mult: int = 1,
        hh_mult: int = 1,
        bias: bool = True,
        recurrent_bias: bool = True,
        prefix_ih: str = "weight_ih",
        prefix_hh: str = "weight_hh",
        prefix_bih: str = "bias_ih",
        prefix_bhh: str = "bias_hh",
    ):
        """Shorthand for the common 2-weight + 2-bias pattern.

        * weight_ih  → shape (ih_mult*H,   I)
        * weight_hh  → shape (hh_mult*H,   H)
        * bias_ih    → shape (ih_mult*H,   )
        * bias_hh    → shape (hh_mult*H,   )
        """
        specs = {
            prefix_ih: ((ih_mult * hidden_size, input_size), True),
            prefix_hh: ((hh_mult * hidden_size, hidden_size), True),
            prefix_bih: ((ih_mult * hidden_size,), bias),
            prefix_bhh: ((hh_mult * hidden_size,), recurrent_bias),
        }
        self._register_tensors(specs)

    def init_weights(self):
        for name, param in self.named_parameters():
            if "weight_ih" in name:
                self.kernel_init(param)
            elif "weight_hh" in name:
                self.recurrent_kernel_init(param)
            elif "bias_ih" in name:
                self.bias_init(param)
            elif "bias_hh" in name:
                self.recurrent_bias_init(param)


class BaseSingleRecurrentCell(BaseRecurrentCell):
    r"""Base class for single-state recurrent cells (e.g., vanilla RNN).

    The hidden state is a single tensor ``h``. The :meth:`forward` of a
    subclass must accept an input and optional hidden state and return the
    next hidden state tensor.

    Shape:
        - Input: ``(N, input_size)`` or ``(input_size,)``
        - State (optional): ``(N, hidden_size)`` or ``(hidden_size,)``
        - Output (new state): same as state

    Notes:
        - If ``state`` is ``None``, it is initialized to zeros on the same
          device/dtype as the input.
        - If a tuple is passed for ``state``, only
          the first element is used.
    """

    def uses_double_state(self) -> bool:
        return False

    def _validate_state(self, state: Optional[Tensor]):
        if state is None:
            return

        if not isinstance(state, Tensor):
            cls = self.__class__.__name__
            raise TypeError(f"{cls}: state must be a Tensor or None, got {type(state)}")

        if state.dim() not in (1, 2):
            cls = self.__class__.__name__
            raise ValueError(f"{cls}: state must be 1D or 2D, got {state.dim()}D instead")

    def _init_state(self, inp: Tensor) -> Tensor:
        batch = inp.shape[0] if inp.dim() == 2 else 1
        return torch.zeros(batch, self.hidden_size, device=inp.device, dtype=inp.dtype)

    def _preprocess_input_and_state(
        self, inp: Tensor, state: Optional[Tensor]
    ) -> Tuple[Tensor, Tensor, bool]:
        """Ensure `inp` dim is correct, and initialize or reshape `state`.

        Returns:
          - inp      : (batch_size, input_size)
          - h        : (batch_size, hidden_size)
          - is_batched: True if original inp was 2D
        """
        is_batched = inp.dim() == 2
        if not is_batched:
            inp = inp.unsqueeze(0)

        if state is None:
            state = self._init_state(inp)
        else:
            state = state if is_batched else state.unsqueeze(0)

        return inp, state, is_batched

    def _check_state(
        self, state: Optional[Union[Tensor, Tuple[Tensor, ...]]]
    ) -> Optional[Tensor]:
        """If user passed a tuple to a single‐state cell, warn and pick the first element.

        Otherwise, return state unmodified.
        """
        if isinstance(state, tuple):
            warnings.warn(
                f"{self.__class__.__name__}.forward() got a tuple for `state`; "
                "using only the first tensor as the hidden state.",
                UserWarning,
                stacklevel=3,
            )
            return state[0]
        return state


class BaseDoubleRecurrentCell(BaseRecurrentCell):
    r"""Base class for double-state recurrent cells (e.g., LSTM-style).

    The state is a tuple ``(h, c)`` for hidden and cell states. The
    :meth:`forward` of a subclass must accept an input and optional
    ``(h, c)`` and return the updated ``(h, c)`` tuple.

    Shape:
        - Input: ``(N, input_size)`` or ``(input_size,)``
        - State (optional): tuple of two tensors, each
          ``(N, hidden_size)`` or ``(hidden_size,)``
        - Output (new state): same shapes as state
    """

    def uses_double_state(self) -> bool:
        return True

    def _validate_states(self, states: Optional[Tuple[Optional[Tensor], Optional[Tensor]]]):
        if states is None:
            return
        if not (isinstance(states, tuple) and len(states) == 2):
            cls = self.__class__.__name__
            raise TypeError(f"{cls}: state must be a tuple of two Tensors or None")
        h, c = states
        for name, t in (("hidden", h), ("cell", c)):
            if t is None:
                continue
            if t.dim() not in (1, 2):
                cls = self.__class__.__name__
                raise ValueError(
                    f"{cls}: {name} state must be 1D or 2D, got {t.dim()}D instead"
                )

    def _preprocess_states(
        self,
        inp: Tensor,
        states: Optional[Tuple[Optional[Tensor], Optional[Tensor]]] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, bool]:
        """- Ensures inp is treated as batched.

        - Initializes or reshapes both h and c to [batch_size, hidden_size]
        - Returns (inp, h, c, was_batched)
        """
        is_batched = inp.dim() == 2
        if not is_batched:
            inp = inp.unsqueeze(0)

        state, c_state = states or (None, None)

        if state is None:
            state = self._init_state(inp)
        elif not is_batched:
            state = state.unsqueeze(0)

        if c_state is None:
            c_state = self._init_state(inp)
        elif not is_batched:
            c_state = c_state.unsqueeze(0)

        assert isinstance(state, Tensor), "state must be a Tensor here"
        assert isinstance(c_state, Tensor), "c_state must be a Tensor here"

        return inp, state, c_state, is_batched

    def _check_states(
        self, states: Optional[Union[Tensor, Tuple[Tensor, ...]]]
    ) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        """Sanity‐check the `states` argument for double‐state cells.

        - If `states` is None, return (None, None).
        - If a single Tensor is passed, warn and treat it as the hidden state (h),
            with cell‐state (c) initialized to None.
        - If a tuple of length exactly 2 is passed, unpack to (h, c).
        - Otherwise (tuple of wrong length), raise an error.
        """
        if states is None:
            return None, None

        if not isinstance(states, tuple):
            warnings.warn(
                f"{self.__class__.__name__}.forward() got a single Tensor for `states`; "
                "treating it as the hidden state and initializing cell‐state to None.",
                UserWarning,
                stacklevel=3,
            )
            return states, None

        if len(states) != 2:
            raise ValueError(
                f"{self.__class__.__name__}.forward(): expected 2‐tuple (h, c), "
                f"got tuple of length {len(states)}"
            )

        state, c_state = states
        return state, c_state


class BaseRecurrentLayer(nn.Module):
    r"""Abstract base class for stacked recurrent *layers*.

    Manages common configuration (e.g., number of layers, dropout, batch
    layout) and provides :meth:`initialize_cells` to build a stack of cell
    instances for subclasses.

    Args:
        input_size (int): Size of each input sample.
        hidden_size (int): Size of the hidden state in each layer.
        num_layers (int, optional): Number of stacked layers. Default: ``1``.
        dropout (float, optional): Dropout probability applied to outputs
            of all but the last layer at each time step. Default: ``0.0``.
        batch_first (bool, optional): If ``True``, input and output tensors
            are provided/returned as ``(N, T, C)``. Default: ``False`` (``(T, N, C)``).

    Attributes:
        cells (nn.ModuleList): Populated by :meth:`initialize_cells` in subclasses.
        dropout (float): As above.
        dropout_layer (nn.Dropout or None): Dropout module if ``dropout > 0``.

    Notes:
        Subclasses should call ``self.initialize_cells(CellClass, **cell_kwargs)``
        in ``__init__`` to build a stack with the first layer taking
        ``input_size`` inputs and subsequent layers taking ``hidden_size`` inputs.
    """

    __constants__ = [
        "input_size",
        "hidden_size",
        "num_layers",
        "batch_first",
        "dropout",
    ]

    input_size: int
    hidden_size: int
    bias: bool
    dropout: float
    batch_first: bool

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        batch_first: bool = False,
    ):
        super(BaseRecurrentLayer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.dropout = dropout

        if self.dropout > 0:
            self.dropout_layer = nn.Dropout(dropout)
        else:
            self.dropout_layer = None

    def __repr__(self) -> str:
        classname = self.__class__.__name__
        args = [str(self.input_size), str(self.hidden_size)]

        # only show if not default
        if self.num_layers != 1:
            args.append(f"num_layers={self.num_layers}")
        if self.dropout != 0.0:
            args.append(f"dropout={self.dropout}")
        if self.batch_first:
            args.append(f"batch_first={self.batch_first}")

        return f"{classname}({', '.join(args)})"

    def initialize_cells(self, cell_class, **kwargs):
        """Helper method to initialize cells for the derived recurrent layer class."""
        layers = [cell_class(self.input_size, self.hidden_size, **kwargs)] + [
            cell_class(self.hidden_size, self.hidden_size, **kwargs)
            for _ in range(1, self.num_layers)
        ]
        self.cells = nn.ModuleList(layers)


class BaseSingleRecurrentLayer(BaseRecurrentLayer):
    r"""Recurrent layer wrapper for single-state cells.

    Iterates a stack of single-state cells over the time dimension.

    Shape:
        - Input:
            - If ``batch_first=False`` (default): ``(T, N, input_size)``
            - If ``batch_first=True``: ``(N, T, input_size)``
        - State (optional): ``(num_layers, N, hidden_size)``
        - Output:
            - Sequence: same layout as input but with ``hidden_size`` features
            - Final state: ``(num_layers, N, hidden_size)``

    Args:
        inp (Tensor): Input sequence.
        state (Tensor, optional): Initial hidden state for each layer.

    Returns:
        Tuple[Tensor, Tensor]: ``(output, h_n)`` where
        ``output`` is the output features for all timesteps and
        ``h_n`` is the final hidden state for each layer.

    Notes:
        - When ``state`` is ``None``, zeros are used.
        - Dropout (if set) is applied to intermediate layer outputs at each
          timestep, not to the last layer.
    """

    def forward(self, inp: Tensor, state: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        if self.batch_first:
            inp = inp.transpose(0, 1)

        seq_len, batch_size, _ = inp.size()
        # init single‐state
        if state is None:
            state = torch.zeros(
                self.num_layers,
                batch_size,
                self.hidden_size,
                dtype=inp.dtype,
                device=inp.device,
            )

        outputs = []
        for t in range(seq_len):
            x = inp[t]
            new_states = []
            for layer_idx, cell in enumerate(self.cells):
                h_prev = state[layer_idx]
                h_new = cell(x, h_prev)
                new_states.append(h_new)
                x = h_new
                if self.dropout_layer and layer_idx < self.num_layers - 1:
                    x = self.dropout_layer(x)

            state = torch.stack(new_states, dim=0)
            outputs.append(x)

        out = torch.stack(outputs, dim=0)
        if self.batch_first:
            out = out.transpose(0, 1)
        return out, state


class BaseDoubleRecurrentLayer(BaseRecurrentLayer):
    r"""Recurrent layer wrapper for double-state cells.

    Iterates a stack of double-state cells (e.g., LSTM-style) over time.

    Shape:
        - Input:
            - If ``batch_first=False`` (default): ``(T, N, input_size)``
            - If ``batch_first=True``: ``(N, T, input_size)``
        - State (optional): tuple ``(h_0, c_0)``, each
          ``(num_layers, N, hidden_size)``
        - Output:
            - Sequence: same layout as input but with ``hidden_size`` features
            - Final state: tuple ``(h_n, c_n)`` with same shapes as ``(h_0, c_0)``

    Args:
        inp (Tensor): Input sequence.
        state (Tuple[Tensor, Tensor], optional): Initial ``(h_0, c_0)``.

    Returns:
        Tuple[Tensor, Tuple[Tensor, Tensor]]: ``(output, (h_n, c_n))``.

    Notes:
        - When ``state`` is ``None``, zeros are used for both ``h`` and ``c``.
        - Dropout (if set) is applied to intermediate layer outputs at each
          timestep, not to the last layer.
    """

    def forward(
        self, inp: Tensor, state: Optional[Tuple[Tensor, Tensor]] = None
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        if self.batch_first:
            inp = inp.transpose(0, 1)

        seq_len, batch_size, _ = inp.size()
        # init double‐state
        if state is None:
            h = torch.zeros(
                self.num_layers,
                batch_size,
                self.hidden_size,
                dtype=inp.dtype,
                device=inp.device,
            )
            c = torch.zeros_like(h)
            state = (h, c)

        outputs = []
        for t in range(seq_len):
            x = inp[t]
            new_h, new_c = [], []
            h_prev, c_prev = state

            for layer_idx, cell in enumerate(self.cells):
                h_i, c_i = cell(x, (h_prev[layer_idx], c_prev[layer_idx]))
                new_h.append(h_i)
                new_c.append(c_i)
                x = h_i
                if self.dropout_layer and layer_idx < self.num_layers - 1:
                    x = self.dropout_layer(x)

            state = (torch.stack(new_h, dim=0), torch.stack(new_c, dim=0))
            outputs.append(x)

        out = torch.stack(outputs, dim=0)
        if self.batch_first:
            out = out.transpose(0, 1)
        return out, state

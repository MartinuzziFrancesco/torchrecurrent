# torchrecurrent

Pytorch implementation of various recurrent layers
found in the literature. All implementations are meant
to be as close to PyTorch's implementation of cells as possible.
Much inspiration has being drawn from
[fastrnns](https://github.com/pytorch/pytorch/blob/main/benchmarks/fastrnns)
approach to building cells.


## Technical notes on implementations

Some free flow thoughts on perusing the literature for the implementation
of recurrent cells and layers.

### Efficiency vs accuracy
 Here accuracy means accuracy in following the literature. 
 
 For example, in the GRU the default implementations in Pytorch and
 Tensorflow seem to be with the reset gate applied _after_ the recurrent matrix
 multiplication with the hidden state. It's confusing to trace the original
 implementation, since in [Cho 2014a](https://arxiv.org/pdf/1406.1078)
 the reset gate is applied _to_ the hidden state and only then it's
 multiplied to the recurrent matrix. This is also the approach followed
 in the empirical comparison with LSTMs of
 [Chung 2014](https://arxiv.org/pdf/1412.3555). But in
 [Cho 2014b](https://arxiv.org/pdf/1409.1259) the reset gate is applied before.
 [Chung 2014](https://arxiv.org/pdf/1412.3555) does (foot)note that the two
 implementations yielded roughly similar results.
  - Pytorch [GRU](https://pytorch.org/docs/stable/generated/torch.nn.GRU.html)
    offers an implementation with the reset gate applied before
    (with a note describing the change).
  - Tensorflow
    [GRU](https://www.tensorflow.org/api_docs/python/tf/keras/layers/GRU)
    offers two variants of the model, with v3 (based on
    [Chung 2014c](https://arxiv.org/abs/1406.1078v3)) being the default,
    so with reset gate application done after the recurrent multiplication.
    They also offer the other (original) variant, which has the order reversed. 
  - Flax also offers the
    [v3](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/nnrecurrent.html#flax.nnx.nn.recurrent.GRUCell)
    as the default
  - This topic has also been discussed in
    [Flux.jl](https://github.com/FluxML/Flux.jl/issues/1671),
    where the offer the original implementation as default,
    while also providing the v3 as a choice. 
  
The default choice being v3 in the python offerings
could be due to the more efficient computation that it provides,
allowing to perform matrix multiplication once for all the gates,
and then splitting. Alternatively, one has to split the matrices
and then compute the multiplication for each gate. The MGUCell
has a similar setup, so I can try to provide a faster alternative
by applying the reset gate after the recurrent multiplication.

Should input and hidden state be concatenated and then multiplied?
Is it more efficient while still being accurate?

### Merging gate computation
The standard approach when dealing with multiple gates seems to compute
the matrix vector multiplication with a larger matrix and then split the result
and feed it to the activation function. Supposedly this helps in computational
speed. However, Flax used distinct matrices in their
[nnx LSTM](https://github.com/google/flax/blob/main/flax/nnx/nn/recurrent.py#L163-L170)
implementation. Flax provides an additional
[OptimizedLSTMCell](https://github.com/google/flax/blob/main/flax/nnx/nn/recurrent.py#L215)
where the have this fusion. The docstrings mention, verbatim
"Note that this cell is often faster than ``LSTMCell`` as long as the
hidden size is roughly <= 2048 units". So far larger models this does not actually
have an impact over computation times. How is the curve for the two implementations
looking? This would be an interesting quick test. Notably, the
[GRU](https://github.com/google/flax/blob/main/flax/nnx/nn/recurrent.py#L518)
only offer the merged gates approach. Was there no difference between distinct matrices
and merged ones? And if so, why is the default LSTM the _not_ optimized one?

### Dropout

Applying dropout in recurrent architecture is apparently tricky, since one can't just put it in after the computational, since the state is saved and propagated to the same cell after. This is known in literature [Zaremba (2015)](https://arxiv.org/abs/1409.2329).

## See also

[LuxRecurrentLayers.jl](https://github.com/MartinuzziFrancesco/LuxRecurrentLayers.jl):
Provides recurrent layers for Lux.jl in Julia.

[RecurrentLayers.jl](https://github.com/MartinuzziFrancesco/RecurrentLayers.jl):
Provides recurrent layers for Flux.jl in Julia.


[ReservoirComputing.jl](https://github.com/SciML/ReservoirComputing.jl):
Reservoir computing utilities for scientific machine learning.
Essentially gradient free trained recurrent neural networks.
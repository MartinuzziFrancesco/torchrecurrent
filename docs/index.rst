TorchRecurrent
==============

Welcome to the **TorchRecurrent** documentation!

**TorchRecurrent** provides a large collection of recurrent neural network (RNN)
cells and layers, all implemented in PyTorch with a consistent API and extended
customization options.

Highlights
----------

- **30+ recurrent cells** (like :class:`torch.nn.LSTMCell`)
  implemented from classic and modern papers
- **30+ high-level recurrent layers** (like :class:`torch.nn.LSTM`)
  wrapping the cells
- **PyTorch-like interface** so you can drop in models with minimal changes
- **Extra options** such as custom initializers, bias settings, and advanced variants

Installation
------------

If you already have PyTorch installed:

.. code-block:: bash

   pip install torchrecurrent

Quick Example
-------------

Using a recurrent cell directly:

.. code-block:: python

   import torch
   from torchrecurrent import LEMCell

   cell = LEMCell(input_size=10, hidden_size=20)
   x = torch.randn(5, 10)   # (time, input_size)
   h = torch.zeros(20)

   outputs = []
   for t in range(x.size(0)):
       h = cell(x[t], h)
       outputs.append(h)

   outputs = torch.stack(outputs, dim=0)  # (time, hidden_size)

Or using the layer abstraction:

.. code-block:: python

   from torchrecurrent import LEM

   rnn = LEM(input_size=10, hidden_size=20, num_layers=2)
   x = torch.randn(3, 5, 10)   # (batch, time, input_size)
   output, hn = rnn(x)


.. toctree::
   :titlesonly:
   :hidden:
   :maxdepth: 2

   models
   api/index

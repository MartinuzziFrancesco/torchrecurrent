Benchmarks Reference
=========================

Synthetic benchmarks for evaluating recurrent architectures. The generators use
batch-first tensors, support reproducible :class:`torch.Generator` instances, and
can return either raw tensors or :class:`torch.utils.data.DataLoader` objects.

``adding_problem`` implements the two-half marker sampling from the canonical
adding task. ``copy_memory`` implements the categorical ``T + 20`` protocol by
default and can one-hot encode inputs for direct use with recurrent layers.
``sequential_mnist`` adapts standard MNIST tensors to the sequential and fixed
permutation variants without introducing a dataset-download dependency.
``penn_treebank`` prepares licensed, preprocessed PTB split files for canonical
word-level language modeling with contiguous truncated-BPTT batches.
``timit`` batches aligned 120-dimensional log-Mel, delta, and acceleration
features for the canonical 180-state frame-classification protocol.

.. autosummary::
   :toctree: ../generated/
   :nosignatures:

   torchrecurrent.benchmarks.adding_problem
   torchrecurrent.benchmarks.copy_memory
   torchrecurrent.benchmarks.penn_treebank
   torchrecurrent.benchmarks.sequential_mnist
   torchrecurrent.benchmarks.timit

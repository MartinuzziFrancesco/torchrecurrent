from .adding import adding_problem
from .copymemory import copy_memory
from .penn_treebank import PennTreebankCorpus, penn_treebank
from .sequential_cifar10 import sequential_cifar10
from .sequential_mnist import sequential_mnist
from .timit import TIMITBatch, TIMITDataset, timit

__all__ = [
    "PennTreebankCorpus",
    "TIMITBatch",
    "TIMITDataset",
    "adding_problem",
    "copy_memory",
    "penn_treebank",
    "sequential_cifar10",
    "sequential_mnist",
    "timit",
]

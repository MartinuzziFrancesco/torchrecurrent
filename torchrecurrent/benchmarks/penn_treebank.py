from collections import Counter
from pathlib import Path
from typing import Dict, NamedTuple, Union

import torch
from torch.utils.data import DataLoader, Dataset


class PennTreebankCorpus(NamedTuple):
    """Data loaders and vocabulary for the Penn Treebank language model task."""

    train: DataLoader
    validation: DataLoader
    test: DataLoader
    vocabulary: Dict[str, int]


class _BatchedLanguageModelDataset(Dataset):
    def __init__(self, tokens: torch.Tensor, batch_size: int, sequence_length: int):
        stream_length = tokens.numel() // batch_size
        if stream_length < 2:
            raise ValueError("each corpus split must contain more tokens than batch_size")
        self.tokens = tokens[: batch_size * stream_length].view(batch_size, stream_length)
        self.sequence_length = sequence_length

    def __len__(self):
        return (self.tokens.shape[1] - 1 + self.sequence_length - 1) // self.sequence_length

    def __getitem__(self, index):
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        start = index * self.sequence_length
        length = min(self.sequence_length, self.tokens.shape[1] - start - 1)
        inputs = self.tokens[:, start : start + length]
        targets = self.tokens[:, start + 1 : start + length + 1]
        return inputs, targets


def _read_tokens(path: Union[str, Path], eos_token: str):
    tokens = []
    with Path(path).open("r", encoding="utf-8") as file:
        for line in file:
            tokens.extend(line.split())
            tokens.append(eos_token)
    return tokens


def _build_vocabulary(tokens, max_vocab_size: int, eos_token: str, unk_token: str):
    counts = Counter(tokens)
    ordered_tokens = sorted(counts, key=lambda token: (-counts[token], token))
    ordered_tokens = [
        token for token in ordered_tokens if token not in (unk_token, eos_token)
    ]
    vocabulary_tokens = [unk_token, eos_token]
    vocabulary_tokens.extend(ordered_tokens[: max_vocab_size - 2])
    return {token: index for index, token in enumerate(vocabulary_tokens)}


def _encode(tokens, vocabulary, unk_token: str, device):
    unknown_index = vocabulary[unk_token]
    return torch.tensor(
        [vocabulary.get(token, unknown_index) for token in tokens],
        dtype=torch.long,
        device=device,
    )


def penn_treebank(
    train_file: Union[str, Path],
    validation_file: Union[str, Path],
    test_file: Union[str, Path],
    batch_size: int = 20,
    sequence_length: int = 35,
    *,
    max_vocab_size: int = 10_000,
    eos_token: str = "<eos>",
    unk_token: str = "<unk>",
    device: torch.device = None,
):
    """Prepare the standard word-level Penn Treebank language-model benchmark.

    This adapter expects the three whitespace-tokenized text files from the
    Mikolov-preprocessed Penn Treebank corpus. It appends ``<eos>`` at each
    newline, builds a training-only vocabulary, maps unseen validation and test
    words to ``<unk>``, and creates contiguous streams for truncated BPTT. The
    default batch size and unroll length follow Zaremba et al. (2014)
    (https://arxiv.org/abs/1409.2329).

    Penn Treebank data is not downloaded or redistributed by this package. The
    original corpus is described by Marcus et al. (1993)
    (https://aclanthology.org/J93-2004/).

    Args:
        train_file: Path to the preprocessed training split.
        validation_file: Path to the preprocessed validation split.
        test_file: Path to the preprocessed test split.
        batch_size: Number of independent contiguous token streams per batch.
        sequence_length: Maximum truncated-BPTT unroll length.
        max_vocab_size: Maximum number of tokens, including ``unk_token``.
        eos_token: Token appended at the end of every input line.
        unk_token: Token used for words outside the training vocabulary.
        device: Device on which to store encoded token tensors.

    Returns:
        :class:`PennTreebankCorpus` containing train, validation, and test data
        loaders plus the token-to-index vocabulary. Each loader yields
        ``(inputs, targets)`` tensors shaped ``(batch_size, time_steps)``. The
        targets are the inputs shifted forward by one token.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if sequence_length < 1:
        raise ValueError("sequence_length must be positive")
    if max_vocab_size < 2:
        raise ValueError("max_vocab_size must be at least 2")
    if eos_token == unk_token:
        raise ValueError("eos_token and unk_token must be different")

    train_tokens = _read_tokens(train_file, eos_token)
    validation_tokens = _read_tokens(validation_file, eos_token)
    test_tokens = _read_tokens(test_file, eos_token)
    if not train_tokens:
        raise ValueError("training split must not be empty")

    vocabulary = _build_vocabulary(train_tokens, max_vocab_size, eos_token, unk_token)

    def make_loader(tokens):
        encoded = _encode(tokens, vocabulary, unk_token, device)
        dataset = _BatchedLanguageModelDataset(encoded, batch_size, sequence_length)
        return DataLoader(dataset, batch_size=None, shuffle=False)

    return PennTreebankCorpus(
        train=make_loader(train_tokens),
        validation=make_loader(validation_tokens),
        test=make_loader(test_tokens),
        vocabulary=vocabulary,
    )

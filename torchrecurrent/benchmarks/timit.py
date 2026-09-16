from typing import NamedTuple, Sequence

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset


class TIMITBatch(NamedTuple):
    """A padded batch of acoustic features, frame labels, and valid lengths."""

    features: torch.Tensor
    targets: torch.Tensor
    lengths: torch.Tensor


class TIMITDataset(Dataset):
    """Validated variable-length TIMIT feature and phone-state sequences."""

    def __init__(
        self,
        features: Sequence[torch.Tensor],
        targets: Sequence[torch.Tensor],
        feature_size: int = 120,
        num_classes: int = 180,
    ):
        if feature_size < 1:
            raise ValueError("feature_size must be positive")
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2")
        if len(features) != len(targets):
            raise ValueError(
                "features and targets must contain the same number of utterances"
            )
        if not features:
            raise ValueError("features and targets must not be empty")

        self.features = []
        self.targets = []
        for index, (utterance, labels) in enumerate(zip(features, targets)):
            if utterance.ndim != 2 or utterance.shape[1] != feature_size:
                raise ValueError(
                    f"features[{index}] must have shape (frames, {feature_size})"
                )
            if not utterance.is_floating_point():
                raise TypeError(f"features[{index}] must have a floating-point dtype")
            if labels.ndim != 1 or labels.shape[0] != utterance.shape[0]:
                raise ValueError(
                    f"targets[{index}] must have one label for every feature frame"
                )
            if utterance.shape[0] == 0:
                raise ValueError(f"features[{index}] must contain at least one frame")
            labels = labels.to(dtype=torch.long)
            if torch.any(labels < 0) or torch.any(labels >= num_classes):
                raise ValueError(
                    f"targets[{index}] must contain class indices in [0, {num_classes})"
                )
            self.features.append(utterance)
            self.targets.append(labels)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index], self.targets[index]


class _PadTIMITBatch:
    def __init__(self, padding_value: float, target_padding_value: int):
        self.padding_value = padding_value
        self.target_padding_value = target_padding_value

    def __call__(self, samples):
        features, targets = zip(*samples)
        lengths = torch.tensor([sequence.shape[0] for sequence in features])
        padded_features = pad_sequence(
            features, batch_first=True, padding_value=self.padding_value
        )
        padded_targets = pad_sequence(
            targets, batch_first=True, padding_value=self.target_padding_value
        )
        return TIMITBatch(padded_features, padded_targets, lengths)


def timit(
    features: Sequence[torch.Tensor],
    targets: Sequence[torch.Tensor],
    batch_size: int = 32,
    shuffle: bool = True,
    *,
    feature_size: int = 120,
    num_classes: int = 180,
    padding_value: float = 0.0,
    target_padding_value: int = -100,
    generator: torch.Generator = None,
    **dataloader_kwargs,
):
    """Prepare aligned TIMIT features for frame-level phone-state recognition.

    The default contract follows the TIMIT experiment of Le et al. (2015),
    Section 4.4 (https://arxiv.org/abs/1504.00941): each frame contains 40 log-Mel
    filterbank coefficients, their deltas, and accelerations (120 features), and
    is aligned to one of 180 Kaldi phone states.

    This function does not download the licensed TIMIT corpus or reproduce the
    external Kaldi alignment pipeline. It batches user-provided aligned feature
    and target tensors for direct use with batch-first recurrent layers.

    Args:
        features: Utterance tensors shaped ``(frames, feature_size)``.
        targets: Corresponding frame-label tensors shaped ``(frames,)``.
        batch_size: Number of utterances per batch.
        shuffle: Whether to shuffle utterances between epochs.
        feature_size: Expected number of acoustic features per frame.
        num_classes: Number of valid phone-state target classes.
        padding_value: Value used to pad acoustic feature sequences.
        target_padding_value: Value used to pad target sequences. The default
            matches :class:`torch.nn.CrossEntropyLoss`'s ``ignore_index``.
        generator: Optional generator used for reproducible shuffling.
        **dataloader_kwargs: Additional arguments passed to
            :class:`torch.utils.data.DataLoader`.

    Returns:
        A data loader yielding :class:`TIMITBatch` objects. Features have shape
        ``(batch, max_frames, feature_size)``, targets have shape
        ``(batch, max_frames)``, and lengths has shape ``(batch,)``.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if 0 <= target_padding_value < num_classes:
        raise ValueError("target_padding_value must not be a valid class index")

    dataset = TIMITDataset(features, targets, feature_size, num_classes)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        collate_fn=_PadTIMITBatch(padding_value, target_padding_value),
        **dataloader_kwargs,
    )

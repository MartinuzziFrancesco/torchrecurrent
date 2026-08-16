import torch
from torch.utils.data import DataLoader, TensorDataset


def sequential_cifar10(
    images: torch.Tensor,
    targets: torch.Tensor,
    permutation: torch.Tensor = None,
    return_dataloader: bool = True,
    batch_size: int = 64,
    shuffle: bool = True,
    *,
    normalize: bool = True,
    dtype: torch.dtype = None,
    device: torch.device = None,
    generator: torch.Generator = None,
    **dataloader_kwargs,
):
    """Convert CIFAR-10 tensors into the sequential or permuted-CIFAR task.

    Every 32 by 32 RGB image becomes a sequence of 1024 three-channel pixel
    inputs in raster-scan order. With a permutation, the same fixed pixel
    ordering is applied to every image, mirroring the permuted-MNIST protocol
    of Arjovsky et al. (2016), Section 5.3
    (https://proceedings.mlr.press/v48/arjovsky16.html), extended to CIFAR-10 as
    used to benchmark long-range recurrent models (e.g. Li et al. (2018) IndRNN,
    https://arxiv.org/abs/1803.04831). The caller supplies the CIFAR-10 tensors
    so this package does not require a dataset-download dependency.

    Args:
        images: CIFAR-10 images with shape ``(N, 32, 32, 3)`` or
            ``(N, 3, 32, 32)``.
        targets: Class indices with shape ``(N,)``.
        permutation: Optional permutation of the integers from 0 through 1023.
            Reuse the same tensor for training and test data.
        return_dataloader: Return a data loader when true, otherwise tensors.
        batch_size: Batch size of the returned data loader.
        shuffle: Whether the returned data loader shuffles samples.
        normalize: Convert integer pixels from ``[0, 255]`` to ``[0, 1]``.
            Floating-point inputs are assumed to be normalized already.
        dtype: Floating-point dtype of the returned inputs. Defaults to the
            current PyTorch default dtype.
        device: Device on which to place inputs and targets.
        generator: Optional generator used by the data loader when shuffling.
        **dataloader_kwargs: Additional arguments passed to
            :class:`torch.utils.data.DataLoader`.

    Returns:
        A data loader, or ``(sequences, targets)`` when
        ``return_dataloader=False``. Sequences have shape ``(N, 1024, 3)`` and
        targets have shape ``(N,)`` with dtype :class:`torch.long`.
    """
    if images.ndim == 4 and images.shape[1] == 3:
        images = images.permute(0, 2, 3, 1)
    if images.ndim != 4 or images.shape[1:] != (32, 32, 3):
        raise ValueError("images must have shape (N, 32, 32, 3) or (N, 3, 32, 32)")
    if targets.ndim != 1 or targets.shape[0] != images.shape[0]:
        raise ValueError("targets must have shape (N,) matching images")

    if dtype is None:
        dtype = torch.get_default_dtype()
    if not dtype.is_floating_point:
        raise TypeError("dtype must be a floating-point dtype")

    integer_pixels = not images.is_floating_point()
    sequences = images.reshape(images.shape[0], 1024, 3)
    sequences = sequences.to(device=device, dtype=dtype)
    if normalize and integer_pixels:
        sequences = sequences / 255

    if permutation is not None:
        if permutation.ndim != 1 or permutation.numel() != 1024:
            raise ValueError("permutation must have shape (1024,)")
        permutation = permutation.to(device=sequences.device, dtype=torch.long)
        expected = torch.arange(1024, device=sequences.device)
        if not torch.equal(torch.sort(permutation).values, expected):
            raise ValueError("permutation must contain every index from 0 through 1023")
        sequences = sequences.index_select(1, permutation)

    targets = targets.to(device=device, dtype=torch.long)
    if not return_dataloader:
        return sequences, targets

    return DataLoader(
        TensorDataset(sequences, targets),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        **dataloader_kwargs,
    )

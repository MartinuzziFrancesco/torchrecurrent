import torch
from torch.utils.data import DataLoader, TensorDataset


def adding_problem(
    sequence_length: int,
    n_samples: int,
    return_dataloader: bool = True,
    batch_size: int = 64,
    shuffle: bool = True,
    *,
    generator: torch.Generator = None,
    dtype: torch.dtype = None,
    device: torch.device = None,
    **dataloader_kwargs,
):
    """Generate the adding problem introduced for long-term memory tests.

    Each input contains a uniform random sequence and a binary indicator
    sequence. One indicator is sampled from each half of the sequence, and the
    regression target is the sum of the two indicated values. This is the
    formulation used by Arjovsky et al. (2016), Section 5.2
    (https://proceedings.mlr.press/v48/arjovsky16.html).

    Args:
        sequence_length: Number of time steps. Must be at least 2.
        n_samples: Number of independent sequences to generate.
        return_dataloader: Return a data loader when true, otherwise tensors.
        batch_size: Batch size of the returned data loader.
        shuffle: Whether the returned data loader shuffles samples.
        generator: Optional random number generator used for reproducibility.
            Only forwarded to the returned data loader's shuffling when it is a
            CPU generator; a non-CPU generator is still used for tensor
            generation but the loader falls back to its own seeding.
        dtype: Floating-point dtype of inputs and targets.
        device: Device on which to create the tensors.
        **dataloader_kwargs: Additional arguments passed to
            :class:`torch.utils.data.DataLoader`.

    Returns:
        A data loader, or ``(inputs, targets)`` when ``return_dataloader=False``.
        Inputs have shape ``(n_samples, sequence_length, 2)`` and targets have
        shape ``(n_samples, 1)``.
    """
    if sequence_length < 2:
        raise ValueError("sequence_length must be at least 2")
    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    if dtype is None:
        dtype = torch.get_default_dtype()
    if not dtype.is_floating_point:
        raise TypeError("dtype must be a floating-point dtype")

    random_sequence = torch.rand(
        n_samples,
        sequence_length,
        1,
        generator=generator,
        dtype=dtype,
        device=device,
    )
    first_half = sequence_length // 2
    first_indices = torch.randint(
        first_half, (n_samples, 1), generator=generator, device=device
    )
    second_indices = torch.randint(
        first_half,
        sequence_length,
        (n_samples, 1),
        generator=generator,
        device=device,
    )
    indices = torch.cat((first_indices, second_indices), dim=1)

    mask_sequence = torch.zeros_like(random_sequence)
    mask_sequence.scatter_(1, indices.unsqueeze(-1), 1)
    targets = random_sequence.squeeze(-1).gather(1, indices).sum(dim=1, keepdim=True)
    inputs = torch.cat((random_sequence, mask_sequence), dim=-1)

    if not return_dataloader:
        return inputs, targets

    # torch.utils.data.DataLoader shuffling always runs its generator on CPU
    # (torch.randperm has no device argument), so a generator created for a
    # non-CPU generation device cannot also drive the loader's shuffling.
    loader_generator = generator
    if generator is not None and generator.device.type != "cpu":
        loader_generator = None

    dataset = TensorDataset(inputs, targets)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=loader_generator,
        **dataloader_kwargs,
    )

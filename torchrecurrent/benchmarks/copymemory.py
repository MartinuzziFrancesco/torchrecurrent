import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset


def copy_memory(
    seq_len: int,
    n_samples: int,
    num_classes: int = 10,
    *,
    memory_length: int = 10,
    return_dataloader: bool = True,
    one_hot: bool = False,
    generator: torch.Generator = None,
    device: torch.device = None,
    **dataloader_kwargs,
):
    """Generate the canonical copy-memory benchmark.

    The first ``memory_length`` tokens are sampled from the content classes.
    They are followed by ``seq_len - 1`` blank tokens, a delimiter, and another
    ``memory_length`` blanks. Targets are blank until the final segment, where
    they reproduce the initial tokens. This follows Arjovsky et al. (2016),
    Section 5.1 (https://proceedings.mlr.press/v48/arjovsky16.html).

    Args:
        seq_len: Time lag ``T`` in the paper. Must be positive. The complete
            sequence length is ``seq_len + 2 * memory_length``.
        n_samples: Number of independent sequences to generate.
        num_classes: Alphabet size. The last two classes are reserved for the
            blank and delimiter tokens, respectively.
        memory_length: Number of content tokens to remember.
        return_dataloader: Return a data loader when true, otherwise tensors.
        one_hot: Convert inputs to floating-point one-hot vectors so they can be
            passed directly to recurrent layers. Targets remain integer class
            indices suitable for :class:`torch.nn.CrossEntropyLoss`.
        generator: Optional random number generator used for reproducibility.
            Only forwarded to the returned data loader's shuffling when it is a
            CPU generator; a non-CPU generator is still used for tensor
            generation but the loader falls back to its own seeding.
        device: Device on which to create the tensors.
        **dataloader_kwargs: Arguments passed to
            :class:`torch.utils.data.DataLoader`.

    Returns:
        A data loader, or ``(inputs, targets)`` when ``return_dataloader=False``.
        Integer inputs and targets have shape ``(n_samples, total_length)``.
        With ``one_hot=True``, inputs have an additional final dimension of size
        ``num_classes``.
    """
    if seq_len < 1:
        raise ValueError("seq_len must be positive")
    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    if num_classes < 3:
        raise ValueError("num_classes must be at least 3")
    if memory_length < 1:
        raise ValueError("memory_length must be positive")

    blank_token = num_classes - 2
    delimiter_token = num_classes - 1
    content = torch.randint(
        blank_token,
        (n_samples, memory_length),
        generator=generator,
        device=device,
    )
    leading_blanks = torch.full(
        (n_samples, seq_len - 1), blank_token, dtype=torch.long, device=device
    )
    delimiter = torch.full((n_samples, 1), delimiter_token, dtype=torch.long, device=device)
    trailing_blanks = torch.full(
        (n_samples, memory_length), blank_token, dtype=torch.long, device=device
    )
    inputs = torch.cat((content, leading_blanks, delimiter, trailing_blanks), dim=1)
    targets = torch.cat(
        (
            torch.full(
                (n_samples, seq_len + memory_length),
                blank_token,
                dtype=torch.long,
                device=device,
            ),
            content,
        ),
        dim=1,
    )

    if one_hot:
        inputs = F.one_hot(inputs, num_classes=num_classes).to(torch.get_default_dtype())
    if not return_dataloader:
        return inputs, targets

    # torch.utils.data.DataLoader shuffling always runs its generator on CPU
    # (torch.randperm has no device argument), so a generator created for a
    # non-CPU generation device cannot also drive the loader's shuffling.
    loader_generator = generator
    if generator is not None and generator.device.type != "cpu":
        loader_generator = None

    return DataLoader(
        TensorDataset(inputs, targets), generator=loader_generator, **dataloader_kwargs
    )

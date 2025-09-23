import torch
from torch.utils.data import TensorDataset, DataLoader


def copy_memory(seq_len: int, n_samples: int, num_classes: int = 10, **kwargs):
    """Generate data for the copy memory benchmark.

    The copy memory task is a synthetic sequence learning problem where a
    model must memorize and reproduce an input sequence after a long delay.
    Each sample consists of:

    - A random sequence of integers (the content to be memorized).
    - A delimiter symbol marking the end of the input.
    - A sequence of zeros acting as distractors.
    - The target sequence requires the model to output padding until the
      delimiter, then reproduce the original random sequence.

    Args:
        seq_len (int): Length of the random sequence to memorize.
        n_samples (int): Number of samples to generate.
        num_classes (int, optional): Number of distinct classes used for the
            random sequence. Defaults to 10. The delimiter token uses the
            value ``num_classes``.
        **kwargs: Additional keyword arguments passed to
            :class:`torch.utils.data.DataLoader` (e.g. ``batch_size``,
            ``shuffle``).

    Returns:
        torch.utils.data.DataLoader: A DataLoader yielding batches of
        ``(input_seq, target_seq)`` where:

        - ``input_seq`` has shape ``(n_samples, 2 * seq_len + 1)`` and contains
          the random sequence, followed by a delimiter token, followed by
          distractor zeros.
        - ``target_seq`` has shape ``(n_samples, 2 * seq_len + 1)`` and
          contains padding + delimiter, followed by the original random
          sequence.
    """
    random_seq = torch.randint(0, num_classes, (n_samples, seq_len))
    delimiter = torch.full((n_samples, 1), num_classes)
    distractor_seq = torch.zeros((n_samples, seq_len), dtype=torch.long)
    input_seq = torch.cat([random_seq, delimiter, distractor_seq], dim=1)
    target_seq = torch.cat(
        [torch.full((n_samples, seq_len + 1), num_classes), random_seq], dim=1
    )

    dataset = TensorDataset(input_seq, target_seq)
    dataloader = DataLoader(dataset, **kwargs)

    return dataloader

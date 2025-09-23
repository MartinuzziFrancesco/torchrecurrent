import torch
from torch.utils.data import TensorDataset, DataLoader


def adding_problem(
    sequence_length: int,
    n_samples: int,
    return_dataloader: bool = True,
    batch_size: int = 64,
    shuffle=True,
):
    """Generate data for the adding problem benchmark.

    The adding problem is a synthetic task where each input sequence
    consists of two features per time step:

      1. A random number sampled uniformly from [0, 1].
      2. A binary mask indicating which two positions in the sequence
         should be summed.

    The target is the sum of the two masked numbers.

    Parameters
    ----------
    sequence_length : int
        Length of each input sequence.
    n_samples : int
        Number of samples to generate.
    return_dataloader : bool, default=True
        If True, return a DataLoader wrapping the dataset.
        If False, return raw tensors instead.
    batch_size : int, default=64
        Batch size used when returning a DataLoader.
    shuffle : bool, default=True
        Whether to shuffle the dataset when returning a DataLoader.

    Returns
    -------
    torch.utils.data.DataLoader or tuple of (torch.Tensor, torch.Tensor)
        - If ``return_dataloader`` is True: a DataLoader yielding batches of
          (inputs, targets).
        - If ``return_dataloader`` is False:
            * inputs: torch.Tensor of shape (n_samples, sequence_length, 2)
            * targets: torch.Tensor of shape (n_samples, 1)
    """
    random_sequence = torch.rand(n_samples, sequence_length, 1)
    mask_sequence = torch.zeros(n_samples, sequence_length, 1)
    targets = torch.zeros(n_samples, 1)

    for i in range(n_samples):
        idx = torch.randperm(sequence_length)[:2]
        mask_sequence[i, idx, 0] = 1
        targets[i] = random_sequence[i, idx, 0].sum()

    inputs = torch.cat((random_sequence, mask_sequence), dim=2)
    if return_dataloader:
        dataset = TensorDataset(inputs, targets)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

        return data_loader
    else:
        return inputs, targets

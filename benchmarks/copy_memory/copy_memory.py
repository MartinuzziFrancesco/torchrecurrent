import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch import Tensor
import sys
import os
import argparse

def generate_copy_memory_data(seq_len: int,
        n_samples: int,
        num_classes: int = 10,
        **kwargs):
    random_seq = torch.randint(0, num_classes, (n_samples, seq_len))
    delimiter = torch.full((n_samples, 1), num_classes)
    distractor_seq = torch.zeros((n_samples, seq_len), dtype=torch.long)
    input_seq = torch.cat([random_seq, delimiter, distractor_seq], dim=1)
    target_seq = torch.cat([torch.full((n_samples, seq_len + 1), num_classes), random_seq], dim=1)

    dataset = TensorDataset(input_seq, target_seq)
    dataloader = DataLoader(dataset, **kwargs)
    
    return dataloader

def get_dataloaders(train_samples: int,
    test_samples: int,
    seq_len: int,
    num_classes: int = 10,
    **kwargs):

    train_loader = generate_copy_memory_data(
        seq_len, train_samples, num_classes=num_classes, **kwargs
    )
    test_loader = generate_copy_memory_data(
        seq_len, test_samples, num_classes=num_classes, **kwargs
    )

    return train_loader, test_loader

class RecurrentModel(nn.Module):
    
def train():

def test():

def main():

if __name__ == '__main__':
    main()
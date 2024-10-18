import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch import Tensor
import sys
import os
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'recurrent_layers')))
from minimal_gated_unit import MGU

import torch

class RecurrentModel(nn.Module):
    def __init__(self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        **kwargs):
        self.hidden_size = hidden_size
        self.rnn = MGU(input_size, hidden_size, **kwargs)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, inp: Tensor):
        output, _ = self.rnn(inp)
        output = output[:, -1, :]
        output = self.fc(output)
        return output


def generate_adding_problem_data(
    sequence_length: int,
    n_samples: int,
    return_dataloader: bool = True,
    batch_size: int = 64,
    shuffle = True):
    """
    Generate data for the adding problem benchmark.

    Parameters:
    - sequence_length (int): Length of each input sequence.
    - n_samples (int): Number of samples to generate.

    Returns:
    - inputs (torch.Tensor): Tensor of shape (n_samples, sequence_length, 2).
                             Each input has two features per time step:
                             - A random number between 0 and 1.
                             - A mask indicator (0 or 1).
    - targets (torch.Tensor): Tensor of shape (n_samples, 1), containing the sum
                              of the two masked numbers in each sequence.
    """
    random_sequence = torch.rand(n_samples, sequence_length, 1)    
    mask_sequence = torch.zeros(n_samples, sequence_length, 1)
    targets = torch.zeros(n_samples, 1)
    
    for i in range(n_samples):
        # Randomly select two unique positions in the sequence
        idx = torch.randperm(sequence_length)[:2]
        mask_sequence[i, idx, 0] = 1  # Set mask to 1 at selected positions
        # Compute the target sum
        targets[i] = random_sequence[i, idx, 0].sum()
    
    # Concatenate X and M along the feature dimension
    inputs = torch.cat((random_sequence, mask_sequence), dim=2)
    if return_dataloader:
        dataset = TensorDataset(inputs, targets)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        
        return data_loader
    else:
        return inputs, targets


def train(args,
    model,
    device,
    train_loader,
    optimizer,
    criterion,
    train_losses,
    epoch):
    model.train()
    total_loss = 0
    for input_data, target_data in train_loader:
        input_data, target_data = input_data.to(device), target_data.to(device)
        #reset gradient
        optimizer.zero_grad()
        #forward pass
        output = model(input_data)
        #calculate loss
        loss = criterion(output, target_data)
        #backward pass
        loss.backward()
        #optmize
        optimizer.step()
        #record loss
        total_loss += loss.item()

        if args.dry_run:
            print('Dry run enabled, breaking after one batch.')
            break
    
    avg_loss = total_loss / len(train_loader)
    train_losses.append(avg_loss)
    print(f'Epoch {epoch}, Training Loss: {avg_loss:.6f}')

def test(args,
    model,
    device,
    test_loader,
    criterion,
    test_losses,
    epoch):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for input_data, target_data in test_loader:
            input_data, target_data = input_data.to(device), target_data.to(device)

            #forward pass
            output = model(input_data)

            #loss
            loss = criterion(output, target_data)

            #record loss
            total_loss += loss

            if args.dry_run:
                print('Dry run enabled, breaking after one batch.')
                break

        avg_loss = total_loss / len(test_loader)
        test_losses.append(avg_loss)
        print(f'Test Loss: {avg_loss:.6f}')


def main():
    # parse arguments
    parser = argparse.ArgumentParser(
        description='Addition problem benchmarks for recurrent layers'
    )
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=20, metavar='N',
                        help='number of epochs to train (default: 20)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--sequence-length', type=int, default=100, metavar='SL',
                        help='length of the input sequences (default: 100)')
    parser.add_argument('--train-samples', type=int, default=5000, metavar='N',
                        help='number of training samples (default: 5000)')
    parser.add_argument('--test-samples', type=int, default=1000, metavar='N',
                        help='number of test samples (default: 1000)')
    parser.add_argument('--hidden-size', type=int, default=128, metavar='N',
                        help='number of hidden units (default: 128)')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--mps', action="store_true", default=False,
                        help="enables MPS training")
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    args = parser.parse_args()

    #set device
    if args.cuda and not args.mps:
        device = torch.device('cuda')
    elif args.mps and not args.cuda:
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    #set seed
    torch.manual_seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(args.seed)

    print(f'Using device: {device}')

    #get data

    train_loader = generate_adding_problem_data(
        sequence_length=args.sequence_length,
        n_samples=args.train_samples,
        batch_size=args.batch_size,
        shuffle=True
    )

    test_loader = generate_adding_problem_data(
        sequence_length=args.sequence_length,
        n_samples=args.test_samples,
        batch_size=args.test_batch_size,
        shuffle=False
    )

    # define model, optimizer and loss
    input_size = 2
    output_size = 1
    model = RecurrentModel(input_size, args.hidden_size, output_size).to(device)
    criterion = nn.MSELoss
    optimizer = optim.Adam(model.parameters(), lr = args.lr)

    #store loss
    train_losses = []
    test_losses = []
    
    #train and validate
    for epoch in range(1, args.epochs+1):
        train(args,
            model,
            device,
            train_loader,
            optimizer,
            criterion,
            train_losses,
            epoch)
        test(args,
             model,
             device,
             test_loader,
             criterion,
             test_losses,
             epoch)
        
        if args.dry_run:
            print('Dry run enabled, stopping training.')
            break

        #visualize loss


if __name__ == '__main__':
    main()
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import sys
import os
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'recurrent_layers')))
from mgu_rnn_cell import MGU

import torch

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


def train(args, model, device, train_loader, optimizer, loss, epoch):
    model.train()
    for input_data, target_data in train_loader:
        input_data, target_data = input_data.to(device), target_data.to(device)
        #reset gradient
        optimizer.zero_grad()
        #forward pass
        output = model(input_data)
        #calculate loss
        loss = loss(output, target_data)
        #backward pass
        loss.backward()
        #optmize
        optimizer.step()
        #visualize loss here

def test(args, model, device, train_loader, optimizer, loss, epoch):
    model.eval()

def main():
    # parse arguments
    parser = argparse.ArgumentParser(
        description='Addition problem benchmarks for recurrent layers'
    )
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--mps', action="store_true", default=False,
                        help="enables MPS training")
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 1)')
    args = parser.parse_args()

    if args.cuda and not args.mps:
        device = "cuda"
    elif args.mps and not args.cuda:
        device = "mps"
    else:
        device = "cpu"

    device = torch.device(device)
    torch.manual_seed(args.seed)

    #get data
    
    #train
    #test

if __name__ == '__main__':
    main()
        

# Define model parameters
input_size = 2  # Sequence A and marker Sequence B
hidden_size = 50
num_layers = 3
seq_len = 100
batch_size = 64
num_epochs = 10
learning_rate = 0.001

# Initialize the models
model_classes = {
    'LSTM': nn.LSTM,
    'MGU': MGU,
    'GRU': nn.GRU
}

models = {name: model_class(
    input_size=input_size,
    hidden_size=hidden_size,
    num_layers=num_layers,
    batch_first=True
) for name, model_class in model_classes.items()}

# Define the loss function
criterion = nn.MSELoss()

# Generate synthetic addition data
def generate_addition_data(seq_len, batch_size):
    # Sequence A: Random values between 0 and 1
    sequence_a = torch.rand(batch_size, seq_len, 1)
    # Sequence B: Two markers (one-hot vectors)
    markers = torch.zeros(batch_size, seq_len, 1)
    indices = torch.randint(0, seq_len, (batch_size, 2))
    for i in range(batch_size):
        markers[i, indices[i, 0], 0] = 1
        markers[i, indices[i, 1], 0] = 1
    # Target: Sum of the two marked values
    target = torch.sum(sequence_a[torch.arange(batch_size).unsqueeze(1), indices], dim=1, keepdim=True)
    # Concatenate Sequence A and markers to create the input
    input_sequence = torch.cat((sequence_a, markers), dim=2)
    return input_sequence, target

# Training loop for each model
for model_name, model in models.items():
    # Define optimizer for each model
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    print(f"\nTraining {model_name} model...")
    for epoch in range(num_epochs):
        input_sequence, target = generate_addition_data(seq_len, batch_size)

        # Zero the gradient
        optimizer.zero_grad()

        # Forward pass
        output, _ = model(input_sequence)
        prediction = output[:, -1, :]  # Take the last output for prediction

        # Compute loss
        loss = criterion(prediction, target)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Print loss for the epoch
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

print("Training complete.")
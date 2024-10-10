import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'recurrent_layers')))
from mgu_rnn_cell import MGU

# Define model parameters
input_size = 2  # Sequence A and marker Sequence B
hidden_size = 50
num_layers = 1
seq_len = 100
batch_size = 64
num_epochs = 10
learning_rate = 0.001

# Initialize the models
model_classes = {
    'MGU': MGU,
    'LSTM': nn.LSTM,
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
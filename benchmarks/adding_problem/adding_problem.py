import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch import Tensor
import argparse
import matplotlib.pyplot as plt
import os
import csv


class RecurrentModel(nn.Module):
    def __init__(self, cell, input_size: int, hidden_size: int, output_size: int, **kwargs):
        super().__init__()
        self.rnn = cell(input_size, hidden_size, batch_first=True, **kwargs)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, inp: Tensor):
        output, _ = self.rnn(inp)
        last = output[:, -1, :]
        return self.fc(last)


def generate_adding_problem_data(
    sequence_length: int,
    n_samples: int,
    return_dataloader: bool = True,
    batch_size: int = 64,
    shuffle=True,
):
    """Generate data for the adding problem benchmark.

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


def train(args, model, device, train_loader, optimizer, criterion, train_losses, epoch):
    model.train()
    total_loss = 0
    for input_data, target_data in train_loader:
        input_data, target_data = input_data.to(device), target_data.to(device)
        optimizer.zero_grad()
        output = model(input_data)
        loss = criterion(output, target_data)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if args.dry_run:
            print("Dry run enabled, breaking after one batch.")
            break

    avg_loss = total_loss / len(train_loader)
    train_losses.append(avg_loss)
    print(f"Epoch {epoch}, Training Loss: {avg_loss:.6f}")


def test(args, model, device, test_loader, criterion, test_losses, epoch):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for input_data, target_data in test_loader:
            input_data, target_data = input_data.to(device), target_data.to(device)
            output = model(input_data)
            loss = criterion(output, target_data)
            total_loss += float(loss.item())
            if args.dry_run:
                print("Dry run enabled, breaking after one batch.")
                break
    avg_loss = total_loss / len(test_loader)
    test_losses.append(avg_loss)
    print(f"Epoch {epoch}, Test Loss: {avg_loss:.6f}")


def plot_learning_curves(train_losses, test_losses, out_png, title="", show=False):
    plt.figure()
    plt.plot(train_losses, label="Train MSE")
    plt.plot(test_losses, label="Test MSE")
    plt.axhline(1.0 / 6.0, linestyle="--", label="Baseline (1/6)")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title(title or "Learning Curves")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(out_png, bbox_inches="tight", dpi=150)
    if show:
        plt.show()
    plt.close()


def save_losses_csv(train_losses, test_losses, out_csv):
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_mse", "test_mse"])
        for i, (tr, te) in enumerate(zip(train_losses, test_losses), start=1):
            w.writerow([i, f"{tr:.8f}", f"{te:.8f}"])


def main():
    parser = argparse.ArgumentParser(
        description="Addition problem benchmarks for recurrent layers"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1000,
        metavar="N",
        help="number of epochs to train (default: 20)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        metavar="LR",
        help="learning rate (default: 0.001)",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        metavar="DO",
        help="dropout (default: 0.2)",
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=2,
        metavar="NL",
        help="num_layers (default: 2)",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=100,
        metavar="SL",
        help="length of the input sequences (default: 100)",
    )
    parser.add_argument(
        "--train-samples",
        type=int,
        default=5000,
        metavar="N",
        help="number of training samples (default: 5000)",
    )
    parser.add_argument(
        "--test-samples",
        type=int,
        default=1000,
        metavar="N",
        help="number of test samples (default: 1000)",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=128,
        metavar="N",
        help="number of hidden units (default: 128)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, metavar="S", help="random seed (default: 42)"
    )
    parser.add_argument(
        "--cuda", action="store_true", default=True, help="enables CUDA training"
    )
    parser.add_argument(
        "--mps", action="store_true", default=False, help="enables MPS training"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="quickly check a single pass",
    )
    parser.add_argument(
        "--outdir", type=str, default="runs/adding_single", help="Where to save plots/logs."
    )
    parser.add_argument(
        "--show",
        action="store_true",
        default=False,
        help="Call plt.show() after saving the plot.",
    )
    parser.add_argument(
        "--save-csv", action="store_true", default=False, help="Also save losses as CSV."
    )
    args = parser.parse_args()

    if args.cuda and torch.cuda.is_available():
        device = torch.device("cuda")
    elif args.mps and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(args.seed)

    print(f"Using device: {device}")

    train_loader = generate_adding_problem_data(
        sequence_length=args.sequence_length,
        n_samples=args.train_samples,
        batch_size=args.batch_size,
        shuffle=True,
    )

    test_loader = generate_adding_problem_data(
        sequence_length=args.sequence_length,
        n_samples=args.test_samples,
        batch_size=args.test_batch_size,
        shuffle=False,
    )

    input_size = 2
    output_size = 1
    model = RecurrentModel(
        nn.GRU,
        input_size,
        args.hidden_size,
        output_size,
        dropout=args.dropout,
        num_layers=args.num_layers,
    ).to(device)
    # model = torch.jit.script(model)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)  # Adam

    train_losses = []
    test_losses = []

    # train and validate
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, criterion, train_losses, epoch)
        test(args, model, device, test_loader, criterion, test_losses, epoch)

        if args.dry_run:
            print("Dry run enabled, stopping training.")
            break

    # ---- visualize & save ----
    os.makedirs(args.outdir, exist_ok=True)
    model_name = type(model.rnn).__name__ if hasattr(model, "rnn") else type(model).__name__
    tag = f"{model_name}_T{args.sequence_length}_H{args.hidden_size}"
    png_path = os.path.join(args.outdir, f"{tag}_learning_curves.png")
    plot_learning_curves(
        train_losses,
        test_losses,
        png_path,
        title=f"{model_name} on Adding Problem (T={args.sequence_length})",
        show=args.show,
    )
    print(f"Saved plot: {png_path}")

    if args.save_csv:
        csv_path = os.path.join(args.outdir, f"{tag}_losses.csv")
        save_losses_csv(train_losses, test_losses, csv_path)
        print(f"Saved CSV: {csv_path}")

    best_test = min(test_losses)
    best_epoch = 1 + int(test_losses.index(best_test))
    print(f"Best Test MSE: {best_test:.6f} @ epoch {best_epoch}")
    print("Note: trivial baseline MSE ≈ 1/6 ≈ 0.1667")


if __name__ == "__main__":
    main()

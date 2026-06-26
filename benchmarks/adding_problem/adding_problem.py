from __future__ import annotations

import argparse
import csv
import inspect
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

import torchrecurrent
from torchrecurrent.benchmarks import adding_problem

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - benchmark convenience path
    plt = None


RECURRENT_LAYERS = [
    torchrecurrent.AntisymmetricRNN,
    torchrecurrent.ATR,
    torchrecurrent.BR,
    torchrecurrent.CFN,
    torchrecurrent.coRNN,
    torchrecurrent.DSGU,
    torchrecurrent.FastGRNN,
    torchrecurrent.FastRNN,
    torchrecurrent.GatedAntisymmetricRNN,
    torchrecurrent.IndRNN,
    torchrecurrent.JANET,
    torchrecurrent.LEM,
    torchrecurrent.LightRU,
    torchrecurrent.LiGRU,
    torchrecurrent.MGU,
    torchrecurrent.MiRU1,
    torchrecurrent.MiRU2,
    torchrecurrent.MultiplicativeLSTM,
    torchrecurrent.MUT1,
    torchrecurrent.MUT2,
    torchrecurrent.MUT3,
    torchrecurrent.NAS,
    torchrecurrent.NBR,
    torchrecurrent.OriginalLSTM,
    torchrecurrent.PeepholeLSTM,
    torchrecurrent.RAN,
    torchrecurrent.ResLSTM,
    torchrecurrent.SCRN,
    torchrecurrent.SGRN,
    torchrecurrent.SGU,
    torchrecurrent.STAR,
    torchrecurrent.tauGRU,
    torchrecurrent.UGRNN,
    torchrecurrent.UnICORNN,
    torchrecurrent.WMCLSTM,
]

TORCH_BASELINES = [nn.RNN, nn.GRU, nn.LSTM]
BASELINE_MSE = 1.0 / 6.0


@dataclass
class BenchmarkResult:
    model: str
    status: str
    params: Optional[int] = None
    best_test_mse: Optional[float] = None
    best_epoch: Optional[int] = None
    final_train_mse: Optional[float] = None
    final_test_mse: Optional[float] = None
    seconds: Optional[float] = None
    error: Optional[str] = None


class RecurrentRegressor(nn.Module):
    def __init__(
        self,
        recurrent_layer: type[nn.Module],
        input_size: int,
        hidden_size: int,
        output_size: int,
        **kwargs,
    ):
        super().__init__()
        self.rnn = recurrent_layer(input_size, hidden_size, batch_first=True, **kwargs)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, inp: Tensor) -> Tensor:
        output, _ = self.rnn(inp)
        return self.fc(output[:, -1, :])


def model_name(layer: type[nn.Module]) -> str:
    return layer.__name__


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(args: argparse.Namespace) -> torch.device:
    if args.device != "auto":
        return torch.device(args.device)
    if torch.cuda.is_available() and not args.no_cuda:
        return torch.device("cuda")
    if args.mps and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def layer_kwargs(layer: type[nn.Module], args: argparse.Namespace) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "dropout": args.dropout,
        "num_layers": args.num_layers,
    }
    signature = inspect.signature(layer)
    if "nonlinearity" in signature.parameters:
        kwargs["nonlinearity"] = "tanh"
    return kwargs


def make_loaders(
    train_inputs: Tensor,
    train_targets: Tensor,
    test_inputs: Tensor,
    test_targets: Tensor,
    batch_size: int,
    test_batch_size: int,
    seed: int,
) -> tuple[DataLoader, DataLoader]:
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(
        TensorDataset(train_inputs, train_targets),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    test_loader = DataLoader(
        TensorDataset(test_inputs, test_targets),
        batch_size=test_batch_size,
        shuffle=False,
    )
    return train_loader, test_loader


def train_epoch(
    model: nn.Module,
    device: torch.device,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    max_batches: Optional[int],
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0
    for input_data, target_data in train_loader:
        input_data = input_data.to(device)
        target_data = target_data.to(device)

        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(input_data), target_data)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        n_batches += 1
        if max_batches is not None and n_batches >= max_batches:
            break
    return total_loss / n_batches


def evaluate(
    model: nn.Module,
    device: torch.device,
    test_loader: DataLoader,
    criterion: nn.Module,
    max_batches: Optional[int],
) -> float:
    model.eval()
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for input_data, target_data in test_loader:
            input_data = input_data.to(device)
            target_data = target_data.to(device)
            total_loss += float(criterion(model(input_data), target_data).item())
            n_batches += 1
            if max_batches is not None and n_batches >= max_batches:
                break
    return total_loss / n_batches


def run_one_model(
    layer: type[nn.Module],
    args: argparse.Namespace,
    device: torch.device,
    train_inputs: Tensor,
    train_targets: Tensor,
    test_inputs: Tensor,
    test_targets: Tensor,
    curves_dir: Path,
) -> BenchmarkResult:
    name = model_name(layer)
    set_seed(args.seed + args.model_seed_offset)
    train_loader, test_loader = make_loaders(
        train_inputs,
        train_targets,
        test_inputs,
        test_targets,
        args.batch_size,
        args.test_batch_size,
        args.seed + args.shuffle_seed_offset,
    )
    model = RecurrentRegressor(
        layer,
        input_size=2,
        hidden_size=args.hidden_size,
        output_size=1,
        **layer_kwargs(layer, args),
    ).to(device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_losses: list[float] = []
    test_losses: list[float] = []
    start = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(
            model,
            device,
            train_loader,
            optimizer,
            criterion,
            args.max_batches,
        )
        test_loss = evaluate(
            model,
            device,
            test_loader,
            criterion,
            args.max_test_batches,
        )
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        if args.verbose:
            print(
                f"{name:>24s} epoch {epoch:03d}: train={train_loss:.6f} test={test_loss:.6f}"
            )

    seconds = time.perf_counter() - start
    best_test = min(test_losses)
    best_epoch = test_losses.index(best_test) + 1
    write_curve_csv(curves_dir / f"{name}_losses.csv", train_losses, test_losses)
    return BenchmarkResult(
        model=name,
        status="ok",
        params=params,
        best_test_mse=best_test,
        best_epoch=best_epoch,
        final_train_mse=train_losses[-1],
        final_test_mse=test_losses[-1],
        seconds=seconds,
    )


def write_curve_csv(
    path: Path, train_losses: list[float], test_losses: list[float]
) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_mse", "test_mse"])
        for epoch, (train_loss, test_loss) in enumerate(
            zip(train_losses, test_losses), start=1
        ):
            writer.writerow([epoch, f"{train_loss:.8f}", f"{test_loss:.8f}"])


def write_summary_csv(path: Path, results: list[BenchmarkResult]) -> None:
    fields = (
        list(asdict(results[0]).keys())
        if results
        else list(BenchmarkResult.__annotations__)
    )
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))


def read_curve(path: Path) -> tuple[list[int], list[float]]:
    epochs: list[int] = []
    losses: list[float] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            losses.append(float(row["test_mse"]))
    return epochs, losses


def svg_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def write_svg_bar(path: Path, results: list[BenchmarkResult]) -> None:
    row_height = 22
    margin_left = 190
    margin_right = 40
    margin_top = 42
    width = 980
    height = margin_top + row_height * len(results) + 48
    values = [r.best_test_mse or 0.0 for r in results]
    max_value = max(values + [BASELINE_MSE]) * 1.08
    plot_width = width - margin_left - margin_right
    baseline_x = margin_left + plot_width * BASELINE_MSE / max_value
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        "<style>text{font-family:Arial,sans-serif;font-size:12px}"
        ".title{font-size:18px;font-weight:700}.axis{fill:#555}</style>",
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="20" y="26" class="title">Adding Problem: best test MSE</text>',
        f'<line x1="{baseline_x:.1f}" x2="{baseline_x:.1f}" y1="{margin_top - 8}" '
        f'y2="{height - 36}" stroke="#d62728" stroke-dasharray="5 4"/>',
        f'<text x="{baseline_x + 4:.1f}" y="{margin_top - 14}" fill="#d62728">'
        "baseline 1/6</text>",
    ]
    for idx, result in enumerate(results):
        y = margin_top + idx * row_height
        value = result.best_test_mse or 0.0
        bar_width = plot_width * value / max_value
        lines.extend(
            [
                f'<text x="{margin_left - 8}" y="{y + 14}" text-anchor="end">'
                f"{idx + 1}. {svg_escape(result.model)}</text>",
                f'<rect x="{margin_left}" y="{y + 3}" width="{bar_width:.1f}" '
                f'height="14" fill="#4c78a8"/>',
                f'<text x="{margin_left + bar_width + 5:.1f}" y="{y + 14}">'
                f"{value:.6f}</text>",
            ]
        )
    lines.append("</svg>")
    path.write_text("\n".join(lines))


def write_svg_curves(
    path: Path, results: list[BenchmarkResult], curves_dir: Path, top_k: int
) -> None:
    width = 1000
    height = 650
    margin_left = 70
    margin_right = 180
    margin_top = 48
    margin_bottom = 55
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    curves = []
    max_epoch = 1
    max_loss = BASELINE_MSE
    min_loss = 0.0
    for result in results:
        epochs, losses = read_curve(curves_dir / f"{result.model}_losses.csv")
        curves.append((result.model, epochs, losses))
        max_epoch = max(max_epoch, max(epochs))
        max_loss = max(max_loss, max(losses))
    max_loss *= 1.08

    def point(epoch: int, loss: float) -> tuple[float, float]:
        x = margin_left + plot_width * (epoch - 1) / max(1, max_epoch - 1)
        y = margin_top + plot_height * (max_loss - loss) / max(1e-12, max_loss - min_loss)
        return x, y

    palette = [
        "#4c78a8",
        "#f58518",
        "#54a24b",
        "#e45756",
        "#72b7b2",
        "#b279a2",
        "#ff9da6",
        "#9d755d",
        "#bab0ac",
        "#8cd17d",
    ]
    top_names = {result.model for result in results[:top_k]}
    top_order = [result.model for result in results[:top_k]]
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        "<style>text{font-family:Arial,sans-serif;font-size:12px}"
        ".title{font-size:18px;font-weight:700}.axis{fill:#555}</style>",
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="20" y="28" class="title">Adding Problem: test curves</text>',
        f'<rect x="{margin_left}" y="{margin_top}" width="{plot_width}" '
        f'height="{plot_height}" fill="none" stroke="#999"/>',
    ]
    baseline_y = point(1, BASELINE_MSE)[1]
    lines.append(
        f'<line x1="{margin_left}" x2="{margin_left + plot_width}" y1="{baseline_y:.1f}" '
        f'y2="{baseline_y:.1f}" stroke="#d62728" stroke-dasharray="5 4"/>'
    )
    for name, epochs, losses in curves:
        points = " ".join(
            f"{x:.1f},{y:.1f}"
            for x, y in (point(epoch, loss) for epoch, loss in zip(epochs, losses))
        )
        if name in top_names:
            color = palette[top_order.index(name) % len(palette)]
            lines.append(
                f'<polyline points="{points}" fill="none" stroke="{color}" '
                'stroke-width="2.2"/>'
            )
        else:
            lines.append(
                f'<polyline points="{points}" fill="none" stroke="#aaa" '
                'stroke-width="0.8" opacity="0.45"/>'
            )
    for idx, result in enumerate(results[:top_k]):
        color = palette[idx % len(palette)]
        y = margin_top + 18 + idx * 18
        lines.extend(
            [
                f'<line x1="{width - margin_right + 10}" x2="{width - margin_right + 34}" '
                f'y1="{y - 4}" y2="{y - 4}" stroke="{color}" stroke-width="2.2"/>',
                f'<text x="{width - margin_right + 40}" y="{y}">'
                f"{svg_escape(result.model)}</text>",
            ]
        )
    lines.extend(
        [
            f'<text x="{margin_left + plot_width / 2}" y="{height - 14}" '
            'text-anchor="middle" class="axis">Epoch</text>',
            f'<text x="18" y="{margin_top + plot_height / 2}" transform="rotate(-90 18 '
            f'{margin_top + plot_height / 2})" text-anchor="middle" class="axis">Test MSE</text>',
            "</svg>",
        ]
    )
    path.write_text("\n".join(lines))


def write_svg_scatter(path: Path, results: list[BenchmarkResult], top_k: int) -> None:
    width = 900
    height = 620
    margin_left = 70
    margin_right = 35
    margin_top = 48
    margin_bottom = 55
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    max_seconds = max((r.seconds or 0.0) for r in results) * 1.08
    max_loss = max([r.best_test_mse or 0.0 for r in results] + [BASELINE_MSE]) * 1.08

    def point(seconds: float, loss: float) -> tuple[float, float]:
        x = margin_left + plot_width * seconds / max(1e-12, max_seconds)
        y = margin_top + plot_height * (max_loss - loss) / max(1e-12, max_loss)
        return x, y

    baseline_y = point(0.0, BASELINE_MSE)[1]
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        "<style>text{font-family:Arial,sans-serif;font-size:12px}"
        ".title{font-size:18px;font-weight:700}.axis{fill:#555}</style>",
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="20" y="28" class="title">Adding Problem: quality/runtime tradeoff</text>',
        f'<rect x="{margin_left}" y="{margin_top}" width="{plot_width}" '
        f'height="{plot_height}" fill="none" stroke="#999"/>',
        f'<line x1="{margin_left}" x2="{margin_left + plot_width}" y1="{baseline_y:.1f}" '
        f'y2="{baseline_y:.1f}" stroke="#d62728" stroke-dasharray="5 4"/>',
    ]
    for idx, result in enumerate(results):
        x, y = point(result.seconds or 0.0, result.best_test_mse or 0.0)
        radius = min(14.0, max(4.0, ((result.params or 0) / 5000) ** 0.5))
        lines.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius:.1f}" fill="#59a14f" opacity="0.75"/>'
        )
        if idx < top_k:
            lines.append(
                f'<text x="{x + 6:.1f}" y="{y - 6:.1f}">{svg_escape(result.model)}</text>'
            )
    lines.extend(
        [
            f'<text x="{margin_left + plot_width / 2}" y="{height - 14}" '
            'text-anchor="middle" class="axis">Training seconds</text>',
            f'<text x="18" y="{margin_top + plot_height / 2}" transform="rotate(-90 18 '
            f'{margin_top + plot_height / 2})" text-anchor="middle" class="axis">Best test MSE</text>',
            "</svg>",
        ]
    )
    path.write_text("\n".join(lines))


def make_svg_plots(outdir: Path, results: list[BenchmarkResult], top_k: int) -> list[Path]:
    ok_results = [r for r in results if r.status == "ok"]
    ok_results.sort(
        key=lambda r: r.best_test_mse if r.best_test_mse is not None else float("inf")
    )
    if not ok_results:
        return []
    curves_dir = outdir / "curves"
    paths = [
        outdir / "best_test_mse.svg",
        outdir / "test_curves_all_models.svg",
        outdir / "best_mse_vs_runtime.svg",
    ]
    write_svg_bar(paths[0], ok_results)
    write_svg_curves(paths[1], ok_results, curves_dir, top_k)
    write_svg_scatter(paths[2], ok_results, top_k)
    return paths


def make_plots(outdir: Path, results: list[BenchmarkResult], top_k: int) -> list[Path]:
    if plt is None:
        return make_svg_plots(outdir, results, top_k)

    ok_results = [r for r in results if r.status == "ok"]
    ok_results.sort(
        key=lambda r: r.best_test_mse if r.best_test_mse is not None else float("inf")
    )
    if not ok_results:
        return []

    plot_paths: list[Path] = []
    curves_dir = outdir / "curves"

    bar_path = outdir / "best_test_mse.png"
    fig_height = max(8.0, 0.28 * len(ok_results))
    plt.figure(figsize=(11, fig_height))
    labels = [r.model for r in ok_results]
    values = [r.best_test_mse for r in ok_results]
    plt.barh(labels, values, color="#4c78a8")
    plt.axvline(BASELINE_MSE, color="#d62728", linestyle="--", label="baseline 1/6")
    plt.gca().invert_yaxis()
    plt.xlabel("Best test MSE")
    plt.title("Adding Problem: best test MSE by recurrent layer")
    plt.legend()
    plt.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    plt.savefig(bar_path, dpi=180)
    plt.close()
    plot_paths.append(bar_path)

    curves_path = outdir / "test_curves_all_models.png"
    top_names = {r.model for r in ok_results[:top_k]}
    plt.figure(figsize=(13, 8))
    for result in ok_results:
        epochs, losses = read_curve(curves_dir / f"{result.model}_losses.csv")
        if result.model in top_names:
            plt.plot(epochs, losses, linewidth=2.2, label=result.model)
        else:
            plt.plot(epochs, losses, color="#9e9e9e", linewidth=0.8, alpha=0.45)
    plt.axhline(BASELINE_MSE, color="#d62728", linestyle="--", label="baseline 1/6")
    plt.xlabel("Epoch")
    plt.ylabel("Test MSE")
    plt.title(f"Adding Problem: test curves, top {min(top_k, len(ok_results))} highlighted")
    plt.legend(ncol=2, fontsize=8)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(curves_path, dpi=180)
    plt.close()
    plot_paths.append(curves_path)

    scatter_path = outdir / "best_mse_vs_runtime.png"
    plt.figure(figsize=(10, 7))
    xs = [r.seconds for r in ok_results]
    ys = [r.best_test_mse for r in ok_results]
    sizes = [max(20, min(300, (r.params or 0) / 200)) for r in ok_results]
    plt.scatter(xs, ys, s=sizes, color="#59a14f", alpha=0.75, edgecolors="#1f1f1f")
    for result in ok_results[:top_k]:
        plt.annotate(result.model, (result.seconds, result.best_test_mse), fontsize=8)
    plt.axhline(BASELINE_MSE, color="#d62728", linestyle="--", label="baseline 1/6")
    plt.xlabel("Training seconds")
    plt.ylabel("Best test MSE")
    plt.title("Adding Problem: quality/runtime tradeoff")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(scatter_path, dpi=180)
    plt.close()
    plot_paths.append(scatter_path)

    return plot_paths


def select_layers(args: argparse.Namespace) -> list[type[nn.Module]]:
    layers = list(RECURRENT_LAYERS)
    if args.include_torch_baselines:
        layers.extend(TORCH_BASELINES)
    if args.models:
        requested = [name.strip() for name in args.models.split(",") if name.strip()]
        by_name = {model_name(layer): layer for layer in layers}
        unknown = sorted(set(requested).difference(by_name))
        if unknown:
            raise ValueError(f"Unknown model(s): {', '.join(unknown)}")
        layers = [by_name[name] for name in requested]
    if args.skip_models:
        skipped = {name.strip() for name in args.skip_models.split(",") if name.strip()}
        layers = [layer for layer in layers if model_name(layer) not in skipped]
    return layers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified adding-problem benchmark for torchrecurrent layers."
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--test-batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=100)
    parser.add_argument("--train-samples", type=int, default=5000)
    parser.add_argument("--test-samples", type=int, default=1000)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-seed-offset", type=int, default=10_000)
    parser.add_argument("--shuffle-seed-offset", type=int, default=20_000)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, or mps")
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--mps", action="store_true", help="Allow MPS when device=auto.")
    parser.add_argument("--models", default="", help="Comma-separated model names to run.")
    parser.add_argument(
        "--skip-models", default="", help="Comma-separated model names to skip."
    )
    parser.add_argument("--include-torch-baselines", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true", default=True)
    parser.add_argument("--fail-fast", dest="continue_on_error", action="store_false")
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--max-test-batches", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--outdir", type=Path, default=Path("/tmp/torchrecurrent_adding_problem")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outdir = args.outdir.expanduser().resolve()
    curves_dir = outdir / "curves"
    outdir.mkdir(parents=True, exist_ok=True)
    curves_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args)
    layers = select_layers(args)
    set_seed(args.seed)
    train_inputs, train_targets = adding_problem(
        sequence_length=args.sequence_length,
        n_samples=args.train_samples,
        return_dataloader=False,
    )
    test_inputs, test_targets = adding_problem(
        sequence_length=args.sequence_length,
        n_samples=args.test_samples,
        return_dataloader=False,
    )

    config = vars(args).copy()
    config["outdir"] = str(outdir)
    config["device"] = str(device)
    config["models"] = [model_name(layer) for layer in layers]
    with (outdir / "config.json").open("w") as f:
        json.dump(config, f, indent=2, sort_keys=True)

    print(f"Output directory: {outdir}")
    print(f"Device: {device}")
    print(f"Models: {len(layers)}")

    results: list[BenchmarkResult] = []
    for idx, layer in enumerate(layers, start=1):
        name = model_name(layer)
        print(f"[{idx:02d}/{len(layers):02d}] {name}")
        try:
            result = run_one_model(
                layer,
                args,
                device,
                train_inputs,
                train_targets,
                test_inputs,
                test_targets,
                curves_dir,
            )
            print(
                f"  best_test_mse={result.best_test_mse:.6f} "
                f"epoch={result.best_epoch} seconds={result.seconds:.1f}"
            )
        except Exception as exc:
            if not args.continue_on_error:
                raise
            result = BenchmarkResult(model=name, status="error", error=repr(exc))
            print(f"  error: {result.error}")
        results.append(result)
        write_summary_csv(outdir / "summary.csv", results)
        with (outdir / "summary.json").open("w") as f:
            json.dump([asdict(result) for result in results], f, indent=2)

    plot_paths = make_plots(outdir, results, args.top_k)
    ok_results = [result for result in results if result.status == "ok"]
    ok_results.sort(
        key=lambda r: r.best_test_mse if r.best_test_mse is not None else float("inf")
    )
    if ok_results:
        print("\nTop results:")
        for rank, result in enumerate(ok_results[: args.top_k], start=1):
            print(
                f"{rank:2d}. {result.model:24s} "
                f"best_test_mse={result.best_test_mse:.6f} "
                f"epoch={result.best_epoch} seconds={result.seconds:.1f}"
            )
    print("\nSaved:")
    print(f"  {outdir / 'summary.csv'}")
    print(f"  {outdir / 'summary.json'}")
    for path in plot_paths:
        print(f"  {path}")
    if plt is None and not plot_paths:
        print("  plots skipped because matplotlib is not installed")


if __name__ == "__main__":
    main()

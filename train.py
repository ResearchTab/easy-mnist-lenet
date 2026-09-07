from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import random
import subprocess
import time
from contextlib import AbstractContextManager, nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

try:
    import psutil
except ImportError:  # System telemetry is optional; experiment metrics are always recorded.
    psutil = None

SPLIT_SEED = 20260906
TRAIN_PER_CLASS = 4_000
VALIDATION_PER_CLASS = 1_000
MNIST_VERSION = "torchvision-MNIST-v1"


class RunLogger(Protocol):
    def log(
        self,
        values: dict[str, float],
        *,
        step: int,
        units: dict[str, str] | None = None,
        directions: dict[str, str] | None = None,
    ) -> None: ...

    def log_system(self, values: dict[str, float], *, step: int, units: dict[str, str]) -> None: ...

    def heartbeat(
        self,
        progress_percent: int,
        *,
        note: str,
        current_step: int,
        total_steps: int,
    ) -> Any: ...

    def log_artifact(
        self, path: Path, *, logical_path: str = "", tags: list[str] | None = None
    ) -> Any: ...

    def set_summary(self, *, interpretation: str, limitations: str) -> Any: ...

    def finish(self, *, outcome: str | None = None) -> None: ...


class LocalRun:
    """Portable development recorder; never used for a user-visible reference run."""

    def log(
        self,
        values: dict[str, float],
        *,
        step: int,
        units: dict[str, str] | None = None,
        directions: dict[str, str] | None = None,
    ) -> None:
        del values, step, units, directions

    def log_system(self, values: dict[str, float], *, step: int, units: dict[str, str]) -> None:
        del values, step, units

    def heartbeat(
        self,
        progress_percent: int,
        *,
        note: str,
        current_step: int,
        total_steps: int,
    ) -> None:
        del progress_percent, note, current_step, total_steps

    def log_artifact(
        self, path: Path, *, logical_path: str = "", tags: list[str] | None = None
    ) -> None:
        del path, logical_path, tags

    def set_summary(self, *, interpretation: str, limitations: str) -> None:
        del interpretation, limitations

    def finish(self, *, outcome: str | None = None) -> None:
        del outcome


class LeNet5(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, padding=2),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120),
            nn.Tanh(),
            nn.Linear(120, 84),
            nn.Tanh(),
            nn.Linear(84, 10),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(inputs))


@dataclass(frozen=True)
class Settings:
    seed: int = 7
    split_seed: int = SPLIT_SEED
    batch_size: int = 128
    epochs: int = 3
    learning_rate: float = 0.01
    momentum: float = 0.9
    log_every: int = 50
    validation_every: int = 50
    num_workers: int = 0


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def class_balanced_split(targets: torch.Tensor, split_seed: int) -> tuple[list[int], list[int]]:
    train_indices: list[int] = []
    validation_indices: list[int] = []
    generator = torch.Generator().manual_seed(split_seed)
    for digit in range(10):
        candidates = torch.where(targets == digit)[0]
        ordered = candidates[torch.randperm(len(candidates), generator=generator)]
        train_indices.extend(ordered[:TRAIN_PER_CLASS].tolist())
        validation_indices.extend(
            ordered[TRAIN_PER_CLASS : TRAIN_PER_CLASS + VALIDATION_PER_CLASS].tolist()
        )
    return train_indices, validation_indices


def create_loaders(
    settings: Settings, data_dir: Path
) -> tuple[DataLoader[Any], DataLoader[Any], DataLoader[Any]]:
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    training_pool = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_data = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
    train_indices, validation_indices = class_balanced_split(
        training_pool.targets, settings.split_seed
    )
    loader_generator = torch.Generator().manual_seed(settings.seed)
    common = {
        "batch_size": settings.batch_size,
        "num_workers": settings.num_workers,
        "pin_memory": False,
    }
    return (
        DataLoader(
            Subset(training_pool, train_indices),
            shuffle=True,
            generator=loader_generator,
            **common,
        ),
        DataLoader(Subset(training_pool, validation_indices), shuffle=False, **common),
        DataLoader(test_data, shuffle=False, **common),
    )


def evaluate(
    model: nn.Module,
    loader: DataLoader[Any],
    criterion: nn.Module,
    *,
    collect_confusion: bool = True,
) -> tuple[float, float, torch.Tensor]:
    model.eval()
    loss_total = 0.0
    correct = 0
    count = 0
    confusion = torch.zeros((10, 10), dtype=torch.int64)
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images)
            loss_total += float(criterion(logits, labels)) * len(labels)
            predictions = logits.argmax(dim=1)
            correct += int((predictions == labels).sum())
            count += len(labels)
            if collect_confusion:
                for expected, predicted in zip(labels.tolist(), predictions.tolist(), strict=True):
                    confusion[expected, predicted] += 1
    return loss_total / count, correct / count, confusion


def class_metrics(confusion: torch.Tensor) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    for digit in range(10):
        true_positive = int(confusion[digit, digit])
        predicted = int(confusion[:, digit].sum())
        expected = int(confusion[digit, :].sum())
        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / expected if expected else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append({"digit": digit, "precision": precision, "recall": recall, "f1": f1})
    return rows


def git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],  # noqa: S607 - Git is optional read-only provenance.
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def write_outputs(
    output_dir: Path,
    settings: Settings,
    metric_history: list[dict[str, float | int | str]],
    summary: dict[str, Any],
    confusion: torch.Tensor,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    metrics_path = output_dir / "metrics.csv"
    with metrics_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["step", "key", "value", "unit", "source"])
        writer.writeheader()
        writer.writerows(metric_history)

    classes_path = output_dir / "per_class_metrics.csv"
    with classes_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["digit", "precision", "recall", "f1"])
        writer.writeheader()
        writer.writerows(class_metrics(confusion))

    confusion_path = output_dir / "confusion_matrix.csv"
    with confusion_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["actual/predicted", *range(10)])
        for digit, row in enumerate(confusion.tolist()):
            writer.writerow([digit, *row])

    manifest_path = output_dir / "reproduction_manifest.json"
    manifest = {
        "command": "python train.py --project EASY --experiment E-1 --seed <seed>",
        "configuration": asdict(settings),
        "dataset": MNIST_VERSION,
        "git_commit": git_commit(),
        "model": "LeNet-5",
        "platform": platform.system(),
        "python": platform.python_version(),
        "split": {"train": 40_000, "validation": 10_000, "test": 10_000, "unused": 10_000},
        "torch": torch.__version__,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return [summary_path, metrics_path, classes_path, confusion_path, manifest_path]


def train(settings: Settings, run: RunLogger, data_dir: Path, output_dir: Path) -> dict[str, Any]:
    if settings.validation_every < 1:
        raise ValueError("validation_every must be at least 1")
    run_started = time.perf_counter()
    seed_everything(settings.seed)
    train_loader, validation_loader, test_loader = create_loaders(settings, data_dir)
    model = LeNet5()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=settings.learning_rate, momentum=settings.momentum
    )
    total_steps = len(train_loader) * settings.epochs
    global_step = 0
    history: list[dict[str, float | int | str]] = []
    metric_history: list[dict[str, float | int | str]] = []

    for epoch in range(settings.epochs):
        model.train()
        epoch_started = time.perf_counter()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_count = 0
        for images, labels in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            batch_accuracy = float((logits.argmax(dim=1) == labels).float().mean())
            epoch_loss += float(loss.detach()) * len(labels)
            epoch_correct += int((logits.argmax(dim=1) == labels).sum())
            epoch_count += len(labels)
            if global_step % settings.log_every == 0:
                batch_loss = float(loss.detach())
                run.log(
                    {"train.loss": batch_loss, "train.accuracy": batch_accuracy},
                    step=global_step,
                    units={"train.loss": "cross_entropy", "train.accuracy": "fraction"},
                    directions={"train.loss": "LOWER", "train.accuracy": "HIGHER"},
                )
                history.append(
                    {
                        "step": global_step,
                        "scope": "train_batch",
                        "loss": batch_loss,
                        "accuracy": batch_accuracy,
                    }
                )
                metric_history.extend(
                    [
                        {
                            "step": global_step,
                            "key": "train.loss",
                            "value": batch_loss,
                            "unit": "cross_entropy",
                            "source": "EXPERIMENT",
                        },
                        {
                            "step": global_step,
                            "key": "train.accuracy",
                            "value": batch_accuracy,
                            "unit": "fraction",
                            "source": "EXPERIMENT",
                        },
                    ]
                )
            if global_step % 100 == 0:
                run.heartbeat(
                    int(100 * global_step / total_steps),
                    note=f"Training epoch {epoch + 1} of {settings.epochs}.",
                    current_step=global_step,
                    total_steps=total_steps,
                )
            global_step += 1
            if global_step % settings.validation_every == 0:
                checkpoint_loss, checkpoint_accuracy, _ = evaluate(
                    model,
                    validation_loader,
                    criterion,
                    collect_confusion=False,
                )
                run.log(
                    {
                        "validation.loss": checkpoint_loss,
                        "validation.accuracy": checkpoint_accuracy,
                    },
                    step=global_step,
                    units={
                        "validation.loss": "cross_entropy",
                        "validation.accuracy": "fraction",
                    },
                    directions={
                        "validation.loss": "LOWER",
                        "validation.accuracy": "HIGHER",
                    },
                )
                metric_history.extend(
                    [
                        {
                            "step": global_step,
                            "key": "validation.loss",
                            "value": checkpoint_loss,
                            "unit": "cross_entropy",
                            "source": "EXPERIMENT",
                        },
                        {
                            "step": global_step,
                            "key": "validation.accuracy",
                            "value": checkpoint_accuracy,
                            "unit": "fraction",
                            "source": "EXPERIMENT",
                        },
                    ]
                )
                history.append(
                    {
                        "step": global_step,
                        "scope": "validation_checkpoint",
                        "loss": checkpoint_loss,
                        "accuracy": checkpoint_accuracy,
                    }
                )
                model.train()

        train_loss = epoch_loss / epoch_count
        train_accuracy = epoch_correct / epoch_count
        validation_loss, validation_accuracy, _ = evaluate(model, validation_loader, criterion)
        epoch_duration = time.perf_counter() - epoch_started
        epoch_step = (epoch + 1) * len(train_loader)
        epoch_metrics = {
            "train.epoch_loss": train_loss,
            "train.epoch_accuracy": train_accuracy,
            "validation.loss": validation_loss,
            "validation.accuracy": validation_accuracy,
            "performance.examples_per_second": epoch_count / epoch_duration,
            "performance.epoch_duration_seconds": epoch_duration,
        }
        epoch_units = {
            "train.epoch_loss": "cross_entropy",
            "train.epoch_accuracy": "fraction",
            "validation.loss": "cross_entropy",
            "validation.accuracy": "fraction",
            "performance.examples_per_second": "examples/second",
            "performance.epoch_duration_seconds": "seconds",
        }
        run.log(
            epoch_metrics,
            step=epoch_step,
            units=epoch_units,
            directions={
                "train.epoch_loss": "LOWER",
                "train.epoch_accuracy": "HIGHER",
                "validation.loss": "LOWER",
                "validation.accuracy": "HIGHER",
                "performance.examples_per_second": "HIGHER",
                "performance.epoch_duration_seconds": "LOWER",
            },
        )
        metric_history.extend(
            {
                "step": epoch_step,
                "key": key,
                "value": float(value),
                "unit": epoch_units[key],
                "source": "EXPERIMENT",
            }
            for key, value in epoch_metrics.items()
        )
        if psutil is not None:
            memory_mb = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
            system_metrics = {
                "system.cpu.percent": psutil.cpu_percent(interval=None),
                "system.memory.used_mb": memory_mb,
            }
            run.log_system(
                system_metrics,
                step=epoch_step,
                units={"system.cpu.percent": "percent", "system.memory.used_mb": "MB"},
            )
            metric_history.extend(
                {
                    "step": epoch_step,
                    "key": key,
                    "value": value,
                    "unit": "percent" if key.endswith("percent") else "MB",
                    "source": "SYSTEM",
                }
                for key, value in system_metrics.items()
            )
        history.extend(
            [
                {
                    "step": epoch_step,
                    "scope": "train",
                    "loss": train_loss,
                    "accuracy": train_accuracy,
                },
                {
                    "step": epoch_step,
                    "scope": "validation",
                    "loss": validation_loss,
                    "accuracy": validation_accuracy,
                },
            ]
        )

    test_loss, test_accuracy, confusion = evaluate(model, test_loader, criterion)
    run_duration = time.perf_counter() - run_started
    final_metrics = {
        "test.loss": test_loss,
        "test.accuracy": test_accuracy,
        "performance.run_duration_seconds": run_duration,
    }
    run.log(
        final_metrics,
        step=total_steps + 1,
        units={
            "test.loss": "cross_entropy",
            "test.accuracy": "fraction",
            "performance.run_duration_seconds": "seconds",
        },
        directions={
            "test.loss": "LOWER",
            "test.accuracy": "HIGHER",
            "performance.run_duration_seconds": "LOWER",
        },
    )
    metric_history.extend(
        [
            {
                "step": total_steps + 1,
                "key": "test.loss",
                "value": test_loss,
                "unit": "cross_entropy",
                "source": "EXPERIMENT",
            },
            {
                "step": total_steps + 1,
                "key": "test.accuracy",
                "value": test_accuracy,
                "unit": "fraction",
                "source": "EXPERIMENT",
            },
            {
                "step": total_steps + 1,
                "key": "performance.run_duration_seconds",
                "value": run_duration,
                "unit": "seconds",
                "source": "EXPERIMENT",
            },
        ]
    )
    history.append(
        {"step": total_steps + 1, "scope": "test", "loss": test_loss, "accuracy": test_accuracy}
    )
    summary = {
        "model": "LeNet-5",
        "seed": settings.seed,
        "split_seed": settings.split_seed,
        "test_accuracy": test_accuracy,
        "test_loss": test_loss,
        "run_duration_seconds": run_duration,
        "validation_accuracy": history[-2]["accuracy"],
        "validation_loss": history[-2]["loss"],
    }
    artifacts = write_outputs(output_dir, settings, metric_history, summary, confusion)
    for artifact in artifacts:
        run.log_artifact(artifact, logical_path="evaluation", tags=["tutorial", "mnist"])
    run.set_summary(
        interpretation=(
            f"LeNet-5 reached {test_accuracy:.2%} accuracy on the official MNIST test set."
        ),
        limitations=(
            "This compact tutorial uses one deterministic split and three epochs. It is not a "
            "robustness, fairness, or production-readiness evaluation."
        ),
    )
    run.heartbeat(
        100,
        note="Training and evaluation completed.",
        current_step=total_steps,
        total_steps=total_steps,
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default="EASY")
    parser.add_argument("--experiment", default="E-1")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--validation-every", type=int, default=50)
    parser.add_argument("--api-url")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--offline", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = Settings(
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_every=args.validation_every,
    )
    context: AbstractContextManager[RunLogger]
    if args.offline:
        context = nullcontext(LocalRun())
    else:
        import researchtab

        context = researchtab.init(
            project=args.project,
            experiment=args.experiment,
            name=f"LeNet seed {settings.seed}",
            config=asdict(settings),
            tags=["tutorial", "mnist", "lenet"],
            seed=settings.seed,
            api_url=args.api_url,
            collect_packages=True,
            dataset_name="MNIST",
            dataset_version=MNIST_VERSION,
            notes="Deterministic class-balanced Research Tab tutorial execution.",
            limitations=(
                "Three-epoch CPU tutorial; no robustness, fairness, or production evaluation."
            ),
        )
    with context as run:
        summary = train(settings, run, args.data_dir, args.output_dir / f"seed-{settings.seed}")
        run.finish(outcome="POSITIVE" if summary["test_accuracy"] >= 0.95 else "NEGATIVE")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()

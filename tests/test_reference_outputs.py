from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path

EXECUTION_COMMIT = "0a93183e5e93adce959839bf02cd1d09128ac14a"
REFERENCE_ROOT = Path(__file__).parents[1] / "reference"
MODELS = ("lenet", "mlp-matched")
SEEDS = (7, 21, 42, 84, 168)


def test_reference_runs_are_finite_successful_and_sanitized() -> None:
    for model in MODELS:
        for seed in SEEDS:
            run_dir = REFERENCE_ROOT / model / f"seed-{seed}"
            summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
            assert summary["model"] == model
            assert summary["seed"] == seed
            assert summary["trainable_parameters"] == 61_706
            assert summary["test_accuracy"] >= 0.95
            assert all(
                math.isfinite(float(summary[key]))
                for key in (
                    "test_accuracy",
                    "test_loss",
                    "validation_accuracy",
                    "validation_loss",
                    "run_duration_seconds",
                )
            )

            with (run_dir / "metrics.csv").open(encoding="utf-8", newline="") as handle:
                metrics = list(csv.DictReader(handle))
            epoch_losses = [
                float(metric["value"])
                for metric in metrics
                if metric["key"] == "train.epoch_loss"
            ]
            validation_accuracy = [
                metric for metric in metrics if metric["key"] == "validation.accuracy"
            ]
            assert len(epoch_losses) == 3
            assert epoch_losses[0] > epoch_losses[1] > epoch_losses[2]
            assert len(validation_accuracy) == 21
            assert all(metric["key"] != "optimizer.learning_rate" for metric in metrics)
            assert {metric["source"] for metric in metrics} == {"EXPERIMENT", "SYSTEM"}

            manifest = json.loads(
                (run_dir / "reproduction_manifest.json").read_text(encoding="utf-8")
            )
            assert manifest["git_commit"] == EXECUTION_COMMIT
            serialized = json.dumps({"summary": summary, "metrics": metrics, "manifest": manifest})
            assert "RESEARCHTAB_API_TOKEN" not in serialized
            assert ":\\" not in serialized
            assert "/Users/" not in serialized


def test_aggregate_summary_matches_the_ten_exact_run_outputs() -> None:
    aggregate = json.loads(
        (REFERENCE_ROOT / "aggregate-summary.json").read_text(encoding="utf-8")
    )
    assert aggregate["executable_commit"] == EXECUTION_COMMIT
    for model in MODELS:
        values = [
            json.loads(
                (REFERENCE_ROOT / model / f"seed-{seed}" / "summary.json").read_text(
                    encoding="utf-8"
                )
            )["test_accuracy"]
            for seed in SEEDS
        ]
        model_summary = aggregate["models"][model]
        assert model_summary["seeds"] == list(SEEDS)
        assert model_summary["n"] == 5
        assert model_summary["test_accuracy_mean"] == statistics.mean(values)
        assert math.isclose(
            model_summary["test_accuracy_sample_sd"], statistics.stdev(values), rel_tol=1e-12
        )

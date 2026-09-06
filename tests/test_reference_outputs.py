from __future__ import annotations

import csv
import json
import math
from pathlib import Path

EXECUTION_COMMIT = "95a33c6d92d5ccdefc1bda07dea10faa03b7c049"
REFERENCE_ROOT = Path(__file__).parents[1] / "reference"


def test_reference_runs_are_finite_successful_and_sanitized() -> None:
    for seed in (7, 21):
        run_dir = REFERENCE_ROOT / f"seed-{seed}"
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
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
            float(metric["value"]) for metric in metrics if metric["key"] == "train.epoch_loss"
        ]
        assert len(epoch_losses) == 3
        assert epoch_losses[0] > epoch_losses[1] > epoch_losses[2]
        assert {metric["source"] for metric in metrics} == {"EXPERIMENT", "SYSTEM"}

        manifest = json.loads((run_dir / "reproduction_manifest.json").read_text(encoding="utf-8"))
        assert manifest["git_commit"] == EXECUTION_COMMIT
        serialized = json.dumps({"summary": summary, "metrics": metrics, "manifest": manifest})
        assert "RESEARCHTAB_API_TOKEN" not in serialized
        assert ":\\" not in serialized
        assert "/Users/" not in serialized

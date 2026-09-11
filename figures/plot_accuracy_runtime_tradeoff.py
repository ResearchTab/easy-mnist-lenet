#!/usr/bin/env python3
"""Plot the Easy Project accuracy/runtime comparison.

The script is deliberately specific to this figure.  The experiments, metrics,
seeds, axes, and styling are fixed so that a Research Desk context either
reproduces the approved result or fails loudly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import statistics
from pathlib import Path
from typing import Any

import matplotlib

# Use the non-interactive backend so the same command works locally and in CI.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# These are the five live CPU runs recorded in Research Desk for both architectures.
EXPECTED_SEEDS = (7, 21, 42, 84, 168)

# Okabe-Ito blue and vermillion remain distinguishable for most forms of
# colour-vision deficiency and also separate well in greyscale.
SERIES = {
    "E-1": {
        "label": "LeNet-5",
        "color": "#0072B2",
        "marker": "o",
    },
    "E-3": {
        "label": "Parameter-matched MLP",
        "color": "#D55E00",
        "marker": "s",
    },
}
ACCURACY_KEY = "test.accuracy"
RUNTIME_KEY = "performance.run_duration_seconds"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", required=True, type=Path)
    parser.add_argument("--pdf", required=True, type=Path)
    parser.add_argument("--png", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    return parser.parse_args()


def fail(message: str) -> None:
    raise ValueError(message)


def sha256(path: Path) -> str:
    """Hash a file without loading a potentially large artifact into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    return {
        "filename": path.name,
        "byteSize": path.stat().st_size,
        "sha256": sha256(path),
    }


def load_snapshot(context_path: Path) -> dict[str, Any]:
    """Accept either a CLI preview envelope or the immutable snapshot itself."""

    with context_path.open("r", encoding="utf-8") as stream:
        context = json.load(stream)
    if isinstance(context, dict) and isinstance(context.get("preview"), dict):
        snapshot = context["preview"].get("snapshot")
    else:
        snapshot = context
    if not isinstance(snapshot, dict):
        fail("Context must contain a preview.snapshot object or be a snapshot object.")
    if snapshot.get("schemaVersion") != 2:
        fail("This figure requires custom-code figure snapshot schemaVersion 2.")
    if not isinstance(snapshot.get("runs"), list):
        fail("Snapshot runs must be an array.")
    return snapshot


def final_metric(run: dict[str, Any], key: str, expected_unit: str) -> float:
    """Return the unique observation at the largest recorded step."""

    metrics = [metric for metric in run.get("metrics", []) if metric.get("key") == key]
    if not metrics:
        fail(f"Run {run.get('runId')} is missing {key}.")
    steps = [metric.get("step") for metric in metrics]
    if any(not isinstance(step, int) for step in steps):
        fail(f"Every {key} observation must have an integer step.")
    # "Final" is defined by the recorded step, not by JSON array order.
    final_step = max(steps)
    final = [metric for metric in metrics if metric.get("step") == final_step]
    if len(final) != 1:
        fail(f"Run {run.get('runId')} must have exactly one final {key} observation.")
    metric = final[0]
    if metric.get("unit") != expected_unit:
        fail(f"{key} must use unit {expected_unit}.")
    value = metric.get("numericValue")
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        fail(f"Final {key} must be finite and numeric.")
    return float(value)


def extract_records(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    """Validate the expected run set and reduce it to values used by the plot."""

    records: list[dict[str, Any]] = []
    seen_run_ids: set[str] = set()
    for run in snapshot["runs"]:
        if not isinstance(run, dict):
            fail("Each snapshot Run must be an object.")
        descriptor = run.get("descriptor")
        if not isinstance(descriptor, dict):
            fail("Every v2 snapshot Run requires a descriptor.")
        experiment_code = descriptor.get("experimentCode")
        seed = descriptor.get("seed")
        status = descriptor.get("status")
        run_id = run.get("runId")
        if experiment_code not in SERIES:
            fail(f"Unexpected experiment code: {experiment_code!r}.")
        if seed not in EXPECTED_SEEDS:
            fail(f"Unexpected seed for {experiment_code}: {seed!r}.")
        if status != "SUCCEEDED":
            fail(f"Run {run_id} is not SUCCEEDED.")
        if not isinstance(run_id, str) or run_id in seen_run_ids:
            fail("Run IDs must be present and unique.")
        seen_run_ids.add(run_id)
        records.append(
            {
                "runId": run_id,
                "experimentCode": experiment_code,
                "seed": seed,
                "accuracyPercent": final_metric(run, ACCURACY_KEY, "fraction") * 100.0,
                "runtimeSeconds": final_metric(run, RUNTIME_KEY, "seconds"),
            }
        )

    # Checking the complete Cartesian product prevents a duplicated Run from
    # silently replacing a missing seed or architecture.
    expected_pairs = {(code, seed) for code in SERIES for seed in EXPECTED_SEEDS}
    actual_pairs = {(record["experimentCode"], record["seed"]) for record in records}
    if actual_pairs != expected_pairs or len(records) != len(expected_pairs):
        fail("Context must contain exactly five unique shared seeds for each architecture.")
    return sorted(records, key=lambda record: (record["seed"], record["experimentCode"]))


def summaries(records: list[dict[str, Any]]) -> dict[str, dict[str, float | int | str]]:
    """Calculate the mean and sample SD shown by the large crossbars."""

    result: dict[str, dict[str, float | int | str]] = {}
    for code, style in SERIES.items():
        group = [record for record in records if record["experimentCode"] == code]
        accuracies = [record["accuracyPercent"] for record in group]
        runtimes = [record["runtimeSeconds"] for record in group]
        result[code] = {
            "label": style["label"],
            "n": len(group),
            "accuracyMeanPercent": statistics.mean(accuracies),
            "accuracySampleSdPercentagePoints": statistics.stdev(accuracies),
            "runtimeMeanSeconds": statistics.mean(runtimes),
            "runtimeSampleSdSeconds": statistics.stdev(runtimes),
        }
    return result


def assert_expected_statistics(summary: dict[str, dict[str, float | int | str]]) -> None:
    """Guard the approved figure against accidental input drift."""

    # Values are kept at full precision.  The tolerance only absorbs harmless
    # floating-point differences; it is much tighter than the plotted precision.
    expected = {
        "E-1": (96.806, 0.26763781496642086, 75.5641363799572, 0.4754235901606291),
        "E-3": (95.756, 0.03361547262793952, 72.86635902002453, 1.0629716152274225),
    }
    fields = (
        "accuracyMeanPercent",
        "accuracySampleSdPercentagePoints",
        "runtimeMeanSeconds",
        "runtimeSampleSdSeconds",
    )
    for code, values in expected.items():
        for field, expected_value in zip(fields, values, strict=True):
            actual = float(summary[code][field])
            if not math.isclose(actual, expected_value, rel_tol=0.0, abs_tol=1e-10):
                fail(f"Unexpected {code} {field}: {actual}.")


def render(records: list[dict[str, Any]], summary: dict[str, dict[str, float | int | str]], pdf_path: Path, png_path: Path) -> None:
    """Render the publication PDF and its matching review PNG."""

    # Keep every visual choice explicit.  Matplotlib defaults can change between
    # releases, whereas this figure should remain byte-stable in a pinned setup.
    matplotlib.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.0,
            "axes.titlesize": 14.0,
            "axes.labelsize": 10.0,
            "axes.edgecolor": "#344054",
            "axes.linewidth": 0.8,
            "xtick.color": "#344054",
            "ytick.color": "#344054",
            "text.color": "#101828",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axis = plt.subplots(figsize=(6.8, 4.6), facecolor="white")
    axis.set_facecolor("#FCFCFD")

    for code, style in SERIES.items():
        group = [record for record in records if record["experimentCode"] == code]

        # Small markers retain every selected run instead of hiding the spread
        # behind the summary marker.
        axis.scatter(
            [record["runtimeSeconds"] for record in group],
            [record["accuracyPercent"] for record in group],
            s=48,
            marker=style["marker"],
            color=style["color"],
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
        item = summary[code]

        # The diamond marks the bivariate mean; bars show one sample SD along
        # each axis.  They are descriptive summaries, not confidence intervals.
        axis.errorbar(
            float(item["runtimeMeanSeconds"]),
            float(item["accuracyMeanPercent"]),
            xerr=float(item["runtimeSampleSdSeconds"]),
            yerr=float(item["accuracySampleSdPercentagePoints"]),
            fmt="D",
            markersize=7.5,
            markerfacecolor=style["color"],
            markeredgecolor="#101828",
            markeredgewidth=0.8,
            ecolor=style["color"],
            elinewidth=1.5,
            capsize=4,
            capthick=1.3,
            zorder=5,
        )

    axis.annotate(
        "LeNet-5 mean",
        (float(summary["E-1"]["runtimeMeanSeconds"]), float(summary["E-1"]["accuracyMeanPercent"])),
        xytext=(-7, 15),
        textcoords="offset points",
        ha="right",
        color=SERIES["E-1"]["color"],
        fontsize=8.2,
        fontweight="bold",
    )
    axis.annotate(
        "Matched MLP mean",
        (float(summary["E-3"]["runtimeMeanSeconds"]), float(summary["E-3"]["accuracyMeanPercent"])),
        xytext=(8, -19),
        textcoords="offset points",
        ha="left",
        color=SERIES["E-3"]["color"],
        fontsize=8.2,
        fontweight="bold",
    )

    # Fixed limits make a regenerated figure directly comparable with the
    # approved preview and avoid autoscaling around an altered input set.
    axis.set_xlim(50.0, 100.0)
    axis.set_ylim(95.5, 97.35)
    axis.set_xticks([50, 60, 70, 80, 90, 100])
    axis.set_yticks([95.5, 96.0, 96.5, 97.0])
    axis.set_xlabel("Recorded run duration (seconds; lower is better)", labelpad=8)
    axis.set_ylabel("Final test accuracy (%; higher is better)", labelpad=8)
    axis.grid(True, color="#E4E7EC", linewidth=0.7)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)

    axis.annotate(
        "Preferred direction",
        xy=(0.035, 0.93),
        xytext=(0.18, 0.81),
        xycoords="axes fraction",
        textcoords="axes fraction",
        arrowprops={"arrowstyle": "->", "color": "#667085", "lw": 1.0},
        color="#667085",
        fontsize=7.7,
        ha="center",
    )

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker=style["marker"],
            linestyle="none",
            markerfacecolor=style["color"],
            markeredgecolor="white",
            markersize=7,
            label=style["label"],
        )
        for style in SERIES.values()
    ]
    legend = axis.legend(
        handles=legend_handles,
        loc="upper right",
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.94,
        borderpad=0.45,
        fontsize=8.4,
    )
    legend.set_zorder(10)

    fig.suptitle(
        "Accuracy-runtime trade-off across matched MNIST runs",
        x=0.13,
        y=0.965,
        ha="left",
        fontweight="bold",
    )
    fig.text(
        0.13,
        0.905,
        "Five user-recorded CPU runs per architecture - matched seeds and training setup",
        ha="left",
        va="top",
        fontsize=8.5,
        color="#475467",
    )
    fig.text(
        0.13,
        0.035,
        "Small markers: runs   |   Large diamonds and crossbars: mean +/- sample SD",
        ha="left",
        va="bottom",
        fontsize=7.3,
        color="#667085",
    )
    fig.subplots_adjust(left=0.13, right=0.97, bottom=0.19, top=0.82)

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    # Dates are intentionally omitted: timestamps would make identical runs
    # produce different PDF hashes.
    pdf_metadata = {
        "Title": "Accuracy-runtime trade-off across matched MNIST runs",
        "Author": "Research Tab",
        "Subject": "Final test accuracy versus recorded CPU run duration",
        "Keywords": "MNIST, accuracy, runtime, LeNet-5, MLP",
        "Creator": Path(__file__).name,
        "Producer": "Matplotlib",
        "CreationDate": None,
        "ModDate": None,
    }
    fig.savefig(pdf_path, format="pdf", metadata=pdf_metadata, facecolor="white")
    fig.savefig(
        png_path,
        format="png",
        dpi=300,
        metadata={"Software": Path(__file__).name},
        facecolor="white",
    )
    plt.close(fig)


def main() -> None:
    args = parse_args()
    snapshot = load_snapshot(args.context)
    records = extract_records(snapshot)
    summary = summaries(records)
    assert_expected_statistics(summary)
    render(records, summary, args.pdf, args.png)

    # Build the QA record after rendering so it can fingerprint both outputs.
    qa = {
        "schemaVersion": 1,
        "sourceSnapshot": {
            "schemaVersion": snapshot["schemaVersion"],
            "contentHash": snapshot.get("contentHash"),
            "runCount": len(records),
            "selectedMetricKeys": [ACCURACY_KEY, RUNTIME_KEY],
        },
        "series": summary,
        "comparison": {
            "accuracyDifferencePercentagePoints": float(summary["E-1"]["accuracyMeanPercent"])
            - float(summary["E-3"]["accuracyMeanPercent"]),
            "runtimeDifferenceSeconds": float(summary["E-1"]["runtimeMeanSeconds"])
            - float(summary["E-3"]["runtimeMeanSeconds"]),
        },
        "environment": {
            "python": platform.python_version(),
            "matplotlib": matplotlib.__version__,
            "backend": matplotlib.get_backend(),
            "font": "DejaVu Sans",
        },
        "files": {
            "script": file_record(Path(__file__).resolve()),
            "context": file_record(args.context.resolve()),
            "pdf": file_record(args.pdf.resolve()),
            "png": file_record(args.png.resolve()),
        },
        "validation": {
            "expectedRunCount": True,
            "expectedSharedSeeds": True,
            "expectedUnits": True,
            "expectedStatistics": True,
            "noExcludedRuns": True,
        },
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    with args.summary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(qa, stream, indent=2, sort_keys=True)
        stream.write("\n")


if __name__ == "__main__":
    main()

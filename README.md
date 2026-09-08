# Easy Project: LeNet on MNIST

This is the executable companion to Research Tab's **Easy Project**. It compares classic LeNet-5
with a parameter-matched multilayer perceptron on a deterministic, class-balanced MNIST split and
sends every complete run to Research Tab with the official `researchtab` Python package.

## What the example records

- sampled training loss and accuracy every 50 optimizer steps;
- validation loss and accuracy every 50 optimizer steps and at each epoch boundary;
- epoch training loss and accuracy;
- final test loss and accuracy;
- throughput and epoch duration, plus opt-in CPU and memory telemetry;
- heartbeats and progress while the model is training;
- the exact configuration, split seed, model seed, package inventory, and Git revision;
- JSON and CSV artifacts for the summary, metric history, class metrics, confusion matrix, and
  reproduction manifest.

Research Tab is the authoritative experiment log. The local files are portable copies, not a
second tracking system.

## Dataset and model

The official 60,000-image MNIST training pool is split deterministically with seed `20260906`:

- 4,000 examples per digit for training: 40,000 total;
- 1,000 examples per digit for validation: 10,000 total;
- 1,000 examples per digit are intentionally unused: 10,000 total;
- the official 10,000-image test set is used without resampling.

LeNet uses `Conv(1,6,5) → AvgPool → Conv(6,16,5) → AvgPool → 120 → 84 → 10`. The comparison
MLP uses `784 → 65 → 76 → 65 → 10` with Tanh activations. Both contain exactly **61,706**
trainable parameters. Both use three epochs, batch size 128, and identical SGD-with-momentum
settings. Learning rate is stored as configuration and is deliberately not logged as a metric.

## Run it

Create a project-scoped experiment-logger token in **API & SDK**, authenticate once, and run:

```bash
researchtab login
python train.py --project EASY --experiment E-1 --model lenet --seed 42
python train.py --project EASY --experiment E-3 --model mlp-matched --seed 42
```

For a self-hosted instance, set `RESEARCHTAB_API_URL` or pass `--api-url`. Tokens are read by the
SDK and are never written to artifacts or logs.

To verify the model without creating a Research Tab run:

```bash
python train.py --offline --model lenet --seed 7
python train.py --offline --model mlp-matched --seed 7
```

`--offline` exists only for development, CI, and validating candidate reference artifacts before
they are curated into the tutorial template. Every run sent to Research Tab is created through
`researchtab.init()`; failures are preserved as failures and are never replaced with synthetic
successful data.

Canonical comparison runs use seeds `7`, `21`, `42`, `84`, and `168` for both architectures and
carry the `aggregate-benchmark-v1` tag. Add `--system-metrics` only when CPU and process-memory
telemetry is appropriate for the machine running the experiment.

Sanitized outputs for the complete matrix are published under [`reference/`](reference/README.md).
[`aggregate-summary.csv`](reference/aggregate-summary.csv) preserves every final run value, while
[`aggregate-summary.json`](reference/aggregate-summary.json) records the exact means and sample
standard deviations used by the Easy Project Paper.

## Interpreting the tutorial

The included reference runs demonstrate Research Tab's workflow. They are training examples, not
scientific evidence supplied by the current user. Create a new run before drawing conclusions for
your own work.

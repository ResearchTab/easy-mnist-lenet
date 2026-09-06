# Easy Project: LeNet on MNIST

This is the executable companion to Research Tab's **Easy Project**. It trains the classic
LeNet-5 architecture on a deterministic, class-balanced MNIST split and sends the complete run to
Research Tab with the official `researchtab` Python package.

## What the example records

- sampled training loss and accuracy every 50 optimizer steps;
- epoch training and validation loss and accuracy;
- final test loss and accuracy;
- learning rate, throughput, epoch duration, CPU usage, and memory usage;
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

LeNet uses `Conv(1,6,5) → AvgPool → Conv(6,16,5) → AvgPool → 120 → 84 → 10`. The default
experiment runs for three epochs with batch size 128 and SGD with momentum.

## Run it

Create a project-scoped experiment-logger token in **API & SDK**, authenticate once, and run:

```bash
researchtab login
python train.py --project EASY --experiment E-1 --seed 42
```

For a self-hosted instance, set `RESEARCHTAB_API_URL` or pass `--api-url`. Tokens are read by the
SDK and are never written to artifacts or logs.

To verify the model without creating a Research Tab run:

```bash
python train.py --offline --seed 7
```

`--offline` exists only for development and CI. Reference and user-visible tutorial executions are
always created through `researchtab.init()`.

## Interpreting the tutorial

The included reference runs demonstrate Research Tab's workflow. They are training examples, not
scientific evidence supplied by the current user. Create a new run before drawing conclusions for
your own work.

# Canonical reference runs

These sanitized artifacts were produced on 2026-09-08 by running `train.py` from clean executable
commit `0a93183e5e93adce959839bf02cd1d09128ac14a` on CPU with seeds 7, 21, 42, 84, and 168 for both
LeNet and the parameter-matched MLP. They contain only reviewable configuration, metric histories,
aggregate evaluation outputs, package versions, and public Git provenance. They contain no token,
credential, hostname, or absolute local path.

All ten runs completed three epochs, contain finite histories, and exceeded 95% accuracy on the
official 10,000-image MNIST test set. `aggregate-summary.csv` and `aggregate-summary.json` report
the exact final values, means, and sample standard deviations; no failed run or synthetic success is
substituted. These are reproducible tutorial references, not scientific evidence produced by the
reader.

The files are organized as `reference/<model>/seed-<seed>/`. The immutable manifests point to the
executable commit above. This summary commit only adds sanitized generated artifacts.
The `reference/seed-7` and `reference/seed-21` directories mirror the corresponding LeNet files for
links created by older Easy Project template versions.

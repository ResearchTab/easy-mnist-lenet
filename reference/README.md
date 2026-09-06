# Canonical reference runs

These sanitized artifacts were produced on 2026-09-06 by running `train.py` from clean commit
`95a33c6d92d5ccdefc1bda07dea10faa03b7c049` on CPU with seeds 7 and 21. They contain only
reviewable configuration, metrics, aggregate evaluation outputs, package versions, and public Git
provenance. They contain no token, credential, hostname, or absolute local path.

Both runs completed three epochs, show declining epoch training loss, and exceeded 95% accuracy on
the official 10,000-image MNIST test set. These are reproducible tutorial references, not scientific
evidence produced by the reader.

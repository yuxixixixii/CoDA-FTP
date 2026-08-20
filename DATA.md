# Data

The strict runner uses a precomputed fused feature CSV containing project and
test metadata, flaky labels, 22 FlakeFlagger expert features, and a serialized
768-dimensional CodeBERT representation:

```text
data/processed_data_with_vocabulary_per_test.csv
```

It contains 21,413 tests from 23 projects and is approximately 268 MB. It is
tracked with Git LFS, not ordinary Git. After cloning the tagged release, run:

```bash
git lfs install
git lfs pull
```

The expected SHA-256 is
`5d31cc607585f34b43ee7eb1c5dac9c5e654f06fd36b7ea19ca7acf24ca0f3b2`.
Retain the file at the path above.

`data/FlakeFlaggerFeaturesTypes.csv` is versioned in the repository and lists
the expert feature columns. The runner consumes the precomputed representation
directly; it does not fetch CodeBERT or re-extract test-method embeddings.

The archive under `revision/results/` permits audit of reported predictions and
selection decisions without redistributing another copy of the input data.

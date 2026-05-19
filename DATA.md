# Data

This artifact contains the precomputed fused dataset used by the released
CoDA-FTP script:

- `data/processed_data_with_vocabulary_per_test.csv`
- `data/FlakeFlaggerFeaturesTypes.csv`

The fused CSV includes project metadata, flaky labels, FlakeFlagger expert
features, and a serialized CodeBERT semantic vector in the
`semantic_representation` column. The released script uses the precomputed
features directly and does not re-extract CodeBERT representations.

## Large file handling

`processed_data_with_vocabulary_per_test.csv` is approximately 268 MB, which
exceeds GitHub's normal 100 MB file limit. The repository is configured to track
this file with Git LFS through `.gitattributes`:

```text
data/processed_data_with_vocabulary_per_test.csv filter=lfs diff=lfs merge=lfs -text
```

Before pushing the repository, install and initialize Git LFS:

```bash
git lfs install
git lfs track "data/processed_data_with_vocabulary_per_test.csv"
git add .gitattributes data/processed_data_with_vocabulary_per_test.csv
```

If Git LFS is not available, do not push the CSV directly. Instead, upload it as
a GitHub Release asset or to an archival service such as Zenodo, and keep the
same relative path after downloading:

```text
data/processed_data_with_vocabulary_per_test.csv
```

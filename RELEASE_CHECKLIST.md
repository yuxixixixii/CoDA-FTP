# Release Checklist

Before uploading this folder to GitHub:

- [x] Confirm the repository license and add a `LICENSE` file.
- [ ] Install Git LFS and track `data/processed_data_with_vocabulary_per_test.csv`.
- [ ] Run `git lfs ls-files` and confirm the fused CSV is listed.
- [ ] Run the Docker reproduction command from `README.md`.
- [ ] Confirm the aggregate output matches:
  - TP = 317
  - FN = 407
  - FP = 138
  - TN = 20,551
  - Precision = 69.7%
  - Recall = 43.8%
  - F1 = 53.8%
  - AUC = 91.6% under pooled ROC-AUC reporting
- [ ] Do not commit `outputs/`, `.venv/`, `__pycache__/`, or local logs.
- [ ] If the data file is not pushed through Git LFS, publish it as a release
      asset and update `DATA.md` and `README.md` with the download URL.

Suggested initial Git commands:

```bash
cd coda_ftp_github
git init
git lfs install
git lfs track "data/processed_data_with_vocabulary_per_test.csv"
git add .
git status
```

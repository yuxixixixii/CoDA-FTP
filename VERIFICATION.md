# Verification for `jss-revision-2026-08-20`

The repository includes an archived, audited output snapshot for the revised
strict source-only evaluation. Verify that snapshot without the large input CSV:

```bash
python -m pip install -r requirements.txt
python revision/scripts/verify_archived_results.py
```

Expected checks:

```text
CoDA-FTP: TP=375, FN=349, FP=228, TN=20461
Flakify (strict): TP=261, FN=463, FP=784, TN=19905
DeepFlaky (strict): TP=419, FN=305, FP=2429, TN=18260
All archived RQ1 checks passed.
```

For a full rerun, install the fused CSV according to [DATA.md](DATA.md) and run
`PYTHON_BIN=python bash scripts/run_main.sh`. The expected primary aggregate is
TP=375, FN=349, FP=228, TN=20,461 (62% precision, 52% recall, 57% F1). Small
floating-point differences can occur across BLAS, XGBoost, and hardware
versions; retain the per-fold source-only decision records when comparing runs.

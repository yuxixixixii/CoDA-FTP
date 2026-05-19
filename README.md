# CoDA-FTP

CoDA-FTP is a conservative domain adaptation pipeline for cross-project flaky
test prediction. For each held-out target project, it selects target-relevant
source projects, aligns the selected source distribution to the target
distribution with CORAL, and trains a regularized XGBoost classifier.

This repository contains a standalone artifact for reproducing the main
CoDA-FTP configuration reported in the paper.

## Repository Layout

```text
.
├── data/
│   ├── FlakeFlaggerFeaturesTypes.csv
│   └── processed_data_with_vocabulary_per_test.csv
├── scripts/
│   └── run_main.sh
├── src/
│   └── coda_ftp.py
├── outputs/
│   └── .gitkeep
├── DATA.md
├── Dockerfile
├── RELEASE_CHECKLIST.md
├── requirements.txt
└── VERIFICATION.md
```

The released script uses precomputed fused features. It does not re-extract
CodeBERT representations.

## Data

The main fused CSV contains:

- project name, test name, and flaky label;
- FlakeFlagger expert features;
- precomputed CodeBERT semantic representation serialized in
  `semantic_representation`.

The fused CSV is about 268 MB and must be uploaded through Git LFS or published
as a release artifact. See `DATA.md`.

## Environment

Python 3.12 is recommended. The verified environment used:

```text
Python 3.12.3
numpy 1.26.4
pandas 2.3.3
scikit-learn 1.6.1
scipy 1.15.3
imbalanced-learn 0.14.1
xgboost 3.2.0
```

Install locally:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run CoDA-FTP

From the repository root:

```bash
bash scripts/run_main.sh
```

Equivalent explicit command:

```bash
python src/coda_ftp.py \
  --data data/processed_data_with_vocabulary_per_test.csv \
  --feature-list data/FlakeFlaggerFeaturesTypes.csv \
  --output-dir outputs/coda_ftp_main \
  --top-k 6 \
  --min-projects 3 \
  --use-coral \
  --coral-reg 0.001 \
  --smote-ratio 0.075 \
  --smote-k-neighbors 3 \
  --threshold 0.65 \
  --n-estimators 100 \
  --scale-pos-weight 3 \
  --max-depth 3 \
  --min-child-weight 10 \
  --gamma 5 \
  --subsample 0.8 \
  --colsample-bytree 0.8 \
  --reg-lambda 10 \
  --random-state 8 \
  --xgb-n-jobs 32 \
  --xgb-tree-method hist
```

The script writes:

- `outputs/coda_ftp_main/prediction_result.csv`
- `outputs/coda_ftp_main/prediction_result_by_project.csv`
- `outputs/coda_ftp_main/prediction_result_per_test.csv`
- `outputs/coda_ftp_main/runtime_seconds.txt`

Expected aggregate result for the bundled data under the verified Linux/Python
3.12 environment:

```text
TP=317, FN=407, FP=138, TN=20551
Precision=69.7%, Recall=43.8%, F1=53.8%, AUC=91.6%
```

Rounded format: Precision 70%, Recall 44%, F1 54%, AUC 92%.

## Docker Reproduction

Docker is recommended when reproducing the released artifact. On Apple Silicon,
keep `--platform linux/amd64` to use the verified x86_64 Linux environment.

```bash
docker build --platform linux/amd64 -t coda-ftp-artifact .

docker run --rm --platform linux/amd64 \
  -v "$PWD/outputs/docker_coda_ftp_main:/artifact/outputs/coda_ftp_main" \
  coda-ftp-artifact
```

The Docker run writes results to `outputs/docker_coda_ftp_main/`.

## Method Summary

For each held-out target project, CoDA-FTP performs:

1. fused representation construction from expert and CodeBERT features;
2. target-aware source selection by standardized centroid distance;
3. CORAL alignment of selected source features to the target feature
   distribution;
4. mild source-side SMOTE for rare flaky tests;
5. regularized XGBoost training on selected aligned sources;
6. fixed-threshold prediction on the target project.

No target-project labels are used for source selection, CORAL alignment, SMOTE,
training, or probability estimation. Target labels are used only after
prediction for evaluation.

## Reproducibility Notes

XGBoost, SMOTE, BLAS/LAPACK, thread scheduling, and platform-level numeric
differences can affect fixed-threshold confusion matrices. Use the pinned
dependency versions and Docker setup for the closest reproduction. See
`VERIFICATION.md` for the verified commands and outputs.

## License

This artifact is released under the Apache License 2.0. See `LICENSE`.

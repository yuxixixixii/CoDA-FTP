# Verification

This verification was run from `coda_ftp_github/` on 2026-05-18.

## Environment

```text
Python 3.12.3
numpy 1.26.4
pandas 2.3.3
scikit-learn 1.6.1
scipy 1.15.3
imbalanced-learn 0.14.1
xgboost 3.2.0
```

## Command

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

## Observed Output

```text
Loaded 21,413 tests, 23 projects, 790 features.
TP=317, FN=407, FP=138, TN=20551
Precision=69.7%, Recall=43.8%, F1=53.8%, AUC=91.6%
Runtime: 56.1s
```

A second independent server rerun with a fresh output directory produced the same aggregate and per-project results:

```text
TP=317, FN=407, FP=138, TN=20551
Precision=69.7%, Recall=43.8%, F1=53.8%, AUC=91.6%
Runtime: 137.7s
```

The script produced:

```text
outputs/coda_ftp_main/prediction_result.csv
outputs/coda_ftp_main/prediction_result_by_project.csv
outputs/coda_ftp_main/prediction_result_per_test.csv
outputs/coda_ftp_main/runtime_seconds.txt
```

The same server environment also reproduced the result with the original project script (`CoDA-FTP.py`) and the same main configuration:

```text
TP=317, FN=407, FP=138, TN=20551
Precision=70%, Recall=44%, F1=54%, AUC=88%
```

The small AUC difference comes from the reporting convention: this standalone release reports pooled ROC-AUC over all target predictions, while the original project script reports the existing aggregate AUC field used by the paper tables.

## Docker Verification

The Docker artifact was built and run locally with the Linux/amd64 platform:

```bash
docker build --platform linux/amd64 -t coda-ftp-artifact .

docker run --rm --platform linux/amd64 \
  -v "$PWD/outputs/docker_coda_ftp_main:/artifact/outputs/coda_ftp_main" \
  coda-ftp-artifact
```

The Docker run produced the same aggregate result:

```text
TP=317, FN=407, FP=138, TN=20551
Precision=69.7%, Recall=43.8%, F1=53.8%, AUC=91.6%
Runtime: 90.3s
```

The resulting image was Linux/amd64:

```text
sha256:937888f5783d0226f2edd0dbbc57cce6a22b986cc6f092f33b2e60573297c7e6 amd64 linux
```

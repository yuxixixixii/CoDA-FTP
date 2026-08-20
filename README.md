# CoDA-FTP replication package

This repository is the replication package for the revised CoDA-FTP manuscript,
*Conservative Domain Adaptation for Cross-Project Flaky-Test Prediction*. The
current revision is deliberately separate from the earlier `CoDA-FTP` release:
it replaces that release's fixed, target-label-informed operating point with
the strict source-only nested evaluation reported in the revision.

**Use the release tag `jss-revision-2026-08-20` when reproducing the revised
manuscript.** The repository history retains the earlier artifact for the
original submission, but its `0.65` fixed-threshold result must not be compared
with, or cited as, a result from the revision.

## What is included

* the executable primary runner in `src/coda_ftp.py`;
* source-only scripts for the primary run, fixed-$k$ sensitivity, component
  diagnostics, TCA, DANN, project-level statistics, and partial-target audit;
* archived primary predictions, fold-specific source-only decisions, strict
  Flakify and DeepFlaky outer-fold metrics, statistical outputs, partial-target
  audit results, and LLM aggregate outputs; and
* a verification program that recomputes the archived RQ1 aggregates and
  checks them against the revised manuscript.

The fused input data are not committed because the CSV is approximately 268 MB.
Obtain it as described in [DATA.md](DATA.md), then place it at
`data/processed_data_with_vocabulary_per_test.csv`.

## Strict evaluation protocol

Each of the 23 projects is held out once. For the held-out target, CoDA-FTP:

1. selects the submitted fixed nearest-six source projects using unlabeled,
   source-fitted z-scored fused features;
2. uses only those source projects in leave-one-source-project-out pseudo-target
   validation to choose one XGBoost configuration and one threshold from
   0.30--0.90; and
3. fits CORAL, SMOTE, and the chosen XGBoost model using the selected sources,
   then reads target labels only for final scoring.

The archived primary result is **TP=375, FN=349, FP=228, TN=20,461**
(62% precision, 52% recall, 57% F1). Full fold-level decisions are in
`revision/results/rq1/coda_ftp/source_only_model_selection.csv`.

## Quick checks

Install the Python dependencies and run the archive check:

```bash
python -m pip install -r requirements.txt
python revision/scripts/verify_archived_results.py
```

It verifies the main CoDA-FTP and strict neural-baseline confusion matrices
against the revision's reported totals. This command is deterministic and does
not need the large fused CSV.

To rerun the strict primary experiment after obtaining the data:

```bash
PYTHON_BIN=python bash scripts/run_main.sh
```

The run is computationally intensive: each outer project performs source-only
pseudo-target validation across five pre-specified XGBoost configurations and
13 candidate thresholds. Outputs are written to `outputs/coda_ftp_primary/`.
The deterministic execution command is also recorded in
`revision/scripts/run_primary_strict_source_only.sh`.

For a containerized run, build without the large data file and mount the
downloaded `data/` directory:

```bash
docker build -t coda-ftp-revision .
docker run --rm \
  -v "$PWD/data:/artifact/data:ro" \
  -v "$PWD/outputs:/artifact/outputs" \
  coda-ftp-revision
```

## Result archive

`revision/results/` is an auditable snapshot of the revision:

| Directory | Contents |
| --- | --- |
| `rq1/coda_ftp/` | primary aggregate, per-project and per-test predictions, and source-only selection records |
| `rq1/baselines/` | strict Flakify/DeepFlaky outer-fold metrics and source-only traditional baseline summaries |
| `statistics/` | exact paired tests and project-cluster bootstrap intervals |
| `rq2/` | the manuscript's strict component/representation and transfer-operator summary tables |
| `rq3/` | partial-target sensitivity audit |
| `llm/` | frozen zero-shot/four-shot aggregate outputs and source-only four-shot examples |

The LLM values are supplementary prompt-only reference results, not matched
model-selection baselines. Their configuration and limitations are documented
in `revision/LLM_PROTOCOL.md`.

## Citations and licenses

The package is distributed under Apache-2.0. Please cite the corresponding
paper and the upstream FlakeFlagger, Flakify, DeepFlaky, CodeBERT, CORAL, TCA,
and DANN work when using their data, concepts, or reproduced architectures.

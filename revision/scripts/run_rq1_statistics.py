#!/usr/bin/env python3
"""Reproducible project-level statistics for RQ1 paired project comparisons.

The unit of analysis is the held-out project. Bootstrap samples resample the
23 project folds as clusters, preserving every test within each sampled fold.
The script reports both strict source-only comparisons and contextual neural
reproduction comparisons; the latter are deliberately labelled as descriptive.
"""
from pathlib import Path

import math

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "result_1/statistical_analysis_source_only_zscore_20260813"
OUT.mkdir(exist_ok=True)
SEED = 20260812
N_BOOT = 10000


def f1(tp, fn, fp):
    denom = 2 * tp + fn + fp
    return 0.0 if denom == 0 else 2 * tp / denom


def rank_biserial(differences):
    """Signed-rank matched-pairs rank-biserial effect size, zeros excluded."""
    values = np.asarray(differences, dtype=float)
    values = values[values != 0]
    if len(values) == 0:
        return 0.0
    ranks = pd.Series(np.abs(values)).rank(method="average").to_numpy()
    positive = ranks[values > 0].sum()
    negative = ranks[values < 0].sum()
    return (positive - negative) / (positive + negative)


def wilcoxon_greater(differences):
    """Exact sign-permutation p value for the Wilcoxon signed-rank statistic.

    Zero differences are excluded. With 17 non-zero project pairs, enumerating
    all $2^{17}$ sign assignments is exact and avoids an asymptotic assumption.
    """
    values = np.asarray(differences, dtype=float)
    values = values[values != 0]
    ranks = pd.Series(np.abs(values)).rank(method="average").to_numpy()
    w_plus = ranks[values > 0].sum()
    n = len(values)
    assignments = np.arange(1 << n, dtype=np.uint32)[:, None]
    null_w_plus = ((assignments >> np.arange(n, dtype=np.uint32)) & 1) @ ranks
    p_greater = np.mean(null_w_plus >= w_plus - 1e-12)
    return w_plus, p_greater


def cluster_bootstrap(frame, left, right, rng):
    """Bootstrap project folds and report macro difference and pooled F1 values."""
    n = len(frame)
    indices = rng.integers(0, n, size=(N_BOOT, n))
    left_counts = frame[[f"{left}_{c}" for c in ("TP", "FN", "FP")]].to_numpy()
    right_counts = frame[[f"{right}_{c}" for c in ("TP", "FN", "FP")]].to_numpy()
    left_f1 = frame[f"{left}_f1"].to_numpy()
    right_f1 = frame[f"{right}_f1"].to_numpy()
    pooled_left = np.array([f1(*left_counts[i].sum(axis=0)) for i in indices])
    pooled_right = np.array([f1(*right_counts[i].sum(axis=0)) for i in indices])
    macro_diff = (left_f1[indices] - right_f1[indices]).mean(axis=1)
    pooled_diff = pooled_left - pooled_right
    q = lambda x: np.quantile(x, [0.025, 0.975])
    return {
        "macro_diff_ci_low": q(macro_diff)[0],
        "macro_diff_ci_high": q(macro_diff)[1],
        "pooled_left_ci_low": q(pooled_left)[0],
        "pooled_left_ci_high": q(pooled_left)[1],
        "pooled_diff_ci_low": q(pooled_diff)[0],
        "pooled_diff_ci_high": q(pooled_diff)[1],
    }


coda = pd.read_csv(
    ROOT / "result_1/coda_ftp_source_only_zscore_20260813/IG_0/prediction_result_by_project.csv"
).set_index("project")[["TP", "FN", "FP"]]
coda.columns = [f"CoDA_{c}" for c in coda.columns]

baselines = pd.read_csv(
    ROOT / "result_1/official_lopo_baselines/flakeflagger_source_only_calibrated/prediction_result_by_project.csv"
)
baselines = baselines[baselines["baseline"].isin(["flakeflagger", "vocabulary_flakeflagger"])]
wide = baselines.pivot(index="project", columns="baseline", values=["TP", "FN", "FP"])
wide.columns = [f"{baseline}_{metric}" for metric, baseline in wide.columns]
wide = wide.rename(columns={
    "flakeflagger_TP": "FF_TP", "flakeflagger_FN": "FF_FN", "flakeflagger_FP": "FF_FP",
    "vocabulary_flakeflagger_TP": "Vocab_TP", "vocabulary_flakeflagger_FN": "Vocab_FN", "vocabulary_flakeflagger_FP": "Vocab_FP",
})

flakify = pd.read_csv(
    ROOT / "result_1/official_lopo_baselines/flakify_queue_b32_e3/combined_project_results.csv"
).set_index("project")[["TP", "FN", "FP"]]
flakify.columns = [f"Flakify_{c}" for c in flakify.columns]

deepflaky = pd.read_csv(
    ROOT / "result_1/classification_result/IG_0.01/prediction_result_by_project.csv"
).set_index("project")[["TP", "FN", "FP"]]
deepflaky.columns = [f"DeepFlaky_{c}" for c in deepflaky.columns]

frame = coda.join(wide).join(flakify).join(deepflaky).sort_index()
if frame.isna().any().any():
    missing = frame.index[frame.isna().any(axis=1)].tolist()
    raise ValueError(f"Missing project-level counts after alignment: {missing}")
for method in ("CoDA", "FF", "Vocab", "Flakify", "DeepFlaky"):
    frame[f"{method}_f1"] = [f1(*row) for row in frame[[f"{method}_{c}" for c in ("TP", "FN", "FP")]].to_numpy()]

rng = np.random.default_rng(SEED)
rows = []
for method, calibration_status in (
        ("FF", "strict source-only"),
        ("Vocab", "strict source-only"),
        ("Flakify", "contextual reproduction"),
        ("DeepFlaky", "contextual reproduction")):
    difference = frame["CoDA_f1"] - frame[f"{method}_f1"]
    w_plus, p_value = wilcoxon_greater(difference)
    boot = cluster_bootstrap(frame, "CoDA", method, rng)
    rows.append({
        "comparison": f"CoDA-FTP vs {method}",
        "calibration_status": calibration_status,
        "n_projects": len(frame),
        "mean_delta_f1": difference.mean(),
        "median_delta_f1": difference.median(),
        "wilcoxon_W_plus": w_plus,
        "wilcoxon_exact_p_one_sided": p_value,
        "rank_biserial": rank_biserial(difference),
        **boot,
    })

summary = pd.DataFrame(rows)
frame.reset_index().to_csv(OUT / "project_level_f1.csv", index=False)
summary.to_csv(OUT / "paired_project_statistics.csv", index=False)

print(summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
print("\nCoDA macro F1:", f"{frame['CoDA_f1'].mean():.6f}")
print("CoDA project-fold bootstrap pooled-F1 95% CI:",
      f"[{summary.iloc[0]['pooled_left_ci_low']:.6f}, {summary.iloc[0]['pooled_left_ci_high']:.6f}]")

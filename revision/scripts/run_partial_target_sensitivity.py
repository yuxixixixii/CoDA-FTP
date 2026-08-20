#!/usr/bin/env python3
"""Strict source-only partial-target-reference sensitivity audit.

For every outer project, a label-blind random subset of target *features* is
available when source projects are selected and when CORAL estimates the target
reference distribution.  The selected sources then undergo the same
leave-one-selected-source-project-out XGBoost/threshold selection as the main
strict protocol.  Target labels are accessed only after all deployment choices
and predictions have been fixed, to score the complete target test suite.

This is a partial-reference availability proxy, not a chronological streaming
experiment: the public dataset does not provide a defensible arrival order for
tests.  The run is resumable at the (fraction, seed, target-project) level.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "experiments/source_only_tuning/CoDA-FTP-source-only-tuning.py"
DATA_PATH = ROOT / "result/processed_data_with_vocabulary_per_test.csv"
FEATURE_PATH = ROOT / "input_data/FlakeFlaggerFeaturesTypes.csv"
THRESHOLDS = [round(value, 2) for value in np.arange(0.30, 0.901, 0.05)]


def load_runner():
    spec = importlib.util.spec_from_file_location("source_only_runner", RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load strict runner from {RUNNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_floats(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def condition_name(fraction: float, seed: int) -> str:
    return f"fraction_{int(round(fraction * 100)):03d}_seed_{seed}"


def condition_seed(project: str, fraction: float, seed: int) -> int:
    # Stable across processes and deliberately independent of labels.
    text = f"{project}|{fraction:.6f}|{seed}"
    return sum((index + 1) * ord(char) for index, char in enumerate(text)) % (2**32)


def safe_source_smote(x_train: np.ndarray, y_train: np.ndarray, random_state: int):
    class_counts = np.bincount(y_train.astype(int), minlength=2)
    desired_minority = int(0.075 * class_counts[0])
    if class_counts[1] <= 3 or desired_minority <= class_counts[1]:
        return x_train, y_train
    try:
        return SMOTE(
            sampling_strategy=0.075,
            k_neighbors=3,
            random_state=random_state,
        ).fit_resample(x_train, y_train)
    except ValueError:
        return x_train, y_train


def fit_outer_model(runner, selected_sources, target_reference, full_target, config, random_state):
    feature_columns = [
        column for column in selected_sources.columns
        if column not in {"flakyStatus", "test_name", "project"}
    ]
    x_train = selected_sources[feature_columns].to_numpy(dtype=np.float64)
    y_train = selected_sources["flakyStatus"].to_numpy(dtype=int)
    x_reference = target_reference[feature_columns].to_numpy(dtype=np.float64)
    x_full_target = full_target[feature_columns].to_numpy(dtype=np.float64)

    # Match the revised main pipeline: fit scaling only on selected sources,
    # then estimate CORAL from the scaled, label-blind reference subset.
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_reference = scaler.transform(x_reference)
    x_full_target = scaler.transform(x_full_target)
    x_train, _ = runner.apply_coral_alignment(x_train, x_reference, reg=1e-3)
    x_train, y_train = safe_source_smote(x_train, y_train, random_state)

    model = runner.build_classifier_model(
        "XGBoost",
        250,
        xgb_n_jobs=1,
        xgb_tree_method="hist",
        xgb_scale_pos_weight=config["xgb_scale_pos_weight"],
        xgb_max_depth=config["xgb_max_depth"],
        xgb_min_child_weight=config["xgb_min_child_weight"],
        xgb_gamma=config["xgb_gamma"],
        xgb_subsample=config["xgb_subsample"],
        xgb_colsample_bytree=config["xgb_colsample_bytree"],
        xgb_reg_lambda=config["xgb_reg_lambda"],
        random_state=random_state,
    ).fit(x_train, y_train)
    return model.predict_proba(x_full_target)[:, 1]


def run_one_condition(data, target_project, fraction, seed, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / f"{target_project}_metrics.json"
    predictions_path = output_dir / f"{target_project}_predictions.csv"
    selections_path = output_dir / f"{target_project}_source_only_model_selection.csv"
    if metrics_path.exists() and predictions_path.exists() and selections_path.exists():
        return {"target_project": target_project, "status": "reused"}

    runner = load_runner()
    full_target = data[data["project"] == target_project].copy()
    source_pool = data[data["project"] != target_project].copy()
    if full_target.empty or source_pool.empty:
        raise RuntimeError(f"Missing outer fold data for {target_project}")

    rng = np.random.default_rng(condition_seed(target_project, fraction, seed))
    reference_count = max(2, int(np.ceil(len(full_target) * fraction)))
    reference_indices = np.sort(
        rng.choice(len(full_target), size=reference_count, replace=False)
    )
    target_reference = full_target.iloc[reference_indices].copy()

    selected_sources, selected_count = runner.select_target_aware_source_projects(
        source_pool,
        target_reference,
        selection_mode="top_k",
        top_k=6,
        top_ratio=0.5,
        min_projects=3,
    )
    selected_projects = sorted(selected_sources["project"].unique().tolist())
    candidates = runner.get_source_only_xgb_candidates("conservative_v1")
    config, selection_rows = runner.calibrate_xgb_threshold_source_only(
        selected_sources,
        candidates,
        THRESHOLDS,
        balance="SMOTE",
        classifier="XGBoost",
        mintree=250,
        xgb_n_jobs=1,
        xgb_tree_method="hist",
        use_coral=True,
        coral_reg=1e-3,
        smote_sampling_strategy=0.075,
        smote_k_neighbors=3,
        feature_scaling="source_zscore",
        random_state=8,
    )
    probabilities = fit_outer_model(
        runner, selected_sources, target_reference, full_target, config, random_state=8
    )
    threshold = float(config["candidate_threshold"])
    y_true = full_target["flakyStatus"].to_numpy(dtype=int)
    y_pred = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    try:
        roc_auc = float(roc_auc_score(y_true, probabilities))
    except ValueError:
        roc_auc = float("nan")
    try:
        pr_auc = float(average_precision_score(y_true, probabilities))
    except ValueError:
        pr_auc = float("nan")

    selection_rows = selection_rows.copy()
    selection_rows["outer_target"] = target_project
    selection_rows["reference_fraction"] = fraction
    selection_rows["reference_seed"] = seed
    selection_rows["reference_tests"] = reference_count
    selection_rows["selected_source_projects"] = ",".join(selected_projects)
    selection_rows.to_csv(selections_path, index=False)

    prediction_df = pd.DataFrame({
        "project": target_project,
        "reference_fraction": fraction,
        "reference_seed": seed,
        "reference_tests": reference_count,
        "test_name": full_target["test_name"].to_numpy(),
        "y_true": y_true,
        "pred_prob": probabilities,
        "pred": y_pred,
    })
    prediction_df.to_csv(predictions_path, index=False)
    metrics = {
        "target_project": target_project,
        "reference_fraction": fraction,
        "reference_seed": seed,
        "reference_tests": reference_count,
        "selected_source_count": int(selected_count),
        "selected_source_projects": selected_projects,
        "selected_xgb_candidate": config["candidate_id"],
        "threshold": threshold,
        "TP": int(tp), "FN": int(fn), "FP": int(fp), "TN": int(tn),
        "precision": precision, "recall": recall, "f1": f1,
        "roc_auc": roc_auc, "pr_auc": pr_auc,
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return {"target_project": target_project, "status": "completed"}


def summarize(output_root: Path):
    rows = []
    for metrics_path in sorted(output_root.glob("fraction_*_seed_*/*_metrics.json")):
        rows.append(json.loads(metrics_path.read_text(encoding="utf-8")))
    if not rows:
        return
    project_metrics = pd.DataFrame(rows)
    project_metrics.to_csv(output_root / "project_metrics.csv", index=False)
    summary_rows = []
    for (fraction, seed), group in project_metrics.groupby(["reference_fraction", "reference_seed"]):
        totals = group[["TP", "FN", "FP", "TN"]].sum()
        precision = totals.TP / (totals.TP + totals.FP) if totals.TP + totals.FP else 0.0
        recall = totals.TP / (totals.TP + totals.FN) if totals.TP + totals.FN else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        summary_rows.append({
            "reference_fraction": fraction, "reference_seed": seed,
            "completed_targets": len(group),
            "TP": int(totals.TP), "FN": int(totals.FN),
            "FP": int(totals.FP), "TN": int(totals.TN),
            "pooled_precision": precision, "pooled_recall": recall, "pooled_f1": f1,
            "mean_project_roc_auc": group.roc_auc.mean(),
            "mean_project_pr_auc": group.pr_auc.mean(),
        })
    pd.DataFrame(summary_rows).sort_values(
        ["reference_fraction", "reference_seed"]
    ).to_csv(output_root / "condition_summary.csv", index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fractions", default="0.25,0.50,0.75,1.00")
    parser.add_argument("--seeds", default="20260819,20260820,20260821")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--projects", default="")
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "result_1/partial_target_strict_source_only_20260819"),
    )
    args = parser.parse_args()
    fractions = parse_floats(args.fractions)
    seeds = parse_ints(args.seeds)
    if not fractions or any(value <= 0 or value > 1 for value in fractions):
        raise ValueError("Fractions must be in (0, 1].")
    if not seeds:
        raise ValueError("At least one label-blind sampling seed is required.")

    runner = load_runner()
    data, _ = runner.build_precomputed_fused_data(DATA_PATH, FEATURE_PATH)
    all_projects = sorted(data["project"].unique().tolist())
    projects = [item.strip() for item in args.projects.split(",") if item.strip()] or all_projects
    unknown = sorted(set(projects) - set(all_projects))
    if unknown:
        raise ValueError(f"Unknown projects: {unknown}")

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    metadata = {
        "protocol": "partial_target_reference_strict_source_only_nested",
        "fractions": fractions,
        "seeds": seeds,
        "fixed_k": 6,
        "feature_scaling": "source_zscore",
        "coral_reg": 0.001,
        "smote_sampling_strategy": 0.075,
        "xgb_candidate_set": "conservative_v1",
        "thresholds": THRESHOLDS,
        "scoring": "complete outer target; outer labels used only after prediction",
    }
    (output_root / "protocol.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    jobs = []
    for fraction in fractions:
        # Full reference is deterministic; do not wastefully repeat it for all seeds.
        active_seeds = seeds[:1] if fraction == 1.0 else seeds
        for seed in active_seeds:
            condition_dir = output_root / condition_name(fraction, seed)
            for project in projects:
                jobs.append((project, fraction, seed, str(condition_dir)))
    print(f"Strict partial-target audit: {len(jobs)} target conditions; workers={args.workers}")

    failures = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(run_one_condition, data, project, fraction, seed, condition_dir)
            for project, fraction, seed, condition_dir in jobs
        ]
        for future in as_completed(futures):
            try:
                result = future.result()
                print(f"{result['status']}: {result['target_project']}", flush=True)
            except Exception as exc:  # preserve completed target checkpoints
                failures.append(str(exc))
                print(f"FAILED: {exc}", flush=True)
    summarize(output_root)
    if failures:
        raise RuntimeError(f"{len(failures)} target conditions failed; rerun safely resumes completed conditions.")


if __name__ == "__main__":
    main()

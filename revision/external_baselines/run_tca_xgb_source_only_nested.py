#!/usr/bin/env python3
"""Strict source-only nested TCA + XGBoost transfer diagnostics.

The script evaluates linear TCA as an MMD-based alternative to CORAL under the
same label-isolation rule as the revised CoDA-FTP primary result.  For every
outer target project, target features are available only for unsupervised TCA;
source-fitted scaling, XGBoost configuration, and decision threshold are
selected exclusively by leave-one-source-project-out pseudo-target validation.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import eigh
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
except Exception as exc:  # pragma: no cover
    raise ImportError("TCA diagnostics require xgboost.") from exc


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ONLY_RUNNER = ROOT / "experiments" / "source_only_tuning" / "CoDA-FTP-source-only-tuning.py"
spec = importlib.util.spec_from_file_location("source_only_runner", SOURCE_ONLY_RUNNER)
if spec is None or spec.loader is None:  # pragma: no cover
    raise ImportError(f"Unable to load strict source-only runner from {SOURCE_ONLY_RUNNER}")
source_only_runner = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = source_only_runner
spec.loader.exec_module(source_only_runner)
SOURCE_ONLY_XGB_CANDIDATE_SETS = source_only_runner.SOURCE_ONLY_XGB_CANDIDATE_SETS
build_precomputed_fused_data = source_only_runner.build_precomputed_fused_data
select_target_aware_source_projects = source_only_runner.select_target_aware_source_projects


def parse_csv_floats(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--precomputed-data-path", default="result/processed_data_with_vocabulary_per_test.csv")
    parser.add_argument("--flakeflagger-features-path", default="input_data/FlakeFlaggerFeaturesTypes.csv")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source-selection", choices=["top_k", "none"], default="top_k")
    parser.add_argument("--source-selection-top-k", type=int, default=6)
    parser.add_argument("--source-selection-min-projects", type=int, default=3)
    parser.add_argument("--tca-dim", type=int, default=128)
    parser.add_argument("--tca-reg", type=float, default=1.0)
    parser.add_argument("--thresholds", default="0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90")
    parser.add_argument("--smote-sampling-strategy", type=float, default=0.075)
    parser.add_argument("--smote-k-neighbors", type=int, default=3)
    parser.add_argument("--xgb-candidate-set", choices=sorted(SOURCE_ONLY_XGB_CANDIDATE_SETS), default="conservative_v1")
    parser.add_argument("--random-state", type=int, default=8)
    parser.add_argument("--xgb-n-jobs", type=int, default=1)
    parser.add_argument("--xgb-tree-method", default="hist")
    return parser.parse_args()


def fit_linear_tca(x_source: np.ndarray, x_target: np.ndarray, dim: int, reg: float) -> tuple[np.ndarray, np.ndarray]:
    """Fit marginal linear TCA without materializing an n-by-n MMD matrix."""
    x_source = np.asarray(x_source, dtype=np.float64)
    x_target = np.asarray(x_target, dtype=np.float64)
    combined = np.vstack([x_source, x_target])
    mean_diff = x_source.mean(axis=0) - x_target.mean(axis=0)
    mmd_scatter = np.outer(mean_diff, mean_diff)
    centered = combined - combined.mean(axis=0, keepdims=True)
    total_scatter = centered.T @ centered
    d = x_source.shape[1]
    left = mmd_scatter + reg * np.eye(d)
    right = total_scatter + 1e-6 * np.eye(d)
    components = max(1, min(dim, d))
    eigvals, eigvecs = eigh(left, right, subset_by_index=[0, components - 1])
    projection = eigvecs[:, np.argsort(eigvals)].astype(np.float32)
    return (x_source @ projection).astype(np.float32), (x_target @ projection).astype(np.float32)


def apply_smote(x: np.ndarray, y: np.ndarray, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    from imblearn.over_sampling import SMOTE

    counts = np.bincount(y.astype(int), minlength=2)
    if counts[0] == 0 or counts[1] == 0 or counts[1] <= args.smote_k_neighbors:
        return x, y
    # imbalanced-learn obtains the desired minority count by truncating the
    # float strategy times the majority count. Match that rule exactly, so a
    # borderline fold is left unchanged instead of asking SMOTE to remove data.
    target_minority_count = int(args.smote_sampling_strategy * counts[0])
    if target_minority_count <= counts[1]:
        return x, y
    return SMOTE(
        sampling_strategy=args.smote_sampling_strategy,
        k_neighbors=args.smote_k_neighbors,
        random_state=args.random_state,
    ).fit_resample(x, y)


def build_model(config: dict[str, float | str], args: argparse.Namespace) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=250,
        eval_metric="logloss",
        n_jobs=args.xgb_n_jobs,
        tree_method=args.xgb_tree_method,
        scale_pos_weight=config["xgb_scale_pos_weight"],
        max_depth=config["xgb_max_depth"],
        min_child_weight=config["xgb_min_child_weight"],
        gamma=config["xgb_gamma"],
        subsample=config["xgb_subsample"],
        colsample_bytree=config["xgb_colsample_bytree"],
        reg_lambda=config["xgb_reg_lambda"],
        random_state=args.random_state,
    )


def fit_predict(train_df: pd.DataFrame, test_df: pd.DataFrame, features: list[str], config: dict, args: argparse.Namespace) -> np.ndarray:
    x_train = train_df[features].to_numpy(dtype=np.float32)
    y_train = train_df["flakyStatus"].to_numpy(dtype=int)
    x_test = test_df[features].to_numpy(dtype=np.float32)
    scaler = StandardScaler().fit(x_train)
    x_train, x_test = scaler.transform(x_train), scaler.transform(x_test)
    x_train, x_test = fit_linear_tca(x_train, x_test, args.tca_dim, args.tca_reg)
    x_train, y_train = apply_smote(x_train, y_train, args)
    return build_model(config, args).fit(x_train, y_train).predict_proba(x_test)[:, 1]


def metric_row(y: np.ndarray, prob: np.ndarray, threshold: float) -> dict[str, float]:
    pred = (prob >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    return {"threshold": threshold, "precision": float(precision), "recall": float(recall), "f1": float(f1)}


def choose_config_and_threshold(selected: pd.DataFrame, features: list[str], args: argparse.Namespace) -> tuple[dict, float, pd.DataFrame]:
    rows: list[dict] = []
    thresholds = parse_csv_floats(args.thresholds)
    projects = sorted(selected["project"].unique())
    for config in SOURCE_ONLY_XGB_CANDIDATE_SETS[args.xgb_candidate_set]:
        all_y: list[int] = []
        all_prob: list[float] = []
        for pseudo_target in projects:
            inner_train = selected[selected["project"] != pseudo_target]
            inner_test = selected[selected["project"] == pseudo_target]
            if inner_train.empty or inner_test.empty:
                continue
            all_y.extend(inner_test["flakyStatus"].astype(int).tolist())
            all_prob.extend(fit_predict(inner_train, inner_test, features, config, args).tolist())
        y = np.asarray(all_y, dtype=int)
        prob = np.asarray(all_prob, dtype=float)
        for threshold in thresholds:
            row = metric_row(y, prob, threshold)
            row.update(config)
            row["candidate_id"] = config["candidate_id"]
            row["validation_projects"] = ",".join(projects)
            row["num_validation_tests"] = len(y)
            rows.append(row)
    candidates = pd.DataFrame(rows)
    best = max(
        rows,
        key=lambda row: (row["f1"], row["precision"], row["threshold"], -row["xgb_max_depth"], -row["xgb_min_child_weight"]),
    )
    config = {key: best[key] for key in SOURCE_ONLY_XGB_CANDIDATE_SETS[args.xgb_candidate_set][0]}
    return config, float(best["threshold"]), candidates


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir) / "IG_0"
    out.mkdir(parents=True, exist_ok=True)
    data, features = build_precomputed_fused_data(args.precomputed_data_path, args.flakeflagger_features_path)
    projects = sorted(data["project"].unique())
    all_predictions: list[dict] = []
    calibration_frames: list[pd.DataFrame] = []
    print(f"Strict source-only TCA: {len(projects)} target projects; source_selection={args.source_selection}")
    for index, target in enumerate(projects, start=1):
        outer_source = data[data["project"] != target].reset_index(drop=True)
        outer_target = data[data["project"] == target].reset_index(drop=True)
        if args.source_selection == "top_k":
            selected, selected_count = select_target_aware_source_projects(
                outer_source, outer_target, selection_mode="top_k", top_k=args.source_selection_top_k,
                min_projects=args.source_selection_min_projects,
            )
        else:
            selected, selected_count = outer_source, outer_source["project"].nunique()
        config, threshold, calibration = choose_config_and_threshold(selected, features, args)
        calibration["outer_target_project"] = target
        calibration_frames.append(calibration)
        prob = fit_predict(selected, outer_target, features, config, args)
        pred = (prob >= threshold).astype(int)
        y = outer_target["flakyStatus"].to_numpy(dtype=int)
        for name, actual, score, prediction in zip(outer_target["test_name"], y, prob, pred):
            all_predictions.append({
                "project": target, "test_name": name, "y_true": int(actual), "pred_prob": float(score), "pred": int(prediction),
                "threshold": threshold, "threshold_policy": "source_only_xgb_cv", "feature_scaling": "source_zscore",
                "source_selection": args.source_selection, "source_selection_num_projects": selected_count,
                "tca_dim": args.tca_dim, "tca_reg": args.tca_reg, **config,
            })
        print(f"[{index}/{len(projects)}] {target}: threshold={threshold:.2f}, config={config['candidate_id']}")
    predictions = pd.DataFrame(all_predictions)
    predictions["Matrix_label"] = np.select(
        [(predictions.y_true == 1) & (predictions.pred == 1), (predictions.y_true == 1) & (predictions.pred == 0), (predictions.y_true == 0) & (predictions.pred == 1)],
        ["TP", "FN", "FP"], default="TN",
    )
    tn, fp, fn, tp = confusion_matrix(predictions.y_true, predictions.pred, labels=[0, 1]).ravel()
    precision, recall, f1, _ = precision_recall_fscore_support(predictions.y_true, predictions.pred, average="binary", zero_division=0)
    auc = roc_auc_score(predictions.y_true, predictions.pred_prob)
    summary = pd.DataFrame([{
        "baseline": "TCA+XGBoost", "protocol": "strict_source_only_nested_LOPO", "feature_scaling": "source_zscore",
        "source_selection": args.source_selection, "source_selection_top_k": args.source_selection_top_k if args.source_selection == "top_k" else "",
        "adaptation": "linear_TCA", "tca_dim": args.tca_dim, "tca_reg": args.tca_reg,
        "threshold_policy": "source_only_xgb_cv", "xgb_candidate_set": args.xgb_candidate_set,
        "TP": tp, "FN": fn, "FP": fp, "TN": tn, "precision": precision, "recall": recall, "f1": f1, "auc": auc,
    }])
    predictions.to_csv(out / "prediction_result_per_test.csv", index=False)
    summary.to_csv(out / "prediction_result.csv", index=False)
    pd.concat(calibration_frames, ignore_index=True).to_csv(out / "source_only_xgb_threshold_cv.csv", index=False)
    per_project = predictions.groupby("project", as_index=False).apply(
        lambda x: pd.Series({
            "TP": int(((x.y_true == 1) & (x.pred == 1)).sum()), "FN": int(((x.y_true == 1) & (x.pred == 0)).sum()),
            "FP": int(((x.y_true == 0) & (x.pred == 1)).sum()), "TN": int(((x.y_true == 0) & (x.pred == 0)).sum()),
        })
    ).reset_index(drop=True)
    per_project.to_csv(out / "prediction_result_by_project.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""CoDA-FTP: Conservative Domain Adaptation for Cross-Project Flaky Test Prediction.

This standalone script reproduces the main CoDA-FTP pipeline:

1. Load precomputed fused test representations.
2. Run leave-one-project-out evaluation.
3. Select target-relevant source projects by centroid distance.
4. Align selected sources to the unlabeled target distribution with CORAL.
5. Train a regularized XGBoost classifier with mild source-side SMOTE.
6. Save aggregate, per-project, and per-test predictions.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix, roc_auc_score
from xgboost import XGBClassifier


METADATA_COLUMNS = ["test_name", "project", "flakyStatus"]


def parse_semantic_representation(value: object) -> np.ndarray:
    """Parse the serialized CodeBERT vector stored in the precomputed CSV."""
    if not isinstance(value, str) or not value.strip():
        return np.array([], dtype=np.float32)
    cleaned = value.replace("[", " ").replace("]", " ").replace("\n", " ")
    return np.fromstring(cleaned, sep=" ", dtype=np.float32)


def load_fused_data(data_path: Path, feature_list_path: Path) -> tuple[pd.DataFrame, list[str]]:
    """Load expert features and semantic vectors into one feature table."""
    raw = pd.read_csv(data_path)
    feature_list = pd.read_csv(feature_list_path)

    missing = [column for column in METADATA_COLUMNS + ["semantic_representation"] if column not in raw.columns]
    if missing:
        raise ValueError(f"Missing required columns in {data_path}: {missing}")
    if "allFeatures" not in feature_list.columns:
        raise ValueError(f"{feature_list_path} must contain an 'allFeatures' column.")

    semantic_vectors = raw["semantic_representation"].apply(parse_semantic_representation)
    vector_length = next((len(vector) for vector in semantic_vectors if len(vector) > 0), 0)
    if vector_length == 0:
        raise ValueError("No valid semantic_representation vector found.")
    semantic_vectors = semantic_vectors.apply(
        lambda vector: vector if len(vector) == vector_length else np.zeros(vector_length, dtype=np.float32)
    )
    semantic_columns = [f"semantic_{index}" for index in range(vector_length)]
    semantic_df = pd.DataFrame(semantic_vectors.tolist(), columns=semantic_columns, index=raw.index)

    expert_columns = [
        column
        for column in feature_list["allFeatures"].dropna().astype(str).unique().tolist()
        if column in raw.columns
    ]
    fused = pd.concat(
        [
            raw[METADATA_COLUMNS + expert_columns].reset_index(drop=True),
            semantic_df.reset_index(drop=True),
        ],
        axis=1,
    )
    fused["flakyStatus"] = fused["flakyStatus"].astype(int)
    # Match the original experimental protocol: rows with missing fused features
    # are removed before leave-one-project-out evaluation.
    fused = fused.dropna().reset_index(drop=True)
    return fused, expert_columns + semantic_columns


def matrix_sqrt(matrix: np.ndarray) -> np.ndarray:
    eigvals, eigvecs = np.linalg.eigh(matrix)
    eigvals = np.clip(eigvals, 1e-12, None)
    return (eigvecs * np.sqrt(eigvals)) @ eigvecs.T


def matrix_inv_sqrt(matrix: np.ndarray) -> np.ndarray:
    eigvals, eigvecs = np.linalg.eigh(matrix)
    eigvals = np.clip(eigvals, 1e-12, None)
    return (eigvecs * (1.0 / np.sqrt(eigvals))) @ eigvecs.T


def coral_align(source: np.ndarray, target: np.ndarray, reg: float) -> tuple[np.ndarray, np.ndarray]:
    """Apply CORAL alignment to source features using target features as reference."""
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if len(source) < 2 or len(target) < 2:
        return source.astype(np.float32), target.astype(np.float32)

    source_mean = source.mean(axis=0, keepdims=True)
    target_mean = target.mean(axis=0, keepdims=True)
    source_centered = source - source_mean
    target_centered = target - target_mean

    dim = source.shape[1]
    source_cov = np.cov(source_centered, rowvar=False) + reg * np.eye(dim)
    target_cov = np.cov(target_centered, rowvar=False) + reg * np.eye(dim)
    transform = matrix_inv_sqrt(source_cov) @ matrix_sqrt(target_cov)
    aligned_source = source_centered @ transform + target_mean
    return aligned_source.astype(np.float32), target.astype(np.float32)


def select_source_projects(
    source_data: pd.DataFrame,
    target_data: pd.DataFrame,
    feature_columns: list[str],
    top_k: int,
    min_projects: int,
) -> tuple[pd.DataFrame, list[tuple[str, float]]]:
    """Select source projects closest to the target centroid in standardized feature space."""
    projects = sorted(source_data["project"].dropna().unique().tolist())
    source_features = source_data[feature_columns].to_numpy(dtype=np.float64)
    target_features = target_data[feature_columns].to_numpy(dtype=np.float64)
    source_mean = source_features.mean(axis=0, keepdims=True)
    source_std = source_features.std(axis=0, keepdims=True) + 1e-8
    target_centroid = ((target_features - source_mean) / source_std).mean(axis=0)

    distances: list[tuple[str, float]] = []
    for project in projects:
        project_features = source_data.loc[source_data["project"] == project, feature_columns].to_numpy(dtype=np.float64)
        project_centroid = ((project_features - source_mean) / source_std).mean(axis=0)
        distances.append((project, float(np.linalg.norm(project_centroid - target_centroid))))

    distances.sort(key=lambda item: item[1])
    num_selected = min(len(projects), max(min_projects, top_k))
    selected_projects = [project for project, _ in distances[:num_selected]]
    return source_data[source_data["project"].isin(selected_projects)].copy(), distances[:num_selected]


def binary_counts(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[int, int, int, int]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return int(tp), int(fn), int(fp), int(tn)


def prf(tp: int, fn: int, fp: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def run_fold(
    data: pd.DataFrame,
    feature_columns: list[str],
    target_project: str,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, dict[str, object]]:
    source_data = data[data["project"] != target_project].copy()
    target_data = data[data["project"] == target_project].copy()
    selected_source, selected_distances = select_source_projects(
        source_data,
        target_data,
        feature_columns,
        top_k=args.top_k,
        min_projects=args.min_projects,
    )
    nearest = ", ".join(f"{project}:{distance:.3f}" for project, distance in selected_distances)
    print(f"[{target_project}] selected {len(selected_distances)} sources: {nearest}", flush=True)

    x_source = selected_source[feature_columns].to_numpy(dtype=np.float32)
    y_source = selected_source["flakyStatus"].to_numpy(dtype=int)
    x_target = target_data[feature_columns].to_numpy(dtype=np.float32)
    y_target = target_data["flakyStatus"].to_numpy(dtype=int)

    if args.use_coral:
        x_source, x_target = coral_align(x_source, x_target, reg=args.coral_reg)

    if args.smote_ratio is not None:
        class_counts = np.bincount(y_source, minlength=2)
        current_ratio = class_counts[1] / class_counts[0] if class_counts[0] else math.inf
        if args.smote_ratio > current_ratio and class_counts[1] > args.smote_k_neighbors:
            smote = SMOTE(
                sampling_strategy=args.smote_ratio,
                k_neighbors=args.smote_k_neighbors,
                random_state=args.random_state,
            )
            x_source, y_source = smote.fit_resample(x_source, y_source)

    model = XGBClassifier(
        n_estimators=args.n_estimators,
        eval_metric="logloss",
        n_jobs=args.xgb_n_jobs,
        tree_method=args.xgb_tree_method,
        scale_pos_weight=args.scale_pos_weight,
        max_depth=args.max_depth,
        min_child_weight=args.min_child_weight,
        gamma=args.gamma,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        reg_lambda=args.reg_lambda,
        random_state=args.random_state,
    )
    model.fit(x_source, y_source)
    pred_prob = model.predict_proba(x_target)[:, 1]
    pred = (pred_prob >= args.threshold).astype(int)
    tp, fn, fp, tn = binary_counts(y_target, pred)
    precision, recall, f1 = prf(tp, fn, fp)
    auc = roc_auc_score(y_target, pred_prob) if len(np.unique(y_target)) == 2 else np.nan

    per_test = pd.DataFrame(
        {
            "project": target_project,
            "test_name": target_data["test_name"].tolist(),
            "y_true": y_target,
            "pred_prob": pred_prob,
            "pred": pred,
        }
    )
    per_test["Matrix_label"] = np.select(
        [
            (per_test["y_true"] == 1) & (per_test["pred"] == 1),
            (per_test["y_true"] == 1) & (per_test["pred"] == 0),
            (per_test["y_true"] == 0) & (per_test["pred"] == 1),
        ],
        ["TP", "FN", "FP"],
        default="TN",
    )
    per_test["threshold"] = args.threshold
    per_test["selected_sources"] = ";".join(project for project, _ in selected_distances)

    project_result = {
        "project": target_project,
        "TP": tp,
        "FN": fn,
        "FP": fp,
        "TN": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "AUC": auc,
        "selected_sources": ";".join(project for project, _ in selected_distances),
    }
    return per_test, project_result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CoDA-FTP under leave-one-project-out evaluation.")
    parser.add_argument("--data", type=Path, default=Path("data/processed_data_with_vocabulary_per_test.csv"))
    parser.add_argument("--feature-list", type=Path, default=Path("data/FlakeFlaggerFeaturesTypes.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/coda_ftp_main"))
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--min-projects", type=int, default=3)
    parser.add_argument("--use-coral", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--coral-reg", type=float, default=1e-3)
    parser.add_argument("--smote-ratio", type=float, default=0.075)
    parser.add_argument("--smote-k-neighbors", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=0.65)
    parser.add_argument("--random-state", type=int, default=8)
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--scale-pos-weight", type=float, default=3.0)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--min-child-weight", type=float, default=10.0)
    parser.add_argument("--gamma", type=float, default=5.0)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample-bytree", type=float, default=0.8)
    parser.add_argument("--reg-lambda", type=float, default=10.0)
    parser.add_argument("--xgb-n-jobs", type=int, default=4)
    parser.add_argument("--xgb-tree-method", default="hist")
    args = parser.parse_args()

    start = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data, feature_columns = load_fused_data(args.data, args.feature_list)
    projects = sorted(data["project"].unique().tolist())
    print(f"Loaded {len(data):,} tests, {len(projects)} projects, {len(feature_columns)} features.")

    per_test_frames: list[pd.DataFrame] = []
    project_rows: list[dict[str, object]] = []
    for project in projects:
        per_test, project_result = run_fold(data, feature_columns, project, args)
        per_test_frames.append(per_test)
        project_rows.append(project_result)

    per_test_df = pd.concat(per_test_frames, ignore_index=True)
    project_df = pd.DataFrame(project_rows)
    tp = int((per_test_df["Matrix_label"] == "TP").sum())
    fn = int((per_test_df["Matrix_label"] == "FN").sum())
    fp = int((per_test_df["Matrix_label"] == "FP").sum())
    tn = int((per_test_df["Matrix_label"] == "TN").sum())
    precision, recall, f1 = prf(tp, fn, fp)
    auc = roc_auc_score(per_test_df["y_true"], per_test_df["pred_prob"])
    aggregate = pd.DataFrame(
        [
            {
                "TP": tp,
                "FN": fn,
                "FP": fp,
                "TN": tn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "AUC": auc,
                "threshold": args.threshold,
                "top_k": args.top_k,
            }
        ]
    )

    per_test_df.to_csv(args.output_dir / "prediction_result_per_test.csv", index=False)
    project_df.to_csv(args.output_dir / "prediction_result_by_project.csv", index=False)
    aggregate.to_csv(args.output_dir / "prediction_result.csv", index=False)

    elapsed = time.perf_counter() - start
    (args.output_dir / "runtime_seconds.txt").write_text(f"{elapsed:.3f}\n", encoding="utf-8")
    print("\n===== CoDA-FTP Summary =====")
    print(f"TP={tp}, FN={fn}, FP={fp}, TN={tn}")
    print(f"Precision={precision * 100:.1f}%, Recall={recall * 100:.1f}%, F1={f1 * 100:.1f}%, AUC={auc * 100:.1f}%")
    print(f"Outputs written to: {args.output_dir}")
    print(f"Runtime: {elapsed:.1f}s")


if __name__ == "__main__":
    main()

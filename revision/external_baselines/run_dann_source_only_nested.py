#!/usr/bin/env python3
"""Strict source-only nested DANN diagnostic for CoDA-FTP.

For each leave-one-project-out fold, this runner preserves the primary
selected-source setting (fixed top-k sources and source-fitted z-scoring), but
replaces CORAL plus XGBoost with a small domain-adversarial neural network
(DANN).  The held-out target feature matrix is used only as unlabeled domain
data.  DANN architecture/training candidates and the decision threshold are
selected by leave-one-selected-source-project-out pseudo-target validation;
outer target labels are read only for final scoring.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score
from sklearn.preprocessing import StandardScaler

try:
    import torch
    from torch import nn
    from torch.autograd import Function
except Exception as exc:  # pragma: no cover
    raise ImportError("DANN requires a PyTorch runtime.") from exc


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ONLY_RUNNER = ROOT / "experiments" / "source_only_tuning" / "CoDA-FTP-source-only-tuning.py"
spec = importlib.util.spec_from_file_location("source_only_runner", SOURCE_ONLY_RUNNER)
if spec is None or spec.loader is None:  # pragma: no cover
    raise ImportError(f"Unable to load source-only runner from {SOURCE_ONLY_RUNNER}")
source_only_runner = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = source_only_runner
spec.loader.exec_module(source_only_runner)
build_precomputed_fused_data = source_only_runner.build_precomputed_fused_data
select_target_aware_source_projects = source_only_runner.select_target_aware_source_projects


@dataclass(frozen=True)
class DannConfig:
    candidate_id: str
    hidden_dim: int
    dropout: float
    max_domain_lambda: float
    epochs: int
    learning_rate: float
    positive_weight: float


# The complete set is deliberately small and pre-specified.  Architecture,
# adversarial strength, epoch budget, and class-loss weight are therefore
# selected within source-only pseudo-target validation, never on outer targets.
DANN_CANDIDATE_SETS: dict[str, list[DannConfig]] = {
    "nested_v1": [
        DannConfig("dann_small_weak", 128, 0.10, 0.10, 30, 1e-3, 3.0),
        DannConfig("dann_medium", 256, 0.10, 0.30, 30, 1e-3, 3.0),
        DannConfig("dann_medium_strong", 256, 0.10, 1.00, 50, 1e-3, 3.0),
    ],
    # A single deterministic setting used solely for a fast smoke test.  It is
    # not a reportable configuration because it does not select architecture.
    "smoke": [DannConfig("dann_smoke", 128, 0.10, 0.10, 3, 1e-3, 3.0)],
}


class GradientReversal(Function):
    @staticmethod
    def forward(ctx, values: torch.Tensor, coefficient: float) -> torch.Tensor:
        ctx.coefficient = coefficient
        return values.view_as(values)

    @staticmethod
    def backward(ctx, gradient: torch.Tensor) -> tuple[torch.Tensor, None]:
        return -ctx.coefficient * gradient, None


class DannNetwork(nn.Module):
    def __init__(self, input_dim: int, config: DannConfig) -> None:
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(input_dim, config.hidden_dim), nn.ReLU(), nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim), nn.ReLU(),
        )
        self.label_head = nn.Linear(config.hidden_dim, 1)
        self.domain_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim), nn.ReLU(), nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 1),
        )

    def label_logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.label_head(self.feature(x)).squeeze(1)

    def domain_logits(self, x: torch.Tensor, coefficient: float) -> torch.Tensor:
        features = GradientReversal.apply(self.feature(x), coefficient)
        return self.domain_head(features).squeeze(1)


def parse_csv_floats(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--precomputed-data-path", default="result/processed_data_with_vocabulary_per_test.csv")
    parser.add_argument("--flakeflagger-features-path", default="input_data/FlakeFlaggerFeaturesTypes.csv")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--targets", default="", help="Optional comma-separated outer target projects.")
    parser.add_argument("--source-selection-top-k", type=int, default=6)
    parser.add_argument("--source-selection-min-projects", type=int, default=3)
    parser.add_argument("--thresholds", default="0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90")
    parser.add_argument("--candidate-set", choices=sorted(DANN_CANDIDATE_SETS), default="nested_v1")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--random-state", type=int, default=20260818)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def cycle_batch(values: torch.Tensor, indices: torch.Tensor, start: int, size: int) -> torch.Tensor:
    count = len(indices)
    take = (torch.arange(size, device=indices.device) + start) % count
    return values[indices[take]]


def fit_dann_predict(
    train_df: pd.DataFrame,
    domain_df: pd.DataFrame,
    features: list[str],
    config: DannConfig,
    args: argparse.Namespace,
    seed: int,
) -> np.ndarray:
    """Train without domain labels and predict probabilities for domain_df."""
    set_seed(seed)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    x_source_raw = train_df[features].to_numpy(dtype=np.float32)
    y_source_raw = train_df["flakyStatus"].to_numpy(dtype=np.float32)
    x_domain_raw = domain_df[features].to_numpy(dtype=np.float32)
    scaler = StandardScaler().fit(x_source_raw)
    x_source = torch.as_tensor(scaler.transform(x_source_raw), dtype=torch.float32, device=device)
    y_source = torch.as_tensor(y_source_raw, dtype=torch.float32, device=device)
    x_domain = torch.as_tensor(scaler.transform(x_domain_raw), dtype=torch.float32, device=device)

    model = DannNetwork(x_source.shape[1], config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
    label_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(config.positive_weight, device=device))
    domain_loss = nn.BCEWithLogitsLoss()
    source_indices = torch.randperm(len(x_source), device=device)
    target_indices = torch.randperm(len(x_domain), device=device)
    steps_per_epoch = max(math.ceil(len(x_source) / args.batch_size), math.ceil(len(x_domain) / args.batch_size))
    total_steps = max(1, config.epochs * steps_per_epoch)
    for epoch in range(config.epochs):
        if epoch:
            source_indices = torch.randperm(len(x_source), device=device)
            target_indices = torch.randperm(len(x_domain), device=device)
        model.train()
        for step in range(steps_per_epoch):
            source_x = cycle_batch(x_source, source_indices, step * args.batch_size, args.batch_size)
            source_y = cycle_batch(y_source.unsqueeze(1), source_indices, step * args.batch_size, args.batch_size).squeeze(1)
            target_x = cycle_batch(x_domain, target_indices, step * args.batch_size, args.batch_size)
            progress = (epoch * steps_per_epoch + step) / total_steps
            coefficient = config.max_domain_lambda * (2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0)
            loss_y = label_loss(model.label_logits(source_x), source_y)
            source_domain = domain_loss(model.domain_logits(source_x, coefficient), torch.zeros(len(source_x), device=device))
            target_domain = domain_loss(model.domain_logits(target_x, coefficient), torch.ones(len(target_x), device=device))
            optimizer.zero_grad(set_to_none=True)
            (loss_y + source_domain + target_domain).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
    model.eval()
    probabilities: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(x_domain), args.batch_size):
            probabilities.append(torch.sigmoid(model.label_logits(x_domain[start:start + args.batch_size])).cpu().numpy())
    return np.concatenate(probabilities)


def metric_row(y: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float]:
    predictions = (probabilities >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(y, predictions, average="binary", zero_division=0)
    return {"threshold": threshold, "precision": float(precision), "recall": float(recall), "f1": float(f1)}


def choose_config_and_threshold(selected: pd.DataFrame, features: list[str], args: argparse.Namespace, outer_index: int) -> tuple[DannConfig, float, pd.DataFrame]:
    rows: list[dict] = []
    thresholds = parse_csv_floats(args.thresholds)
    projects = sorted(selected["project"].unique())
    for config_index, config in enumerate(DANN_CANDIDATE_SETS[args.candidate_set]):
        validation_y: list[int] = []
        validation_probabilities: list[float] = []
        for pseudo_index, pseudo_target in enumerate(projects):
            inner_source = selected[selected["project"] != pseudo_target].reset_index(drop=True)
            inner_target = selected[selected["project"] == pseudo_target].reset_index(drop=True)
            probabilities = fit_dann_predict(
                inner_source, inner_target, features, config, args,
                args.random_state + outer_index * 1000 + config_index * 100 + pseudo_index,
            )
            validation_y.extend(inner_target["flakyStatus"].astype(int).tolist())
            validation_probabilities.extend(probabilities.tolist())
        y = np.asarray(validation_y, dtype=int)
        probabilities = np.asarray(validation_probabilities, dtype=float)
        for threshold in thresholds:
            row = metric_row(y, probabilities, threshold)
            row.update(config.__dict__)
            row["validation_projects"] = ",".join(projects)
            row["num_validation_tests"] = len(y)
            rows.append(row)
    best = max(rows, key=lambda row: (row["f1"], row["precision"], row["threshold"], -row["hidden_dim"], -row["epochs"]))
    config = next(item for item in DANN_CANDIDATE_SETS[args.candidate_set] if item.candidate_id == best["candidate_id"])
    return config, float(best["threshold"]), pd.DataFrame(rows)


def summarize_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    tn, fp, fn, tp = confusion_matrix(predictions.y_true, predictions.pred, labels=[0, 1]).ravel()
    precision, recall, f1, _ = precision_recall_fscore_support(predictions.y_true, predictions.pred, average="binary", zero_division=0)
    return pd.DataFrame([{
        "baseline": "selected-source DANN", "protocol": "strict_source_only_nested_LOPO",
        "feature_scaling": "source_zscore", "source_selection": "top_k", "source_selection_top_k": 6,
        "adaptation": "DANN", "TP": int(tp), "FN": int(fn), "FP": int(fp), "TN": int(tn),
        "precision": float(precision), "recall": float(recall), "f1": float(f1),
        "auc": float(roc_auc_score(predictions.y_true, predictions.pred_prob)),
    }])


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("A CUDA device was requested but PyTorch CUDA is unavailable.")
    out = Path(args.output_dir) / "IG_0"
    out.mkdir(parents=True, exist_ok=True)
    data, features = build_precomputed_fused_data(args.precomputed_data_path, args.flakeflagger_features_path)
    all_projects = sorted(data["project"].unique())
    requested = [item.strip() for item in args.targets.split(",") if item.strip()]
    targets = requested or all_projects
    unknown = sorted(set(targets) - set(all_projects))
    if unknown:
        raise ValueError(f"Unknown target projects: {unknown}")
    prediction_rows: list[dict] = []
    calibration_frames: list[pd.DataFrame] = []
    print(f"Strict source-only DANN: {len(targets)} targets; candidates={args.candidate_set}; device={args.device}", flush=True)
    for outer_index, target in enumerate(targets, start=1):
        outer_source = data[data["project"] != target].reset_index(drop=True)
        outer_target = data[data["project"] == target].reset_index(drop=True)
        selected, selected_count = select_target_aware_source_projects(
            outer_source, outer_target, selection_mode="top_k", top_k=args.source_selection_top_k,
            min_projects=args.source_selection_min_projects,
        )
        config, threshold, calibration = choose_config_and_threshold(selected, features, args, all_projects.index(target) + 1)
        calibration["outer_target_project"] = target
        calibration_frames.append(calibration)
        probabilities = fit_dann_predict(selected, outer_target, features, config, args, args.random_state + (all_projects.index(target) + 1) * 10000)
        y = outer_target["flakyStatus"].to_numpy(dtype=int)
        predicted = (probabilities >= threshold).astype(int)
        for name, actual, probability, decision in zip(outer_target["test_name"], y, probabilities, predicted):
            prediction_rows.append({
                "project": target, "test_name": name, "y_true": int(actual), "pred_prob": float(probability), "pred": int(decision),
                "threshold": threshold, "threshold_policy": "source_only_dann_cv", "feature_scaling": "source_zscore",
                "source_selection": "top_k", "source_selection_num_projects": selected_count, **config.__dict__,
            })
        print(f"[{outer_index}/{len(targets)}] {target}: config={config.candidate_id}, threshold={threshold:.2f}", flush=True)
    predictions = pd.DataFrame(prediction_rows)
    predictions["Matrix_label"] = np.select(
        [(predictions.y_true == 1) & (predictions.pred == 1), (predictions.y_true == 1) & (predictions.pred == 0), (predictions.y_true == 0) & (predictions.pred == 1)],
        ["TP", "FN", "FP"], default="TN",
    )
    predictions.to_csv(out / "prediction_result_per_test.csv", index=False)
    predictions.groupby("project", as_index=False).apply(
        lambda frame: pd.Series({
            "TP": int(((frame.y_true == 1) & (frame.pred == 1)).sum()), "FN": int(((frame.y_true == 1) & (frame.pred == 0)).sum()),
            "FP": int(((frame.y_true == 0) & (frame.pred == 1)).sum()), "TN": int(((frame.y_true == 0) & (frame.pred == 0)).sum()),
        })
    ).reset_index(drop=True).to_csv(out / "prediction_result_by_project.csv", index=False)
    summary = summarize_predictions(predictions)
    summary.to_csv(out / "prediction_result.csv", index=False)
    pd.concat(calibration_frames, ignore_index=True).to_csv(out / "source_only_dann_model_selection.csv", index=False)
    with (out / "run_metadata.json").open("w") as handle:
        json.dump(vars(args), handle, indent=2)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()

import time
import warnings
import numpy as np
import os
from pathlib import Path
import argparse
import itertools
import multiprocessing as mp
import hashlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.ensemble import BalancedRandomForestClassifier, EasyEnsembleClassifier
from sklearn.metrics import roc_curve, auc
from sklearn import svm
import math
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer, AutoModel

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except Exception:
    LGBMClassifier = None

try:
    from catboost import CatBoostClassifier
except Exception:
    CatBoostClassifier = None

try:
    import torch
except Exception:
    torch = None

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

PROJECT_PARALLEL_DATA = None
METHOD_NAME = "CoDA-FTP"

# A compact, pre-specified XGBoost candidate set for the source-only protocol.
# It spans conservative, default, and moderately less-conservative complexity
# while keeping the model family and the 250-tree budget fixed.  The first
# candidate is the configuration reported in the original main experiment.
SOURCE_ONLY_XGB_CANDIDATE_SETS = {
    "conservative_v1": [
        {
            "candidate_id": "reported_regularized",
            "xgb_scale_pos_weight": 3.0,
            "xgb_max_depth": 3,
            "xgb_min_child_weight": 10.0,
            "xgb_gamma": 5.0,
            "xgb_subsample": 0.8,
            "xgb_colsample_bytree": 0.8,
            "xgb_reg_lambda": 10.0,
        },
        {
            "candidate_id": "shallower_conservative",
            "xgb_scale_pos_weight": 2.0,
            "xgb_max_depth": 2,
            "xgb_min_child_weight": 15.0,
            "xgb_gamma": 5.0,
            "xgb_subsample": 0.8,
            "xgb_colsample_bytree": 0.8,
            "xgb_reg_lambda": 10.0,
        },
        {
            "candidate_id": "lower_weight_regularized",
            "xgb_scale_pos_weight": 2.0,
            "xgb_max_depth": 3,
            "xgb_min_child_weight": 10.0,
            "xgb_gamma": 5.0,
            "xgb_subsample": 0.8,
            "xgb_colsample_bytree": 0.8,
            "xgb_reg_lambda": 10.0,
        },
        {
            "candidate_id": "moderate_complexity",
            "xgb_scale_pos_weight": 3.0,
            "xgb_max_depth": 3,
            "xgb_min_child_weight": 5.0,
            "xgb_gamma": 2.0,
            "xgb_subsample": 0.8,
            "xgb_colsample_bytree": 0.8,
            "xgb_reg_lambda": 5.0,
        },
        {
            "candidate_id": "deeper_regularized",
            "xgb_scale_pos_weight": 3.0,
            "xgb_max_depth": 4,
            "xgb_min_child_weight": 5.0,
            "xgb_gamma": 1.0,
            "xgb_subsample": 0.8,
            "xgb_colsample_bytree": 0.8,
            "xgb_reg_lambda": 5.0,
        },
    ],
    # Used only for the component ablation that removes the conservative
    # regularization package.  Its candidates retain the model family and
    # 250-tree budget while setting gamma and L2 regularization to zero and
    # relaxing the depth/min-child controls.  The source-only procedure still
    # selects the candidate and threshold without outer-target labels.
    "unregularized_v1": [
        {
            "candidate_id": "xgb_default_like",
            "xgb_scale_pos_weight": 3.0,
            "xgb_max_depth": 6,
            "xgb_min_child_weight": 1.0,
            "xgb_gamma": 0.0,
            "xgb_subsample": 1.0,
            "xgb_colsample_bytree": 1.0,
            "xgb_reg_lambda": 0.0,
        },
        {
            "candidate_id": "shallow_unregularized",
            "xgb_scale_pos_weight": 3.0,
            "xgb_max_depth": 3,
            "xgb_min_child_weight": 1.0,
            "xgb_gamma": 0.0,
            "xgb_subsample": 1.0,
            "xgb_colsample_bytree": 1.0,
            "xgb_reg_lambda": 0.0,
        },
        {
            "candidate_id": "lower_weight_unregularized",
            "xgb_scale_pos_weight": 1.0,
            "xgb_max_depth": 6,
            "xgb_min_child_weight": 1.0,
            "xgb_gamma": 0.0,
            "xgb_subsample": 1.0,
            "xgb_colsample_bytree": 1.0,
            "xgb_reg_lambda": 0.0,
        },
        {
            "candidate_id": "subsampled_unregularized",
            "xgb_scale_pos_weight": 3.0,
            "xgb_max_depth": 6,
            "xgb_min_child_weight": 1.0,
            "xgb_gamma": 0.0,
            "xgb_subsample": 0.8,
            "xgb_colsample_bytree": 0.8,
            "xgb_reg_lambda": 0.0,
        },
        {
            "candidate_id": "deep_unregularized",
            "xgb_scale_pos_weight": 3.0,
            "xgb_max_depth": 8,
            "xgb_min_child_weight": 1.0,
            "xgb_gamma": 0.0,
            "xgb_subsample": 1.0,
            "xgb_colsample_bytree": 1.0,
            "xgb_reg_lambda": 0.0,
        },
    ],
}


def init_project_parallel_worker(data):
    global PROJECT_PARALLEL_DATA
    PROJECT_PARALLEL_DATA = data


def parse_gpu_ids(gpu_ids):
    if gpu_ids is None or str(gpu_ids).strip() == "":
        return []
    return [int(item.strip()) for item in str(gpu_ids).split(",") if item.strip() != ""]


def parse_csv_strings(value):
    return [item.strip() for item in str(value).split(",") if item.strip() != ""]


def parse_csv_floats(value):
    return [float(item.strip()) for item in str(value).split(",") if item.strip() != ""]


def parse_csv_ints(value):
    return [int(item.strip()) for item in str(value).split(",") if item.strip() != ""]


def get_source_only_xgb_candidates(candidate_set):
    try:
        return [dict(candidate) for candidate in SOURCE_ONLY_XGB_CANDIDATE_SETS[candidate_set]]
    except KeyError as exc:
        available = ", ".join(sorted(SOURCE_ONLY_XGB_CANDIDATE_SETS))
        raise ValueError(
            f"Unknown source-only XGBoost candidate set '{candidate_set}'. "
            f"Available sets: {available}."
        ) from exc


def normalize_classifier_name(classifier):
    normalized = str(classifier).strip().lower().replace("-", "_")
    aliases = {
        "xgb": "XGBoost",
        "xgboost": "XGBoost",
        "lgbm": "LightGBM",
        "lightgbm": "LightGBM",
        "cat": "CatBoost",
        "catboost": "CatBoost",
        "brf": "BalancedRF",
        "balanced_rf": "BalancedRF",
        "balancedrandomforest": "BalancedRF",
        "balanced_random_forest": "BalancedRF",
        "easyensemble": "EasyEnsemble",
        "easy_ensemble": "EasyEnsemble",
        "lr": "LogReg",
        "logreg": "LogReg",
        "logistic": "LogReg",
        "logistic_regression": "LogReg",
        "linear_svm": "LinearSVM",
        "linearsvm": "LinearSVM",
        "linsvm": "LinearSVM",
        "svm_linear": "LinearSVM",
        "rf": "RF",
        "random_forest": "RF",
        "svm": "SVM",
        "dt": "DT",
        "mlp": "MLP",
        "ada": "Ada",
        "nb": "NB",
        "knn": "KNN",
    }
    return aliases.get(normalized, str(classifier).strip())


def parse_csv_classifiers(value):
    return [normalize_classifier_name(item) for item in parse_csv_strings(value)]


def normalize_random_state(value):
    return None if value is not None and int(value) < 0 else int(value)


def parse_optional_csv_floats(value):
    if value is None or str(value).strip() == "":
        return [None]
    return [None if item.strip().lower() in ("none", "auto") else float(item.strip())
            for item in str(value).split(",") if item.strip() != ""]


def compute_binary_metrics(y_true, y_prob, threshold):
    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    _, f1, precision, recall = get_scores(int(tn), int(fp), int(fn), int(tp))
    return {
        "threshold": float(threshold),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def choose_threshold_from_metrics(metric_rows):
    if not metric_rows:
        raise ValueError("No source-only threshold metrics were computed.")
    return max(
        metric_rows,
        key=lambda row: (
            row["f1"],
            row["precision"],
            row["threshold"],
        ),
    )


def resolve_model_path(model_path):
    path_text = str(model_path).strip()
    local_path = Path(path_text).expanduser()
    if local_path.exists():
        return str(local_path.resolve())

    looks_like_path = (
        path_text.startswith(".")
        or path_text.startswith("/")
        or path_text.startswith("~")
    )
    if looks_like_path:
        candidates = [
            str(Path.cwd() / path_text),
            str(Path.cwd() / "codebert-base"),
            str(Path.cwd().parent / "codebert-base"),
            "/home/zhangyu/AdvDeepFlaky/codebert-base",
            "/home/zhangyu/AdvDeepFlaky/DeepFlaky-main/codebert-base",
        ]
        existing_candidates = [candidate for candidate in candidates if Path(candidate).exists()]
        hint = (
            f"Existing candidate(s): {existing_candidates}"
            if existing_candidates
            else "No common local CodeBERT directory was found."
        )
        raise FileNotFoundError(
            f"CodeBERT model path does not exist: {model_path}. "
            f"{hint} Pass a valid local directory or use a HuggingFace model id "
            "such as microsoft/codebert-base."
        )

    return path_text



def extract_codebert_features(
        text_data,
        model_path,
        batch_size=64,
        max_length=510,
        use_fp16=False,
        cache_dir=None,
        local_files_only=False):
    if torch is None or not hasattr(torch, "device") or not hasattr(torch, "no_grad"):
        raise RuntimeError("Fresh CodeBERT extraction requires a working PyTorch installation.")
    device = torch.device("cuda" if hasattr(torch, "cuda") and torch.cuda.is_available() else "cpu")
    model_path = resolve_model_path(model_path)
    print(f"Loading CodeBERT model from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )
    model = AutoModel.from_pretrained(
        model_path,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    ).to(device)
    model.eval()

    all_features = []
    texts = text_data.fillna("").astype(str).tolist()

    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start:start + batch_size]
            tokens = tokenizer(
                batch_texts,
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            input_ids = tokens["input_ids"].to(device)
            attention_mask = tokens["attention_mask"].to(device)

            if use_fp16 and device.type == "cuda":
                with torch.cuda.amp.autocast():
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            cls_features = outputs.last_hidden_state[:, 0, :].detach().float().cpu().numpy()
            all_features.append(cls_features)

            if (start // batch_size + 1) % 20 == 0 or start + batch_size >= len(texts):
                print(f"CodeBERT extraction: {min(start + batch_size, len(texts))}/{len(texts)}")

    del model
    if hasattr(torch, "cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()

    return np.concatenate(all_features, axis=0)


def build_fresh_codebert_semantic_data(
        dataset_path,
        model_path,
        batch_size=64,
        max_length=510,
        use_fp16=False,
        cache_dir=None,
        local_files_only=False):
    raw_data = pd.read_csv(dataset_path)
    required_columns = ["project", "class_name", "test_name", "flaky", "final_code"]
    missing_columns = [col for col in required_columns if col not in raw_data.columns]
    if missing_columns:
        raise ValueError(f"Missing required CodeBERT dataset columns: {missing_columns}")

    raw_data = raw_data.dropna(subset=["project", "class_name", "test_name", "flaky", "final_code"]).reset_index(drop=True)
    raw_data["test_name"] = (
        raw_data["project"].astype(str).str.lower()
        + "."
        + raw_data["class_name"].astype(str).str.lower()
        + "."
        + raw_data["test_name"].astype(str).str.lower()
        + ".row"
        + raw_data.index.astype(str)
    )

    features = extract_codebert_features(
        raw_data["final_code"],
        model_path=model_path,
        batch_size=batch_size,
        max_length=max_length,
        use_fp16=use_fp16,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )
    semantic_columns = [f"semantic_{i}" for i in range(features.shape[1])]
    semantic_df = pd.DataFrame(features, columns=semantic_columns, index=raw_data.index)

    semantic_processed_data = pd.concat(
        [
            raw_data[["test_name", "project"]].reset_index(drop=True),
            raw_data["flaky"].astype(int).rename("flakyStatus").reset_index(drop=True),
            semantic_df.reset_index(drop=True),
        ],
        axis=1,
    )
    return semantic_processed_data, semantic_columns


def parse_semantic_representation(semantic_str):
    try:
        if isinstance(semantic_str, str) and semantic_str.strip():
            semantic_str = semantic_str.replace("[", " ").replace("]", " ").replace("\n", " ")
            return np.fromstring(semantic_str, sep=" ")
        return np.array([], dtype=float)
    except Exception as e:
        print(f"Error parsing semantic representation: {e}")
        return np.array([], dtype=float)


def build_precomputed_fused_data(processed_data_path, flakeflagger_features_path):
    main_data = pd.read_csv(processed_data_path)
    flakeflagger_features = pd.read_csv(flakeflagger_features_path)

    semantic_vectors = main_data["semantic_representation"].apply(parse_semantic_representation)
    vector_length = 0
    for vec in semantic_vectors:
        if len(vec) > 0:
            vector_length = len(vec)
            break
    if vector_length == 0:
        raise ValueError("No valid semantic_representation vector found in precomputed data.")

    semantic_vectors = semantic_vectors.apply(
        lambda vec: vec if len(vec) == vector_length else np.zeros(vector_length, dtype=float)
    )
    semantic_columns = [f"semantic_{i}" for i in range(vector_length)]
    semantic_df = pd.DataFrame(semantic_vectors.tolist(), columns=semantic_columns, index=main_data.index).fillna(0)

    expert_columns = [
        col for col in flakeflagger_features.allFeatures.dropna().astype(str).unique().tolist()
        if col in main_data.columns
    ]
    metadata_columns = ["test_name", "project", "flakyStatus"]
    fused_data = pd.concat(
        [
            main_data[metadata_columns + expert_columns].reset_index(drop=True),
            semantic_df.reset_index(drop=True),
        ],
        axis=1,
    )
    feature_columns = expert_columns + semantic_columns
    return fused_data, feature_columns


def _matrix_sqrt(matrix):
    eigvals, eigvecs = np.linalg.eigh(matrix)
    eigvals = np.clip(eigvals, 1e-12, None)
    return (eigvecs * np.sqrt(eigvals)) @ eigvecs.T


def _matrix_inv_sqrt(matrix):
    eigvals, eigvecs = np.linalg.eigh(matrix)
    eigvals = np.clip(eigvals, 1e-12, None)
    return (eigvecs * (1.0 / np.sqrt(eigvals))) @ eigvecs.T


def apply_coral_alignment(X_source, X_target, reg=1e-3):
    X_source = np.asarray(X_source, dtype=np.float64)
    X_target = np.asarray(X_target, dtype=np.float64)
    if len(X_source) < 2 or len(X_target) < 2:
        return X_source.astype(np.float32), X_target.astype(np.float32)

    source_mean = X_source.mean(axis=0, keepdims=True)
    target_mean = X_target.mean(axis=0, keepdims=True)
    source_centered = X_source - source_mean
    target_centered = X_target - target_mean

    dim = X_source.shape[1]
    source_cov = np.cov(source_centered, rowvar=False) + reg * np.eye(dim)
    target_cov = np.cov(target_centered, rowvar=False) + reg * np.eye(dim)

    transform = _matrix_inv_sqrt(source_cov) @ _matrix_sqrt(target_cov)
    aligned_source = source_centered @ transform + target_mean
    return aligned_source.astype(np.float32), X_target.astype(np.float32)


def select_target_aware_source_projects(
        train_data,
        target_data,
        selection_mode="none",
        top_k=10,
        top_ratio=0.5,
        min_projects=3):
    source_projects = sorted(train_data["project"].dropna().unique().tolist())
    if selection_mode == "none" or len(source_projects) == 0:
        return train_data, len(source_projects)

    if selection_mode == "random_k":
        num_selected = max(int(min_projects), int(top_k))
        num_selected = min(len(source_projects), num_selected)
        target_values = target_data["project"].dropna().astype(str).unique().tolist()
        target_project = target_values[0] if target_values else "unknown-target"
        seed_text = f"{os.environ.get('CODA_RANDOM6_REPEAT', '0')}::{target_project}"
        seed = int(hashlib.md5(seed_text.encode("utf-8")).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        selected_projects = sorted(rng.choice(source_projects, size=num_selected, replace=False).tolist())
        print(
            f"Random source selection: mode=random_k, selected={num_selected}/{len(source_projects)}, "
            f"repeat={os.environ.get('CODA_RANDOM6_REPEAT', '0')}, target={target_project}, "
            f"projects={selected_projects}"
        )
        return train_data[train_data["project"].isin(selected_projects)].copy(), num_selected

    feature_columns = [
        col for col in train_data.columns
        if col not in ["flakyStatus", "test_name", "project"]
    ]
    if len(feature_columns) == 0:
        return train_data, len(source_projects)

    source_features = train_data[feature_columns].to_numpy(dtype=np.float64)
    target_features = target_data[feature_columns].to_numpy(dtype=np.float64)
    source_mean = source_features.mean(axis=0, keepdims=True)
    source_std = source_features.std(axis=0, keepdims=True) + 1e-8
    target_centroid = ((target_features - source_mean) / source_std).mean(axis=0)

    project_distances = []
    for project in source_projects:
        project_features = train_data.loc[
            train_data["project"] == project,
            feature_columns,
        ].to_numpy(dtype=np.float64)
        project_centroid = ((project_features - source_mean) / source_std).mean(axis=0)
        distance = float(np.linalg.norm(project_centroid - target_centroid))
        project_distances.append((project, distance))
    project_distances.sort(key=lambda item: item[1])

    if selection_mode == "top_k":
        num_selected = int(top_k)
    elif selection_mode == "top_ratio":
        num_selected = int(math.ceil(len(source_projects) * float(top_ratio)))
    else:
        raise ValueError(f"Unsupported source selection mode: {selection_mode}")
    num_selected = max(int(min_projects), num_selected)
    num_selected = min(len(source_projects), num_selected)

    selected_projects = [project for project, _ in project_distances[:num_selected]]
    selected_distances = ", ".join(
        f"{project}:{distance:.3f}"
        for project, distance in project_distances[:min(num_selected, 8)]
    )
    print(
        f"Target-aware source selection: mode={selection_mode}, "
        f"selected={num_selected}/{len(source_projects)}, "
        f"nearest=[{selected_distances}]"
    )
    return train_data[train_data["project"].isin(selected_projects)].copy(), num_selected


# %%
def get_scores(tn, fp, fn, tp):
    if (tp == 0):
        accuracy = (tp + tn) / (tn + fp + fn + tp)
        Precision = 0
        Recall = 0
        F1 = 0
    else:
        accuracy = (tp + tn) / (tn + fp + fn + tp)
        Precision = tp / (tp + fp)
        Recall = tp / (tp + fn)
        F1 = 2 * ((Precision * Recall) / (Precision + Recall))
    return accuracy, F1, Precision, Recall


# %%
def generateConfusionMatrixByProject(data, processed_data=None):
    filter_columns = ['cross_validation', 'balance_type', 'IG_min', 'numTrees', 'classifier', 'features_structure']
    for optional_column in [
        "threshold",
        "threshold_policy",
        "feature_scaling",
        "xgb_scale_pos_weight",
        "xgb_max_depth",
        "xgb_min_child_weight",
        "xgb_gamma",
        "xgb_subsample",
        "xgb_colsample_bytree",
        "xgb_reg_lambda",
        "source_selection",
        "source_selection_top_k",
        "source_selection_top_ratio",
        "source_selection_min_projects",
        "source_selection_num_projects",
        "smote_sampling_strategy",
        "smote_k_neighbors",
        "random_state",
    ]:
        if optional_column in data.columns:
            filter_columns.append(optional_column)

    filter_data = data[filter_columns]
    filter_data = filter_data.drop_duplicates()
    df_columns = filter_columns + ["project", "TP", "FN", "FP", "TN", "Precision", "Recall", "F1"]
    result = pd.DataFrame(columns=df_columns)

    if "project" in data.columns and data["project"].notna().any():
        updated_data = data.copy()
    else:
        data_with_project_name = processed_data[['project', 'test_name']]
        updated_data = pd.merge(data, data_with_project_name, on='test_name', how='left')

    for index, row in filter_data.iterrows():
        data_per_result = updated_data.copy()
        for filter_column in filter_columns:
            data_per_result = data_per_result[data_per_result[filter_column] == row[filter_column]]

        for proj in data_per_result.project.unique():
            specific_project = data_per_result[data_per_result["project"] == proj]
            TP = len(specific_project[specific_project["Matrix_label"] == "TP"])
            FN = len(specific_project[specific_project["Matrix_label"] == "FN"])
            FP = len(specific_project[specific_project["Matrix_label"] == "FP"])
            TN = len(specific_project[specific_project["Matrix_label"] == "TN"])
            accuracy, F1, Precision, Recall = get_scores(TN, FP, FN, TP)
            new_row = pd.Series(
                [row[col] for col in filter_columns]
                + [proj, TP, FN, FP, TN, str(round(((Precision) * 100))) + "%",
                 str(round(((Recall) * 100))) + "%", str(round(((F1) * 100))) + "%"], index=result.columns)
            result = pd.concat([result, new_row.to_frame().T], ignore_index=True)

    return result


def build_classifier_model(classifier, mintree, xgb_n_jobs=1, xgb_tree_method=None,
                           xgb_scale_pos_weight=None, xgb_max_depth=None,
                           xgb_min_child_weight=None, xgb_gamma=None,
                           xgb_subsample=None, xgb_colsample_bytree=None,
                           xgb_reg_lambda=None, random_state=0):
    positive_class_weight = (
        None if xgb_scale_pos_weight is None else float(xgb_scale_pos_weight)
    )
    sklearn_class_weight = (
        None if positive_class_weight is None else {0: 1.0, 1: positive_class_weight}
    )
    if (classifier == 'DT'):
        return DecisionTreeClassifier(
            criterion='entropy',
            max_depth=xgb_max_depth,
            class_weight=sklearn_class_weight,
            random_state=random_state,
        )
    if (classifier == 'RF'):
        return RandomForestClassifier(
            criterion="entropy",
            n_estimators=mintree,
            n_jobs=xgb_n_jobs,
            max_depth=xgb_max_depth,
            min_samples_leaf=int(xgb_min_child_weight) if xgb_min_child_weight is not None else 1,
            max_features=xgb_colsample_bytree,
            class_weight=sklearn_class_weight,
            random_state=random_state,
        )
    if (classifier == 'MLP'):
        return MLPClassifier(hidden_layer_sizes=(13, 13, 13), max_iter=500, random_state=random_state)
    if (classifier == 'SVM'):
        return make_pipeline(
            StandardScaler(),
            svm.SVC(
                gamma='scale',
                probability=True,
                class_weight=sklearn_class_weight,
                random_state=random_state,
            ),
        )
    if (classifier == 'Ada'):
        return AdaBoostClassifier(n_estimators=100, random_state=random_state)
    if (classifier == 'NB'):
        return GaussianNB()
    if (classifier == 'KNN'):
        return KNeighborsClassifier(n_neighbors=7)
    if (classifier == 'XGBoost'):
        if XGBClassifier is None:
            raise ImportError("XGBoost is not installed. Install xgboost or remove XGBoost from --classifiers.")
        xgb_kwargs = {"eval_metric": "logloss", "n_jobs": xgb_n_jobs, "random_state": random_state}
        if xgb_tree_method:
            xgb_kwargs["tree_method"] = xgb_tree_method
        if xgb_scale_pos_weight is not None:
            xgb_kwargs["scale_pos_weight"] = xgb_scale_pos_weight
        if xgb_max_depth is not None:
            xgb_kwargs["max_depth"] = xgb_max_depth
        if xgb_min_child_weight is not None:
            xgb_kwargs["min_child_weight"] = xgb_min_child_weight
        if xgb_gamma is not None:
            xgb_kwargs["gamma"] = xgb_gamma
        if xgb_subsample is not None:
            xgb_kwargs["subsample"] = xgb_subsample
        if xgb_colsample_bytree is not None:
            xgb_kwargs["colsample_bytree"] = xgb_colsample_bytree
        if xgb_reg_lambda is not None:
            xgb_kwargs["reg_lambda"] = xgb_reg_lambda
        return XGBClassifier(**xgb_kwargs)
    if (classifier == 'LightGBM'):
        if LGBMClassifier is None:
            raise ImportError("LightGBM is not installed. Install lightgbm or remove LightGBM from --classifiers.")
        lgbm_kwargs = {
            "objective": "binary",
            "n_estimators": mintree,
            "learning_rate": 0.05,
            "n_jobs": xgb_n_jobs,
            "random_state": random_state,
            "verbose": -1,
        }
        if xgb_max_depth is not None:
            lgbm_kwargs["max_depth"] = xgb_max_depth
            lgbm_kwargs["num_leaves"] = max(2, min(2 ** int(xgb_max_depth), 31))
        if xgb_min_child_weight is not None:
            lgbm_kwargs["min_child_samples"] = max(1, int(xgb_min_child_weight))
        if xgb_gamma is not None:
            lgbm_kwargs["min_split_gain"] = xgb_gamma
        if xgb_subsample is not None:
            lgbm_kwargs["subsample"] = xgb_subsample
            lgbm_kwargs["subsample_freq"] = 1
        if xgb_colsample_bytree is not None:
            lgbm_kwargs["colsample_bytree"] = xgb_colsample_bytree
        if xgb_reg_lambda is not None:
            lgbm_kwargs["reg_lambda"] = xgb_reg_lambda
        if positive_class_weight is not None:
            lgbm_kwargs["scale_pos_weight"] = positive_class_weight
        return LGBMClassifier(**lgbm_kwargs)
    if (classifier == 'CatBoost'):
        if CatBoostClassifier is None:
            raise ImportError("CatBoost is not installed. Install catboost or remove CatBoost from --classifiers.")
        cat_kwargs = {
            "iterations": mintree,
            "learning_rate": 0.05,
            "loss_function": "Logloss",
            "eval_metric": "AUC",
            "verbose": False,
            "allow_writing_files": False,
            "thread_count": xgb_n_jobs,
        }
        if random_state is not None:
            cat_kwargs["random_seed"] = random_state
        if xgb_max_depth is not None:
            cat_kwargs["depth"] = int(xgb_max_depth)
        if xgb_reg_lambda is not None:
            cat_kwargs["l2_leaf_reg"] = xgb_reg_lambda
        if positive_class_weight is not None:
            cat_kwargs["scale_pos_weight"] = positive_class_weight
        return CatBoostClassifier(**cat_kwargs)
    if (classifier == 'BalancedRF'):
        return BalancedRandomForestClassifier(
            criterion="entropy",
            n_estimators=mintree,
            n_jobs=xgb_n_jobs,
            max_depth=xgb_max_depth,
            min_samples_leaf=int(xgb_min_child_weight) if xgb_min_child_weight is not None else 1,
            max_features=xgb_colsample_bytree,
            replacement=True,
            bootstrap=False,
            random_state=random_state,
        )
    if (classifier == 'EasyEnsemble'):
        return EasyEnsembleClassifier(
            n_estimators=min(10, mintree),
            n_jobs=xgb_n_jobs,
            random_state=random_state,
        )
    if (classifier == 'LogReg'):
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=0.5,
                class_weight=sklearn_class_weight,
                max_iter=5000,
                solver="liblinear",
                random_state=random_state,
            ),
        )
    if (classifier == 'LinearSVM'):
        return make_pipeline(
            StandardScaler(),
            CalibratedClassifierCV(
                estimator=svm.LinearSVC(
                    C=0.5,
                    class_weight=sklearn_class_weight,
                    dual="auto",
                    max_iter=5000,
                    random_state=random_state,
                ),
                cv=3,
            ),
        )
    raise ValueError(f"Unsupported classifier: {classifier}")


def fit_predict_project_split(train_data, test_data, target_project, balance, classifier, mintree,
                              xgb_n_jobs=1, xgb_tree_method=None, use_coral=False,
                              coral_reg=1e-3, xgb_scale_pos_weight=None, xgb_max_depth=None,
                              xgb_min_child_weight=None, xgb_gamma=None, xgb_subsample=None,
                              xgb_colsample_bytree=None, xgb_reg_lambda=None,
                              smote_sampling_strategy=None, smote_k_neighbors=5,
                              feature_scaling="none",
                              random_state=0):
    x_train = train_data.drop(['flakyStatus', 'test_name', 'project'], axis=1)
    y_train = train_data['flakyStatus'].values.ravel()
    x_test = test_data.drop(['flakyStatus', 'test_name', 'project'], axis=1)
    y_test = test_data['flakyStatus'].values.ravel()
    test_names_as_list = test_data['test_name'].tolist()
    x_train_final = x_train.values
    x_test_final = x_test.values

    # Fit scaling statistics exclusively on the current training projects.  In
    # source-only calibration, this means the pseudo-target is transformed but
    # never contributes to the mean or standard deviation; the same holds for
    # the outer target fold.  Scaling precedes CORAL so both operations see the
    # same source-fitted coordinate system.
    if feature_scaling == "source_zscore":
        source_scaler = StandardScaler()
        x_train_final = source_scaler.fit_transform(x_train_final)
        x_test_final = source_scaler.transform(x_test_final)
        print(f"Applied source-fitted z-score scaling for target project {target_project}")
    elif feature_scaling != "none":
        raise ValueError(f"Unsupported feature_scaling: {feature_scaling}")

    if use_coral:
        try:
            x_train_final, x_test_final = apply_coral_alignment(
                x_train_final,
                x_test_final,
                reg=coral_reg,
            )
            print(f"Applied CORAL alignment for target project {target_project}")
        except Exception as e:
            print(f"CORAL alignment failed for {target_project}: {e}, continuing without CORAL")

    if (balance == "SMOTE"):
        class_counts_before = np.bincount(y_train.astype(int), minlength=2)
        target_minority_count = (
            int(smote_sampling_strategy * class_counts_before[0])
            if isinstance(smote_sampling_strategy, float) and class_counts_before[0] > 0
            else None
        )
        if (
            target_minority_count is not None
            and target_minority_count <= class_counts_before[1]
        ):
            print(
                f"SMOTE skipped: requested strategy={smote_sampling_strategy} would produce "
                f"minority count={target_minority_count}, not above the current "
                f"count={class_counts_before[1]}"
            )
        elif class_counts_before[1] <= smote_k_neighbors:
            print(
                f"SMOTE skipped: minority count={class_counts_before[1]} <= "
                f"k_neighbors={smote_k_neighbors}"
            )
        else:
            oversample = SMOTE(
                sampling_strategy=smote_sampling_strategy if smote_sampling_strategy is not None else "auto",
                k_neighbors=smote_k_neighbors,
                random_state=random_state,
            )
            try:
                x_train_final, y_train = oversample.fit_resample(x_train_final, y_train)
                class_counts_after = np.bincount(y_train.astype(int), minlength=2)
                print(
                    f"SMOTE: strategy={smote_sampling_strategy if smote_sampling_strategy is not None else 'auto'}, "
                    f"k_neighbors={smote_k_neighbors}, "
                    f"class_counts_before={class_counts_before.tolist()}, "
                    f"class_counts_after={class_counts_after.tolist()}"
                )
            except ValueError as exc:
                # A requested ratio can equal the observed ratio after integer
                # rounding.  In that case imbalanced-learn rejects SMOTE even
                # though no resampling is needed; retain the original sources
                # rather than dropping the corresponding inner validation fold.
                print(f"SMOTE skipped after boundary check: {exc}")
    elif (balance == "undersampling"):
        undersampling = RandomUnderSampler(random_state=random_state)
        x_train_final, y_train = undersampling.fit_resample(x_train_final, y_train)

    model = build_classifier_model(
        classifier,
        mintree,
        xgb_n_jobs=xgb_n_jobs,
        xgb_tree_method=xgb_tree_method,
        xgb_scale_pos_weight=xgb_scale_pos_weight,
        xgb_max_depth=xgb_max_depth,
        xgb_min_child_weight=xgb_min_child_weight,
        xgb_gamma=xgb_gamma,
        xgb_subsample=xgb_subsample,
        xgb_colsample_bytree=xgb_colsample_bytree,
        xgb_reg_lambda=xgb_reg_lambda,
        random_state=random_state,
    )
    final_model = model.fit(x_train_final, y_train)
    pred_probs = final_model.predict_proba(x_test_final)[:, 1]

    try:
        false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, pred_probs)
        auc_score = auc(false_positive_rate, true_positive_rate)
    except:
        auc_score = 0

    return pred_probs, y_test, test_names_as_list, auc_score


def calibrate_threshold_source_only(
        selected_source_data,
        candidate_thresholds,
        balance,
        classifier,
        mintree,
        xgb_n_jobs=1,
        xgb_tree_method=None,
        use_coral=False,
        coral_reg=1e-3,
        xgb_scale_pos_weight=None,
        xgb_max_depth=None,
        xgb_min_child_weight=None,
        xgb_gamma=None,
        xgb_subsample=None,
        xgb_colsample_bytree=None,
        xgb_reg_lambda=None,
        smote_sampling_strategy=None,
        smote_k_neighbors=5,
        feature_scaling="none",
        random_state=0):
    """
    Select a decision threshold using only source-project labels.

    The outer target project is not used here. Each selected source project is
    treated as a pseudo-target once; the remaining selected sources train the
    model, and the held-out source project provides validation labels.
    """
    source_projects = sorted(selected_source_data["project"].dropna().unique().tolist())
    if len(source_projects) < 2:
        fallback = candidate_thresholds[0] if candidate_thresholds else 0.5
        print(
            f"Source-only threshold calibration skipped: only {len(source_projects)} "
            f"source project(s). Falling back to threshold={fallback}."
        )
        return float(fallback), pd.DataFrame()

    validation_probs = []
    validation_labels = []
    validation_projects = []

    for pseudo_target in source_projects:
        inner_train = selected_source_data[selected_source_data["project"] != pseudo_target]
        inner_valid = selected_source_data[selected_source_data["project"] == pseudo_target]
        if inner_train.empty or inner_valid.empty:
            continue
        try:
            pred_probs, y_valid, _, _ = fit_predict_project_split(
                inner_train,
                inner_valid,
                pseudo_target,
                balance,
                classifier,
                mintree,
                xgb_n_jobs=xgb_n_jobs,
                xgb_tree_method=xgb_tree_method,
                use_coral=use_coral,
                coral_reg=coral_reg,
                xgb_scale_pos_weight=xgb_scale_pos_weight,
                xgb_max_depth=xgb_max_depth,
                xgb_min_child_weight=xgb_min_child_weight,
                xgb_gamma=xgb_gamma,
                xgb_subsample=xgb_subsample,
                xgb_colsample_bytree=xgb_colsample_bytree,
                xgb_reg_lambda=xgb_reg_lambda,
                smote_sampling_strategy=smote_sampling_strategy,
                smote_k_neighbors=smote_k_neighbors,
                feature_scaling=feature_scaling,
                random_state=random_state,
            )
            validation_probs.extend(pred_probs.tolist())
            validation_labels.extend(y_valid.tolist())
            validation_projects.extend([pseudo_target] * len(y_valid))
        except Exception as e:
            print(f"Source-only threshold calibration fold failed for {pseudo_target}: {e}")

    if len(validation_labels) == 0:
        fallback = candidate_thresholds[0] if candidate_thresholds else 0.5
        print(
            "Source-only threshold calibration produced no validation predictions. "
            f"Falling back to threshold={fallback}."
        )
        return float(fallback), pd.DataFrame()

    metric_rows = [
        compute_binary_metrics(validation_labels, validation_probs, threshold)
        for threshold in candidate_thresholds
    ]
    best = choose_threshold_from_metrics(metric_rows)
    metrics_df = pd.DataFrame(metric_rows)
    metrics_df["validation_projects"] = ",".join(source_projects)
    metrics_df["num_validation_tests"] = len(validation_labels)
    print(
        "Source-only threshold calibration: "
        f"selected={best['threshold']:.2f}, "
        f"source_cv_P={best['precision'] * 100:.1f}%, "
        f"source_cv_R={best['recall'] * 100:.1f}%, "
        f"source_cv_F1={best['f1'] * 100:.1f}%"
    )
    return float(best["threshold"]), metrics_df


def calibrate_xgb_threshold_source_only(
        selected_source_data,
        candidate_xgb_configs,
        candidate_thresholds,
        balance,
        classifier,
        mintree,
        xgb_n_jobs=1,
        xgb_tree_method=None,
        use_coral=False,
        coral_reg=1e-3,
        smote_sampling_strategy=None,
        smote_k_neighbors=5,
        feature_scaling="none",
        random_state=0):
    """Choose XGBoost configuration and threshold using only source labels.

    For each outer target, the sources selected with the target's *unlabeled*
    features are split leave-one-source-project-out.  The outer target labels
    are never read during this routine.  Every candidate configuration is
    evaluated on exactly the same pseudo-target folds and threshold grid.
    """
    source_projects = sorted(selected_source_data["project"].dropna().unique().tolist())
    if len(source_projects) < 2 or not candidate_xgb_configs or not candidate_thresholds:
        raise ValueError("Source-only XGBoost tuning requires at least two source projects and nonempty candidates.")

    rows = []
    for config in candidate_xgb_configs:
        validation_probs = []
        validation_labels = []
        for pseudo_target in source_projects:
            inner_train = selected_source_data[selected_source_data["project"] != pseudo_target]
            inner_valid = selected_source_data[selected_source_data["project"] == pseudo_target]
            if inner_train.empty or inner_valid.empty:
                continue
            try:
                pred_probs, y_valid, _, _ = fit_predict_project_split(
                    inner_train,
                    inner_valid,
                    pseudo_target,
                    balance,
                    classifier,
                    mintree,
                    xgb_n_jobs=xgb_n_jobs,
                    xgb_tree_method=xgb_tree_method,
                    use_coral=use_coral,
                    coral_reg=coral_reg,
                    xgb_scale_pos_weight=config["xgb_scale_pos_weight"],
                    xgb_max_depth=config["xgb_max_depth"],
                    xgb_min_child_weight=config["xgb_min_child_weight"],
                    xgb_gamma=config["xgb_gamma"],
                    xgb_subsample=config["xgb_subsample"],
                    xgb_colsample_bytree=config["xgb_colsample_bytree"],
                    xgb_reg_lambda=config["xgb_reg_lambda"],
                    smote_sampling_strategy=smote_sampling_strategy,
                    smote_k_neighbors=smote_k_neighbors,
                    feature_scaling=feature_scaling,
                    random_state=random_state,
                )
                validation_probs.extend(pred_probs.tolist())
                validation_labels.extend(y_valid.tolist())
            except Exception as exc:
                print(
                    "Source-only XGBoost tuning fold failed: "
                    f"candidate={config['candidate_id']}, pseudo_target={pseudo_target}: {exc}"
                )

        if not validation_labels:
            continue
        for threshold in candidate_thresholds:
            metrics = compute_binary_metrics(validation_labels, validation_probs, threshold)
            metrics.update(config)
            metrics["candidate_threshold"] = float(threshold)
            metrics["validation_projects"] = ",".join(source_projects)
            metrics["num_validation_tests"] = len(validation_labels)
            rows.append(metrics)

    if not rows:
        raise RuntimeError("Source-only XGBoost tuning produced no validation predictions.")

    best = max(
        rows,
        key=lambda row: (
            row["f1"],
            row["precision"],
            row["candidate_threshold"],
            -row["xgb_max_depth"],
            -row["xgb_min_child_weight"],
        ),
    )
    metrics_df = pd.DataFrame(rows)
    print(
        "Source-only XGBoost tuning: "
        f"selected={best['candidate_id']}, threshold={best['candidate_threshold']:.2f}, "
        f"source_cv_P={best['precision'] * 100:.1f}%, "
        f"source_cv_R={best['recall'] * 100:.1f}%, "
        f"source_cv_F1={best['f1'] * 100:.1f}%"
    )
    return dict(best), metrics_df


def calibrate_topk_threshold_source_only(
        outer_source_data,
        candidate_top_ks,
        candidate_thresholds,
        balance,
        classifier,
        mintree,
        xgb_n_jobs=1,
        xgb_tree_method=None,
        use_coral=False,
        coral_reg=1e-3,
        source_selection="top_k",
        source_selection_top_ratio=0.5,
        source_selection_min_projects=3,
        xgb_scale_pos_weight=None,
        xgb_max_depth=None,
        xgb_min_child_weight=None,
        xgb_gamma=None,
        xgb_subsample=None,
        xgb_colsample_bytree=None,
        xgb_reg_lambda=None,
        smote_sampling_strategy=None,
        smote_k_neighbors=5,
        feature_scaling="none",
        random_state=0):
    """
    Select source-set size and decision threshold using only outer-source labels.

    For one outer target project, this routine never reads the outer target labels.
    It treats each source project as a pseudo-target, selects source projects for
    that pseudo-target from the remaining source projects, trains the same
    adaptation/classification pipeline, and evaluates candidate top-k/threshold
    pairs on the pseudo-target labels.
    """
    source_projects = sorted(outer_source_data["project"].dropna().unique().tolist())
    candidate_top_ks = sorted({int(k) for k in candidate_top_ks if int(k) > 0})
    candidate_thresholds = [float(t) for t in candidate_thresholds]
    if len(source_projects) < 3 or not candidate_top_ks or not candidate_thresholds:
        fallback_k = candidate_top_ks[0] if candidate_top_ks else source_selection_min_projects
        fallback_threshold = candidate_thresholds[0] if candidate_thresholds else 0.5
        print(
            "Source-only nested tuning skipped: insufficient source projects or candidates. "
            f"Falling back to top_k={fallback_k}, threshold={fallback_threshold}."
        )
        return int(fallback_k), float(fallback_threshold), pd.DataFrame()

    rows = []
    for candidate_top_k in candidate_top_ks:
        validation_probs = []
        validation_labels = []
        validation_projects = []

        for pseudo_target in source_projects:
            inner_train_pool = outer_source_data[outer_source_data["project"] != pseudo_target]
            inner_valid = outer_source_data[outer_source_data["project"] == pseudo_target]
            if inner_train_pool.empty or inner_valid.empty:
                continue

            try:
                if source_selection == "none":
                    inner_train = inner_train_pool.copy()
                    selected_count = inner_train["project"].nunique()
                else:
                    inner_train, selected_count = select_target_aware_source_projects(
                        inner_train_pool,
                        inner_valid,
                        selection_mode=source_selection,
                        top_k=candidate_top_k,
                        top_ratio=source_selection_top_ratio,
                        min_projects=source_selection_min_projects,
                    )

                pred_probs, y_valid, _, _ = fit_predict_project_split(
                    inner_train,
                    inner_valid,
                    pseudo_target,
                    balance,
                    classifier,
                    mintree,
                    xgb_n_jobs=xgb_n_jobs,
                    xgb_tree_method=xgb_tree_method,
                    use_coral=use_coral,
                    coral_reg=coral_reg,
                    xgb_scale_pos_weight=xgb_scale_pos_weight,
                    xgb_max_depth=xgb_max_depth,
                    xgb_min_child_weight=xgb_min_child_weight,
                    xgb_gamma=xgb_gamma,
                    xgb_subsample=xgb_subsample,
                    xgb_colsample_bytree=xgb_colsample_bytree,
                    xgb_reg_lambda=xgb_reg_lambda,
                    smote_sampling_strategy=smote_sampling_strategy,
                    smote_k_neighbors=smote_k_neighbors,
                    feature_scaling=feature_scaling,
                    random_state=random_state,
                )
                validation_probs.extend(pred_probs.tolist())
                validation_labels.extend(y_valid.tolist())
                validation_projects.extend([pseudo_target] * len(y_valid))
            except Exception as e:
                print(
                    "Source-only nested tuning fold failed: "
                    f"top_k={candidate_top_k}, pseudo_target={pseudo_target}: {e}"
                )

        if len(validation_labels) == 0:
            continue

        for threshold in candidate_thresholds:
            metrics = compute_binary_metrics(validation_labels, validation_probs, threshold)
            metrics.update({
                "candidate_top_k": int(candidate_top_k),
                "candidate_threshold": float(threshold),
                "validation_projects": ",".join(source_projects),
                "num_validation_tests": len(validation_labels),
            })
            rows.append(metrics)

    if not rows:
        fallback_k = candidate_top_ks[0]
        fallback_threshold = candidate_thresholds[0]
        print(
            "Source-only nested tuning produced no validation predictions. "
            f"Falling back to top_k={fallback_k}, threshold={fallback_threshold}."
        )
        return int(fallback_k), float(fallback_threshold), pd.DataFrame()

    metrics_df = pd.DataFrame(rows)
    best = max(
        rows,
        key=lambda row: (
            row["f1"],
            row["precision"],
            row["candidate_threshold"],
            -row["candidate_top_k"],
        ),
    )
    print(
        "Source-only nested tuning: "
        f"selected_top_k={best['candidate_top_k']}, "
        f"selected_threshold={best['candidate_threshold']:.2f}, "
        f"source_cv_P={best['precision'] * 100:.1f}%, "
        f"source_cv_R={best['recall'] * 100:.1f}%, "
        f"source_cv_F1={best['f1'] * 100:.1f}%"
    )
    return int(best["candidate_top_k"]), float(best["candidate_threshold"]), metrics_df


def calibrate_topk_xgb_threshold_source_only(
        outer_source_data,
        candidate_top_ks,
        candidate_xgb_configs,
        candidate_thresholds,
        balance,
        classifier,
        mintree,
        xgb_n_jobs=1,
        xgb_tree_method=None,
        use_coral=False,
        coral_reg=1e-3,
        source_selection="top_k",
        source_selection_top_ratio=0.5,
        source_selection_min_projects=3,
        smote_sampling_strategy=None,
        smote_k_neighbors=5,
        feature_scaling="none",
        random_state=0):
    """Jointly choose source-set size, XGBoost configuration, and threshold.

    Every candidate is evaluated only on leave-one-source-project-out
    pseudo-target folds of ``outer_source_data``.  In particular, the outer
    target labels are unavailable while choosing *all* three deployment
    decisions.  Project selection for a pseudo-target may use its feature
    matrix, but never its labels.
    """
    source_projects = sorted(outer_source_data["project"].dropna().unique().tolist())
    candidate_top_ks = sorted({int(k) for k in candidate_top_ks if int(k) > 0})
    candidate_thresholds = [float(t) for t in candidate_thresholds]
    if (len(source_projects) < 3 or not candidate_top_ks
            or not candidate_xgb_configs or not candidate_thresholds):
        raise ValueError(
            "Joint source-only tuning requires at least three source projects "
            "and nonempty top-k, XGBoost, and threshold candidate sets."
        )

    validation_by_candidate = {}
    for candidate_top_k in candidate_top_ks:
        for config in candidate_xgb_configs:
            key = (int(candidate_top_k), config["candidate_id"])
            validation_by_candidate[key] = {"probs": [], "labels": []}

        for pseudo_target in source_projects:
            inner_train_pool = outer_source_data[
                outer_source_data["project"] != pseudo_target
            ]
            inner_valid = outer_source_data[
                outer_source_data["project"] == pseudo_target
            ]
            if inner_train_pool.empty or inner_valid.empty:
                continue
            try:
                if source_selection == "none":
                    inner_train = inner_train_pool.copy()
                else:
                    inner_train, _ = select_target_aware_source_projects(
                        inner_train_pool,
                        inner_valid,
                        selection_mode=source_selection,
                        top_k=candidate_top_k,
                        top_ratio=source_selection_top_ratio,
                        min_projects=source_selection_min_projects,
                    )
            except Exception as exc:
                print(
                    "Joint source-only tuning selection failed: "
                    f"top_k={candidate_top_k}, pseudo_target={pseudo_target}: {exc}"
                )
                continue

            for config in candidate_xgb_configs:
                key = (int(candidate_top_k), config["candidate_id"])
                try:
                    pred_probs, y_valid, _, _ = fit_predict_project_split(
                        inner_train,
                        inner_valid,
                        pseudo_target,
                        balance,
                        classifier,
                        mintree,
                        xgb_n_jobs=xgb_n_jobs,
                        xgb_tree_method=xgb_tree_method,
                        use_coral=use_coral,
                        coral_reg=coral_reg,
                        xgb_scale_pos_weight=config["xgb_scale_pos_weight"],
                        xgb_max_depth=config["xgb_max_depth"],
                        xgb_min_child_weight=config["xgb_min_child_weight"],
                        xgb_gamma=config["xgb_gamma"],
                        xgb_subsample=config["xgb_subsample"],
                        xgb_colsample_bytree=config["xgb_colsample_bytree"],
                        xgb_reg_lambda=config["xgb_reg_lambda"],
                        smote_sampling_strategy=smote_sampling_strategy,
                        smote_k_neighbors=smote_k_neighbors,
                        feature_scaling=feature_scaling,
                        random_state=random_state,
                    )
                    validation_by_candidate[key]["probs"].extend(pred_probs.tolist())
                    validation_by_candidate[key]["labels"].extend(y_valid.tolist())
                except Exception as exc:
                    print(
                        "Joint source-only tuning fold failed: "
                        f"top_k={candidate_top_k}, candidate={config['candidate_id']}, "
                        f"pseudo_target={pseudo_target}: {exc}"
                    )

    rows = []
    for candidate_top_k in candidate_top_ks:
        for config in candidate_xgb_configs:
            record = validation_by_candidate[(int(candidate_top_k), config["candidate_id"])]
            if not record["labels"]:
                continue
            for threshold in candidate_thresholds:
                metrics = compute_binary_metrics(record["labels"], record["probs"], threshold)
                metrics.update(config)
                metrics.update({
                    "candidate_top_k": int(candidate_top_k),
                    "candidate_threshold": float(threshold),
                    "validation_projects": ",".join(source_projects),
                    "num_validation_tests": len(record["labels"]),
                })
                rows.append(metrics)

    if not rows:
        raise RuntimeError("Joint source-only tuning produced no validation predictions.")

    best = max(
        rows,
        key=lambda row: (
            row["f1"],
            row["precision"],
            row["candidate_threshold"],
            -row["xgb_max_depth"],
            -row["xgb_min_child_weight"],
            -row["candidate_top_k"],
        ),
    )
    metrics_df = pd.DataFrame(rows)
    print(
        "Joint source-only nested tuning: "
        f"selected_top_k={best['candidate_top_k']}, "
        f"selected={best['candidate_id']}, "
        f"threshold={best['candidate_threshold']:.2f}, "
        f"source_cv_P={best['precision'] * 100:.1f}%, "
        f"source_cv_R={best['recall'] * 100:.1f}%, "
        f"source_cv_F1={best['f1'] * 100:.1f}%"
    )
    return dict(best), metrics_df


def run_one_project_fold(data, target_project, balance, classifier, mintree, Features_type, ig,
                         xgb_n_jobs=1, xgb_tree_method=None, use_coral=False,
                         coral_reg=1e-3, threshold=0.6,
                         threshold_policy="fixed", threshold_candidates=None,
                         xgb_scale_pos_weight=None, xgb_max_depth=None,
                         xgb_min_child_weight=None, xgb_gamma=None,
                         xgb_subsample=None, xgb_colsample_bytree=None,
                         xgb_reg_lambda=None, smote_sampling_strategy=None,
                         smote_k_neighbors=5, source_selection="none",
                         feature_scaling="none",
                         source_selection_top_k=10, source_selection_top_ratio=0.5,
                         source_selection_min_projects=3,
                         source_only_tune=False,
                         source_selection_top_k_candidates=None,
                         source_only_xgb_tune=False,
                         source_only_xgb_candidates=None,
                         source_only_joint_tune=False,
                         source_only_staged_tune=False,
                         random_state=0,
                         gpu_id=None):
    if data is None:
        data = PROJECT_PARALLEL_DATA

    if torch is not None and hasattr(torch, "cuda") and gpu_id is not None and torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
        print(f"Testing on project: {target_project} using CUDA device {gpu_id}")
    else:
        print(f"Testing on project: {target_project}")

    # 训练集：除当前项目外的所有数据
    train_data = data[data['project'] != target_project]
    # 测试集：当前项目的数据
    test_data = data[data['project'] == target_project]
    source_selection_num_projects = train_data["project"].nunique()
    effective_source_selection_top_k = int(source_selection_top_k)
    effective_threshold = float(threshold)
    threshold_display = threshold
    effective_threshold_policy = threshold_policy
    calibration_metrics = pd.DataFrame()
    effective_xgb_config = {
        "candidate_id": "fixed_reported",
        "xgb_scale_pos_weight": xgb_scale_pos_weight,
        "xgb_max_depth": xgb_max_depth,
        "xgb_min_child_weight": xgb_min_child_weight,
        "xgb_gamma": xgb_gamma,
        "xgb_subsample": xgb_subsample,
        "xgb_colsample_bytree": xgb_colsample_bytree,
        "xgb_reg_lambda": xgb_reg_lambda,
    }

    if source_only_tune:
        candidates = threshold_candidates if threshold_candidates else [threshold]
        top_k_candidates = (
            source_selection_top_k_candidates
            if source_selection_top_k_candidates
            else [source_selection_top_k]
        )
        effective_source_selection_top_k, effective_threshold, calibration_metrics = (
            calibrate_topk_threshold_source_only(
                train_data,
                top_k_candidates,
                candidates,
                balance,
                classifier,
                mintree,
                xgb_n_jobs=xgb_n_jobs,
                xgb_tree_method=xgb_tree_method,
                use_coral=use_coral,
                coral_reg=coral_reg,
                source_selection=source_selection,
                source_selection_top_ratio=source_selection_top_ratio,
                source_selection_min_projects=source_selection_min_projects,
                xgb_scale_pos_weight=xgb_scale_pos_weight,
                xgb_max_depth=xgb_max_depth,
                xgb_min_child_weight=xgb_min_child_weight,
                xgb_gamma=xgb_gamma,
                xgb_subsample=xgb_subsample,
                xgb_colsample_bytree=xgb_colsample_bytree,
                xgb_reg_lambda=xgb_reg_lambda,
                smote_sampling_strategy=smote_sampling_strategy,
                smote_k_neighbors=smote_k_neighbors,
                feature_scaling=feature_scaling,
                random_state=random_state,
            )
        )
        threshold_display = "source_only_nested_cv"
        effective_threshold_policy = "source_only_nested_cv"

    if source_only_joint_tune:
        candidates = threshold_candidates if threshold_candidates else [threshold]
        top_k_candidates = (
            source_selection_top_k_candidates
            if source_selection_top_k_candidates
            else [source_selection_top_k]
        )
        effective_xgb_config, calibration_metrics = calibrate_topk_xgb_threshold_source_only(
            train_data,
            top_k_candidates,
            source_only_xgb_candidates,
            candidates,
            balance,
            classifier,
            mintree,
            xgb_n_jobs=xgb_n_jobs,
            xgb_tree_method=xgb_tree_method,
            use_coral=use_coral,
            coral_reg=coral_reg,
            source_selection=source_selection,
            source_selection_top_ratio=source_selection_top_ratio,
            source_selection_min_projects=source_selection_min_projects,
            smote_sampling_strategy=smote_sampling_strategy,
            smote_k_neighbors=smote_k_neighbors,
            feature_scaling=feature_scaling,
            random_state=random_state,
        )
        effective_source_selection_top_k = int(
            effective_xgb_config["candidate_top_k"]
        )
        effective_threshold = float(effective_xgb_config["candidate_threshold"])
        threshold_display = "source_only_joint_nested_cv"
        effective_threshold_policy = "source_only_joint_nested_cv"

    if source_only_staged_tune:
        if not source_only_xgb_candidates:
            raise ValueError("Staged source-only tuning requires pre-specified XGBoost candidates.")
        candidates = threshold_candidates if threshold_candidates else [threshold]
        top_k_candidates = (
            source_selection_top_k_candidates
            if source_selection_top_k_candidates
            else [source_selection_top_k]
        )
        # Stage 1 chooses the source-set size with the reported regularized
        # configuration as a pre-specified calibration model.  Stage 2 below
        # then chooses the deployed XGBoost configuration and threshold only
        # within the selected sources.  Neither stage reads outer-target labels.
        calibration_config = source_only_xgb_candidates[0]
        effective_source_selection_top_k, _, topk_metrics = (
            calibrate_topk_threshold_source_only(
                train_data,
                top_k_candidates,
                candidates,
                balance,
                classifier,
                mintree,
                xgb_n_jobs=xgb_n_jobs,
                xgb_tree_method=xgb_tree_method,
                use_coral=use_coral,
                coral_reg=coral_reg,
                source_selection=source_selection,
                source_selection_top_ratio=source_selection_top_ratio,
                source_selection_min_projects=source_selection_min_projects,
                xgb_scale_pos_weight=calibration_config["xgb_scale_pos_weight"],
                xgb_max_depth=calibration_config["xgb_max_depth"],
                xgb_min_child_weight=calibration_config["xgb_min_child_weight"],
                xgb_gamma=calibration_config["xgb_gamma"],
                xgb_subsample=calibration_config["xgb_subsample"],
                xgb_colsample_bytree=calibration_config["xgb_colsample_bytree"],
                xgb_reg_lambda=calibration_config["xgb_reg_lambda"],
                smote_sampling_strategy=smote_sampling_strategy,
                smote_k_neighbors=smote_k_neighbors,
                feature_scaling=feature_scaling,
                random_state=random_state,
            )
        )
        if not topk_metrics.empty:
            topk_metrics = topk_metrics.copy()
            topk_metrics["selection_stage"] = "top_k_screen"
        calibration_metrics = topk_metrics

    if source_selection != "none":
        train_data, source_selection_num_projects = select_target_aware_source_projects(
            train_data,
            test_data,
            selection_mode=source_selection,
            top_k=effective_source_selection_top_k,
            top_ratio=source_selection_top_ratio,
            min_projects=source_selection_min_projects,
        )

    if source_only_xgb_tune:
        if source_only_tune or source_only_joint_tune:
            raise ValueError("--source-only-xgb-tune cannot be combined with source-only joint tuning.")
        candidates = threshold_candidates if threshold_candidates else [threshold]
        effective_xgb_config, calibration_metrics = calibrate_xgb_threshold_source_only(
            train_data,
            source_only_xgb_candidates,
            candidates,
            balance,
            classifier,
            mintree,
            xgb_n_jobs=xgb_n_jobs,
            xgb_tree_method=xgb_tree_method,
            use_coral=use_coral,
            coral_reg=coral_reg,
            smote_sampling_strategy=smote_sampling_strategy,
            smote_k_neighbors=smote_k_neighbors,
            feature_scaling=feature_scaling,
            random_state=random_state,
        )
        effective_threshold = float(effective_xgb_config["candidate_threshold"])
        threshold_display = "source_only_xgb_cv"
        effective_threshold_policy = "source_only_xgb_cv"
    elif source_only_staged_tune:
        candidates = threshold_candidates if threshold_candidates else [threshold]
        effective_xgb_config, xgb_metrics = calibrate_xgb_threshold_source_only(
            train_data,
            source_only_xgb_candidates,
            candidates,
            balance,
            classifier,
            mintree,
            xgb_n_jobs=xgb_n_jobs,
            xgb_tree_method=xgb_tree_method,
            use_coral=use_coral,
            coral_reg=coral_reg,
            smote_sampling_strategy=smote_sampling_strategy,
            smote_k_neighbors=smote_k_neighbors,
            feature_scaling=feature_scaling,
            random_state=random_state,
        )
        if not xgb_metrics.empty:
            xgb_metrics = xgb_metrics.copy()
            xgb_metrics["selection_stage"] = "xgb_threshold_finalization"
        calibration_metrics = pd.concat(
            [frame for frame in (calibration_metrics, xgb_metrics) if not frame.empty],
            ignore_index=True,
        )
        effective_threshold = float(effective_xgb_config["candidate_threshold"])
        threshold_display = "source_only_staged_nested_cv"
        effective_threshold_policy = "source_only_staged_nested_cv"
    elif (not source_only_tune) and (not source_only_joint_tune) and (not source_only_staged_tune) and threshold_policy == "source_only_cv":
        candidates = threshold_candidates if threshold_candidates else [threshold]
        effective_threshold, calibration_metrics = calibrate_threshold_source_only(
            train_data,
            candidates,
            balance,
            classifier,
            mintree,
            xgb_n_jobs=xgb_n_jobs,
            xgb_tree_method=xgb_tree_method,
            use_coral=use_coral,
            coral_reg=coral_reg,
            xgb_scale_pos_weight=xgb_scale_pos_weight,
            xgb_max_depth=xgb_max_depth,
            xgb_min_child_weight=xgb_min_child_weight,
            xgb_gamma=xgb_gamma,
            xgb_subsample=xgb_subsample,
            xgb_colsample_bytree=xgb_colsample_bytree,
            xgb_reg_lambda=xgb_reg_lambda,
            smote_sampling_strategy=smote_sampling_strategy,
            smote_k_neighbors=smote_k_neighbors,
            feature_scaling=feature_scaling,
            random_state=random_state,
        )
        threshold_display = "source_only_cv"
        effective_threshold_policy = "source_only_cv"
    elif (not source_only_tune) and (not source_only_joint_tune) and (not source_only_staged_tune) and threshold_policy != "fixed":
        raise ValueError(f"Unsupported threshold_policy: {threshold_policy}")

    pred_probs, y_test, test_names_as_list, auc_score = fit_predict_project_split(
        train_data,
        test_data,
        target_project,
        balance,
        classifier,
        mintree,
        xgb_n_jobs=xgb_n_jobs,
        xgb_tree_method=xgb_tree_method,
        use_coral=use_coral,
        coral_reg=coral_reg,
        xgb_scale_pos_weight=effective_xgb_config["xgb_scale_pos_weight"],
        xgb_max_depth=effective_xgb_config["xgb_max_depth"],
        xgb_min_child_weight=effective_xgb_config["xgb_min_child_weight"],
        xgb_gamma=effective_xgb_config["xgb_gamma"],
        xgb_subsample=effective_xgb_config["xgb_subsample"],
        xgb_colsample_bytree=effective_xgb_config["xgb_colsample_bytree"],
        xgb_reg_lambda=effective_xgb_config["xgb_reg_lambda"],
        smote_sampling_strategy=smote_sampling_strategy,
        smote_k_neighbors=smote_k_neighbors,
        feature_scaling=feature_scaling,
        random_state=random_state,
    )
    preds = (pred_probs >= effective_threshold).astype(int)

    actual_status = y_test.tolist()
    result_rows = []
    for i in range(len(test_names_as_list)):
        result_rows.append([
            "PerProject",
            balance,
            ig,
            mintree,
            classifier,
            Features_type,
            target_project,
            test_names_as_list[i],
            int(actual_status[i]),
            int(preds[i]),
            float(pred_probs[i]),
            threshold_display,
            effective_threshold_policy,
            feature_scaling,
            float(effective_threshold),
            effective_xgb_config["xgb_scale_pos_weight"],
            effective_xgb_config["xgb_max_depth"],
            effective_xgb_config["xgb_min_child_weight"],
            effective_xgb_config["xgb_gamma"],
            effective_xgb_config["xgb_subsample"],
            effective_xgb_config["xgb_colsample_bytree"],
            effective_xgb_config["xgb_reg_lambda"],
            source_selection,
            effective_source_selection_top_k,
            source_selection_top_ratio,
            source_selection_min_projects,
            source_selection_num_projects,
            smote_sampling_strategy,
            smote_k_neighbors,
            random_state,
            "TP" if actual_status[i] == 1 and preds[i] == 1 else "FN" if actual_status[i] == 1 and preds[
                i] == 0 else "FP" if
            actual_status[i] == 0 and preds[i] == 1 else "TN",
        ])

    tn, fp, fn, tp = confusion_matrix(y_test, preds, labels=[0, 1]).ravel()

    return {
        "target_project": target_project,
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "auc_score": auc_score,
        "effective_threshold": float(effective_threshold),
        "effective_source_selection_top_k": int(effective_source_selection_top_k),
        "effective_xgb_config": effective_xgb_config,
        "calibration_metrics": calibration_metrics,
        "result_rows": result_rows,
    }


def predict_RF_perProject(data, project_name, balance, classifier, mintree, Features_type, ig, result_by_test_name,
                          project_n_jobs=1, gpu_ids=None, xgb_n_jobs=1, xgb_tree_method=None,
                          use_coral=False, coral_reg=1e-3, threshold=0.6,
                          threshold_policy="fixed", threshold_candidates=None,
                          xgb_scale_pos_weight=None, xgb_max_depth=None,
                          xgb_min_child_weight=None, xgb_gamma=None,
                          xgb_subsample=None, xgb_colsample_bytree=None,
                          xgb_reg_lambda=None, smote_sampling_strategy=None,
                          smote_k_neighbors=5, source_selection="none",
                          feature_scaling="none",
                          source_selection_top_k=10, source_selection_top_ratio=0.5,
                          source_selection_min_projects=3, source_only_tune=False,
                          source_selection_top_k_candidates=None,
                          source_only_xgb_tune=False, source_only_xgb_candidates=None,
                          source_only_joint_tune=False,
                          source_only_staged_tune=False,
                          random_state=0,
                          checkpoint_dir=None):
    """
    逐项目验证函数 - 每次将一个项目作为测试集，其他项目作为训练集
    """
    data = data.dropna()

    # 确保包含项目列
    if 'project' in data.columns and data.columns.duplicated().any():
        print("Warning: Duplicate column names found. Removing duplicates...")
        # 删除重复的列，只保留第一个
        data = data.loc[:, ~data.columns.duplicated()]

    # 获取所有项目名称
    all_projects = data['project'].unique()

    TN_total = FP_total = FN_total = TP_total = 0
    auc_scores_total = []
    calibration_metrics_total = []

    gpu_id_list = parse_gpu_ids(gpu_ids)

    def consume_project_output(project_output):
        nonlocal TN_total, FP_total, FN_total, TP_total, result_by_test_name, calibration_metrics_total
        TN_total += project_output["TN"]
        FP_total += project_output["FP"]
        FN_total += project_output["FN"]
        TP_total += project_output["TP"]
        auc_scores_total.append(project_output["auc_score"])
        calibration_metrics = project_output.get("calibration_metrics")
        if calibration_metrics is not None and not calibration_metrics.empty:
            calibration_metrics = calibration_metrics.copy()
            calibration_metrics["target_project"] = project_output["target_project"]
            calibration_metrics["selected_threshold"] = project_output.get("effective_threshold")
            calibration_metrics["selected_source_selection_top_k"] = project_output.get(
                "effective_source_selection_top_k"
            )
            selected_xgb_config = project_output.get("effective_xgb_config", {})
            calibration_metrics["selected_xgb_candidate_id"] = selected_xgb_config.get("candidate_id")
            for parameter_name in (
                    "xgb_scale_pos_weight", "xgb_max_depth", "xgb_min_child_weight",
                    "xgb_gamma", "xgb_subsample", "xgb_colsample_bytree", "xgb_reg_lambda"):
                calibration_metrics[f"selected_{parameter_name}"] = selected_xgb_config.get(parameter_name)
            calibration_metrics_total.append(calibration_metrics)
        result_by_test_name = pd.concat(
            [
                result_by_test_name,
                pd.DataFrame(project_output["result_rows"], columns=result_by_test_name.columns),
            ],
            ignore_index=True,
        )
        if checkpoint_dir is not None:
            checkpoint_path = Path(checkpoint_dir)
            checkpoint_path.mkdir(parents=True, exist_ok=True)
            target_name = str(project_output["target_project"]).replace("/", "_")
            pd.DataFrame(project_output["result_rows"], columns=result_by_test_name.columns).to_csv(
                checkpoint_path / f"{target_name}_prediction_result_per_test.csv",
                index=False,
            )
            if calibration_metrics is not None and not calibration_metrics.empty:
                calibration_metrics.to_csv(
                    checkpoint_path / f"{target_name}_source_only_model_selection.csv",
                    index=False,
                )
            with open(checkpoint_path / "progress.log", "a", encoding="utf-8") as progress_file:
                progress_file.write(
                    f"{time.strftime('%Y-%m-%d %H:%M:%S')} finished "
                    f"{project_output['target_project']} "
                    f"TP={project_output['TP']} FN={project_output['FN']} "
                    f"FP={project_output['FP']} TN={project_output['TN']} "
                    f"threshold={project_output.get('effective_threshold')} "
                    f"top_k={project_output.get('effective_source_selection_top_k')} "
                    f"xgb_candidate={project_output.get('effective_xgb_config', {}).get('candidate_id')}\n"
                )

    common_kwargs = dict(
        data=None if project_n_jobs > 1 else data,
        balance=balance,
        classifier=classifier,
        mintree=mintree,
        Features_type=Features_type,
        ig=ig,
        xgb_n_jobs=xgb_n_jobs,
        xgb_tree_method=xgb_tree_method,
        use_coral=use_coral,
        coral_reg=coral_reg,
        threshold=threshold,
        threshold_policy=threshold_policy,
        threshold_candidates=threshold_candidates,
        xgb_scale_pos_weight=xgb_scale_pos_weight,
        xgb_max_depth=xgb_max_depth,
        xgb_min_child_weight=xgb_min_child_weight,
        xgb_gamma=xgb_gamma,
        xgb_subsample=xgb_subsample,
        xgb_colsample_bytree=xgb_colsample_bytree,
        xgb_reg_lambda=xgb_reg_lambda,
        source_selection=source_selection,
        source_selection_top_k=source_selection_top_k,
        source_selection_top_ratio=source_selection_top_ratio,
        source_selection_min_projects=source_selection_min_projects,
        source_only_tune=source_only_tune,
        source_selection_top_k_candidates=source_selection_top_k_candidates,
        source_only_xgb_tune=source_only_xgb_tune,
        source_only_xgb_candidates=source_only_xgb_candidates,
        source_only_joint_tune=source_only_joint_tune,
        source_only_staged_tune=source_only_staged_tune,
        smote_sampling_strategy=smote_sampling_strategy,
        smote_k_neighbors=smote_k_neighbors,
        feature_scaling=feature_scaling,
        random_state=random_state,
    )

    if project_n_jobs > 1:
        print(f"Running per-project folds in parallel: project_n_jobs={project_n_jobs}, gpu_ids={gpu_id_list}")
        with ProcessPoolExecutor(
            max_workers=project_n_jobs,
            mp_context=mp.get_context("spawn"),
            initializer=init_project_parallel_worker,
            initargs=(data,),
        ) as executor:
            futures = []
            for task_index, target_project in enumerate(all_projects):
                gpu_id = gpu_id_list[task_index % len(gpu_id_list)] if len(gpu_id_list) > 0 else None
                futures.append(
                    executor.submit(
                        run_one_project_fold,
                        target_project=target_project,
                        gpu_id=gpu_id,
                        **common_kwargs,
                    )
                )
            for future in as_completed(futures):
                consume_project_output(future.result())
    else:
        for task_index, target_project in enumerate(all_projects):
            gpu_id = gpu_id_list[task_index % len(gpu_id_list)] if len(gpu_id_list) > 0 else None
            consume_project_output(
                run_one_project_fold(
                    target_project=target_project,
                    gpu_id=gpu_id,
                    **common_kwargs,
                )
            )

    # 计算平均性能指标
    accuracy, F1, Precision, Recall = get_scores(TN_total, FP_total, FN_total, TP_total)
    auc_scores_total = [0 if math.isnan(x) else x for x in auc_scores_total]
    avg_auc = sum(auc_scores_total) / len(auc_scores_total) if auc_scores_total else 0
    calibration_metrics_df = (
        pd.concat(calibration_metrics_total, ignore_index=True)
        if len(calibration_metrics_total) > 0 else pd.DataFrame()
    )

    return TN_total, FP_total, FN_total, TP_total, round((Precision * 100)), round(((Recall) * 100)), round(
        (F1 * 100)), round((avg_auc * 100)), result_by_test_name, calibration_metrics_df


# %%
def get_only_specific_columns_V1(full_data, specificColumns, wanted_columns):
    copy_fullData = full_data.copy()
    lst = []
    for i in specificColumns:
        lst.append(i)
    for j in wanted_columns:
        lst.append(j)
    seen = set()
    available_columns = []
    for col in lst:
        if col in full_data.columns and col not in seen:
            available_columns.append(col)
            seen.add(col)
    copy_fullData = copy_fullData[available_columns]
    return copy_fullData


# %%
execution_time = time.time()

if __name__ == '__main__':
    warnings.simplefilter("ignore")
    print(os.getcwd())

    parser = argparse.ArgumentParser(
        description="CoDA-FTP: Conservative Domain Adaptation for cross-project flaky test prediction"
    )
    parser.add_argument("--feature-mode", choices=["fresh_codebert", "precomputed_fused"], default="fresh_codebert")
    parser.add_argument("--precomputed-data-path", default="result/processed_data_with_vocabulary_per_test.csv")
    parser.add_argument("--flakeflagger-features-path", default="input_data/FlakeFlaggerFeaturesTypes.csv")
    parser.add_argument("--codebert-data-path", default="dataset_for_semantic_extract/FlakeFlagger/FlakeFlagger_dataset.csv")
    parser.add_argument("--codebert-model-path", default="microsoft/codebert-base")
    parser.add_argument("--codebert-batch-size", type=int, default=64)
    parser.add_argument("--codebert-max-length", type=int, default=510)
    parser.add_argument("--codebert-fp16", action="store_true")
    parser.add_argument("--codebert-cache-dir", default=None)
    parser.add_argument("--codebert-local-files-only", action="store_true")
    parser.add_argument("--output-dir", default="result_1/coda_ftp_result/")
    parser.add_argument("--project-n-jobs", type=int, default=1)
    parser.add_argument("--gpu-ids", default="")
    parser.add_argument("--classifiers", default="XGBoost",
                        help=("Comma-separated classifiers to evaluate. Supported: XGBoost, LightGBM, "
                              "CatBoost, BalancedRF, EasyEnsemble, LogReg, LinearSVM, RF, SVM, DT, "
                              "MLP, Ada, NB, KNN."))
    parser.add_argument("--classifier", dest="classifiers",
                        help="Alias for --classifiers when running a single classifier.")
    parser.add_argument("--xgb-n-jobs", type=int, default=1)
    parser.add_argument("--xgb-tree-method", default="")
    parser.add_argument("--xgb-max-depth", type=int, default=None)
    parser.add_argument("--xgb-min-child-weight", type=float, default=None)
    parser.add_argument("--xgb-gamma", type=float, default=None)
    parser.add_argument("--xgb-subsample", type=float, default=None)
    parser.add_argument("--xgb-colsample-bytree", type=float, default=None)
    parser.add_argument("--xgb-reg-lambda", type=float, default=None)
    parser.add_argument("--torch-num-threads", type=int, default=0)
    parser.add_argument("--torch-num-interop-threads", type=int, default=0)
    parser.add_argument("--use-coral", action="store_true")
    parser.add_argument("--coral-reg", type=float, default=1e-3)
    parser.add_argument("--feature-scaling", choices=["none", "source_zscore"], default="none",
                        help=(
                            "Feature preprocessing before CORAL/SMOTE/model fitting. "
                            "source_zscore fits mean and standard deviation on the current source training "
                            "projects only, then transforms the pseudo-target or outer target."
                        ))
    parser.add_argument("--source-selection",
                        choices=["none", "top_k", "top_ratio", "random_k"],
                        default="none",
                        help="Select source projects by feature similarity, or use deterministic random_k.")
    parser.add_argument("--source-selection-top-k", type=int, default=10)
    parser.add_argument("--source-selection-top-ks", default="",
                        help="Comma-separated top_k values for source-selection grid search.")
    parser.add_argument("--source-selection-top-ratio", type=float, default=0.5)
    parser.add_argument("--source-selection-min-projects", type=int, default=3)
    parser.add_argument("--balance", default="SMOTE",
                        help="Comma-separated balance modes: SMOTE, undersampling, none")
    parser.add_argument("--smote-sampling-strategies", default="",
                        help="Comma-separated SMOTE sampling_strategy values; empty/auto keeps full SMOTE")
    parser.add_argument("--smote-sampling-strategy", dest="smote_sampling_strategies",
                        help="Alias for --smote-sampling-strategies when running a single value")
    parser.add_argument("--smote-k-neighbors", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=8,
                        help="Use -1 to leave stochastic components unseeded.")
    parser.add_argument("--random-states", default="",
                        help="Comma-separated random_state values for seed sensitivity runs.")
    parser.add_argument("--thresholds", default="0.6",
                        help="Comma-separated decision thresholds, e.g. 0.5,0.6")
    parser.add_argument("--threshold", dest="thresholds",
                        help="Alias for --thresholds when running a single threshold")
    parser.add_argument("--threshold-policy",
                        choices=["fixed", "source_only_cv"],
                        default="fixed",
                        help=(
                            "fixed: evaluate the supplied threshold(s). "
                            "source_only_cv: choose one threshold per target fold using only "
                            "leave-one-source-project-out validation inside the selected sources."
                        ))
    parser.add_argument("--source-only-tune", action="store_true",
                        help=(
                            "Jointly choose source-selection top_k and decision threshold for each "
                            "outer target fold using only source-project pseudo-target validation. "
                            "This overrides fixed/source_only_cv threshold selection for the final "
                            "outer prediction."
                        ))
    parser.add_argument("--source-only-xgb-tune", action="store_true",
                        help=(
                            "Choose one pre-specified XGBoost configuration and one decision threshold "
                            "per outer target fold using only leave-one-source-project-out validation "
                            "inside the fixed selected source set."
                        ))
    parser.add_argument("--source-only-xgb-candidate-set",
                        choices=sorted(SOURCE_ONLY_XGB_CANDIDATE_SETS),
                        default="conservative_v1",
                        help="Pre-specified XGBoost candidate set used by --source-only-xgb-tune.")
    parser.add_argument("--source-only-joint-tune", action="store_true",
                        help=(
                            "Jointly choose source-selection top_k, one pre-specified XGBoost "
                            "configuration, and one decision threshold for each outer target fold "
                            "using only leave-one-source-project-out pseudo-target validation."
                        ))
    parser.add_argument("--source-only-staged-tune", action="store_true",
                        help=(
                            "First choose source-selection top_k with a pre-specified calibration "
                            "XGBoost configuration and source-only pseudo-target validation, then "
                            "choose the deployed XGBoost configuration and threshold within the "
                            "selected sources, again using source-only validation."
                        ))
    parser.add_argument("--xgb-scale-pos-weights", default="",
                        help="Comma-separated XGBoost scale_pos_weight values; empty means unset")
    parser.add_argument("--xgb-scale-pos-weight", dest="xgb_scale_pos_weights",
                        help="Alias for --xgb-scale-pos-weights when running a single value")
    args = parser.parse_args()

    if sum((args.source_only_tune, args.source_only_xgb_tune,
            args.source_only_joint_tune, args.source_only_staged_tune)) > 1:
        parser.error(
            "The source-only tuning modes are mutually exclusive"
        )
    if (args.source_only_xgb_tune or args.source_only_joint_tune
            or args.source_only_staged_tune) and args.threshold_policy != "source_only_cv":
        parser.error(
            "--source-only-xgb-tune, --source-only-joint-tune, and --source-only-staged-tune require "
            "--threshold-policy source_only_cv"
        )

    if torch is not None and hasattr(torch, "set_num_threads") and args.torch_num_threads > 0:
        torch.set_num_threads(args.torch_num_threads)
    if torch is not None and hasattr(torch, "set_num_interop_threads") and args.torch_num_interop_threads > 0:
        torch.set_num_interop_threads(args.torch_num_interop_threads)
    if torch is not None and hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    if args.feature_mode == "fresh_codebert":
        # 重新从 final_code 提取 CodeBERT semantic representation。
        # 本路径不使用 FlakeFlagger 专家特征、token BoW、旧 semantic_representation 或全局 IG。
        experiment_data, feature_columns = build_fresh_codebert_semantic_data(
            dataset_path=args.codebert_data_path,
            model_path=args.codebert_model_path,
            batch_size=args.codebert_batch_size,
            max_length=args.codebert_max_length,
            use_fp16=args.codebert_fp16,
            cache_dir=args.codebert_cache_dir,
            local_files_only=args.codebert_local_files_only,
        )
        feature_label = "CodeBERT-Semantic"
    else:
        # 使用旧预处理结果中的 FlakeFlagger 专家特征 + 旧 semantic_representation。
        # 这里不重新提取 CodeBERT，便于对比“原融合特征 + CORAL”的影响。
        experiment_data, feature_columns = build_precomputed_fused_data(
            processed_data_path=args.precomputed_data_path,
            flakeflagger_features_path=args.flakeflagger_features_path,
        )
        feature_label = "CoDA-FTP-Fused"

    output_dir = args.output_dir
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    result_by_test_name_columns = [
        "cross_validation",
        "balance_type",
        "IG_min",
        "numTrees",
        "classifier",
        "features_structure",
        "project",
        "test_name",
        "y_true",
        "pred",
        "pred_prob",
        "threshold",
        "threshold_policy",
        "feature_scaling",
        "calibrated_threshold",
        "xgb_scale_pos_weight",
        "xgb_max_depth",
        "xgb_min_child_weight",
        "xgb_gamma",
        "xgb_subsample",
        "xgb_colsample_bytree",
        "xgb_reg_lambda",
        "source_selection",
        "source_selection_top_k",
        "source_selection_top_ratio",
        "source_selection_min_projects",
        "source_selection_num_projects",
        "smote_sampling_strategy",
        "smote_k_neighbors",
        "random_state",
        "Matrix_label",
    ]
    df_columns = ["Model", "cross_validation", "balance_type", "numTrees", "features_structure", "IG_min",
                  "num_satsifiedFeatures", "classifier", "threshold", "threshold_policy", "feature_scaling", "xgb_scale_pos_weight",
                  "xgb_max_depth", "xgb_min_child_weight", "xgb_gamma", "xgb_subsample",
                  "xgb_colsample_bytree", "xgb_reg_lambda", "source_selection", "source_selection_top_k", "source_selection_top_ratio",
                  "source_selection_min_projects", "source_selection_num_projects",
                  "smote_sampling_strategy", "smote_k_neighbors", "random_state",
                  "TP", "FN", "FP", "TN", "precision", "recall", "F1_score",
                  "AUC"]

    ##=========================================================##
    # 参数设置
    balance = parse_csv_strings(args.balance)
    classifier = parse_csv_classifiers(args.classifiers)
    treeSize = [250]
    minIGList = [0]
    thresholds_to_run = parse_csv_floats(args.thresholds)
    xgb_scale_pos_weights = (
        parse_csv_floats(args.xgb_scale_pos_weights)
        if args.xgb_scale_pos_weights and str(args.xgb_scale_pos_weights).strip()
        else [None]
    )
    source_selection_top_ks = (
        parse_csv_ints(args.source_selection_top_ks)
        if args.source_selection_top_ks and str(args.source_selection_top_ks).strip()
        else [args.source_selection_top_k]
    )
    source_selection_top_ks_for_runs = (
        [args.source_selection_top_k]
        if (args.source_only_tune or args.source_only_joint_tune or args.source_only_staged_tune)
        else source_selection_top_ks
    )
    random_states = (
        parse_csv_ints(args.random_states)
        if args.random_states and str(args.random_states).strip()
        else [args.random_state]
    )
    random_states = [normalize_random_state(random_state) for random_state in random_states]
    smote_sampling_strategies = parse_optional_csv_floats(args.smote_sampling_strategies)
    source_only_xgb_candidates = (
        get_source_only_xgb_candidates(args.source_only_xgb_candidate_set)
        if (args.source_only_xgb_tune or args.source_only_joint_tune
            or args.source_only_staged_tune) else None
    )
    ##=========================================================##

    for ig in minIGList:
        Path(output_dir + "IG_" + str(ig)).mkdir(parents=True, exist_ok=True)
        experiment_data_full = experiment_data.copy()

        result = pd.DataFrame(columns=df_columns)
        result_by_test_name = pd.DataFrame(columns=result_by_test_name_columns)

        for mintree in treeSize:
            for bal in balance:
                for cl in classifier:
                    for xgb_scale_pos_weight in xgb_scale_pos_weights:
                        for smote_sampling_strategy in smote_sampling_strategies:
                            if bal != "SMOTE" and smote_sampling_strategy is not None:
                                continue
                            for source_selection_top_k, random_state in itertools.product(
                                    source_selection_top_ks_for_runs,
                                    random_states,
                            ):

                                selected_features = feature_columns + ["flakyStatus", "test_name", "project"]
                                combined_data = get_only_specific_columns_V1(
                                    experiment_data_full,
                                    selected_features,
                                    ["flakyStatus", "test_name", "project"],
                                )
                                run_result_by_test_name = pd.DataFrame(columns=result_by_test_name_columns)

                                _, _, _, _, _, _, _, auc_score, run_result_by_test_name, calibration_metrics_df = predict_RF_perProject(
                                    combined_data, 'project', bal, cl, mintree, feature_label, ig, run_result_by_test_name,
                                    project_n_jobs=args.project_n_jobs,
                                    gpu_ids=args.gpu_ids,
                                    xgb_n_jobs=args.xgb_n_jobs,
                                    xgb_tree_method=args.xgb_tree_method or None,
                                    use_coral=args.use_coral,
                                    coral_reg=args.coral_reg,
                                    threshold=thresholds_to_run[0],
                                    threshold_policy=args.threshold_policy,
                                    threshold_candidates=thresholds_to_run,
                                    feature_scaling=args.feature_scaling,
                                    xgb_scale_pos_weight=xgb_scale_pos_weight,
                                    xgb_max_depth=args.xgb_max_depth,
                                    xgb_min_child_weight=args.xgb_min_child_weight,
                                    xgb_gamma=args.xgb_gamma,
                                    xgb_subsample=args.xgb_subsample,
                                    xgb_colsample_bytree=args.xgb_colsample_bytree,
                                    xgb_reg_lambda=args.xgb_reg_lambda,
                                    source_selection=args.source_selection,
                                    source_selection_top_k=source_selection_top_k,
                                    source_only_tune=args.source_only_tune,
                                    source_selection_top_k_candidates=source_selection_top_ks,
                                    source_only_xgb_tune=args.source_only_xgb_tune,
                                    source_only_xgb_candidates=source_only_xgb_candidates,
                                    source_only_joint_tune=args.source_only_joint_tune,
                                    source_only_staged_tune=args.source_only_staged_tune,
                                    source_selection_top_ratio=args.source_selection_top_ratio,
                                    source_selection_min_projects=args.source_selection_min_projects,
                                    smote_sampling_strategy=smote_sampling_strategy,
                                    smote_k_neighbors=args.smote_k_neighbors,
                                    random_state=random_state,
                                    checkpoint_dir=(
                                        output_dir + "IG_" + str(ig) + "/checkpoints"
                                        if (args.source_only_tune or args.source_only_xgb_tune
                                            or args.source_only_joint_tune or args.project_n_jobs > 1)
                                            or args.source_only_staged_tune
                                        else None
                                    ),
                                )

                                if args.source_only_tune:
                                    threshold_runs = [("source_only_nested_cv", run_result_by_test_name.copy())]
                                elif args.source_only_joint_tune:
                                    threshold_runs = [("source_only_joint_nested_cv", run_result_by_test_name.copy())]
                                elif args.source_only_staged_tune:
                                    threshold_runs = [("source_only_staged_nested_cv", run_result_by_test_name.copy())]
                                elif args.source_only_xgb_tune:
                                    threshold_runs = [("source_only_xgb_cv", run_result_by_test_name.copy())]
                                elif args.threshold_policy == "source_only_cv":
                                    threshold_runs = [("source_only_cv", run_result_by_test_name.copy())]
                                else:
                                    threshold_runs = []
                                    for threshold in thresholds_to_run:
                                        threshold_rows = run_result_by_test_name.copy()
                                        threshold_rows["threshold"] = threshold
                                        threshold_rows["threshold_policy"] = "fixed"
                                        threshold_rows["feature_scaling"] = args.feature_scaling
                                        threshold_rows["calibrated_threshold"] = threshold
                                        threshold_rows["source_selection"] = args.source_selection
                                        threshold_rows["source_selection_top_k"] = source_selection_top_k
                                        threshold_rows["source_selection_top_ratio"] = args.source_selection_top_ratio
                                        threshold_rows["source_selection_min_projects"] = args.source_selection_min_projects
                                        threshold_rows["smote_sampling_strategy"] = smote_sampling_strategy
                                        threshold_rows["smote_k_neighbors"] = args.smote_k_neighbors
                                        threshold_rows["random_state"] = random_state
                                        threshold_rows["pred"] = (threshold_rows["pred_prob"] >= threshold).astype(int)
                                        threshold_rows["Matrix_label"] = np.select(
                                            [
                                                (threshold_rows["y_true"] == 1) & (threshold_rows["pred"] == 1),
                                                (threshold_rows["y_true"] == 1) & (threshold_rows["pred"] == 0),
                                                (threshold_rows["y_true"] == 0) & (threshold_rows["pred"] == 1),
                                            ],
                                            ["TP", "FN", "FP"],
                                            default="TN",
                                        )
                                        threshold_runs.append((threshold, threshold_rows))

                                for threshold_display, threshold_rows in threshold_runs:
                                    dynamic_source_selection = (
                                        args.source_only_tune or args.source_only_joint_tune
                                        or args.source_only_staged_tune
                                    )
                                    effective_top_k_for_rows = (
                                        int(threshold_rows["source_selection_top_k"].iloc[0])
                                        if (not dynamic_source_selection)
                                        and "source_selection_top_k" in threshold_rows.columns
                                        and len(threshold_rows) > 0
                                        else source_selection_top_k
                                    )
                                    effective_threshold_policy_for_rows = (
                                        threshold_rows["threshold_policy"].iloc[0]
                                        if "threshold_policy" in threshold_rows.columns
                                        and len(threshold_rows) > 0 else args.threshold_policy
                                    )
                                    top_k_summary_for_result = (
                                        "source_only_nested_cv"
                                        if dynamic_source_selection
                                        else effective_top_k_for_rows
                                    )
                                    threshold_rows["source_selection"] = args.source_selection
                                    threshold_rows["feature_scaling"] = args.feature_scaling
                                    # Each parallel outer fold can choose a different k.  Preserve
                                    # the fold-level value emitted by run_one_project_fold instead
                                    # of broadcasting the first completed project's value.
                                    if not dynamic_source_selection:
                                        threshold_rows["source_selection_top_k"] = effective_top_k_for_rows
                                    threshold_rows["source_selection_top_ratio"] = args.source_selection_top_ratio
                                    threshold_rows["source_selection_min_projects"] = args.source_selection_min_projects
                                    threshold_rows["smote_sampling_strategy"] = smote_sampling_strategy
                                    threshold_rows["smote_k_neighbors"] = args.smote_k_neighbors
                                    threshold_rows["random_state"] = random_state

                                    TP = int((threshold_rows["Matrix_label"] == "TP").sum())
                                    FN = int((threshold_rows["Matrix_label"] == "FN").sum())
                                    FP = int((threshold_rows["Matrix_label"] == "FP").sum())
                                    TN = int((threshold_rows["Matrix_label"] == "TN").sum())
                                    accuracy, F1, Precision, Recall = get_scores(TN, FP, FN, TP)
                                    Precision = round(Precision * 100)
                                    Recall = round(Recall * 100)
                                    f1 = round(F1 * 100)

                                    result_by_test_name = pd.concat(
                                        [result_by_test_name, threshold_rows],
                                        ignore_index=True,
                                    )

                                    new_row = pd.Series([
                                        "PerProject", "PerProject", bal, mintree, feature_label, ig,
                                        combined_data.shape[1] - 3, cl, threshold_display,
                                        effective_threshold_policy_for_rows,
                                        args.feature_scaling,
                                        threshold_rows["xgb_scale_pos_weight"].iloc[0],
                                        threshold_rows["xgb_max_depth"].iloc[0],
                                        threshold_rows["xgb_min_child_weight"].iloc[0],
                                        threshold_rows["xgb_gamma"].iloc[0],
                                        threshold_rows["xgb_subsample"].iloc[0],
                                        threshold_rows["xgb_colsample_bytree"].iloc[0],
                                        threshold_rows["xgb_reg_lambda"].iloc[0],
                                        args.source_selection, top_k_summary_for_result,
                                        args.source_selection_top_ratio, args.source_selection_min_projects,
                                        "source_only_nested_cv"
                                        if args.source_only_tune else (
                                            "source_only_joint_nested_cv"
                                            if args.source_only_joint_tune else (
                                            "source_only_staged_nested_cv"
                                            if args.source_only_staged_tune else (
                                            int(threshold_rows["source_selection_num_projects"].iloc[0])
                                            if "source_selection_num_projects" in threshold_rows.columns
                                            and len(threshold_rows) > 0 else 0
                                            )
                                            )
                                        ),
                                        smote_sampling_strategy, args.smote_k_neighbors, random_state,
                                        TP, FN, FP, TN, Precision, Recall, f1, auc_score
                                    ], index=result.columns)
                                    result = pd.concat([result, new_row.to_frame().T], ignore_index=True)

                                    print("\n===== Final Summary =====")
                                    print(f"Method: {METHOD_NAME}")
                                    print(f"Feature set: {feature_label}")
                                    print(f"Classifier: {cl}")
                                    print(f"CORAL: {args.use_coral}, reg={args.coral_reg}")
                                    print(f"Feature scaling: {args.feature_scaling}")
                                    print(
                                        f"Source selection: {args.source_selection}, "
                                        f"top_k={top_k_summary_for_result}, "
                                        f"top_ratio={args.source_selection_top_ratio}, "
                                        f"min_projects={args.source_selection_min_projects}"
                                    )
                                    print(
                                        f"Balance: {bal}, smote_sampling_strategy={smote_sampling_strategy}, "
                                        f"smote_k_neighbors={args.smote_k_neighbors}, "
                                        f"threshold_policy={effective_threshold_policy_for_rows}, "
                                        f"threshold={threshold_display}, "
                                        f"xgb_scale_pos_weight={xgb_scale_pos_weight}"
                                    )
                                    print(
                                        "Classifier regularization/control args: "
                                        f"max_depth={args.xgb_max_depth}, "
                                        f"min_child_weight={args.xgb_min_child_weight}, "
                                        f"gamma={args.xgb_gamma}, "
                                        f"subsample={args.xgb_subsample}, "
                                        f"colsample_bytree={args.xgb_colsample_bytree}, "
                                        f"reg_lambda={args.xgb_reg_lambda}, "
                                        f"random_state={random_state}"
                                    )
                                    print(f"Output dir: {output_dir}IG_{ig}/")
                                    print(f"TP: {TP}, FN: {FN}, FP: {FP}, TN: {TN}")
                                    print(f"Precision: {Precision}%, Recall: {Recall}%, F1: {f1}%, AUC: {auc_score}%")
                                    print("=========================\n")

                                if (args.source_only_tune or args.source_only_xgb_tune
                                    or args.source_only_joint_tune
                                    or args.source_only_staged_tune
                                    or args.threshold_policy == "source_only_cv") and not calibration_metrics_df.empty:
                                    calibration_metrics_df.to_csv(
                                        output_dir + "IG_" + str(ig) + '/source_only_model_selection.csv',
                                        index=False,
                                    )

                    result_by_test_name.to_csv(
                        output_dir + "IG_" + str(ig) + '/prediction_result_per_test.csv',
                        index=False,
                    )
                    result.to_csv(output_dir + "IG_" + str(ig) + '/prediction_result.csv', index=False)

                    confusion_matrix_by_project = generateConfusionMatrixByProject(result_by_test_name)
                    confusion_matrix_by_project.to_csv(
                        output_dir + "IG_" + str(ig) + '/prediction_result_by_project.csv',
                        index=False,
                    )

    print("Execution time: %s seconds" % (time.time() - execution_time))

#!/usr/bin/env python3
"""Zero-shot Ollama LLM baseline for flaky test prediction.

The runner is intentionally conservative:
- no training examples are provided to the model;
- only Java test code is shown;
- predictions are written incrementally so the run can resume.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score


PROMPT_TEMPLATE = """You are given a Java test case.

A flaky test is a test that may pass or fail nondeterministically without any code changes.
A non-flaky test is a test whose result is expected to be deterministic under the same code and environment.

Predict whether the given test case is flaky or non-flaky.

Return exactly one label and do not explain your answer:
flaky
non-flaky

Test code:
{code}
"""

FEWSHOT_PROMPT_TEMPLATE = """You are given Java test cases labeled as flaky or non-flaky.

A flaky test is a test that may pass or fail nondeterministically without any code changes.
A non-flaky test is a test whose result is expected to be deterministic under the same code and environment.

Examples:

{examples}

Now predict the label of the following Java test case.

Return exactly one label and do not explain your answer:
flaky
non-flaky

Test code:
{code}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", default="dataset_for_semantic_extract/FlakeFlagger/FlakeFlagger_dataset.csv")
    parser.add_argument("--model", default="qwen2.5-coder:1.5b")
    parser.add_argument("--output-dir", default="result_1/llm_baselines/qwen2_5_coder_1_5b_full")
    parser.add_argument("--code-column", default="final_code")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434/api/generate")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--num-predict", type=int, default=8)
    parser.add_argument("--think", choices=["true", "false", "default"], default="false")
    parser.add_argument("--request-timeout", type=int, default=180)
    parser.add_argument("--max-chars", type=int, default=8000)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--prompt-mode", choices=["zero-shot", "few-shot"], default="zero-shot")
    parser.add_argument("--fewshot-flaky", type=int, default=2)
    parser.add_argument("--fewshot-nonflaky", type=int, default=2)
    parser.add_argument("--fewshot-random-state", type=int, default=42)
    return parser.parse_args()


def normalize_label(text: str) -> tuple[int, str, bool]:
    cleaned = text.strip().lower()
    cleaned = re.sub(r"[^a-z\\-\\s]", " ", cleaned)
    tokens = cleaned.split()
    joined = " ".join(tokens)
    if joined == "flaky" or (tokens and tokens[0] == "flaky"):
        return 1, "flaky", False
    if joined == "non-flaky" or joined == "non flaky" or (tokens[:2] == ["non", "flaky"]):
        return 0, "non-flaky", False
    if "non-flaky" in cleaned or "non flaky" in cleaned:
        return 0, "non-flaky", False
    if re.search(r"\\bflaky\\b", cleaned):
        return 1, "flaky", False
    return 0, "invalid->non-flaky", True


def build_fewshot_examples(
    dataset: pd.DataFrame,
    target_project: str,
    args: argparse.Namespace,
) -> tuple[str, list[dict]]:
    source = dataset[dataset["project"].astype(str) != str(target_project)]
    flaky = source[source["flaky"].astype(int).eq(1)]
    nonflaky = source[source["flaky"].astype(int).eq(0)]
    flaky_sample = flaky.sample(n=min(args.fewshot_flaky, len(flaky)), random_state=args.fewshot_random_state)
    nonflaky_sample = nonflaky.sample(n=min(args.fewshot_nonflaky, len(nonflaky)), random_state=args.fewshot_random_state + 1)
    examples_df = pd.concat([flaky_sample, nonflaky_sample], ignore_index=False)
    blocks = []
    records = []
    for idx, (_, row) in enumerate(examples_df.iterrows(), start=1):
        label = "flaky" if int(row["flaky"]) == 1 else "non-flaky"
        code = str(row.get(args.code_column, "") or "")[: args.max_chars]
        blocks.append(f"Example {idx}:\nTest code:\n{code}\nLabel: {label}")
        records.append(
            {
                "target_project": target_project,
                "example_id": idx,
                "example_project": row.get("project", ""),
                "example_class_name": row.get("class_name", ""),
                "example_test_name": row.get("test_name", ""),
                "example_label": label,
            }
        )
    return "\n\n".join(blocks), records


def build_fewshot_context(dataset: pd.DataFrame, args: argparse.Namespace, output_dir: Path) -> dict[str, str]:
    blocks = {}
    records = []
    for project in sorted(dataset["project"].dropna().astype(str).unique()):
        block, project_records = build_fewshot_examples(dataset, project, args)
        blocks[project] = block
        records.extend(project_records)
    if records:
        output_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(records).to_csv(output_dir / "fewshot_examples_by_project.csv", index=False)
    return blocks


def build_prompt(args: argparse.Namespace, code: str, example_block: str = "") -> str:
    code = code[: args.max_chars]
    if args.prompt_mode == "few-shot":
        return FEWSHOT_PROMPT_TEMPLATE.format(examples=example_block, code=code)
    return PROMPT_TEMPLATE.format(code=code)


def call_ollama(args: argparse.Namespace, prompt: str) -> tuple[str, float]:
    payload = {
        "model": args.model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": args.temperature,
            "num_predict": args.num_predict,
        },
    }
    if args.think != "default":
        payload["think"] = args.think == "true"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(args.ollama_url, data=data, headers={"Content-Type": "application/json"})
    start = time.time()
    with urllib.request.urlopen(req, timeout=args.request_timeout) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    return str(body.get("response", "")), time.time() - start


def safe_call_ollama(args: argparse.Namespace, prompt: str, retries: int = 2) -> tuple[str, float, str]:
    total_latency = 0.0
    last_error = ""
    for attempt in range(retries + 1):
        try:
            response, latency = call_ollama(args, prompt)
            return response, latency + total_latency, ""
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = repr(exc)
            total_latency += 0.0
            time.sleep(2 + attempt)
    return "", total_latency, last_error


def load_completed(path: Path) -> set[int]:
    if not path.exists():
        return set()
    completed = set()
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            try:
                completed.add(int(row["sample_id"]))
            except (KeyError, ValueError):
                pass
    return completed


def append_row(path: Path, fieldnames: list[str], row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)
        fh.flush()


def predict_one(args: argparse.Namespace, sample_id: int, row: dict, fewshot_blocks: dict[str, str] | None = None) -> dict:
    code = str(row.get(args.code_column, "") or "")
    project = str(row.get("project", ""))
    example_block = fewshot_blocks.get(project, "") if fewshot_blocks else ""
    prompt = build_prompt(args, code, example_block)
    response, latency, error = safe_call_ollama(args, prompt)
    pred, pred_label, invalid = normalize_label(response)
    return {
        "sample_id": int(sample_id),
        "model": args.model,
        "project": row.get("project", ""),
        "class_name": row.get("class_name", ""),
        "test_name": row.get("test_name", ""),
        "y_true": int(row.get("flaky", 0)),
        "pred": int(pred),
        "pred_label": pred_label,
        "invalid": int(invalid),
        "latency_sec": round(latency, 6),
        "error": error,
        "raw_response": response.replace("\n", " ")[:500],
    }


def write_summaries(output_dir: Path, per_test_path: Path) -> None:
    df = pd.read_csv(per_test_path)
    y_true = df["y_true"].astype(int)
    y_pred = df["pred"].astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    invalid_rate = float(df["invalid"].mean()) if len(df) else 0.0
    avg_latency = float(df["latency_sec"].mean()) if len(df) else 0.0

    summary = pd.DataFrame(
        [
            {
                "model": df["model"].iloc[0] if len(df) else "",
                "tests": len(df),
                "TP": int(tp),
                "FN": int(fn),
                "FP": int(fp),
                "TN": int(tn),
                "Precision": precision,
                "Recall": recall,
                "F1": f1,
                "InvalidRate": invalid_rate,
                "AvgLatencySec": avg_latency,
            }
        ]
    )
    summary.to_csv(output_dir / "prediction_result.csv", index=False)

    rows = []
    for project, group in df.groupby("project"):
        yy = group["y_true"].astype(int)
        pp = group["pred"].astype(int)
        tn, fp, fn, tp = confusion_matrix(yy, pp, labels=[0, 1]).ravel()
        p = precision_score(yy, pp, zero_division=0)
        r = recall_score(yy, pp, zero_division=0)
        f = f1_score(yy, pp, zero_division=0)
        rows.append(
            {
                "project": project,
                "tests": len(group),
                "TP": int(tp),
                "FN": int(fn),
                "FP": int(fp),
                "TN": int(tn),
                "Precision": p,
                "Recall": r,
                "F1": f,
                "InvalidRate": float(group["invalid"].mean()),
                "AvgLatencySec": float(group["latency_sec"].mean()),
            }
        )
    pd.DataFrame(rows).to_csv(output_dir / "prediction_result_by_project.csv", index=False)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    per_test_path = output_dir / "prediction_result_per_test.csv"
    dataset = pd.read_csv(args.dataset_path)
    fewshot_blocks = build_fewshot_context(dataset, args, output_dir) if args.prompt_mode == "few-shot" else None
    if args.limit is not None:
        dataset = dataset.head(args.limit)

    completed = load_completed(per_test_path)
    fields = [
        "sample_id",
        "model",
        "project",
        "class_name",
        "test_name",
        "y_true",
        "pred",
        "pred_label",
        "invalid",
        "latency_sec",
        "error",
        "raw_response",
    ]

    total = len(dataset)
    pending_samples = [
        (int(sample_id), row.to_dict())
        for sample_id, row in dataset.iterrows()
        if int(sample_id) not in completed
    ]

    if args.workers <= 1:
        for sample_id, row in pending_samples:
            result = predict_one(args, sample_id, row, fewshot_blocks)
            append_row(per_test_path, fields, result)
            completed.add(sample_id)
            done = len(completed)
            if done % 50 == 0 or done == total:
                print(f"Processed {done}/{total}", flush=True)
                write_summaries(output_dir, per_test_path)
            if args.sleep:
                time.sleep(args.sleep)
    else:
        print(f"Running with workers={args.workers}", flush=True)
        iterator = iter(pending_samples)
        futures = {}
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            for _ in range(args.workers):
                try:
                    sample_id, row = next(iterator)
                except StopIteration:
                    break
                futures[executor.submit(predict_one, args, sample_id, row, fewshot_blocks)] = sample_id

            while futures:
                done_futures, _ = wait(futures, return_when=FIRST_COMPLETED)
                for future in done_futures:
                    sample_id = futures.pop(future)
                    result = future.result()
                    append_row(per_test_path, fields, result)
                    completed.add(sample_id)
                    done = len(completed)
                    if done % 50 == 0 or done == total:
                        print(f"Processed {done}/{total}", flush=True)
                        write_summaries(output_dir, per_test_path)
                    if args.sleep:
                        time.sleep(args.sleep)
                    try:
                        next_sample_id, next_row = next(iterator)
                    except StopIteration:
                        continue
                    futures[executor.submit(predict_one, args, next_sample_id, next_row, fewshot_blocks)] = next_sample_id

    write_summaries(output_dir, per_test_path)
    print(f"Done. Results saved to {output_dir}", flush=True)


if __name__ == "__main__":
    main()

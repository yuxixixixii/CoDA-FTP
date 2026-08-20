#!/usr/bin/env python3
"""Verify the archived revision aggregates without the large fused CSV."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "revision" / "results"


def totals(frame: pd.DataFrame) -> dict[str, int]:
    return {key: int(frame[key].sum()) for key in ("TP", "FN", "FP", "TN")}


def assert_totals(label: str, observed: dict[str, int], expected: dict[str, int]) -> None:
    if observed != expected:
        raise SystemExit(f"{label} mismatch: observed {observed}; expected {expected}")
    print(f"{label}: TP={observed['TP']}, FN={observed['FN']}, FP={observed['FP']}, TN={observed['TN']}")


def main() -> None:
    primary = pd.read_csv(RESULTS / "rq1" / "coda_ftp" / "prediction_result.csv").iloc[0]
    assert_totals(
        "CoDA-FTP",
        {key: int(primary[key]) for key in ("TP", "FN", "FP", "TN")},
        {"TP": 375, "FN": 349, "FP": 228, "TN": 20461},
    )

    baseline_root = RESULTS / "rq1" / "baselines"
    for label, directory, expected in (
        ("Flakify (strict)", "flakify_strict", {"TP": 261, "FN": 463, "FP": 784, "TN": 19905}),
        ("DeepFlaky (strict)", "deepflaky_strict", {"TP": 419, "FN": 305, "FP": 2429, "TN": 18260}),
    ):
        rows = [pd.read_csv(path) for path in sorted((baseline_root / directory).glob("*/outer_metrics.csv"))]
        if len(rows) != 23:
            raise SystemExit(f"{label} archive has {len(rows)} outer folds; expected 23")
        assert_totals(label, totals(pd.concat(rows, ignore_index=True)), expected)

    traditional = pd.read_csv(baseline_root / "flakeflagger_source_only_by_project.csv")
    for label, method, expected in (
        ("FlakeFlagger (source-only)", "flakeflagger", {"TP": 225, "FN": 499, "FP": 2406, "TN": 18283}),
        ("Vocabulary + FlakeFlagger (source-only)", "vocabulary_flakeflagger", {"TP": 163, "FN": 561, "FP": 470, "TN": 20219}),
    ):
        assert_totals(label, totals(traditional[traditional["baseline"] == method]), expected)

    print("All archived RQ1 checks passed.")


if __name__ == "__main__":
    main()

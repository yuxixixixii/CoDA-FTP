# Supplementary prompt-only LLM reference protocol

The LLM rows are intentionally supplementary references. They are not included
in the main baseline table, paired tests, or comparative RQ1 claim because they
do not fine-tune on source data, use retrieval, select a threshold, or perform
matched hyperparameter selection.

For all archived runs, temperature was 0, output was parsed as a binary label,
and the test-method input was truncated to 8,000 characters. Four-shot runs use
two flaky and two non-flaky examples drawn only from source projects with seed
42; the held-out target labels are never used to choose those examples.

| Archive file | Prompting | Model tag recorded at inference |
| --- | --- | --- |
| `qwen2_5_coder_1_5b_zero_shot.csv` | zero-shot | `qwen2.5-coder:1.5b` |
| `qwen2_5_coder_7b_zero_shot.csv` | zero-shot | `qwen2.5-coder:7b` |
| `qwen3_6_35b_zero_shot.csv` | zero-shot | `qwen3.6:35b` |
| `qwen2_5_coder_1_5b_four_shot.csv` | source-only four-shot | `qwen2.5-coder:1.5b` |
| `qwen2_5_coder_7b_four_shot.csv` | source-only four-shot | `qwen2.5-coder:7b` |
| `qwen3_6_35b_four_shot.csv` | source-only four-shot | `qwen3.6:35b` |
| `qwen3_coder_next_four_shot.csv` | source-only four-shot | `qwen3-coder-next:latest` |

The `qwen3-coder-next:latest` tag is mutable and its historical model digest was
not recorded. It is therefore an observational snapshot from the revision
period, not an exactly rerunnable model identity. The values should be read
only as bounded prompt-only reference evidence. Pre-training contamination also
cannot be excluded because the public test code may have appeared in training
data.

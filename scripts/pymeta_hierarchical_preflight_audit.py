#!/usr/bin/env python3
"""Read-only comprehensive PyMETA hierarchical preflight and tokenization audit."""

import collections
import json
import sys
from pathlib import Path

from transformers import AutoTokenizer

import pymeta_hierarchical_training_engine as engine
from pymeta_hierarchical_collator import HierarchicalDataCollator


def audit_dataset(path, expected_rows, allowed_labels, tokenizer, max_length):
    records = engine.load_records(path, expected_rows, allowed_labels)
    counter = collections.Counter()
    lengths = []
    prompt_lengths = []
    supervised_lengths = []
    over_limit = []

    audit_collator = HierarchicalDataCollator(
        tokenizer,
        max_length=None,  # Measure all rows first; never truncate.
    )

    for row_index, record in enumerate(records):
        target = json.loads(record["messages"][2]["content"])
        counter[target["label"]] += 1

        input_ids, labels = audit_collator._prepare_example(record)
        sequence_length = len(input_ids)
        prompt_length = sum(token == -100 for token in labels)
        supervised_length = sum(token != -100 for token in labels)

        lengths.append(sequence_length)
        prompt_lengths.append(prompt_length)
        supervised_lengths.append(supervised_length)

        if sequence_length > max_length:
            over_limit.append({
                "row_index": row_index,
                "sequence_length": sequence_length,
            })
        if supervised_length <= 0:
            raise RuntimeError(
                f"No assistant-supervised tokens in {path}, row {row_index}."
            )
        if (row_index + 1) % 2000 == 0:
            print(
                f"Tokenization audit {Path(path).name}: "
                f"{row_index + 1}/{len(records)}",
                flush=True,
            )

    return records, {
        "file": str(path),
        "rows": len(records),
        "label_counts": dict(counter),
        "min_sequence_tokens": min(lengths),
        "max_sequence_tokens": max(lengths),
        "mean_sequence_tokens": sum(lengths) / len(lengths),
        "max_prompt_tokens": max(prompt_lengths),
        "min_supervised_assistant_tokens": min(supervised_lengths),
        "max_supervised_assistant_tokens": max(supervised_lengths),
        "rows_over_max_length": over_limit,
        "truncation_enabled": False,
        "assistant_only_loss": True,
    }


def main():
    # This is a CPU-safe tokenizer/data audit. Validate the frozen experiment
    # specification directly; do not gate it on CUDA availability or GPU VRAM.
    spec = json.loads(engine.SPEC_PATH.read_text(encoding="utf-8"))
    engine.verify_experiment_specification(spec)

    if not engine.TOKENIZER_PATH.is_dir():
        raise FileNotFoundError(
            f"Missing local tokenizer directory: {engine.TOKENIZER_PATH}"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        engine.TOKENIZER_PATH,
        local_files_only=True,
    )
    if tokenizer.pad_token_id is None:
        raise RuntimeError("Tokenizer pad_token_id is missing.")
    if not getattr(tokenizer, "chat_template", None):
        raise RuntimeError("Tokenizer chat_template is missing.")

    dataset_reports = {}
    loaded = {}

    for stage_name in ("stage1", "stage2"):
        stage_spec = spec[stage_name]
        stage_config = engine.STAGES[stage_name]

        for split in ("train", "validation"):
            path = stage_config[
                "train_path" if split == "train" else "validation_path"
            ]
            expected_rows = stage_config[
                "train_rows" if split == "train" else "validation_rows"
            ]
            expected_hash = stage_config[
                "train_sha256" if split == "train" else "validation_sha256"
            ]
            actual_hash = engine.verify_file(path, expected_hash)

            records, report = audit_dataset(
                path,
                expected_rows,
                stage_config["classes"],
                tokenizer,
                engine.MAX_LENGTH,
            )

            expected_counts = stage_spec[f"{split}_counts"]
            if report["label_counts"] != expected_counts:
                raise RuntimeError(
                    f"{stage_name} {split} label counts differ from the spec."
                )

            if report["rows_over_max_length"]:
                first = report["rows_over_max_length"][0]
                raise RuntimeError(
                    f"{stage_name} {split} contains a sequence above "
                    f"{engine.MAX_LENGTH} tokens: {first}"
                )

            report["sha256"] = actual_hash
            report["maximum_allowed_tokens"] = engine.MAX_LENGTH
            dataset_reports[f"{stage_name}_{split}"] = report
            loaded[f"{stage_name}_{split}"] = records

            print(
                f"{stage_name} {split}: PASS | "
                f"rows={report['rows']} | "
                f"max_tokens={report['max_sequence_tokens']} | "
                f"labels={report['label_counts']}"
            )

    # Verify that Stage 2 is precisely the ordered Non-Explicit subset of Stage 1.
    stage1_train = loaded["stage1_train"]
    stage2_train = loaded["stage2_train"]
    stage1_train_non_explicit = [
        row for row in stage1_train
        if json.loads(row["messages"][2]["content"])["label"] == "Non-Explicit"
    ]
    if len(stage1_train_non_explicit) != len(stage2_train):
        raise RuntimeError("Stage 2 train row count is not the Stage 1 Non-Explicit subset.")
    if any(
        left["messages"][1]["content"] != right["messages"][1]["content"]
        for left, right in zip(stage1_train_non_explicit, stage2_train)
    ):
        raise RuntimeError("Stage 2 train prompt order/content differs from Stage 1 subset.")

    stage1_dev = loaded["stage1_validation"]
    stage2_dev = loaded["stage2_validation"]
    stage1_dev_non_explicit = [
        row for row in stage1_dev
        if json.loads(row["messages"][2]["content"])["label"] == "Non-Explicit"
    ]
    if len(stage1_dev_non_explicit) != len(stage2_dev):
        raise RuntimeError("Stage 2 validation row count is not the Stage 1 Non-Explicit subset.")
    if any(
        left["messages"][1]["content"] != right["messages"][1]["content"]
        for left, right in zip(stage1_dev_non_explicit, stage2_dev)
    ):
        raise RuntimeError("Stage 2 validation prompt order/content differs from Stage 1 subset.")

    stage2_system_prompts = {
        row["messages"][0]["content"] for row in stage2_dev
    }
    if len(stage2_system_prompts) != 1:
        raise RuntimeError("Stage 2 validation system prompts are inconsistent.")
    stage2_system_prompt = next(iter(stage2_system_prompts))

    # Cascade Stage 2 may receive any dev row routed by Stage 1, including
    # gold Explicit Error rows. Audit all 4,863 possible routed prompts.
    stage2_cascade_prompt_lengths = []
    for row_index, stage1_record in enumerate(stage1_dev):
        stage2_messages = [
            {"role": "system", "content": stage2_system_prompt},
            {
                "role": "user",
                "content": stage1_record["messages"][1]["content"],
            },
        ]
        prompt_text = tokenizer.apply_chat_template(
            stage2_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_ids = tokenizer(
            prompt_text,
            add_special_tokens=False,
            truncation=False,
        )["input_ids"]
        prompt_length = len(prompt_ids)
        stage2_cascade_prompt_lengths.append(prompt_length)
        if prompt_length > engine.MAX_LENGTH:
            raise RuntimeError(
                f"Stage 2 cascade prompt row {row_index} has {prompt_length} "
                f"tokens, exceeding {engine.MAX_LENGTH}."
            )
        if (row_index + 1) % 1000 == 0:
            print(
                f"Stage 2 cascade prompt audit: {row_index + 1}/{len(stage1_dev)}",
                flush=True,
            )

    report = {
        "project": "PyMETA",
        "audit": "hierarchical_tokenization_and_data_pipeline",
        "status": "PASS",
        "max_sequence_length": engine.MAX_LENGTH,
        "truncation": False,
        "padding": "dynamic right padding",
        "loss_masking": "system and user prompt tokens masked; assistant response supervised",
        "stage2_subset_integrity": "PASS",
        "stage2_all_dev_cascade_prompts": {
            "rows_checked": len(stage2_cascade_prompt_lengths),
            "max_prompt_tokens": max(stage2_cascade_prompt_lengths),
            "rows_over_max_length": 0,
            "max_new_tokens": engine.MAX_NEW_TOKENS,
        },
        "datasets": dataset_reports,
        "test_records_accessed": False,
    }

    output_path = (
        engine.ROOT / "outputs" / "hierarchical_preflight_audit.json"
    )
    if output_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing audit report: {output_path}"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nALL HIERARCHICAL TOKENIZATION AND DATA AUDITS PASSED")
    print("Audit report:", output_path)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nPREFLIGHT AUDIT FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise

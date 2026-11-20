#!/usr/bin/env python3
"""Combine measured Stage 1 and Stage 2 runtime records after both stages finish."""

import json
from pathlib import Path

import pymeta_hierarchical_training_engine as engine


def read_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def main():
    summaries = {}
    for stage_name in ("stage1", "stage2"):
        path = engine.STAGES[stage_name]["output_dir"] / "training_summary.json"
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing completed training summary for {stage_name}: {path}"
            )

        summary = read_json(path)
        if summary.get("stage") != stage_name:
            raise RuntimeError(f"Training summary stage mismatch: {path}")
        if summary.get("completed_epochs") != engine.EPOCHS:
            raise RuntimeError(f"{stage_name} training is not fully complete.")
        if summary.get("test_records_accessed") is not False:
            raise RuntimeError(f"{stage_name} summary does not certify test isolation.")

        elapsed = summary.get("elapsed_training_seconds")
        if not isinstance(elapsed, (int, float)) or elapsed < 0:
            raise RuntimeError(f"Invalid elapsed time in {path}")

        summaries[stage_name] = summary

    stage1_seconds = float(summaries["stage1"]["elapsed_training_seconds"])
    stage2_seconds = float(summaries["stage2"]["elapsed_training_seconds"])
    total_seconds = stage1_seconds + stage2_seconds

    report = {
        "project": "PyMETA",
        "record": "measured_hierarchical_training_time",
        "stage1_seconds": stage1_seconds,
        "stage1_hours": stage1_seconds / 3600.0,
        "stage2_seconds": stage2_seconds,
        "stage2_hours": stage2_seconds / 3600.0,
        "total_training_seconds": total_seconds,
        "total_training_hours": total_seconds / 3600.0,
        "stage1_successful_optimizer_updates": summaries["stage1"][
            "successful_optimizer_updates"
        ],
        "stage2_successful_optimizer_updates": summaries["stage2"][
            "successful_optimizer_updates"
        ],
        "total_successful_optimizer_updates": (
            summaries["stage1"]["successful_optimizer_updates"]
            + summaries["stage2"]["successful_optimizer_updates"]
        ),
        "stage1_overflow_attempts": summaries["stage1"]["skipped_overflow_attempts"],
        "stage2_overflow_attempts": summaries["stage2"]["skipped_overflow_attempts"],
        "timing_scope": (
            "Sum of per-stage accumulated active process time recorded by the "
            "training engine. Abruptly interrupted work after the last verified "
            "checkpoint may not be included."
        ),
        "test_records_accessed": False,
    }

    output_path = engine.ROOT / "outputs" / "hierarchical_training_time_report.json"
    if output_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing training-time report: {output_path}"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("HIERARCHICAL TRAINING TIME REPORT")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print("Research record:", output_path)


if __name__ == "__main__":
    main()

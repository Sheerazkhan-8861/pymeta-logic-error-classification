#!/usr/bin/env python3
"""CPU-only unit tests for PyMETA cascade metrics and deterministic selection."""

import json

from pymeta_hierarchical_cascade_evaluator import (
    THREE_CLASSES,
    compute_three_class_metrics,
    select_best_pair,
    strict_parse,
    validate_full_dev_inputs,
)


def main():
    assert strict_parse('{"label": "Logic Error"}', THREE_CLASSES) == "Logic Error"
    assert strict_parse('{"label": "Logic Error", "extra": 1}', THREE_CLASSES) is None
    assert strict_parse("not-json", THREE_CLASSES) is None
    assert strict_parse('{"label": "Unknown"}', THREE_CLASSES) is None
    print("TEST 1 — Strict JSON parsing: PASS")

    gold = [
        "No Error",
        "Logic Error",
        "Explicit Error",
        "No Error",
        "Explicit Error",
        "Logic Error",
    ]
    predictions = [
        "Logic Error",
        "Logic Error",
        None,
        "No Error",
        "Logic Error",
        "Explicit Error",
    ]
    metrics = compute_three_class_metrics(gold, predictions)

    assert metrics["invalid_predictions"] == 1
    assert abs(metrics["invalid_output_rate"] - (1 / 6)) < 1e-12
    assert abs(metrics["LER_over"] - 0.5) < 1e-12
    assert metrics["confusion_matrix"] == [
        [1, 1, 0],
        [0, 1, 1],
        [0, 1, 0],
    ]
    assert metrics["gold_support"] == {
        "No Error": 2,
        "Logic Error": 2,
        "Explicit Error": 2,
    }
    print("TEST 2 — Three-class metrics and invalid-output handling: PASS")

    candidates = [
        {
            "pair": "a",
            "order_index": 0,
            "metrics": {
                "macro_f1": 0.80,
                "LER_over": 0.20,
                "logic_error_recall": 0.70,
            },
        },
        {
            "pair": "b",
            "order_index": 1,
            "metrics": {
                "macro_f1": 0.80,
                "LER_over": 0.15,
                "logic_error_recall": 0.60,
            },
        },
        {
            "pair": "c",
            "order_index": 2,
            "metrics": {
                "macro_f1": 0.80,
                "LER_over": 0.15,
                "logic_error_recall": 0.75,
            },
        },
        {
            "pair": "d",
            "order_index": 3,
            "metrics": {
                "macro_f1": 0.80,
                "LER_over": 0.15,
                "logic_error_recall": 0.75,
            },
        },
    ]
    assert select_best_pair(candidates)["pair"] == "c"
    print("TEST 3 — Macro-F1 / LER_over / Logic recall tie-breaks: PASS")

    flat = []
    stage1 = []
    stage2 = []
    ordered_gold = (
        ["No Error"] * 1875
        + ["Logic Error"] * 1495
        + ["Explicit Error"] * 1493
    )
    stage2_system = "Stage 2 system prompt"
    for row_index, label in enumerate(ordered_gold):
        user_prompt = f"prompt-{row_index}"
        flat.append({
            "messages": [
                {"role": "system", "content": "flat system"},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": json.dumps({"label": label})},
            ]
        })
        stage1_label = "Explicit Error" if label == "Explicit Error" else "Non-Explicit"
        stage1.append({
            "messages": [
                {"role": "system", "content": "stage1 system"},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": json.dumps({"label": stage1_label})},
            ]
        })
        if label != "Explicit Error":
            stage2.append({
                "messages": [
                    {"role": "system", "content": stage2_system},
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": json.dumps({"label": label})},
                ]
            })

    validated_gold, returned_stage2_system = validate_full_dev_inputs(
        flat, stage1, stage2
    )
    assert validated_gold == ordered_gold
    assert returned_stage2_system == stage2_system
    print("TEST 4 — Full-dev cascade split and target alignment: PASS")

    print("\nALL CASCADE METRIC TESTS PASSED")


if __name__ == "__main__":
    main()

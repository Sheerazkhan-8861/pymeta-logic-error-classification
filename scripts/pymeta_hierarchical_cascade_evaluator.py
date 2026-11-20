#!/usr/bin/env python3
"""Full-original-dev cascade evaluation and deterministic epoch-pair selection."""

import argparse
import collections
import gc
import json
import shutil
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

import pymeta_hierarchical_training_engine as engine


THREE_CLASSES = ["No Error", "Logic Error", "Explicit Error"]
STAGE1_CLASSES = ["Explicit Error", "Non-Explicit"]
STAGE2_CLASSES = ["Logic Error", "No Error"]


def read_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".temporary")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    temporary.replace(path)


def validate_full_dev_inputs(flat_records, stage1_records, stage2_records):
    if len(flat_records) != 4863 or len(stage1_records) != 4863:
        raise RuntimeError("Cascade evaluation requires all 4,863 official dev rows.")
    if len(stage2_records) != 3370:
        raise RuntimeError("Stage 2 validation must contain 3,370 gold Non-Explicit rows.")

    gold = []
    for index, record in enumerate(flat_records):
        messages = record.get("messages")
        if not isinstance(messages, list) or len(messages) != 3:
            raise RuntimeError(f"Invalid full-dev record at row {index}.")
        target = json.loads(messages[2]["content"])
        if set(target) != {"label"} or target["label"] not in THREE_CLASSES:
            raise RuntimeError(f"Invalid three-class gold label at row {index}.")
        gold.append(target["label"])

        if messages[1]["content"] != stage1_records[index]["messages"][1]["content"]:
            raise RuntimeError(
                f"Full-dev and Stage 1 prompt mismatch at row {index}."
            )

    expected_gold_counts = {
        "No Error": 1875,
        "Logic Error": 1495,
        "Explicit Error": 1493,
    }
    if dict(collections.Counter(gold)) != expected_gold_counts:
        raise RuntimeError(
            "Full-dev three-class label counts differ from the frozen dataset audit."
        )

    for row_index, (flat_record, stage1_record) in enumerate(
        zip(flat_records, stage1_records)
    ):
        flat_label = json.loads(flat_record["messages"][2]["content"])["label"]
        stage1_label = json.loads(stage1_record["messages"][2]["content"])["label"]
        expected_stage1_label = (
            "Explicit Error" if flat_label == "Explicit Error" else "Non-Explicit"
        )
        if stage1_label != expected_stage1_label:
            raise RuntimeError(
                f"Stage 1 validation target mismatch with original dev at row {row_index}."
            )

    stage1_non_explicit = [
        record
        for record in stage1_records
        if json.loads(record["messages"][2]["content"])["label"] == "Non-Explicit"
    ]
    if len(stage1_non_explicit) != len(stage2_records):
        raise RuntimeError("Stage 2 validation is not the full Non-Explicit dev subset.")

    flat_non_explicit = [
        record
        for record in flat_records
        if json.loads(record["messages"][2]["content"])["label"] != "Explicit Error"
    ]
    for index, (stage1_record, stage2_record, flat_record) in enumerate(
        zip(stage1_non_explicit, stage2_records, flat_non_explicit)
    ):
        if stage1_record["messages"][1]["content"] != stage2_record["messages"][1]["content"]:
            raise RuntimeError(
                f"Stage 2 validation prompt order mismatch at subset row {index}."
            )
        stage2_label = json.loads(stage2_record["messages"][2]["content"])["label"]
        flat_label = json.loads(flat_record["messages"][2]["content"])["label"]
        if stage2_label != flat_label:
            raise RuntimeError(
                f"Stage 2 validation target mismatch with original dev at subset row {index}."
            )

    stage2_systems = {
        record["messages"][0]["content"] for record in stage2_records
    }
    if len(stage2_systems) != 1:
        raise RuntimeError("Stage 2 validation has inconsistent system prompts.")

    return gold, next(iter(stage2_systems))


def strict_parse(text, allowed):
    try:
        value = json.loads(text.strip())
    except (json.JSONDecodeError, TypeError, AttributeError):
        return None
    if not isinstance(value, dict) or set(value) != {"label"}:
        return None
    if value["label"] not in allowed:
        return None
    return value["label"]


def load_inference_model(adapter_path, tokenizer):
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        engine.MODEL_PATH,
        local_files_only=True,
        quantization_config=quantization_config,
        dtype=torch.float16,
        device_map="auto",
        attn_implementation="sdpa",
    )
    model = PeftModel.from_pretrained(
        base_model,
        adapter_path,
        is_trainable=False,
    )
    model.config.use_cache = True
    model.eval()
    return model


def generate_one(model, tokenizer, messages, allowed):
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    device = model.get_input_embeddings().weight.device
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,
        truncation=False,
    ).to(device)

    prompt_length = int(inputs["input_ids"].shape[1])
    if prompt_length > engine.MAX_LENGTH:
        raise RuntimeError(
            f"Cascade prompt has {prompt_length} tokens, above "
            f"{engine.MAX_LENGTH}; truncation is prohibited."
        )

    with torch.inference_mode():
        with torch.autocast("cuda", dtype=torch.float16):
            output = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=engine.MAX_NEW_TOKENS,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

    generated = output[0, prompt_length:]
    decoded = tokenizer.decode(generated, skip_special_tokens=True)
    return strict_parse(decoded, allowed), decoded


def compute_three_class_metrics(gold, predictions):
    if len(gold) != len(predictions) or not gold:
        raise ValueError("Invalid cascade metric inputs.")

    index = {label: i for i, label in enumerate(THREE_CLASSES)}
    confusion = [[0, 0, 0] for _ in THREE_CLASSES]
    invalid_rows = []
    correct = 0

    for row, (actual, predicted) in enumerate(zip(gold, predictions)):
        if predicted not in index:
            invalid_rows.append(row)
            continue
        confusion[index[actual]][index[predicted]] += 1
        if actual == predicted:
            correct += 1

    precision = []
    recall = []
    f1 = []
    supports = []

    for class_index, label in enumerate(THREE_CLASSES):
        tp = confusion[class_index][class_index]
        fp = sum(
            confusion[row][class_index]
            for row in range(3)
            if row != class_index
        )
        support = sum(actual == label for actual in gold)
        fn = sum(
            actual == label and predicted != label
            for actual, predicted in zip(gold, predictions)
        )

        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / support if support else 0.0
        score = (2 * p * r / (p + r)) if (p + r) else 0.0

        precision.append(p)
        recall.append(r)
        f1.append(score)
        supports.append(support)

    no_error_idx = index["No Error"]
    logic_idx = index["Logic Error"]
    explicit_idx = index["Explicit Error"]

    no_error_to_logic = (
        confusion[no_error_idx][logic_idx] / supports[no_error_idx]
        if supports[no_error_idx] else 0.0
    )
    explicit_to_logic = (
        confusion[explicit_idx][logic_idx] / supports[explicit_idx]
        if supports[explicit_idx] else 0.0
    )
    ler_over = (no_error_to_logic + explicit_to_logic) / 2.0

    logic = index["Logic Error"]
    return {
        "accuracy": correct / len(gold),
        "macro_precision": sum(precision) / 3,
        "macro_recall": sum(recall) / 3,
        "macro_f1": sum(f1) / 3,
        "logic_error_precision": precision[logic],
        "logic_error_recall": recall[logic],
        "logic_error_f1": f1[logic],
        "LER_over": ler_over,
        "invalid_output_rate": len(invalid_rows) / len(gold),
        "invalid_predictions": len(invalid_rows),
        "confusion_matrix": confusion,
        "confusion_matrix_labels": THREE_CLASSES,
        "gold_support": dict(zip(THREE_CLASSES, supports)),
        "invalid_row_indices": invalid_rows,
    }


def select_best_pair(pair_results):
    """Return the best pair using unrounded metrics and deterministic tie order."""
    if not pair_results:
        raise ValueError("No cascade pair results were supplied.")
    return min(
        pair_results,
        key=lambda item: (
            -item["metrics"]["macro_f1"],
            item["metrics"]["LER_over"],
            -item["metrics"]["logic_error_recall"],
            item["order_index"],
        ),
    )


def load_snapshot_metrics(stage_name, epoch_number):
    snapshot = (
        engine.STAGES[stage_name]["output_dir"] / f"epoch_{epoch_number}"
    )
    engine.verify_epoch_snapshot(snapshot, stage_name, epoch_number)
    metrics = read_json(snapshot / "validation_metrics.json")
    if "validation_loss" not in metrics:
        raise RuntimeError(
            f"{stage_name} epoch {epoch_number} has no validation_loss."
        )
    return snapshot, metrics


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate all four PyMETA hierarchical epoch pairs on official dev."
    )
    parser.add_argument(
        "--output-dir",
        default=str(engine.ROOT / "outputs" / "hierarchical_validation" / "cascade_selection"),
    )
    parser.add_argument(
        "--best-pair-dir",
        default=str(engine.ROOT / "checkpoints" / "hierarchical_best_pair"),
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    best_pair_dir = Path(args.best_pair_dir)

    if output_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite cascade output directory: {output_dir}"
        )
    if best_pair_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite best-pair directory: {best_pair_dir}"
        )

    engine.verify_environment()
    flat_dev_path = engine.ROOT / "data" / "flat_validation.json"
    if not flat_dev_path.is_file():
        raise FileNotFoundError(
            f"Required original three-class dev file is missing: {flat_dev_path}"
        )

    stage1_config = engine.STAGES["stage1"]
    stage2_config = engine.STAGES["stage2"]

    stage1_records = engine.load_records(
        stage1_config["validation_path"],
        stage1_config["validation_rows"],
        STAGE1_CLASSES,
    )
    stage2_records = engine.load_records(
        stage2_config["validation_path"],
        stage2_config["validation_rows"],
        STAGE2_CLASSES,
    )
    flat_records = read_json(flat_dev_path)
    gold, stage2_system_prompt = validate_full_dev_inputs(
        flat_records, stage1_records, stage2_records
    )

    snapshot_tokenizer_hashes = {}
    all_snapshots = {}
    for stage_name in ("stage1", "stage2"):
        for epoch in (1, 2):
            snapshot = (
                engine.STAGES[stage_name]["output_dir"] / f"epoch_{epoch}"
            )
            manifest = engine.verify_epoch_snapshot(snapshot, stage_name, epoch)
            all_snapshots[(stage_name, epoch)] = snapshot
            inventory = manifest["checkpoint_files"]
            snapshot_tokenizer_hashes[(stage_name, epoch)] = {
                name: digest
                for name, digest in inventory.items()
                if name.startswith("tokenizer/")
            }

    reference_tokenizer_hashes = snapshot_tokenizer_hashes[("stage1", 1)]
    if not reference_tokenizer_hashes:
        raise RuntimeError("Epoch snapshots do not contain tokenizer files.")
    if any(
        value != reference_tokenizer_hashes
        for value in snapshot_tokenizer_hashes.values()
    ):
        raise RuntimeError("Tokenizer files differ across hierarchical epoch snapshots.")

    stage1_predictions = {}
    stage1_raw_outputs = {}
    stage1_metrics = {}
    stage1_snapshots = {}

    # Stage 1: generate over every original dev row for both epoch adapters.
    for epoch in (1, 2):
        snapshot, metrics = load_snapshot_metrics("stage1", epoch)
        stage1_snapshots[epoch] = snapshot
        stage1_metrics[epoch] = metrics

        tokenizer = AutoTokenizer.from_pretrained(
            snapshot / "tokenizer",
            local_files_only=True,
        )
        if tokenizer.pad_token_id is None or not tokenizer.chat_template:
            raise RuntimeError("Stage 1 snapshot tokenizer is invalid.")
        model = load_inference_model(snapshot / "adapter", tokenizer)
        outputs = []
        raw_outputs = []
        for row_index, record in enumerate(stage1_records):
            prediction, raw_output = generate_one(
                model,
                tokenizer,
                record["messages"][:2],
                STAGE1_CLASSES,
            )
            outputs.append(prediction)
            raw_outputs.append(raw_output)
            if (row_index + 1) % 250 == 0:
                print(
                    f"Stage 1 epoch {epoch}: {row_index + 1}/{len(stage1_records)}",
                    flush=True,
                )
        stage1_predictions[epoch] = outputs
        stage1_raw_outputs[epoch] = raw_outputs

        del model
        gc.collect()
        torch.cuda.empty_cache()

    # Stage 2 is only called for rows routed as Non-Explicit by at least
    # one Stage 1 epoch. This union supports all four epoch combinations.
    routed_union = sorted({
        row_index
        for epoch_predictions in stage1_predictions.values()
        for row_index, prediction in enumerate(epoch_predictions)
        if prediction == "Non-Explicit"
    })

    stage2_predictions = {}
    stage2_raw_outputs = {}
    stage2_metrics = {}
    stage2_snapshots = {}

    for epoch in (1, 2):
        snapshot, metrics = load_snapshot_metrics("stage2", epoch)
        stage2_snapshots[epoch] = snapshot
        stage2_metrics[epoch] = metrics

        tokenizer = AutoTokenizer.from_pretrained(
            snapshot / "tokenizer",
            local_files_only=True,
        )
        if tokenizer.pad_token_id is None or not tokenizer.chat_template:
            raise RuntimeError("Stage 2 snapshot tokenizer is invalid.")
        model = load_inference_model(snapshot / "adapter", tokenizer)
        predictions_for_epoch = {}
        raw_outputs_for_epoch = {}

        for count, row_index in enumerate(routed_union, start=1):
            messages = [
                {"role": "system", "content": stage2_system_prompt},
                {
                    "role": "user",
                    "content": stage1_records[row_index]["messages"][1]["content"],
                },
            ]
            prediction, raw_output = generate_one(
                model,
                tokenizer,
                messages,
                STAGE2_CLASSES,
            )
            predictions_for_epoch[row_index] = prediction
            raw_outputs_for_epoch[row_index] = raw_output

            if count % 250 == 0:
                print(
                    f"Stage 2 epoch {epoch}: {count}/{len(routed_union)}",
                    flush=True,
                )

        stage2_predictions[epoch] = predictions_for_epoch
        stage2_raw_outputs[epoch] = raw_outputs_for_epoch
        del model
        gc.collect()
        torch.cuda.empty_cache()

    pair_order = [(1, 1), (1, 2), (2, 1), (2, 2)]
    pair_results = []

    for order_index, (stage1_epoch, stage2_epoch) in enumerate(pair_order):
        final_predictions = []
        prediction_records = []

        for row_index, (gold_label, stage1_prediction) in enumerate(
            zip(gold, stage1_predictions[stage1_epoch])
        ):
            stage2_prediction = None

            if stage1_prediction == "Explicit Error":
                final_prediction = "Explicit Error"
            elif stage1_prediction == "Non-Explicit":
                stage2_prediction = stage2_predictions[stage2_epoch].get(row_index)
                final_prediction = stage2_prediction
            else:
                final_prediction = None

            final_predictions.append(final_prediction)
            prediction_records.append({
                "row_index": row_index,
                "gold": gold_label,
                "stage1_prediction": stage1_prediction,
                "stage1_raw_output": stage1_raw_outputs[stage1_epoch][row_index],
                "stage2_prediction": stage2_prediction,
                "stage2_raw_output": (
                    stage2_raw_outputs[stage2_epoch].get(row_index)
                    if stage1_prediction == "Non-Explicit"
                    else None
                ),
                "final_prediction": final_prediction,
            })

        metrics = compute_three_class_metrics(gold, final_predictions)
        metrics.update({
            "stage1_epoch": stage1_epoch,
            "stage2_epoch": stage2_epoch,
            "stage1_validation_loss": stage1_metrics[stage1_epoch]["validation_loss"],
            "stage2_validation_loss": stage2_metrics[stage2_epoch]["validation_loss"],
            "validation_loss": {
                "stage1_assistant_token_weighted": stage1_metrics[stage1_epoch]["validation_loss"],
                "stage2_assistant_token_weighted": stage2_metrics[stage2_epoch]["validation_loss"],
                "combined": None,
            },
            "validation_loss_note": (
                "Stage-specific assistant-token-weighted losses are reported separately; "
                "no combined cascade-loss formula was specified."
            ),
            "selection_order_index": order_index,
        })

        pair_name = f"stage1_epoch_{stage1_epoch}_stage2_epoch_{stage2_epoch}"
        write_json(output_dir / f"{pair_name}_metrics.json", metrics)
        write_json(output_dir / f"{pair_name}_predictions.json", prediction_records)

        pair_results.append({
            "pair": pair_name,
            "stage1_epoch": stage1_epoch,
            "stage2_epoch": stage2_epoch,
            "metrics": metrics,
            "order_index": order_index,
        })

    # Compare unrounded values. Earlier pair wins exact ties by final key.
    best = select_best_pair(pair_results)

    best_pair_temporary = best_pair_dir.with_name(
        best_pair_dir.name + ".temporary"
    )
    if best_pair_temporary.exists():
        raise FileExistsError(
            f"Refusing to overwrite interrupted best-pair directory: {best_pair_temporary}"
        )
    best_pair_temporary.mkdir(parents=True)
    shutil.copytree(
        stage1_snapshots[best["stage1_epoch"]] / "adapter",
        best_pair_temporary / "stage1_adapter",
    )
    shutil.copytree(
        stage1_snapshots[best["stage1_epoch"]] / "tokenizer",
        best_pair_temporary / "stage1_tokenizer",
    )
    shutil.copytree(
        stage2_snapshots[best["stage2_epoch"]] / "adapter",
        best_pair_temporary / "stage2_adapter",
    )
    shutil.copytree(
        stage2_snapshots[best["stage2_epoch"]] / "tokenizer",
        best_pair_temporary / "stage2_tokenizer",
    )

    report = {
        "project": "PyMETA",
        "evaluation": "full_original_development_set_hierarchical_cascade",
        "development_rows": len(gold),
        "candidate_pairs": [
            {
                "pair": item["pair"],
                "stage1_epoch": item["stage1_epoch"],
                "stage2_epoch": item["stage2_epoch"],
                "metrics": item["metrics"],
            }
            for item in pair_results
        ],
        "selection_rule": [
            "higher three_class_macro_f1",
            "lower LER_over",
            "higher logic_error_recall",
            "earlier deterministic pair order",
        ],
        "best_pair": best["pair"],
        "best_pair_metrics": best["metrics"],
        "best_pair_adapter_directory": str(best_pair_dir),
        "test_records_accessed": False,
    }

    write_json(output_dir / "cascade_selection_report.json", report)
    write_json(
        best_pair_temporary / "best_pair_manifest.json",
        {
            "best_pair": best["pair"],
            "stage1_epoch": best["stage1_epoch"],
            "stage2_epoch": best["stage2_epoch"],
            "metrics": best["metrics"],
            "test_records_accessed": False,
        },
    )
    best_pair_temporary.replace(best_pair_dir)

    print("\nCASCADE EVALUATION COMPLETED")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

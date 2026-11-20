#!/usr/bin/env python3
"""Sequential zero-shot evaluation of three pinned models on PyMETA dev."""

import gc
import hashlib
import json
import sys
import time
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "flat_validation.json"
OUTPUT_ROOT = ROOT / "outputs" / "external_zero_shot"

EXPECTED_SHA256 = (
    "b90c97b5668452c1da1e49b3ab5d78ba80c5bc7bc745276836eaa7d9e18874da"
)

CLASSES = ["No Error", "Logic Error", "Explicit Error"]

EXPECTED_COUNTS = {
    "No Error": 1875,
    "Logic Error": 1495,
    "Explicit Error": 1493,
}

MAX_LENGTH = 1536
MAX_NEW_TOKENS = 16

MODELS = [
    {
        "name": "DeepSeek",
        "model_id": "deepseek-ai/deepseek-coder-6.7b-instruct",
        "revision": "e5d64addd26a6a1db0f9b863abf6ee3141936807",
    },
    {
        "name": "Granite",
        "model_id": "ibm-granite/granite-4.1-8b",
        "revision": "1504002f650e656a0a3789d99574df12e3e94ed0",
    },
    {
        "name": "Llama",
        "model_id": "meta-llama/Llama-3.1-8B-Instruct",
        "revision": "0e9e39f249a16976918f6564b8830bc894c89659",
    },
]


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".temporary")

    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        handle.flush()

    temporary.replace(path)


def strict_parse(text):
    try:
        value = json.loads(text.strip())
    except (json.JSONDecodeError, TypeError, AttributeError):
        return None

    if not isinstance(value, dict) or set(value) != {"label"}:
        return None

    if value["label"] not in CLASSES:
        return None

    return value["label"]


def validate_dataset(records):
    if len(records) != 4863:
        raise RuntimeError(
            f"Expected 4863 development records, found {len(records)}."
        )

    gold = []
    system_prompts = set()

    for index, record in enumerate(records):
        messages = record.get("messages")

        if not isinstance(messages, list) or len(messages) != 3:
            raise RuntimeError(f"Invalid message structure at row {index}.")

        expected_roles = ["system", "user", "assistant"]
        actual_roles = [message.get("role") for message in messages]

        if actual_roles != expected_roles:
            raise RuntimeError(f"Invalid message roles at row {index}.")

        if not all(
            isinstance(message.get("content"), str)
            for message in messages
        ):
            raise RuntimeError(f"Non-string message content at row {index}.")

        system_prompts.add(messages[0]["content"])

        target = strict_parse(messages[2]["content"])

        if target is None:
            raise RuntimeError(f"Invalid gold label at row {index}.")

        gold.append(target)

    if len(system_prompts) != 1:
        raise RuntimeError("Development records have inconsistent system prompts.")

    counts = dict(Counter(gold))

    if counts != EXPECTED_COUNTS:
        raise RuntimeError(
            f"Development label counts differ from audit: {counts}"
        )

    return gold, next(iter(system_prompts))


def compute_metrics(gold, predictions):
    if len(gold) != len(predictions) or not gold:
        raise ValueError("Invalid metric inputs.")

    index = {label: i for i, label in enumerate(CLASSES)}
    confusion = [[0, 0, 0] for _ in CLASSES]

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

    for class_index, label in enumerate(CLASSES):
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
        score = 2 * p * r / (p + r) if (p + r) else 0.0

        precision.append(p)
        recall.append(r)
        f1.append(score)
        supports.append(support)

    no_error = index["No Error"]
    logic_error = index["Logic Error"]
    explicit_error = index["Explicit Error"]

    no_error_to_logic = (
        confusion[no_error][logic_error] / supports[no_error]
        if supports[no_error] else 0.0
    )

    explicit_to_logic = (
        confusion[explicit_error][logic_error] / supports[explicit_error]
        if supports[explicit_error] else 0.0
    )

    return {
        "accuracy": correct / len(gold),
        "macro_precision": sum(precision) / 3,
        "macro_recall": sum(recall) / 3,
        "macro_f1": sum(f1) / 3,
        "logic_error_precision": precision[logic_error],
        "logic_error_recall": recall[logic_error],
        "logic_error_f1": f1[logic_error],
        "LER_over": (no_error_to_logic + explicit_to_logic) / 2,
        "invalid_output_rate": len(invalid_rows) / len(gold),
        "invalid_predictions": len(invalid_rows),
        "confusion_matrix": confusion,
        "confusion_matrix_labels": CLASSES,
        "gold_support": dict(zip(CLASSES, supports)),
        "invalid_row_indices": invalid_rows,
    }


def load_model(model_config, quantization_config):
    model_id = model_config["model_id"]
    revision = model_config["revision"]

    print("Loading tokenizer:", model_id, flush=True)

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        revision=revision,
        trust_remote_code=False,
        local_files_only=True,
    )

    if not tokenizer.chat_template:
        raise RuntimeError("Tokenizer has no chat template.")

    if tokenizer.eos_token_id is None:
        raise RuntimeError("Tokenizer has no EOS token ID.")

    print("Loading 4-bit model:", model_id, flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        trust_remote_code=False,
        local_files_only=True,
        quantization_config=quantization_config,
        dtype=torch.float16,
        device_map="auto",
        attn_implementation="sdpa",
    )

    model.config.use_cache = True
    model.eval()

    return model, tokenizer


def generate_one(model, tokenizer, messages):
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,
        truncation=False,
    )

    prompt_length = int(inputs["input_ids"].shape[1])

    if prompt_length > MAX_LENGTH:
        raise RuntimeError(
            f"Prompt length {prompt_length} exceeds {MAX_LENGTH}. "
            "Truncation is prohibited."
        )

    device = model.get_input_embeddings().weight.device
    inputs = inputs.to(device)

    pad_token_id = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id
    )

    with torch.inference_mode():
        with torch.autocast("cuda", dtype=torch.float16):
            output = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=MAX_NEW_TOKENS,
                pad_token_id=pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

    generated = output[0, prompt_length:]

    decoded = tokenizer.decode(
        generated,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    return strict_parse(decoded), decoded, prompt_length, len(generated)


def evaluate_model(model_config, records, gold, run_dir, quantization_config):
    name = model_config["name"]

    print("\n" + "=" * 75, flush=True)
    print("STARTING:", name, flush=True)
    print("Model:", model_config["model_id"], flush=True)
    print("Revision:", model_config["revision"], flush=True)

    model = None
    tokenizer = None

    predictions = []
    started = time.perf_counter()
    peak_memory_gib = None

    prediction_path = run_dir / name.lower() / "predictions.jsonl"
    prediction_path.parent.mkdir(parents=True, exist_ok=True)

    status = {
        "model": name,
        "model_id": model_config["model_id"],
        "revision": model_config["revision"],
        "status": "RUNNING",
        "completed_rows": 0,
        "total_rows": len(records),
    }

    status_path = prediction_path.parent / "status.json"
    write_json_atomic(status_path, status)

    try:
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.reset_peak_memory_stats()

        model, tokenizer = load_model(
            model_config,
            quantization_config,
        )

        print("MODEL LOAD: PASS", flush=True)

        with open(prediction_path, "w", encoding="utf-8") as output_file:
            for row_index, record in enumerate(records):
                messages = record["messages"][:2]
                row_started = time.perf_counter()

                prediction, raw_output, prompt_tokens, generated_tokens = (
                    generate_one(model, tokenizer, messages)
                )

                elapsed = time.perf_counter() - row_started

                predictions.append(prediction)

                output_record = {
                    "row_index": row_index,
                    "gold": gold[row_index],
                    "prediction": prediction,
                    "raw_output": raw_output,
                    "prompt_tokens": prompt_tokens,
                    "generated_tokens": generated_tokens,
                    "generation_seconds": round(elapsed, 4),
                }

                output_file.write(
                    json.dumps(output_record, ensure_ascii=False) + "\n"
                )

                # Flush every prediction to preserve progress.
                output_file.flush()

                status["completed_rows"] = row_index + 1

                if (row_index + 1) % 50 == 0:
                    # Persist progress periodically for interruption recovery.
                    write_json_atomic(status_path, status)

                    elapsed_total = time.perf_counter() - started
                    rate = (row_index + 1) / elapsed_total

                    print(
                        f"{name}: {row_index + 1}/{len(records)} "
                        f"| {rate:.3f} rows/sec "
                        f"| last prediction {elapsed:.2f}s",
                        flush=True,
                    )

        torch.cuda.synchronize()

        peak_memory_gib = round(
            torch.cuda.max_memory_allocated() / (1024 ** 3), 3
        )

        metrics = compute_metrics(gold, predictions)

        metrics.update({
            "model": name,
            "model_id": model_config["model_id"],
            "revision": model_config["revision"],
            "evaluation_records": len(records),
            "peak_allocated_gpu_memory_gib": peak_memory_gib,
            "elapsed_seconds": round(time.perf_counter() - started, 2),
            "generation_protocol": {
                "zero_shot": True,
                "do_sample": False,
                "max_new_tokens": MAX_NEW_TOKENS,
                "max_input_tokens": MAX_LENGTH,
                "truncation": False,
                "strict_json": True,
                "retry_or_repair": False,
            },
        })

        write_json_atomic(
            prediction_path.parent / "metrics.json",
            metrics,
        )

        status.update({
            "status": "PASS",
            "completed_rows": len(records),
            "peak_allocated_gpu_memory_gib": peak_memory_gib,
            "elapsed_seconds": metrics["elapsed_seconds"],
        })

        print("\nMODEL EVALUATION COMPLETE:", name, flush=True)
        print(json.dumps({
            "macro_f1": metrics["macro_f1"],
            "accuracy": metrics["accuracy"],
            "logic_error_recall": metrics["logic_error_recall"],
            "LER_over": metrics["LER_over"],
            "invalid_output_rate": metrics["invalid_output_rate"],
        }, indent=2), flush=True)

        return status

    except KeyboardInterrupt:
        status.update({
            "status": "INTERRUPTED",
            "completed_rows": len(predictions),
            "error": "Evaluation manually interrupted.",
        })
        print("\\nMODEL EVALUATION INTERRUPTED:", name, flush=True)
        raise

    except Exception as error:
        status.update({
            "status": "FAILED",
            "completed_rows": len(predictions),
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        })

        print("\nMODEL EVALUATION FAILED:", name, flush=True)
        traceback.print_exc()

        return status

    finally:
        del model
        del tokenizer
        gc.collect()
        torch.cuda.empty_cache()

        status["gpu_memory_after_cleanup_gib"] = round(
            torch.cuda.memory_allocated() / (1024 ** 3), 3
        )

        write_json_atomic(status_path, status)

        print(
            name,
            "GPU memory after cleanup:",
            status["gpu_memory_after_cleanup_gib"],
            "GiB",
            flush=True,
        )


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")

    if not DATA_PATH.is_file():
        raise FileNotFoundError(DATA_PATH)

    actual_hash = sha256_file(DATA_PATH)

    print("Development dataset SHA-256:", actual_hash, flush=True)

    if actual_hash != EXPECTED_SHA256:
        raise RuntimeError(
            "Development dataset hash mismatch. Evaluation aborted."
        )

    with open(DATA_PATH, "r", encoding="utf-8") as handle:
        records = json.load(handle)

    gold, system_prompt = validate_dataset(records)

    print("Development rows:", len(records), flush=True)
    print("Gold label counts:", dict(Counter(gold)), flush=True)
    print("System prompt SHA-256:", hashlib.sha256(
        system_prompt.encode("utf-8")
    ).hexdigest(), flush=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = OUTPUT_ROOT / f"run_{timestamp}"

    if run_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing run: {run_dir}"
        )

    run_dir.mkdir(parents=True, exist_ok=False)

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    manifest = {
        "run_timestamp_utc": timestamp,
        "dataset": str(DATA_PATH.relative_to(ROOT)),
        "dataset_sha256": actual_hash,
        "dataset_records": len(records),
        "dataset_split": "official development",
        "test_data_accessed": False,
        "gold_label_counts": dict(Counter(gold)),
        "system_prompt_sha256": hashlib.sha256(
            system_prompt.encode("utf-8")
        ).hexdigest(),
        "models": MODELS,
        "quantization": {
            "load_in_4bit": True,
            "quant_type": "nf4",
            "double_quantization": True,
            "compute_dtype": "float16",
        },
        "generation": {
            "zero_shot": True,
            "do_sample": False,
            "max_new_tokens": MAX_NEW_TOKENS,
            "max_input_tokens": MAX_LENGTH,
            "truncation": False,
            "strict_json": True,
            "retry_or_repair": False,
            "prompt_source": "first two messages of each flat validation record",
        },
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
        },
    }

    write_json_atomic(run_dir / "run_manifest.json", manifest)

    summary = {
        "run_timestamp_utc": timestamp,
        "run_directory": str(run_dir),
        "models": [],
    }

    write_json_atomic(run_dir / "run_summary.json", summary)

    for model_config in MODELS:
        result = evaluate_model(
            model_config,
            records,
            gold,
            run_dir,
            quantization_config,
        )

        summary["models"].append(result)
        write_json_atomic(run_dir / "run_summary.json", summary)

    summary["completed_timestamp_utc"] = (
        datetime.now(timezone.utc).isoformat()
    )

    summary["all_models_passed"] = all(
        result["status"] == "PASS"
        for result in summary["models"]
    )

    write_json_atomic(run_dir / "run_summary.json", summary)

    print("\n" + "=" * 75, flush=True)
    print("FINAL RUN SUMMARY", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    print("\nOutput directory:", run_dir, flush=True)


if __name__ == "__main__":
    main()

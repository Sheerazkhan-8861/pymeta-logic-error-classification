
#!/usr/bin/env python3

"""
PyMETA Hierarchical Training Engine
-----------------------------------
Separate Stage 1 / Stage 2 QLoRA training.

Usage:
    python scripts/pymeta_hierarchical_training_engine.py --stage stage1
    python scripts/pymeta_hierarchical_training_engine.py --stage stage2

Training requires:
    PYMETA_START_TRAINING=YES

Do not start training until the project preflight is approved.
"""

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    get_cosine_schedule_with_warmup,
)
from peft import PeftModel
from bitsandbytes.optim import PagedAdamW8bit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.pymeta_hierarchical_collator import HierarchicalDataCollator


# ============================================================
# 1. FROZEN PROJECT CONFIGURATION
# ============================================================

ROOT = Path.home() / "pymeta_project"

MODEL_ID = "Qwen/Qwen2.5-Coder-7B-Instruct"
MODEL_REVISION = "c03e6d358207e414f1eca0bb1891e29f1db0e242"
MODEL_PATH = ROOT / "model_cache" / "Qwen2.5-Coder-7B-Instruct"
INITIAL_ADAPTER = ROOT / "checkpoints" / "initial_adapter"
TOKENIZER_PATH = ROOT / "tokenizer"

SPEC_PATH = ROOT / "config" / "hierarchical_experiment_spec_v1.json"

STAGES = {
    "stage1": {
        "train_path": ROOT / "data" / "hierarchical_stage1_train.json",
        "validation_path": ROOT / "data" / "hierarchical_stage1_validation.json",
        "train_sha256": "6b52a0d69a5018388d91b1991c05ce652602ea954cbd7c7575480ac10eb7ffe0",
        "validation_sha256": "c1d9cfbccf7ec83ece36b5932ace2a5e02e97fc0408df3029d69ceb99c280c27",
        "train_rows": 38918,
        "validation_rows": 4863,
        "classes": ["Explicit Error", "Non-Explicit"],
        "steps_per_epoch": 2433,
        "total_steps": 4866,
        "warmup_steps": 146,
        "output_dir": ROOT / "checkpoints" / "hierarchical_stage1",
    },
    "stage2": {
        "train_path": ROOT / "data" / "hierarchical_stage2_train.json",
        "validation_path": ROOT / "data" / "hierarchical_stage2_validation.json",
        "train_sha256": "16080b67b459a378d47028c39190a94779457d56fcd3c96f0798caf2b764280a",
        "validation_sha256": "8bc82eed5743fb3cba64d91d07ee31921fcf315c0a8600a44defdcfc092a8906",
        "train_rows": 27901,
        "validation_rows": 3370,
        "classes": ["Logic Error", "No Error"],
        "steps_per_epoch": 1744,
        "total_steps": 3488,
        "warmup_steps": 105,
        "output_dir": ROOT / "checkpoints" / "hierarchical_stage2",
    },
}

EPOCHS = 2
ACCUMULATION = 16
MAX_LENGTH = 1536
MAX_NEW_TOKENS = 16

LEARNING_RATE = 0.0002
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0

SEED = 42
CHECKPOINT_INTERVAL = 250
MAX_OVERFLOW_RETRIES = 5

USE_FP16 = True


# ============================================================
# 2. GENERAL UTILITIES
# ============================================================

def sha256_file(path):
    digest = hashlib.sha256()

    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def verify_file(path, expected_hash):
    if not path.is_file():
        raise FileNotFoundError(f"Missing file: {path}")

    actual_hash = sha256_file(path)

    if actual_hash != expected_hash:
        raise RuntimeError(
            f"SHA-256 mismatch for {path}\n"
            f"Expected: {expected_hash}\n"
            f"Actual:   {actual_hash}"
        )

    return actual_hash


def load_records(path, expected_rows, allowed_classes):
    with open(path, "r", encoding="utf-8") as handle:
        records = json.load(handle)

    if not isinstance(records, list):
        raise ValueError(f"Expected a JSON list: {path}")

    if len(records) != expected_rows:
        raise ValueError(
            f"Row-count mismatch for {path}: "
            f"{len(records)} != {expected_rows}"
        )

    for index, record in enumerate(records):
        messages = record.get("messages")

        if not isinstance(messages, list) or len(messages) != 3:
            raise ValueError(f"Invalid messages at row {index}")

        if [m.get("role") for m in messages] != [
            "system", "user", "assistant"
        ]:
            raise ValueError(f"Invalid message roles at row {index}")

        try:
            target = json.loads(messages[2]["content"])
        except (json.JSONDecodeError, TypeError):
            raise ValueError(f"Invalid target JSON at row {index}")

        if set(target.keys()) != {"label"}:
            raise ValueError(f"Invalid target structure at row {index}")

        if target["label"] not in allowed_classes:
            raise ValueError(
                f"Unexpected target at row {index}: {target['label']}"
            )

    return records


def capture_rng(generator):
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
        "generator": generator.get_state(),
    }


def restore_rng(state, generator):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state_all(state["cuda"])
    generator.set_state(state["generator"])


def seed_everything():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary = path.with_suffix(path.suffix + ".temporary")

    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)

    os.replace(temporary, path)


# ============================================================
# 3. INPUT AND ENVIRONMENT PREFLIGHT
# ============================================================

def verify_experiment_specification(specification):
    """Fail closed if the executable configuration differs from the frozen spec."""
    if specification.get("project") != "PyMETA":
        raise RuntimeError("Experiment specification project mismatch.")
    if specification.get("version") != 1:
        raise RuntimeError("Unexpected experiment specification version.")
    base_model = specification.get("base_model", {})
    if base_model.get("model_id") != MODEL_ID:
        raise RuntimeError("Base model ID differs from the frozen specification.")
    if base_model.get("revision") != MODEL_REVISION:
        raise RuntimeError("Base model revision differs from the frozen specification.")
    if Path(base_model.get("local_path", "")) != Path("model_cache/Qwen2.5-Coder-7B-Instruct"):
        raise RuntimeError("Local model path differs from the frozen specification.")
    if Path(base_model.get("initial_adapter", "")) != Path("checkpoints/initial_adapter"):
        raise RuntimeError("Initial adapter path differs from the frozen specification.")

    training = specification.get("training", {})
    expected_training = {
        "epochs": EPOCHS,
        "gradient_accumulation": ACCUMULATION,
        "max_sequence_length": MAX_LENGTH,
        "max_new_tokens": MAX_NEW_TOKENS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "max_gradient_norm": MAX_GRAD_NORM,
        "seed": SEED,
        "checkpoint_interval": CHECKPOINT_INTERVAL,
        "overflow_retry_limit": MAX_OVERFLOW_RETRIES,
        "scheduler": "cosine",
        "optimizer": "PagedAdamW8bit",
        "quantization": "4-bit NF4 with double quantization",
        "compute_dtype": "float16",
        "loss": "Assistant-only causal language-model loss",
    }
    for key, expected_value in expected_training.items():
        if training.get(key) != expected_value:
            raise RuntimeError(
                f"Training configuration mismatch for {key}: "
                f"spec={training.get(key)!r}, engine={expected_value!r}"
            )

    if training.get("warmup_ratio") != 0.03:
        raise RuntimeError("Warmup ratio differs from the frozen specification.")
    if not USE_FP16:
        raise RuntimeError("Engine FP16 mode is disabled but the spec requires float16.")

    lora = specification.get("lora", {})
    adapter_config_path = INITIAL_ADAPTER / "adapter_config.json"
    if not adapter_config_path.is_file():
        raise FileNotFoundError(
            f"Missing initial adapter configuration: {adapter_config_path}"
        )
    with open(adapter_config_path, "r", encoding="utf-8") as handle:
        adapter_config = json.load(handle)

    expected_lora = {
        "r": lora.get("rank"),
        "lora_alpha": lora.get("alpha"),
        "lora_dropout": lora.get("dropout"),
        "bias": lora.get("bias"),
    }
    for key, expected_value in expected_lora.items():
        if adapter_config.get(key) != expected_value:
            raise RuntimeError(
                f"Initial adapter LoRA mismatch for {key}: "
                f"adapter={adapter_config.get(key)!r}, spec={expected_value!r}"
            )

    actual_targets = adapter_config.get("target_modules")
    expected_targets = lora.get("target_modules")
    if not isinstance(actual_targets, (list, set, tuple)):
        raise RuntimeError("Initial adapter target_modules is missing or invalid.")
    if set(actual_targets) != set(expected_targets or []):
        raise RuntimeError("Initial adapter target_modules differ from the spec.")

    for stage_name, stage_config in STAGES.items():
        stage_spec = specification.get(stage_name, {})
        expected_values = {
            "train_file": str(stage_config["train_path"].relative_to(ROOT)),
            "validation_file": str(stage_config["validation_path"].relative_to(ROOT)),
            "train_sha256": stage_config["train_sha256"],
            "validation_sha256": stage_config["validation_sha256"],
            "train_rows": stage_config["train_rows"],
            "validation_rows": stage_config["validation_rows"],
            "labels": stage_config["classes"],
            "steps_per_epoch": stage_config["steps_per_epoch"],
            "total_steps": stage_config["total_steps"],
            "warmup_steps": stage_config["warmup_steps"],
            "output_dir": str(stage_config["output_dir"].relative_to(ROOT)),
        }
        for key, expected_value in expected_values.items():
            if stage_spec.get(key) != expected_value:
                raise RuntimeError(
                    f"{stage_name} specification mismatch for {key}: "
                    f"spec={stage_spec.get(key)!r}, engine={expected_value!r}"
                )

        calculated_steps = math.ceil(
            stage_config["train_rows"] / ACCUMULATION
        )
        if stage_config["steps_per_epoch"] != calculated_steps:
            raise RuntimeError(
                f"{stage_name} steps_per_epoch does not match accumulation plan."
            )
        if stage_config["total_steps"] != EPOCHS * calculated_steps:
            raise RuntimeError(
                f"{stage_name} total_steps does not match epoch plan."
            )
        if stage_config["warmup_steps"] != round(
            stage_config["total_steps"] * training["warmup_ratio"]
        ):
            raise RuntimeError(
                f"{stage_name} warmup_steps does not match warmup ratio."
            )

    cascade = specification.get("cascade_selection", {})
    expected_cascade = {
        "validation_file": "data/flat_validation.json",
        "candidate_combinations": 4,
        "primary_metric": "three_class_macro_f1",
        "tie_breakers": [
            "lower_LER_over",
            "higher_logic_error_recall",
            "earlier_deterministic_pair",
        ],
        "test_data_access_allowed": False,
    }
    for key, expected_value in expected_cascade.items():
        if cascade.get(key) != expected_value:
            raise RuntimeError(
                f"Cascade-selection specification mismatch for {key}."
            )

    reproducibility = specification.get("reproducibility", {})
    required_reproducibility = {
        "flat_baseline_must_remain_unchanged": True,
        "official_question_id_split_only": True,
        "random_split_prohibited": True,
        "test_data_during_training_or_selection": False,
        "stage2_initialization_from_stage1_prohibited": True,
        "strict_json_output": True,
        "invalid_predictions_count_as_errors": True,
    }
    for key, expected_value in required_reproducibility.items():
        if reproducibility.get(key) != expected_value:
            raise RuntimeError(
                f"Reproducibility constraint mismatch: {key}"
            )


def verify_environment():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Training cannot start.")

    if not MODEL_PATH.is_dir():
        raise FileNotFoundError(f"Missing model: {MODEL_PATH}")
    if not (MODEL_PATH / "config.json").is_file():
        raise FileNotFoundError(f"Missing base-model config: {MODEL_PATH / 'config.json'}")
    model_weight_files = (
        list(MODEL_PATH.glob("*.safetensors"))
        + list(MODEL_PATH.glob("*.bin"))
    )
    if not model_weight_files:
        raise FileNotFoundError(
            f"No local model weight files found in: {MODEL_PATH}"
        )

    if not INITIAL_ADAPTER.is_dir():
        raise FileNotFoundError(
            f"Missing initial adapter: {INITIAL_ADAPTER}"
        )

    if not TOKENIZER_PATH.is_dir():
        raise FileNotFoundError(
            f"Missing tokenizer: {TOKENIZER_PATH}"
        )

    if not SPEC_PATH.is_file():
        raise FileNotFoundError(
            f"Missing experiment specification: {SPEC_PATH}"
        )

    with open(SPEC_PATH, "r", encoding="utf-8") as handle:
        specification = json.load(handle)

    verify_experiment_specification(specification)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA became unavailable during preflight.")

    actual_gpu_name = torch.cuda.get_device_name(0)
    actual_gpu_memory_mib = round(
        torch.cuda.get_device_properties(0).total_memory / 1024**2
    )
    expected_hardware = specification.get("hardware", {})
    if actual_gpu_name != expected_hardware.get("gpu"):
        raise RuntimeError(
            f"GPU differs from frozen experiment hardware: {actual_gpu_name}"
        )
    expected_memory_mib = expected_hardware.get("vram_mib")
    if (
        isinstance(expected_memory_mib, bool)
        or not isinstance(expected_memory_mib, int)
        or expected_memory_mib <= 0
    ):
        raise RuntimeError(
            "Frozen experiment hardware has an invalid vram_mib value."
        )

    # CUDA/driver versions can report a small difference in total device
    # memory for the same physical GPU. Treat the frozen value as a reference
    # capacity and enforce a 3% lower-bound tolerance; larger GPUs are valid.
    minimum_acceptable_memory_mib = math.ceil(expected_memory_mib * 0.97)
    if actual_gpu_memory_mib < minimum_acceptable_memory_mib:
        raise RuntimeError(
            "GPU memory is below the frozen hardware capacity tolerance: "
            f"actual={actual_gpu_memory_mib} MiB; "
            f"reference={expected_memory_mib} MiB; "
            f"minimum_acceptable={minimum_acceptable_memory_mib} MiB "
            "(97% of reference)"
        )

    print("CUDA:", actual_gpu_name)
    print(
        "GPU memory:",
        f"{actual_gpu_memory_mib} MiB "
        f"({actual_gpu_memory_mib / 1024:.2f} GiB)",
    )
    print(
        "Frozen hardware VRAM reference:",
        f"{expected_memory_mib} MiB (3% lower-bound tolerance)",
    )
    print("PyTorch:", torch.__version__)
    print("Experiment specification: PASS")


def verify_stage_inputs(stage_name):
    config = STAGES[stage_name]

    print("\nVerifying frozen dataset hashes...")

    train_hash = verify_file(
        config["train_path"],
        config["train_sha256"],
    )

    validation_hash = verify_file(
        config["validation_path"],
        config["validation_sha256"],
    )

    print("Training dataset SHA-256: PASS")
    print("Validation dataset SHA-256: PASS")

    return train_hash, validation_hash


# ============================================================
# 4. MODEL AND TOKENIZER
# ============================================================

def initialize_model(adapter_path):
    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_PATH,
        local_files_only=True,
    )

    if tokenizer.pad_token_id is None:
        raise RuntimeError("Tokenizer has no pad_token_id.")

    collator = HierarchicalDataCollator(tokenizer, max_length=MAX_LENGTH)

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
        quantization_config=quantization_config,
        dtype=torch.float16,
        device_map="auto",
        attn_implementation="sdpa",
    )

    model = PeftModel.from_pretrained(
        base_model,
        adapter_path,
        is_trainable=True,
    )

    model.config.use_cache = False

    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    model.enable_input_require_grads()

    return model, tokenizer, collator


# ============================================================
# 5. PREDICTION PARSING AND BINARY METRICS
# ============================================================

def parse_prediction(text, allowed_classes):
    try:
        parsed = json.loads(text.strip())

        if not isinstance(parsed, dict):
            return None

        if set(parsed.keys()) != {"label"}:
            return None

        if parsed["label"] not in allowed_classes:
            return None

        return parsed["label"]

    except (json.JSONDecodeError, TypeError, AttributeError):
        return None


def calculate_metrics(gold, predictions, classes):
    if len(gold) != len(predictions) or not gold:
        raise ValueError("Invalid evaluation inputs.")

    confusion = [[0, 0], [0, 0]]
    invalid = 0

    for actual, predicted in zip(gold, predictions):
        if predicted not in classes:
            invalid += 1
            continue

        confusion[classes.index(actual)][classes.index(predicted)] += 1

    f1_values = []

    for index in range(2):
        tp = confusion[index][index]

        fp = sum(
            confusion[row][index]
            for row in range(2)
            if row != index
        )

        fn = sum(
            actual == classes[index] and predicted != classes[index]
            for actual, predicted in zip(gold, predictions)
        )

        denominator = 2 * tp + fp + fn

        f1_values.append(
            2 * tp / denominator if denominator else 0.0
        )

    accuracy = sum(
        actual == predicted
        for actual, predicted in zip(gold, predictions)
    ) / len(gold)

    return {
        "macro_f1": float(np.mean(f1_values)),
        "accuracy": float(accuracy),
        "invalid_output_rate": float(invalid / len(gold)),
        "invalid_predictions": invalid,
        "confusion_matrix": confusion,
    }


# ============================================================
# 6. AUTOREGRESSIVE VALIDATION
# ============================================================

def evaluate(model, tokenizer, records, classes, collator):
    """Greedy autoregressive metrics plus assistant-token-weighted validation loss."""
    was_training = model.training
    model.eval()

    gold = []
    predictions = []
    weighted_loss = 0.0
    supervised_token_count = 0

    device = model.get_input_embeddings().weight.device

    try:
        for index, record in enumerate(records):
            # Teacher-forced validation loss; prompt tokens are masked by collator.
            loss_batch = collator([record])
            active_labels = int((loss_batch["labels"] != -100).sum().item())
            loss_batch = {
                key: value.to(device)
                for key, value in loss_batch.items()
            }

            with torch.inference_mode():
                with torch.autocast("cuda", dtype=torch.float16):
                    loss_output = model(**loss_batch)

            weighted_loss += float(loss_output.loss.item()) * active_labels
            supervised_token_count += active_labels

            messages = record["messages"]
            prompt = tokenizer.apply_chat_template(
                messages[:2],
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                add_special_tokens=False,
                truncation=False,
            ).to(device)

            prompt_length = int(inputs["input_ids"].shape[1])
            if prompt_length > MAX_LENGTH:
                raise ValueError(
                    f"Validation prompt at row {index} has {prompt_length} "
                    f"tokens, exceeding {MAX_LENGTH}; truncation is prohibited."
                )

            with torch.inference_mode():
                with torch.autocast("cuda", dtype=torch.float16):
                    output = model.generate(
                        **inputs,
                        do_sample=False,
                        max_new_tokens=MAX_NEW_TOKENS,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )

            generated = output[0, prompt_length:]
            decoded = tokenizer.decode(
                generated,
                skip_special_tokens=True,
            )
            predictions.append(parse_prediction(decoded, classes))

            target = json.loads(messages[2]["content"])
            gold.append(target["label"])

            if (index + 1) % 250 == 0:
                print(
                    f"Validation: {index + 1}/{len(records)}",
                    flush=True,
                )

        metrics = calculate_metrics(gold, predictions, classes)
        metrics["validation_loss"] = (
            weighted_loss / supervised_token_count
            if supervised_token_count
            else None
        )
        metrics["validation_supervised_tokens"] = supervised_token_count
        return metrics

    finally:
        model.train(was_training)



# ============================================================
# 7. CHECKPOINT MANIFEST AND RECOVERY
# ============================================================

def checkpoint_inventory(directory):
    """SHA-256 inventory of every checkpoint file except manifest."""
    inventory = {}

    for path in sorted(directory.rglob("*")):
        if path.is_file():
            relative = str(path.relative_to(directory))

            if relative != "manifest.json":
                inventory[relative] = sha256_file(path)

    if not inventory:
        raise RuntimeError(f"Checkpoint contains no files: {directory}")

    return inventory


def checkpoint_manifest(
    config,
    stage_name,
    train_hash,
    validation_hash,
    state,
    inventory,
):
    return {
        "project": "PyMETA",
        "checkpoint_format_version": 3,
        "stage": stage_name,
        "experiment_spec_sha256": sha256_file(SPEC_PATH),
        "train_sha256": train_hash,
        "validation_sha256": validation_hash,
        "epoch": state["epoch"],
        "next_position": state["next_position"],
        "global_step": state["global_step"],
        "test_records_accessed": False,
        "training_started": bool(state["global_step"] > 0),
        "elapsed_training_seconds": float(state.get("elapsed_training_seconds", 0.0)),
        "checkpoint_files": inventory,
    }


def verify_recovery_checkpoint(
    recovery_dir,
    config,
    stage_name,
    train_hash,
    validation_hash,
):
    """Verify checkpoint identity, required files, state and SHA-256 inventory."""
    manifest_path = recovery_dir / "manifest.json"
    state_path = recovery_dir / "training_state.pt"
    adapter_path = recovery_dir / "adapter"
    tokenizer_path = recovery_dir / "tokenizer"

    required_paths = (
        manifest_path,
        state_path,
        adapter_path,
        tokenizer_path,
    )

    for path in required_paths:
        if not path.exists():
            raise RuntimeError(f"Incomplete recovery checkpoint: {path}")

    for directory in (adapter_path, tokenizer_path):
        if not directory.is_dir() or not any(p.is_file() for p in directory.rglob("*")):
            raise RuntimeError(f"Checkpoint directory is empty or invalid: {directory}")

    try:
        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Invalid checkpoint manifest: {exc}") from exc

    manifest_version = manifest.get("checkpoint_format_version")
    if manifest_version not in (2, 3):
        raise RuntimeError("Unsupported recovery checkpoint format version.")

    expected = {
        "project": "PyMETA",
        "stage": stage_name,
        "experiment_spec_sha256": sha256_file(SPEC_PATH),
        "train_sha256": train_hash,
        "validation_sha256": validation_hash,
        "test_records_accessed": False,
    }

    for key, expected_value in expected.items():
        if manifest.get(key) != expected_value:
            raise RuntimeError(f"Recovery manifest mismatch: {key}")

    recorded_inventory = manifest.get("checkpoint_files")
    if not isinstance(recorded_inventory, dict) or not recorded_inventory:
        raise RuntimeError("Checkpoint SHA-256 inventory is missing or invalid.")

    actual_inventory = checkpoint_inventory(recovery_dir)
    if recorded_inventory != actual_inventory:
        missing = sorted(set(recorded_inventory) - set(actual_inventory))
        unexpected = sorted(set(actual_inventory) - set(recorded_inventory))
        changed = sorted(
            name
            for name in set(recorded_inventory) & set(actual_inventory)
            if recorded_inventory[name] != actual_inventory[name]
        )
        raise RuntimeError(
            "Checkpoint SHA-256 verification failed. "
            f"Missing={missing}; Unexpected={unexpected}; Changed={changed}"
        )

    state = torch.load(
        state_path,
        map_location="cpu",
        weights_only=False,
    )

    required_state = {
        "optimizer",
        "scheduler",
        "scaler",
        "epoch",
        "permutation",
        "next_position",
        "global_step",
        "skipped_overflow_attempts",
        "history",
        "rng",
    }

    if not isinstance(state, dict) or not required_state.issubset(state):
        raise RuntimeError("Recovery state is missing required fields.")

    if manifest_version == 3 and "elapsed_training_seconds" not in state:
        raise RuntimeError("Version 3 checkpoint is missing elapsed training time.")
    state.setdefault("elapsed_training_seconds", 0.0)

    for key in ("epoch", "next_position", "global_step"):
        if state[key] != manifest.get(key):
            raise RuntimeError(f"Checkpoint state mismatch: {key}")

    for state_key in ("optimizer", "scheduler", "scaler"):
        if not isinstance(state[state_key], dict):
            raise RuntimeError(f"Checkpoint {state_key} state is invalid.")

    stage_config = STAGES[stage_name]
    train_rows = stage_config["train_rows"]

    permutation = state["permutation"]
    if (
        not isinstance(permutation, torch.Tensor)
        or permutation.ndim != 1
        or permutation.dtype not in (torch.int32, torch.int64)
        or permutation.numel() != train_rows
    ):
        raise RuntimeError("Checkpoint permutation has an invalid shape or dtype.")

    expected_permutation = torch.arange(train_rows, dtype=permutation.dtype)
    if not torch.equal(torch.sort(permutation).values, expected_permutation):
        raise RuntimeError("Checkpoint permutation is not a complete row permutation.")

    epoch = state["epoch"]
    next_position = state["next_position"]
    global_step = state["global_step"]

    if not isinstance(epoch, int) or not 0 <= epoch <= EPOCHS:
        raise RuntimeError("Checkpoint epoch is outside the training plan.")
    if not isinstance(next_position, int) or not 0 <= next_position <= train_rows:
        raise RuntimeError("Checkpoint next_position is outside the training data.")
    if not isinstance(global_step, int) or not 0 <= global_step <= stage_config["total_steps"]:
        raise RuntimeError("Checkpoint global_step is outside the training plan.")

    scheduler_last_epoch = state["scheduler"].get("last_epoch")
    if scheduler_last_epoch is not None and scheduler_last_epoch != global_step:
        raise RuntimeError(
            "Checkpoint scheduler position does not match global_step: "
            f"{scheduler_last_epoch} != {global_step}"
        )

    if epoch == EPOCHS:
        expected_position = train_rows
    else:
        updates_in_epoch = global_step - epoch * stage_config["steps_per_epoch"]
        if not 0 <= updates_in_epoch <= stage_config["steps_per_epoch"]:
            raise RuntimeError("Checkpoint epoch/global_step relationship is invalid.")
        expected_position = min(updates_in_epoch * ACCUMULATION, train_rows)

    if next_position != expected_position:
        raise RuntimeError(
            "Checkpoint next_position is inconsistent with epoch/global_step: "
            f"{next_position} != {expected_position}"
        )

    if not isinstance(state["history"], list):
        raise RuntimeError("Checkpoint training history is invalid.")
    if len(state["history"]) != epoch:
        raise RuntimeError("Checkpoint history length does not match completed epochs.")

    skipped = state["skipped_overflow_attempts"]
    if not isinstance(skipped, int) or skipped < 0:
        raise RuntimeError("Checkpoint overflow retry count is invalid.")

    elapsed = state.get("elapsed_training_seconds", 0.0)
    if not isinstance(elapsed, (int, float)) or elapsed < 0:
        raise RuntimeError("Checkpoint elapsed training time is invalid.")

    expected_training_started = (
        bool(global_step > 0) if manifest_version == 3 else True
    )
    if manifest.get("training_started") != expected_training_started:
        raise RuntimeError("Checkpoint manifest training_started flag is inconsistent.")
    if manifest_version == 3:
        if manifest.get("elapsed_training_seconds") != float(elapsed):
            raise RuntimeError("Checkpoint manifest elapsed time is inconsistent.")

    rng = state["rng"]
    required_rng = {"python", "numpy", "torch", "cuda", "generator"}
    if not isinstance(rng, dict) or not required_rng.issubset(rng):
        raise RuntimeError("Checkpoint RNG state is incomplete.")

    print(
        f"Recovery checkpoint integrity: PASS "
        f"({stage_name}, {len(actual_inventory)} files)"
    )
    return state


def recover_interrupted_checkpoint(
    recovery_dir,
    config,
    stage_name,
    train_hash,
    validation_hash,
):
    """
    Keep a verified current checkpoint or restore a verified .previous.
    Invalid recovery data is preserved; it is never silently discarded.
    """
    backup = recovery_dir.with_name(recovery_dir.name + ".previous")
    temporary = recovery_dir.with_name(recovery_dir.name + ".temporary")
    corrupt = recovery_dir.with_name(recovery_dir.name + ".corrupt")

    current_valid = False
    backup_valid = False

    if recovery_dir.exists():
        try:
            verify_recovery_checkpoint(
                recovery_dir, config, stage_name, train_hash, validation_hash
            )
            current_valid = True
        except Exception as exc:
            print(f"Current checkpoint invalid: {exc}")

    if backup.exists():
        try:
            verify_recovery_checkpoint(
                backup, config, stage_name, train_hash, validation_hash
            )
            backup_valid = True
        except Exception as exc:
            print(f"Previous checkpoint invalid: {exc}")

    if current_valid:
        if backup.exists():
            if backup_valid:
                shutil.rmtree(backup)
            else:
                invalid_backup = backup.with_name(backup.name + ".corrupt")
                if invalid_backup.exists():
                    raise RuntimeError(
                        "Current checkpoint is valid, but the invalid previous "
                        f"checkpoint cannot be preserved because this path exists: {invalid_backup}"
                    )
                os.replace(backup, invalid_backup)
                print("Invalid previous checkpoint preserved at:", invalid_backup)
        if temporary.exists():
            shutil.rmtree(temporary)
        print("Recovery selection: current checkpoint retained.")
        return True

    if backup_valid:
        if recovery_dir.exists():
            if corrupt.exists():
                raise RuntimeError(
                    f"Cannot preserve invalid checkpoint because this path exists: {corrupt}"
                )
            os.replace(recovery_dir, corrupt)

        try:
            os.replace(backup, recovery_dir)
            verify_recovery_checkpoint(
                recovery_dir, config, stage_name, train_hash, validation_hash
            )
        except Exception:
            # If restoration fails, preserve the failed restore and put the
            # original invalid directory back where possible.
            failed_restore = recovery_dir.with_name(recovery_dir.name + ".failed_restore")
            if recovery_dir.exists() and not failed_restore.exists():
                os.replace(recovery_dir, failed_restore)
            if corrupt.exists() and not recovery_dir.exists():
                os.replace(corrupt, recovery_dir)
            raise

        if temporary.exists():
            shutil.rmtree(temporary)
        if corrupt.exists():
            print("Invalid checkpoint preserved at:", corrupt)
        print("Recovery selection: previous checkpoint restored and verified.")
        return True

    if recovery_dir.exists() or backup.exists():
        raise RuntimeError(
            "No verified recovery checkpoint is available. "
            "Existing recovery data has been preserved; refusing to overwrite it."
        )

    if temporary.exists():
        shutil.rmtree(temporary)

    print("No existing recovery checkpoint found.")
    return False


def save_recovery_checkpoint(
    recovery_dir,
    model,
    tokenizer,
    optimizer,
    scheduler,
    scaler,
    state,
    generator,
    config,
    stage_name,
    train_hash,
    validation_hash,
):
    """
    Create and verify a full checkpoint before replacing the old one.
    Keep .previous until the replacement passes verification.
    """

    temporary = recovery_dir.with_name(
        recovery_dir.name + ".temporary"
    )
    backup = recovery_dir.with_name(
        recovery_dir.name + ".previous"
    )

    # Resolve any interrupted replacement before starting another.
    recover_interrupted_checkpoint(
        recovery_dir, config, stage_name,
        train_hash, validation_hash,
    )

    if temporary.exists():
        shutil.rmtree(temporary)

    temporary.mkdir(parents=True)

    model.save_pretrained(temporary / "adapter")
    tokenizer.save_pretrained(temporary / "tokenizer")

    training_state = {
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": state["epoch"],
        "permutation": state["permutation"],
        "next_position": state["next_position"],
        "global_step": state["global_step"],
        "skipped_overflow_attempts": state["skipped_overflow_attempts"],
        "history": state["history"],
        "elapsed_training_seconds": float(state.get("elapsed_training_seconds", 0.0)),
        "rng": capture_rng(generator),
    }

    torch.save(training_state, temporary / "training_state.pt")

    inventory = checkpoint_inventory(temporary)

    manifest = checkpoint_manifest(
        config,
        stage_name,
        train_hash,
        validation_hash,
        state,
        inventory,
    )

    write_json(temporary / "manifest.json", manifest)

    # First verification: before installation.
    verify_recovery_checkpoint(
        temporary, config, stage_name,
        train_hash, validation_hash,
    )

    moved_old = False

    try:
        if recovery_dir.exists():
            os.replace(recovery_dir, backup)
            moved_old = True

        os.replace(temporary, recovery_dir)

        # Second verification: after installation.
        verify_recovery_checkpoint(
            recovery_dir, config, stage_name,
            train_hash, validation_hash,
        )

        # Delete old checkpoint only after new one is verified.
        if backup.exists():
            shutil.rmtree(backup)

        print("New recovery checkpoint installed and verified.")

    except Exception:
        if recovery_dir.exists():
            shutil.rmtree(recovery_dir)

        if moved_old and backup.exists():
            os.replace(backup, recovery_dir)

            verify_recovery_checkpoint(
                recovery_dir, config, stage_name,
                train_hash, validation_hash,
            )

            print("Previous verified checkpoint restored.")

        raise


# ============================================================
# 8. EPOCH ADAPTER SNAPSHOTS
# ============================================================

def epoch_snapshot_inventory(directory):
    inventory = {}
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.name != "snapshot_manifest.json":
            inventory[str(path.relative_to(directory))] = sha256_file(path)
    if not inventory:
        raise RuntimeError(f"Epoch snapshot contains no files: {directory}")
    return inventory


def verify_epoch_snapshot(epoch_dir, stage_name, epoch_number):
    manifest_path = epoch_dir / "snapshot_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"Epoch snapshot manifest is missing: {manifest_path}")

    for required in (
        epoch_dir / "adapter",
        epoch_dir / "tokenizer",
        epoch_dir / "validation_metrics.json",
    ):
        if not required.exists():
            raise RuntimeError(f"Epoch snapshot is incomplete: {required}")

    for directory in (epoch_dir / "adapter", epoch_dir / "tokenizer"):
        if not directory.is_dir() or not any(path.is_file() for path in directory.rglob("*")):
            raise RuntimeError(f"Epoch snapshot directory is empty: {directory}")

    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    if manifest.get("stage") != stage_name:
        raise RuntimeError("Epoch snapshot stage mismatch.")
    if manifest.get("epoch") != epoch_number:
        raise RuntimeError("Epoch snapshot number mismatch.")

    actual = epoch_snapshot_inventory(epoch_dir)
    if manifest.get("checkpoint_files") != actual:
        raise RuntimeError("Epoch snapshot SHA-256 verification failed.")

    return manifest


def save_epoch_adapter(
    model,
    tokenizer,
    output_dir,
    stage_name,
    epoch_number,
    metrics,
):
    """Atomically replace an orphaned epoch snapshot without losing the prior copy."""
    epoch_dir = output_dir / f"epoch_{epoch_number}"
    temporary = output_dir / f"epoch_{epoch_number}.temporary"
    backup = output_dir / f"epoch_{epoch_number}.previous"
    corrupt = output_dir / f"epoch_{epoch_number}.corrupt"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve a previous interrupted snapshot replacement.
    if backup.exists():
        if epoch_dir.exists():
            try:
                verify_epoch_snapshot(epoch_dir, stage_name, epoch_number)
                shutil.rmtree(backup)
            except Exception:
                if corrupt.exists():
                    raise RuntimeError(
                        f"Cannot preserve invalid epoch snapshot; path exists: {corrupt}"
                    )
                os.replace(epoch_dir, corrupt)
                os.replace(backup, epoch_dir)
                verify_epoch_snapshot(epoch_dir, stage_name, epoch_number)
        else:
            os.replace(backup, epoch_dir)
            verify_epoch_snapshot(epoch_dir, stage_name, epoch_number)

    if temporary.exists():
        shutil.rmtree(temporary)

    temporary.mkdir(parents=True)
    moved_old = False

    try:
        model.save_pretrained(temporary / "adapter")
        tokenizer.save_pretrained(temporary / "tokenizer")
        write_json(
            temporary / "validation_metrics.json",
            metrics,
        )

        inventory = epoch_snapshot_inventory(temporary)
        write_json(
            temporary / "snapshot_manifest.json",
            {
                "stage": stage_name,
                "epoch": epoch_number,
                "validation_metrics": metrics,
                "checkpoint_files": inventory,
            },
        )

        verify_epoch_snapshot(temporary, stage_name, epoch_number)

        if epoch_dir.exists():
            verify_epoch_snapshot(epoch_dir, stage_name, epoch_number)
            os.replace(epoch_dir, backup)
            moved_old = True

        os.replace(temporary, epoch_dir)
        verify_epoch_snapshot(epoch_dir, stage_name, epoch_number)

        if backup.exists():
            shutil.rmtree(backup)

        print(f"Epoch {epoch_number} adapter saved and verified: {epoch_dir}")

    except Exception:
        if epoch_dir.exists():
            failed = output_dir / f"epoch_{epoch_number}.failed"
            if failed.exists():
                raise RuntimeError(
                    f"Failed epoch snapshot preserved; path exists: {failed}"
                )
            os.replace(epoch_dir, failed)

        if moved_old and backup.exists():
            os.replace(backup, epoch_dir)
            verify_epoch_snapshot(epoch_dir, stage_name, epoch_number)
            print("Previous epoch snapshot restored.")

        raise



# ============================================================
# 9. TRAINING OBJECTS
# ============================================================

def create_training_objects(model, config):
    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    if not trainable_parameters:
        raise RuntimeError("No trainable model parameters.")

    optimizer = PagedAdamW8bit(
        trainable_parameters,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config["warmup_steps"],
        num_training_steps=config["total_steps"],
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        init_scale=65536,
        enabled=True,
    )

    return optimizer, scheduler, scaler


# ============================================================
# 10. TRAINING SETUP
# ============================================================

def setup_training(stage_name):
    config = STAGES[stage_name]

    verify_environment()

    train_hash, validation_hash = verify_stage_inputs(stage_name)

    train_records = load_records(
        config["train_path"],
        config["train_rows"],
        config["classes"],
    )

    validation_records = load_records(
        config["validation_path"],
        config["validation_rows"],
        config["classes"],
    )

    print("Training rows:", len(train_records))
    print("Validation rows:", len(validation_records))
    print("Stage labels:", config["classes"])

    output_dir = config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    recovery_dir = output_dir / "recovery"

    has_recovery = recover_interrupted_checkpoint(
        recovery_dir,
        config,
        stage_name,
        train_hash,
        validation_hash,
    )

    if has_recovery:
        print("\nVerified recovery checkpoint detected.")
        saved_state = verify_recovery_checkpoint(
            recovery_dir,
            config,
            stage_name,
            train_hash,
            validation_hash,
        )
        adapter_path = recovery_dir / "adapter"
    else:
        saved_state = None
        adapter_path = INITIAL_ADAPTER

    seed_everything()

    model, tokenizer, collator = initialize_model(adapter_path)

    optimizer, scheduler, scaler = create_training_objects(
        model,
        config,
    )

    generator = torch.Generator()
    generator.manual_seed(SEED)

    if saved_state is not None:
        optimizer.load_state_dict(saved_state["optimizer"])
        scheduler.load_state_dict(saved_state["scheduler"])
        scaler.load_state_dict(saved_state["scaler"])

        state = {
            key: saved_state[key]
            for key in [
                "epoch",
                "permutation",
                "next_position",
                "global_step",
                "skipped_overflow_attempts",
                "history",
                "elapsed_training_seconds",
            ]
        }

        restore_rng(saved_state["rng"], generator)

        print("Full recovery state restored.")

    else:
        permutation = torch.randperm(
            len(train_records),
            generator=generator,
        )

        state = {
            "epoch": 0,
            "permutation": permutation,
            "next_position": 0,
            "global_step": 0,
            "skipped_overflow_attempts": 0,
            "history": [],
            "elapsed_training_seconds": 0.0,
        }

        save_recovery_checkpoint(
            recovery_dir,
            model,
            tokenizer,
            optimizer,
            scheduler,
            scaler,
            state,
            generator,
            config,
            stage_name,
            train_hash,
            validation_hash,
        )

        verify_recovery_checkpoint(
            recovery_dir,
            config,
            stage_name,
            train_hash,
            validation_hash,
        )

        print("Initial full recovery checkpoint created and verified.")

    return (
        config,
        train_records,
        validation_records,
        model,
        tokenizer,
        collator,
        optimizer,
        scheduler,
        scaler,
        generator,
        state,
        recovery_dir,
        train_hash,
        validation_hash,
    )


# ============================================================
# 11. ONE ACCUMULATION GROUP
# ============================================================

def train_accumulation_group(
    model,
    optimizer,
    scaler,
    collator,
    records,
    indices,
):
    group_size = len(indices)

    if group_size == 0:
        raise ValueError("Empty accumulation group.")

    optimizer.zero_grad(set_to_none=True)

    device = model.get_input_embeddings().weight.device

    for index in indices:
        batch = collator([records[int(index)]])

        batch = {
            key: value.to(device)
            for key, value in batch.items()
        }

        with torch.autocast("cuda", dtype=torch.float16):
            output = model(**batch)
            loss = output.loss / group_size

        if not torch.isfinite(loss.detach()):
            raise FloatingPointError(
                "Non-finite training loss. Aborting safely."
            )

        scaler.scale(loss).backward()

    scaler.unscale_(optimizer)

    gradients_finite = True

    for parameter in model.parameters():
        if parameter.grad is not None:
            if not torch.isfinite(parameter.grad).all():
                gradients_finite = False
                break

    if gradients_finite:
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            MAX_GRAD_NORM,
        )

    old_scale = scaler.get_scale()

    scaler.step(optimizer)
    scaler.update()

    new_scale = scaler.get_scale()

    successful = gradients_finite and new_scale >= old_scale

    optimizer.zero_grad(set_to_none=True)

    return successful


# ============================================================
# 12. MAIN TRAINING LOOP
# ============================================================

def train(stage_name):
    if os.environ.get("PYMETA_START_TRAINING") != "YES":
        raise RuntimeError(
            "TRAINING BLOCKED. "
            "Set PYMETA_START_TRAINING=YES only after explicit approval."
        )

    segment_start_time = time.monotonic()

    (
        config,
        train_records,
        validation_records,
        model,
        tokenizer,
        collator,
        optimizer,
        scheduler,
        scaler,
        generator,
        state,
        recovery_dir,
        train_hash,
        validation_hash,
    ) = setup_training(stage_name)

    output_dir = config["output_dir"]

    elapsed_before_resume = float(state.get("elapsed_training_seconds", 0.0))

    def update_elapsed_training_time():
        state["elapsed_training_seconds"] = (
            elapsed_before_resume + time.monotonic() - segment_start_time
        )

    summary_path = output_dir / "training_summary.json"
    if state["epoch"] >= EPOCHS and summary_path.is_file():
        print("Training is already complete and its summary exists; no changes made.")
        return

    if state["epoch"] >= EPOCHS:
        print("Training is already complete; finalizing the training summary.")
    else:
        print("\n" + "=" * 65)
        print("HIERARCHICAL TRAINING STARTED")
        print("Stage:", stage_name)
        print("Epochs:", EPOCHS)
        print("Training rows:", len(train_records))
        print("Accumulation:", ACCUMULATION)
        print("Planned total successful updates:", config["total_steps"])
        print("=" * 65)

    model.train()

    while state["epoch"] < EPOCHS:
        epoch_number = state["epoch"] + 1
        permutation = state["permutation"]
        position = state["next_position"]

        print(
            f"\nEpoch {epoch_number}/{EPOCHS} "
            f"starting at position {position}"
        )

        while position < len(permutation):
            indices = permutation[
                position:position + ACCUMULATION
            ]

            rng_snapshot = capture_rng(generator)

            successful = False
            retries = 0

            while not successful:
                successful = train_accumulation_group(
                    model,
                    optimizer,
                    scaler,
                    collator,
                    train_records,
                    indices,
                )

                if successful:
                    scheduler.step()

                    state["global_step"] += 1
                    position += len(indices)

                    state["next_position"] = position

                    break

                retries += 1
                state["skipped_overflow_attempts"] += 1

                print(
                    f"Overflow retry {retries}/"
                    f"{MAX_OVERFLOW_RETRIES} "
                    f"at position {position}",
                    flush=True,
                )

                if retries > MAX_OVERFLOW_RETRIES:
                    raise RuntimeError(
                        "Maximum overflow retries exceeded. "
                        "Last verified recovery checkpoint is preserved."
                    )

                restore_rng(rng_snapshot, generator)

            if state["global_step"] % 25 == 0:
                print(
                    f"Epoch {epoch_number} | "
                    f"Position {position}/{len(permutation)} | "
                    f"Successful updates {state['global_step']} | "
                    f"Loss scale {scaler.get_scale()}",
                    flush=True,
                )

            if state["global_step"] % CHECKPOINT_INTERVAL == 0:
                update_elapsed_training_time()
                save_recovery_checkpoint(
                    recovery_dir,
                    model,
                    tokenizer,
                    optimizer,
                    scheduler,
                    scaler,
                    state,
                    generator,
                    config,
                    stage_name,
                    train_hash,
                    validation_hash,
                )

                print("Verified progress checkpoint saved.")

        if position != len(permutation):
            raise RuntimeError("Epoch ended at an unexpected position.")

        if state["global_step"] != epoch_number * config["steps_per_epoch"]:
            raise RuntimeError(
                "Successful optimizer update count does not match "
                "the frozen training plan."
            )

        print(f"\nEpoch {epoch_number} training completed.")
        print("Running validation...")

        metrics = evaluate(
            model,
            tokenizer,
            validation_records,
            config["classes"],
            collator,
        )

        print(
            json.dumps(metrics, indent=2),
            flush=True,
        )

        save_epoch_adapter(
            model,
            tokenizer,
            output_dir,
            stage_name,
            epoch_number,
            metrics,
        )

        state["history"].append({
            "epoch": epoch_number,
            "global_step": state["global_step"],
            "validation_metrics": metrics,
        })

        state["epoch"] += 1

        if state["epoch"] < EPOCHS:
            state["permutation"] = torch.randperm(
                len(train_records),
                generator=generator,
            )

            state["next_position"] = 0

        else:
            state["next_position"] = len(train_records)

        update_elapsed_training_time()
        save_recovery_checkpoint(
            recovery_dir,
            model,
            tokenizer,
            optimizer,
            scheduler,
            scaler,
            state,
            generator,
            config,
            stage_name,
            train_hash,
            validation_hash,
        )

        print(f"Epoch {epoch_number} recovery checkpoint saved.")

    if state["global_step"] != config["total_steps"]:
        raise RuntimeError(
            "Final successful update count does not match the frozen plan."
        )

    if scheduler.state_dict()["last_epoch"] != config["total_steps"]:
        raise RuntimeError(
            "Scheduler step count does not match successful updates."
        )

    update_elapsed_training_time()
    final_record = {
        "stage": stage_name,
        "completed_epochs": EPOCHS,
        "successful_optimizer_updates": state["global_step"],
        "skipped_overflow_attempts": state["skipped_overflow_attempts"],
        "elapsed_training_seconds": float(state["elapsed_training_seconds"]),
        "elapsed_training_hours": float(state["elapsed_training_seconds"] / 3600.0),
        "timing_scope": (
            "Accumulated active process time captured at verified checkpoints; "
            "time since the last verified checkpoint may be absent after an abrupt crash."
        ),
        "history": state["history"],
        "test_records_accessed": False,
    }

    write_json(
        output_dir / "training_summary.json",
        final_record,
    )

    print("\n" + "=" * 65)
    print("HIERARCHICAL STAGE TRAINING COMPLETED")
    print(json.dumps(final_record, indent=2))
    print("=" * 65)


# ============================================================
# 13. COMMAND-LINE ENTRY POINT
# ============================================================


def prepare_checkpoint_only(stage_name):
    """Create and verify a real epoch-zero checkpoint without training."""
    config = STAGES[stage_name]
    recovery_dir = config["output_dir"] / "recovery"

    protected_paths = [
        recovery_dir,
        recovery_dir.with_name(recovery_dir.name + ".previous"),
        recovery_dir.with_name(recovery_dir.name + ".previous.corrupt"),
        recovery_dir.with_name(recovery_dir.name + ".temporary"),
        recovery_dir.with_name(recovery_dir.name + ".corrupt"),
        recovery_dir.with_name(recovery_dir.name + ".failed_restore"),
    ]

    existing = [str(path) for path in protected_paths if path.exists()]
    if existing:
        raise RuntimeError(
            "Checkpoint-only preparation refuses to overwrite existing "
            "recovery data: " + ", ".join(existing)
        )

    print("\\nPreparing full epoch-zero recovery checkpoint.")
    print("Stage:", stage_name)
    print("Training authorization is not required for this operation.")

    result = setup_training(stage_name)

    (
        loaded_config,
        train_records,
        validation_records,
        model,
        tokenizer,
        collator,
        optimizer,
        scheduler,
        scaler,
        generator,
        state,
        loaded_recovery_dir,
        train_hash,
        validation_hash,
    ) = result

    if loaded_recovery_dir != recovery_dir:
        raise RuntimeError("Unexpected recovery directory returned.")

    saved_state = verify_recovery_checkpoint(
        recovery_dir,
        loaded_config,
        stage_name,
        train_hash,
        validation_hash,
    )

    if (
        saved_state["epoch"] != 0
        or saved_state["global_step"] != 0
        or saved_state["next_position"] != 0
        or saved_state["elapsed_training_seconds"] != 0.0
    ):
        raise RuntimeError(
            "Prepared checkpoint is not a clean epoch-zero checkpoint."
        )

    with open(recovery_dir / "manifest.json", "r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    if manifest.get("training_started") is not False:
        raise RuntimeError("Checkpoint manifest incorrectly indicates training.")

    if manifest.get("test_records_accessed") is not False:
        raise RuntimeError("Checkpoint manifest test-access flag is invalid.")

    print("\\nFULL INITIAL RECOVERY CHECKPOINT CREATED AND VERIFIED")
    print("Stage:", stage_name)
    print("Recovery directory:", recovery_dir)
    print("Epoch:", saved_state["epoch"])
    print("Global step:", saved_state["global_step"])
    print("Next position:", saved_state["next_position"])
    print("Training started:", manifest["training_started"])
    print("Test records accessed:", manifest["test_records_accessed"])
    print("Checkpoint files:", len(manifest["checkpoint_files"]))
    print("CHECKPOINT-ONLY PREPARATION: PASS")



def main():
    parser = argparse.ArgumentParser(
        description="PyMETA hierarchical QLoRA training engine"
    )

    parser.add_argument(
        "--stage",
        required=True,
        choices=["stage1", "stage2"],
    )

    parser.add_argument(
        "--check-inputs",
        action="store_true",
        help="Verify stage datasets and configuration without training.",
    )

    parser.add_argument(
        "--prepare-checkpoint-only",
        action="store_true",
        help="Create and verify an epoch-zero recovery checkpoint without training.",
    )

    args = parser.parse_args()

    if args.check_inputs and args.prepare_checkpoint_only:
        parser.error("--check-inputs and --prepare-checkpoint-only cannot be combined.")

    if args.check_inputs:
        verify_environment()
        verify_stage_inputs(args.stage)

        config = STAGES[args.stage]

        train_records = load_records(
            config["train_path"],
            config["train_rows"],
            config["classes"],
        )

        validation_records = load_records(
            config["validation_path"],
            config["validation_rows"],
            config["classes"],
        )

        print("Training records:", len(train_records))
        print("Validation records:", len(validation_records))
        print("INPUT CHECK: PASS")
        return

    if args.prepare_checkpoint_only:
        prepare_checkpoint_only(args.stage)
        return

    train(args.stage)


if __name__ == "__main__":
    main()

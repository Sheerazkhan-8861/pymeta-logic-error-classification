
from __future__ import annotations

import os
import json
import random
import hashlib
import shutil
import math
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    get_cosine_schedule_with_warmup,
)
from peft import PeftModel
from bitsandbytes.optim import PagedAdamW8bit


# ============================================================
# 1. FROZEN PROJECT CONFIGURATION
# ============================================================

ROOT = Path.home() / "pymeta_project"

MODEL_PATH = ROOT / "model_cache/Qwen2.5-Coder-7B-Instruct"
INITIAL_ADAPTER = ROOT / "checkpoints/initial_adapter"
TOKENIZER_PATH = ROOT / "tokenizer"

TRAIN_FILE = ROOT / "data/flat_train_overlay_cleaned_1536.json"
DEV_FILE = ROOT / "data/flat_validation.json"

CONFIG_FILE = ROOT / "config/training_config_frozen.json"
SPEC_FILE = ROOT / "config/training_loop_implementation_spec_v1.json"
AMENDMENT_FILE = ROOT / "config/overflow_retry_policy_amendment_v1_1.json"

OUTPUT_DIR = ROOT / "outputs/training"
RECOVERY_DIR = ROOT / "checkpoints/training_recovery"
BEST_DIR = ROOT / "checkpoints/best_checkpoint"

CONFIG_HASH = "33d374c03ff32634405f604e7f0a9d7abff9eb037f9260d563e8c227ba2c272c"
TRAIN_HASH = "5e063da34c597edeab081e84a7e73eb2fab5a5fc5c56a20dba0b09a125f232a1"
DEV_HASH = "b90c97b5668452c1da1e49b3ab5d78ba80c5bc7bc745276836eaa7d9e18874da"

SPEC_HASH = "945ccbf8f8f2798e151a5aab13760e062c47c425f797f11ba7aa3221bb1202d1"
AMENDMENT_HASH = "b92c4d1a45e467893cd12fd1a052bb287e5c446981f795fb88faacc8016a80b9"

CLASSES = ["No Error", "Logic Error", "Explicit Error"]

EPOCHS = 2
ACCUMULATION = 16
MAX_LENGTH = 1536
MAX_NEW_TOKENS = 16

LEARNING_RATE = 0.0002
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 146
TOTAL_STEPS = 4866
MAX_GRAD_NORM = 1.0

MAX_OVERFLOW_RETRIES = 5
CHECKPOINT_INTERVAL = 250


# ============================================================
# 2. HASH AND INPUT VERIFICATION
# ============================================================

def sha256(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


def verify_file(path, expected_hash):
    if not path.is_file():
        raise FileNotFoundError(path)

    actual = sha256(path)

    if actual != expected_hash:
        raise RuntimeError(
            f"Hash mismatch: {path}\n"
            f"Expected: {expected_hash}\n"
            f"Actual: {actual}"
        )


def verify_inputs():
    verify_file(CONFIG_FILE, CONFIG_HASH)
    verify_file(TRAIN_FILE, TRAIN_HASH)
    verify_file(DEV_FILE, DEV_HASH)
    verify_file(SPEC_FILE, SPEC_HASH)
    verify_file(AMENDMENT_FILE, AMENDMENT_HASH)

    with open(SPEC_FILE, encoding="utf-8") as f:
        spec = json.load(f)

    with open(AMENDMENT_FILE, encoding="utf-8") as f:
        amendment = json.load(f)

    assert spec["training"]["epochs"] == 2
    assert spec["training"]["training_examples"] == 38918
    assert spec["training"]["gradient_accumulation"] == 16
    assert spec["training"]["total_planned_successful_updates"] == 4866

    assert amendment["policy"]["retry_limit"]["max_overflow_retries_per_group"] == 5

    print("[PASS] Frozen configuration hash")
    print("[PASS] Training data hash")
    print("[PASS] Validation data hash")
    print("[PASS] Implementation specification hashes")


# ============================================================
# 3. RNG MANAGEMENT
# ============================================================

def capture_rng(generator):
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
        "data_generator": generator.get_state(),
    }


def restore_rng(state, generator):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state_all(state["cuda"])
    generator.set_state(state["data_generator"])


# ============================================================
# 4. DATA AND COLLATOR
# ============================================================

def load_records(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def initialize_collator(tokenizer):
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    import pymeta_collator

    pymeta_collator.tokenizer = tokenizer
    pymeta_collator.MAX_LENGTH = MAX_LENGTH

    return pymeta_collator.pymeta_collator


# ============================================================
# 5. MODEL INITIALIZATION
# ============================================================

def initialize_model(adapter_path):
    tokenizer = AutoTokenizer.from_pretrained(
        str(TOKENIZER_PATH),
        local_files_only=True,
    )

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    base = AutoModelForCausalLM.from_pretrained(
        str(MODEL_PATH),
        local_files_only=True,
        quantization_config=quantization,
        dtype=torch.float16,
        device_map="auto",
        attn_implementation="sdpa",
    )

    model = PeftModel.from_pretrained(
        base,
        str(adapter_path),
        is_trainable=True,
    )

    model.config.use_cache = False

    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    model.enable_input_require_grads()

    return model, tokenizer


# ============================================================
# 6. STRICT JSON OUTPUT PARSER
# ============================================================

def parse_prediction(text):
    try:
        parsed = json.loads(text.strip())

        if not isinstance(parsed, dict):
            return None

        if set(parsed.keys()) != {"label"}:
            return None

        if parsed["label"] not in CLASSES:
            return None

        return parsed["label"]

    except (json.JSONDecodeError, TypeError, AttributeError):
        return None


# ============================================================
# 7. VALIDATION METRICS
# ============================================================

def calculate_metrics(gold, predictions):
    if len(gold) != len(predictions) or not gold:
        raise ValueError("Invalid evaluation inputs.")

    if any(x not in CLASSES for x in gold):
        raise ValueError("Invalid gold label.")

    confusion = [[0] * 3 for _ in range(3)]
    invalid = 0

    for actual, predicted in zip(gold, predictions):
        if predicted is None:
            invalid += 1
            continue

        if predicted not in CLASSES:
            invalid += 1
            continue

        confusion[CLASSES.index(actual)][CLASSES.index(predicted)] += 1

    f1 = []

    for i in range(3):
        tp = confusion[i][i]
        fp = sum(confusion[j][i] for j in range(3) if j != i)

        # Invalid predictions are false negatives for the gold class.
        fn = sum(
            actual == CLASSES[i] and predicted != CLASSES[i]
            for actual, predicted in zip(gold, predictions)
        )

        denominator = 2 * tp + fp + fn
        f1.append(2 * tp / denominator if denominator else 0.0)

    def rate(actual_label, predicted_label):
        count = sum(x == actual_label for x in gold)

        if count == 0:
            raise ValueError(f"No gold examples for {actual_label}")

        return sum(
            a == actual_label and p == predicted_label
            for a, p in zip(gold, predictions)
        ) / count

    return {
        "macro_f1": float(np.mean(f1)),
        "LER_over": (
            rate("No Error", "Logic Error")
            + rate("Explicit Error", "Logic Error")
        ) / 2,
        "logic_error_recall": rate("Logic Error", "Logic Error"),
        "invalid_output_rate": invalid / len(gold),
        "invalid_predictions": invalid,
        "confusion_matrix": confusion,
    }


def better_checkpoint(candidate, current):
    if current is None:
        return True

    if candidate["macro_f1"] != current["macro_f1"]:
        return candidate["macro_f1"] > current["macro_f1"]

    if candidate["LER_over"] != current["LER_over"]:
        return candidate["LER_over"] < current["LER_over"]

    if candidate["logic_error_recall"] != current["logic_error_recall"]:
        return candidate["logic_error_recall"] > current["logic_error_recall"]

    return False


# ============================================================
# 8. AUTOREGRESSIVE VALIDATION
# ============================================================

def evaluate(model, tokenizer, records):
    model.eval()

    gold = []
    predictions = []

    device = model.get_input_embeddings().weight.device

    for index, record in enumerate(records):
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
        ).to(device)

        with torch.inference_mode():
            with torch.autocast("cuda", dtype=torch.float16):
                output = model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=MAX_NEW_TOKENS,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

        generated = output[0, inputs["input_ids"].shape[1]:]

        decoded = tokenizer.decode(
            generated,
            skip_special_tokens=True,
        )

        predictions.append(parse_prediction(decoded))

        target = json.loads(messages[2]["content"])
        gold.append(target["label"])

        if (index + 1) % 250 == 0:
            print(f"Validation: {index + 1}/{len(records)}")

    metrics = calculate_metrics(gold, predictions)

    model.train()
    return metrics


# ============================================================
# 9. CHECKPOINT CREATION AND INTEGRITY
# ============================================================

def save_checkpoint(
    model,
    tokenizer,
    optimizer,
    scheduler,
    scaler,
    generator,
    state,
):
    temporary = RECOVERY_DIR.with_name(
        RECOVERY_DIR.name + ".temporary"
    )

    if temporary.exists():
        shutil.rmtree(temporary)

    temporary.mkdir(parents=True)

    model.save_pretrained(temporary / "adapter")
    tokenizer.save_pretrained(temporary / "tokenizer")

    payload = {
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": state["epoch"],
        "permutation": state["permutation"],
        "next_position": state["next_position"],
        "global_step": state["global_step"],
        "skipped_attempts": state["skipped_attempts"],
        "history": state["history"],
        "best_metrics": state["best_metrics"],
        "rng": capture_rng(generator),
    }

    torch.save(payload, temporary / "training_state.pt")

    manifest = {
        "checkpoint_type": "full_training_recovery",
        "config_sha256": CONFIG_HASH,
        "train_sha256": TRAIN_HASH,
        "validation_sha256": DEV_HASH,
        "epoch": state["epoch"],
        "global_step": state["global_step"],
        "next_position": state["next_position"],
        "training_started": state["global_step"] > 0,
        "test_records_accessed": False,
    }

    with open(temporary / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    if RECOVERY_DIR.exists():
        backup = RECOVERY_DIR.with_name(RECOVERY_DIR.name + ".previous")

        if backup.exists():
            shutil.rmtree(backup)

        RECOVERY_DIR.rename(backup)
        temporary.rename(RECOVERY_DIR)
        shutil.rmtree(backup)
    else:
        temporary.rename(RECOVERY_DIR)

    verify_checkpoint()


def verify_checkpoint():
    manifest_path = RECOVERY_DIR / "manifest.json"
    state_path = RECOVERY_DIR / "training_state.pt"

    if not manifest_path.is_file() or not state_path.is_file():
        raise RuntimeError("Recovery checkpoint files missing.")

    with open(manifest_path) as f:
        manifest = json.load(f)

    if manifest["config_sha256"] != CONFIG_HASH:
        raise RuntimeError("Checkpoint configuration mismatch.")

    if manifest["train_sha256"] != TRAIN_HASH:
        raise RuntimeError("Checkpoint training data mismatch.")

    if manifest["validation_sha256"] != DEV_HASH:
        raise RuntimeError("Checkpoint validation data mismatch.")

    required = [
        RECOVERY_DIR / "adapter/adapter_config.json",
        RECOVERY_DIR / "adapter/adapter_model.safetensors",
        RECOVERY_DIR / "tokenizer/tokenizer.json",
    ]

    if any(not p.is_file() for p in required):
        raise RuntimeError("Adapter or tokenizer checkpoint incomplete.")

    state = torch.load(
        state_path,
        map_location="cpu",
        weights_only=False,
    )

    for key in [
        "optimizer", "scheduler", "scaler",
        "epoch", "permutation", "next_position",
        "global_step", "skipped_attempts",
        "history", "best_metrics", "rng",
    ]:
        if key not in state:
            raise RuntimeError(f"Checkpoint state missing: {key}")

    print("[PASS] Full recovery checkpoint independently inspected")
    print("[PASS] Adapter and tokenizer present")
    print("[PASS] Optimizer, scheduler and scaler states present")
    print("[PASS] RNG and training position present")


# ============================================================
# 10. TRAINING SETUP
# ============================================================

def create_training_objects(adapter_path):
    model, tokenizer = initialize_model(adapter_path)

    optimizer = PagedAdamW8bit(
        [p for p in model.parameters() if p.requires_grad],
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=WARMUP_STEPS,
        num_training_steps=TOTAL_STEPS,
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        init_scale=65536.0,
        enabled=True,
    )

    return model, tokenizer, optimizer, scheduler, scaler


# ============================================================
# 11. RESUME OR CREATE INITIAL RECOVERY CHECKPOINT
# ============================================================

def setup_training():
    verify_inputs()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU unavailable.")

    train_records = load_records(TRAIN_FILE)
    dev_records = load_records(DEV_FILE)

    if len(train_records) != 38918:
        raise RuntimeError("Unexpected training record count.")

    if len(dev_records) != 4863:
        raise RuntimeError("Unexpected validation record count.")

    if RECOVERY_DIR.exists():
        verify_checkpoint()
        adapter_path = RECOVERY_DIR / "adapter"
    else:
        adapter_path = INITIAL_ADAPTER

    model, tokenizer, optimizer, scheduler, scaler = (
        create_training_objects(adapter_path)
    )

    generator = torch.Generator()
    generator.manual_seed(42)

    if RECOVERY_DIR.exists():
        state = torch.load(
            RECOVERY_DIR / "training_state.pt",
            map_location="cpu",
            weights_only=False,
        )

        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])

        restore_rng(state["rng"], generator)

    else:
        permutation = torch.randperm(
            len(train_records),
            generator=generator,
        ).tolist()

        state = {
            "epoch": 0,
            "permutation": permutation,
            "next_position": 0,
            "global_step": 0,
            "skipped_attempts": 0,
            "history": [],
            "best_metrics": None,
        }

        save_checkpoint(
            model, tokenizer, optimizer, scheduler,
            scaler, generator, state,
        )

        print("[PASS] Initial full training recovery checkpoint created.")

    return (
        model, tokenizer, optimizer, scheduler, scaler,
        generator, state, train_records, dev_records,
    )


# ============================================================
# 12. OVERFLOW-SAFE TRAINING
# ============================================================

def train():
    (
        model, tokenizer, optimizer, scheduler, scaler,
        generator, state, train_records, dev_records,
    ) = setup_training()

    collator = initialize_collator(tokenizer)

    model.train()
    device = model.get_input_embeddings().weight.device

    while state["epoch"] < EPOCHS:
        epoch = state["epoch"]
        permutation = state["permutation"]
        position = state["next_position"]

        while position < len(permutation):
            group = permutation[position:position + ACCUMULATION]
            group_size = len(group)

            rng_snapshot = capture_rng(generator)
            retries = 0

            while True:
                optimizer.zero_grad(set_to_none=True)

                for index in group:
                    batch = collator([train_records[index]])

                    batch = {
                        key: value.to(device)
                        for key, value in batch.items()
                    }

                    with torch.autocast("cuda", dtype=torch.float16):
                        output = model(**batch)
                        loss = output.loss / group_size

                    scaler.scale(loss).backward()

                scaler.unscale_(optimizer)

                gradients_finite = all(
                    parameter.grad is None
                    or torch.isfinite(parameter.grad).all().item()
                    for parameter in model.parameters()
                    if parameter.requires_grad
                )

                old_scale = scaler.get_scale()

                if gradients_finite:
                    clip_grad_norm_(
                        [
                            p for p in model.parameters()
                            if p.requires_grad
                        ],
                        MAX_GRAD_NORM,
                    )

                scaler.step(optimizer)
                scaler.update()

                new_scale = scaler.get_scale()

                if gradients_finite and new_scale >= old_scale:
                    scheduler.step()
                    state["global_step"] += 1
                    position += group_size
                    state["next_position"] = position

                    optimizer.zero_grad(set_to_none=True)
                    break

                state["skipped_attempts"] += 1
                retries += 1
                optimizer.zero_grad(set_to_none=True)

                if retries > MAX_OVERFLOW_RETRIES:
                    raise RuntimeError(
                        "Maximum overflow retries exceeded. "
                        "Last verified checkpoint remains available."
                    )

                restore_rng(rng_snapshot, generator)

                print(
                    f"Overflow retry {retries}/{MAX_OVERFLOW_RETRIES} "
                    f"at epoch {epoch + 1}, position {position}"
                )

            if state["global_step"] % CHECKPOINT_INTERVAL == 0:
                save_checkpoint(
                    model, tokenizer, optimizer, scheduler,
                    scaler, generator, state,
                )

            if state["global_step"] % 10 == 0:
                print(
                    f"Epoch {epoch + 1}/{EPOCHS} | "
                    f"Update {state['global_step']}/{TOTAL_STEPS} | "
                    f"Position {position}/{len(permutation)}"
                )

        print(f"Epoch {epoch + 1}: validation started.")

        metrics = evaluate(model, tokenizer, dev_records)

        state["history"].append({
            "epoch": epoch + 1,
            "global_step": state["global_step"],
            "metrics": metrics,
        })

        print("Validation metrics:", json.dumps(metrics, indent=2))

        if better_checkpoint(metrics, state["best_metrics"]):
            state["best_metrics"] = metrics

            temporary_best = BEST_DIR.with_name(BEST_DIR.name + ".temporary")

            if temporary_best.exists():
                shutil.rmtree(temporary_best)

            temporary_best.mkdir(parents=True)

            model.save_pretrained(temporary_best / "adapter")
            tokenizer.save_pretrained(temporary_best / "tokenizer")

            with open(temporary_best / "metrics.json", "w") as f:
                json.dump(metrics, f, indent=2)

            if BEST_DIR.exists():
                shutil.rmtree(BEST_DIR)

            temporary_best.rename(BEST_DIR)

            print("[PASS] Best validation checkpoint updated.")

        state["epoch"] += 1
        state["next_position"] = 0

        if state["epoch"] < EPOCHS:
            next_permutation = torch.randperm(
                len(train_records),
                generator=generator,
            ).tolist()

            state["permutation"] = next_permutation

        save_checkpoint(
            model, tokenizer, optimizer, scheduler,
            scaler, generator, state,
        )

        position = 0
        model.train()

    if state["global_step"] != TOTAL_STEPS:
        raise RuntimeError(
            f"Expected {TOTAL_STEPS} successful updates, "
            f"got {state['global_step']}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_DIR / "training_history.json", "w") as f:
        json.dump(state["history"], f, indent=2)

    print("=" * 65)
    print("TRAINING COMPLETE")
    print("Successful optimizer updates:", state["global_step"])
    print("Skipped overflow attempts:", state["skipped_attempts"])
    print("Best validation metrics:", state["best_metrics"])
    print("Test dataset accessed: NO")
    print("=" * 65)


if __name__ == "__main__":
    if os.environ.get("PYMETA_START_TRAINING") != "YES":
        raise SystemExit(
            "Training launch locked. Set PYMETA_START_TRAINING=YES "
            "only after independent preflight approval."
        )

    train()
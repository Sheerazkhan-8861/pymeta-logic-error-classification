
"""You are an expert Python programming evaluator.

Your task is to classify a student's Python code submission
into exactly one of the following three categories:

1. No Error
   The submitted code correctly solves the programming question.

2. Logic Error
   The submitted code is syntactically valid and does not contain
   an explicit programming error, but its logic or algorithm does
   not correctly solve the programming question.

3. Explicit Error
   The submitted code contains an explicit programming error,
   such as a syntax error, indentation error, undefined name,
   type error, or another detectable programming error.

Evaluate the student's submission using the programming question
and reference solution as context.

Important instructions:
- Do not assume code is correct merely because it looks plausible.
- Distinguish logical mistakes from explicit programming errors.
- Do not invent execution results.
- Do not use information that is not provided in the input.
- Select exactly one category.
- Do not provide explanations or additional text.

Return your answer in exactly this JSON format:

{"label": "<one allowed label>"}

The label must be exactly one of:
"No Error", "Logic Error", "Explicit Error".
"""

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel


# ============================================================
# PyMETA — FROZEN TEST EVALUATOR
# ============================================================

ROOT = Path.home() / "pymeta_project"

MODEL_PATH = ROOT / "model_cache/Qwen2.5-Coder-7B-Instruct"
TOKENIZER_PATH = ROOT / "tokenizer"

TRAIN_FILE = ROOT / "data/flat_train_overlay_cleaned_1536.json"
DEV_FILE = ROOT / "data/flat_validation.json"
TEST_FILE = ROOT / "data/official_test_source/test.csv"
METADATA_FILE = ROOT / "data/official_metadata/question_groups.csv"

ADAPTER_PATH = ROOT / "checkpoints/best_checkpoint/adapter"

OUTPUT_DIR = ROOT / "outputs/test_evaluation"

EXPECTED_TEST_SHA256 = (
    "314fd14743cde7fbb029fa9585cce3da859e3d5da14963d7cfe985a543af5832"
)

EXPECTED_METADATA_SHA256 = (
    "24c1e36cd7052fe478726647b8d54d96a4d4bdfa4bd68026e07e4f03715d152e"
)

EXPECTED_TEST_ROWS = 4865
MAX_LENGTH = 1536
MAX_NEW_TOKENS = 16

CLASSES = ["No Error", "Logic Error", "Explicit Error"]

EXPECTED_TEST_CLASS_COUNTS = {
    "No Error": 1510,
    "Logic Error": 1813,
    "Explicit Error": 1542,
}

SYSTEM_PROMPT = 'You are an expert Python programming evaluator.\n\nYour task is to classify a student\'s Python code submission\ninto exactly one of the following three categories:\n\n1. No Error\n   The submitted code correctly solves the programming question.\n\n2. Logic Error\n   The submitted code is syntactically valid and does not contain\n   an explicit programming error, but its logic or algorithm does\n   not correctly solve the programming question.\n\n3. Explicit Error\n   The submitted code contains an explicit programming error,\n   such as a syntax error, indentation error, undefined name,\n   type error, or another detectable programming error.\n\nEvaluate the student\'s submission using the programming question\nand reference solution as context.\n\nImportant instructions:\n- Do not assume code is correct merely because it looks plausible.\n- Distinguish logical mistakes from explicit programming errors.\n- Do not invent execution results.\n- Do not use information that is not provided in the input.\n- Select exactly one category.\n- Do not provide explanations or additional text.\n\nReturn your answer in exactly this JSON format:\n\n{"label": "<one allowed label>"}\n\nThe label must be exactly one of:\n"No Error", "Logic Error", "Explicit Error".\n'

USER_TEMPLATE = """Classify the following student submission.

Question:
{question}

Reference Solution:
{exceptedAnswer}

Student Code:
{studentAnswer}

Return only the required JSON classification.
"""


# ============================================================
# GENERAL UTILITIES
# ============================================================

def sha256_file(path):
    digest = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def require_file(path):
    if not path.is_file():
        raise FileNotFoundError(f"Required file missing: {path}")


def verify_hash(path, expected):
    require_file(path)

    actual = sha256_file(path)

    print(f"\nFile: {path}")
    print(f"Expected SHA256: {expected}")
    print(f"Actual SHA256:   {actual}")

    if actual != expected:
        raise RuntimeError(f"SHA256 mismatch: {path}")

    print("[PASS] SHA256 verified")


def map_gold_label(raw_label):
    if raw_label == "No error":
        return "No Error"

    if raw_label == "LogicError":
        return "Logic Error"

    if not raw_label:
        raise ValueError("Missing all_errortype2 label")

    return "Explicit Error"


def clean_question_overlay(question):
    """
    Historical approved transformation:
    clean U+0336 crossed-out characters in QUESTION only.
    Preserve all other fields exactly.
    """
    paired = re.findall(r".\u0336", question)

    cleaned = re.sub(r".\u0336", "", question)
    cleaned = cleaned.replace("\u0336", "")

    paired_count = len(paired)
    total_marks = question.count("\u0336")
    isolated_count = total_marks - paired_count

    return cleaned, paired_count, isolated_count


def normalize_question_id(value):
    value = str(value).strip()

    if value.endswith(".0"):
        value = value[:-2]

    return value


# ============================================================
# DATASET METADATA AUDIT
# ============================================================

def load_and_validate_metadata():
    verify_hash(METADATA_FILE, EXPECTED_METADATA_SHA256)

    with open(METADATA_FILE, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    required = {"questionId", "split"}

    if not required.issubset(reader.fieldnames or []):
        raise RuntimeError("Official metadata is missing required columns")

    if len(rows) != 153:
        raise RuntimeError(
            f"Expected 153 metadata QuestionIDs, found {len(rows)}"
        )

    mapping = {}

    for row in rows:
        qid = normalize_question_id(row["questionId"])
        split = row["split"].strip().lower()

        if qid in mapping:
            raise RuntimeError(f"Duplicate QuestionID in metadata: {qid}")

        if split not in {"train", "dev", "test"}:
            raise RuntimeError(f"Unexpected split for QID {qid}: {split}")

        mapping[qid] = split

    split_counts = Counter(mapping.values())

    expected = {
        "train": 133,
        "dev": 11,
        "test": 9,
    }

    if dict(split_counts) != expected:
        raise RuntimeError(
            f"Unexpected metadata split counts: {dict(split_counts)}"
        )

    print("\n[PASS] Official metadata structure")
    print("QuestionID count:", len(mapping))
    print("Split counts:", dict(split_counts))

    return mapping


# ============================================================
# MODEL AND TOKENIZER
# ============================================================

def load_tokenizer():
    require_file(TOKENIZER_PATH / "tokenizer.json")

    print("\\nLoading tokenizer independently...")

    return AutoTokenizer.from_pretrained(
        TOKENIZER_PATH,
        local_files_only=True,
    )


def load_inference_model(tokenizer=None):
    require_file(MODEL_PATH / "config.json")
    require_file(TOKENIZER_PATH / "tokenizer.json")
    require_file(ADAPTER_PATH / "adapter_config.json")
    require_file(ADAPTER_PATH / "adapter_model.safetensors")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Refusing inference.")

    if tokenizer is None:
        tokenizer = load_tokenizer()

    print("Loading 4-bit base model...")

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
        quantization_config=quant_config,
        torch_dtype=torch.float16,
        device_map="auto",
        attn_implementation="sdpa",
    )

    print("Loading best adapter in inference mode...")

    model = PeftModel.from_pretrained(
        base_model,
        ADAPTER_PATH,
        is_trainable=False,
    )

    model.eval()
    model.config.use_cache = False

    for parameter in model.parameters():
        if parameter.requires_grad:
            raise RuntimeError(
                "Trainable parameter found in inference model"
            )

    print("[PASS] Model loaded in inference-only mode")
    print("GPU:", torch.cuda.get_device_name(0))

    return model, tokenizer


# ============================================================
# PROMPT AND TOKENIZATION
# ============================================================

def make_messages(question, reference, student_code):
    user_content = USER_TEMPLATE.format(
        question=question,
        exceptedAnswer=reference,
        studentAnswer=student_code,
    )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]



def tokenize_prompt(tokenizer, messages, enforce_max_length=True):
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    encoded = tokenizer(
        rendered,
        add_special_tokens=False,
        truncation=False,
        return_tensors="pt",
    )

    input_length = encoded["input_ids"].shape[1]

    if enforce_max_length and input_length > MAX_LENGTH:
        raise RuntimeError(
            f"Prompt exceeds {MAX_LENGTH} tokens: {input_length}. "
            "Truncation is prohibited."
        )

    return encoded, input_length

# ============================================================
# STRICT OUTPUT PARSER
# ============================================================

def parse_prediction(text):
    try:
        parsed = json.loads(text.strip())
    except (json.JSONDecodeError, TypeError):
        return None

    if not isinstance(parsed, dict):
        return None

    if set(parsed.keys()) != {"label"}:
        return None

    label = parsed["label"]

    if label not in CLASSES:
        return None

    return label


# ============================================================
# METRICS — MATCH VALIDATION LOGIC
# ============================================================

def calculate_metrics(gold, predictions):
    if len(gold) != len(predictions):
        raise ValueError("Gold/prediction length mismatch")

    if not gold:
        raise ValueError("Empty evaluation dataset")

    confusion = [[0] * 3 for _ in range(3)]

    class_to_index = {
        label: i for i, label in enumerate(CLASSES)
    }

    invalid_count = 0

    for actual, predicted in zip(gold, predictions):
        if actual not in class_to_index:
            raise ValueError(f"Invalid gold label: {actual}")

        if predicted not in class_to_index:
            invalid_count += 1
            continue

        confusion[
            class_to_index[actual]
        ][
            class_to_index[predicted]
        ] += 1

    f1_scores = []

    for i in range(3):
        tp = confusion[i][i]

        fp = sum(confusion[r][i] for r in range(3)) - tp

        # Invalid predictions are false negatives for their gold class.
        fn = sum(confusion[i]) - tp + sum(
            1
            for actual, predicted in zip(gold, predictions)
            if actual == CLASSES[i] and predicted not in class_to_index
        )

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0

        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0.0
        )

        f1_scores.append(f1)

    no_error_index = class_to_index["No Error"]
    logic_index = class_to_index["Logic Error"]
    explicit_index = class_to_index["Explicit Error"]

    no_error_total = sum(confusion[no_error_index]) + sum(
        1 for a, p in zip(gold, predictions)
        if a == "No Error" and p not in class_to_index
    )

    explicit_total = sum(confusion[explicit_index]) + sum(
        1 for a, p in zip(gold, predictions)
        if a == "Explicit Error" and p not in class_to_index
    )

    logic_over_no_error = (
        confusion[no_error_index][logic_index] / no_error_total
        if no_error_total else 0.0
    )

    logic_over_explicit = (
        confusion[explicit_index][logic_index] / explicit_total
        if explicit_total else 0.0
    )

    logic_total = sum(confusion[logic_index]) + sum(
        1 for a, p in zip(gold, predictions)
        if a == "Logic Error" and p not in class_to_index
    )

    logic_recall = (
        confusion[logic_index][logic_index] / logic_total
        if logic_total else 0.0
    )

    correct = sum(
        1 for a, p in zip(gold, predictions)
        if a == p
    )

    return {
        "accuracy": correct / len(gold),
        "macro_f1": sum(f1_scores) / 3,
        "class_f1": dict(zip(CLASSES, f1_scores)),
        "LER_over": (
            logic_over_no_error + logic_over_explicit
        ) / 2,
        "logic_error_recall": logic_recall,
        "invalid_output_rate": invalid_count / len(gold),
        "invalid_predictions": invalid_count,
        "confusion_matrix": confusion,
        "confusion_matrix_rows": "Actual",
        "confusion_matrix_columns": "Predicted",
    }


# ============================================================
# INFERENCE
# ============================================================


def generate_one(model, tokenizer, encoded, prompt_length):
    device = model.get_input_embeddings().weight.device

    encoded = {
        key: value.to(device)
        for key, value in encoded.items()
    }

    with torch.inference_mode():
        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
        ):
            output = model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=MAX_NEW_TOKENS,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

    generated_ids = output[0][encoded["input_ids"].shape[1]:]

    raw_output = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    )

    prediction = parse_prediction(raw_output)

    return prediction, raw_output, prompt_length

# ============================================================
# VALIDATION-ONLY DRY RUN
# ============================================================

def run_preflight():
    print("=" * 70)
    print("PyMETA TEST EVALUATOR — PREFLIGHT")
    print("=" * 70)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")

    verify_hash(TEST_FILE, EXPECTED_TEST_SHA256)

    metadata = load_and_validate_metadata()

    if len(metadata) != 153:
        raise RuntimeError("Metadata audit failed")

    require_file(TRAIN_FILE)
    require_file(DEV_FILE)

    print("\nChecking frozen processed dataset hashes...")

    verify_hash(
        TRAIN_FILE,
        "5e063da34c597edeab081e84a7e73eb2fab5a5fc5c56a20dba0b09a125f232a1",
    )

    verify_hash(
        DEV_FILE,
        "b90c97b5668452c1da1e49b3ab5d78ba80c5bc7bc745276836eaa7d9e18874da",
    )

    print("\nLoading validation artifact for dry run...")

    with open(DEV_FILE, "r", encoding="utf-8") as f:
        validation_records = json.load(f)

    if len(validation_records) != 4863:
        raise RuntimeError("Unexpected validation record count")

    sample = validation_records[0]["messages"]

    if len(sample) != 3:
        raise RuntimeError("Unexpected validation message structure")

    if sample[0]["content"] != SYSTEM_PROMPT:
        raise RuntimeError(
            "Evaluator system prompt does not exactly match training"
        )

    model, tokenizer = load_inference_model()

    question = sample[1]["content"]

    # Use the existing validation example unchanged.
    encoded, prompt_length = tokenize_prompt(
        tokenizer,
        sample[:2],
    )

    print("\nValidation prompt token length:", prompt_length)

    if prompt_length > MAX_LENGTH:
        raise RuntimeError("Validation dry-run prompt too long")

    print("\nPerforming one validation-only generation...")

    device = model.get_input_embeddings().weight.device

    encoded = {
        key: value.to(device)
        for key, value in encoded.items()
    }

    with torch.inference_mode():
        with torch.autocast("cuda", dtype=torch.float16):
            output = model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=MAX_NEW_TOKENS,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

    generated_ids = output[0][encoded["input_ids"].shape[1]:]

    raw = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    )

    print("Dry-run raw output:", repr(raw))
    print("Parsed prediction:", parse_prediction(raw))

    del model
    torch.cuda.empty_cache()

    print("\n" + "=" * 70)
    print("PREFLIGHT: PASS")
    print("Test CSV rows accessed: NO")
    print("Test labels accessed: NO")
    print("Training started: NO")
    print("Test evaluation started: NO")
    print("=" * 70)


# ============================================================
# FINAL TEST EVALUATION
# ============================================================

def run_evaluation():
    print("=" * 70)
    print("PyMETA — FINAL TEST EVALUATION")
    print("=" * 70)

    # Hash verification MUST happen before CSV row reading.
    verify_hash(TEST_FILE, EXPECTED_TEST_SHA256)

    metadata = load_and_validate_metadata()

    print("\nLoading official test rows after explicit confirmation...")

    with open(TEST_FILE, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        required = {
            "questionId",
            "question",
            "exceptedAnswer",
            "studentAnswer",
            "all_errortype2",
        }

        if not required.issubset(reader.fieldnames or []):
            raise RuntimeError("Test CSV missing required columns")

        rows = list(reader)

    if len(rows) != EXPECTED_TEST_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_TEST_ROWS} test rows, found {len(rows)}"
        )

    gold = []
    question_ids = []
    transformed_question_count = 0
    paired_marks_total = 0
    isolated_marks_total = 0

    prepared = []

    for index, row in enumerate(rows):
        qid = normalize_question_id(row["questionId"])

        if qid not in metadata:
            raise RuntimeError(f"Unknown test QuestionID: {qid}")

        if metadata[qid] != "test":
            raise RuntimeError(
                f"Split leakage: QID {qid} belongs to {metadata[qid]}"
            )

        question = row["question"]
        reference = row["exceptedAnswer"]
        student_code = row["studentAnswer"]

        if question is None or reference is None or student_code is None:
            raise RuntimeError(f"Missing permitted input at row {index}")

        # Preserve the official question exactly; historical overlay cleaning is unverified.
        cleaned_question = question
        paired = 0
        isolated = 0

        if paired or isolated:
            transformed_question_count += 1

        paired_marks_total += paired
        isolated_marks_total += isolated

        label = map_gold_label(row["all_errortype2"])

        gold.append(label)
        question_ids.append(qid)

        prepared.append({
            "row_index": index,
            "question": cleaned_question,
            "reference": reference,
            "student_code": student_code,
            "gold": label,
        })

    qid_counts = Counter(question_ids)

    expected_test_qids = {
        qid for qid, split in metadata.items()
        if split == "test"
    }

    if set(question_ids) != expected_test_qids:
        raise RuntimeError("Test QuestionID set differs from official metadata")

    observed_class_counts = dict(Counter(gold))

    if observed_class_counts != EXPECTED_TEST_CLASS_COUNTS:
        raise RuntimeError(
            "Test class distribution mismatch. "
            f"Observed: {observed_class_counts}"
        )

    print("\n[PASS] Test row count:", len(prepared))
    print("[PASS] Test QuestionID count:", len(qid_counts))
    print("[PASS] All QuestionIDs belong to official test split")
    print("[PASS] Gold label mapping and class counts")
    print("Class counts:", observed_class_counts)
    print("Questions with overlay marks:", transformed_question_count)
    print("Paired overlay marks:", paired_marks_total)
    print("Isolated overlay marks:", isolated_marks_total)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    predictions_path = OUTPUT_DIR / "test_predictions.jsonl"
    metrics_path = OUTPUT_DIR / "test_metrics.json"
    audit_path = OUTPUT_DIR / "test_evaluation_audit.json"

    if predictions_path.exists() or metrics_path.exists() or audit_path.exists():
        raise RuntimeError(
            "Evaluation output already exists. Refusing to overwrite."
        )


    # --------------------------------------------------------
    # FULL TEST PROMPT LENGTH AUDIT — BEFORE MODEL LOADING
    # --------------------------------------------------------

    tokenizer = load_tokenizer()

    audited_prepared = []
    over_limit = []
    prompt_lengths = []

    if len(prepared) != EXPECTED_TEST_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_TEST_ROWS} prepared records, "
            f"found {len(prepared)}"
        )

    print("\nAuditing every test prompt before model loading...")

    for index, record in enumerate(prepared):
        messages = make_messages(
            record["question"],
            record["reference"],
            record["student_code"],
        )

        encoded, prompt_length = tokenize_prompt(
            tokenizer,
            messages,
            enforce_max_length=False,
        )

        prompt_lengths.append(prompt_length)

        audited_record = dict(record)
        audited_record["encoded"] = encoded
        audited_record["prompt_length"] = prompt_length
        audited_prepared.append(audited_record)

        if prompt_length > MAX_LENGTH:
            over_limit.append({
                "row_index": record["row_index"],
                "questionId": question_ids[index],
                "prompt_tokens": prompt_length,
            })

    if len(audited_prepared) != EXPECTED_TEST_ROWS:
        raise RuntimeError("Full prompt audit is incomplete.")

    print("[AUDIT] Prompts checked:", len(audited_prepared))
    print("[AUDIT] Maximum allowed:", MAX_LENGTH)
    print("[AUDIT] Minimum prompt length:", min(prompt_lengths))
    print("[AUDIT] Maximum prompt length:", max(prompt_lengths))
    print("[AUDIT] Over-limit prompts:", len(over_limit))

    if over_limit:
        print("\nOVER-LIMIT PROMPTS (first 50):")
        for item in over_limit[:50]:
            print(item)

        raise RuntimeError(
            f"Prompt audit FAILED: {len(over_limit)} prompts exceed "
            f"{MAX_LENGTH} tokens. Model loading and prediction "
            "were prevented."
        )

    print("[PASS] All test prompts are within the sequence limit.")
    print("[PASS] Truncation was not used.")

    # Model loading happens only after the complete audit passes.
    model, tokenizer = load_inference_model(tokenizer=tokenizer)

    predictions = []
    invalid_count = 0

    with open(predictions_path, "w", encoding="utf-8") as out:
        for index, record in enumerate(audited_prepared):
            prediction, raw, prompt_length = generate_one(
                model,
                tokenizer,
                record["encoded"],
                record["prompt_length"],
            )

            predictions.append(prediction)

            if prediction is None:
                invalid_count += 1

            output_record = {
                "row_index": record["row_index"],
                "questionId": question_ids[index],
                "gold": record["gold"],
                "prediction": prediction,
                "raw_output": raw,
                "prompt_tokens": prompt_length,
            }

            out.write(json.dumps(output_record, ensure_ascii=False) + "\n")
            out.flush()

            if (index + 1) % 250 == 0:
                print(
                    f"Completed {index + 1}/{len(prepared)} "
                    f"| invalid={invalid_count}"
                )

    metrics = calculate_metrics(gold, predictions)

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    audit = {
        "project": "PyMETA",
        "evaluation": "Official held-out test evaluation",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_repository": "CircleCat/pymeta",
        "dataset_revision": "db66bd04916cdbefee9ad66774b4598e5f2a3ba0",
        "test_file": str(TEST_FILE),
        "test_sha256": EXPECTED_TEST_SHA256,
        "metadata_sha256": EXPECTED_METADATA_SHA256,
        "test_records": len(prepared),
        "test_question_ids": len(qid_counts),
        "class_counts": observed_class_counts,
        "question_preprocessing": {
            "field": "question",
            "transformation_applied": False,
            "policy": "Official test question preserved unchanged",
            "reason": (
                "Historical training overlay-cleaning procedure "
                "could not be independently verified."
            ),
            "reproducibility_limitation": True,
        },
        "qid50_training_exception_applied_to_test": False,
        "permitted_model_inputs": [
            "question",
            "exceptedAnswer",
            "studentAnswer",
        ],
        "max_sequence_length": MAX_LENGTH,
        "truncation": False,
        "max_new_tokens": MAX_NEW_TOKENS,
        "generation": "greedy",
        "adapter": str(ADAPTER_PATH),
        "prompt_token_min": min(prompt_lengths),
        "prompt_token_max": max(prompt_lengths),
        "metrics_file": str(metrics_path),
        "predictions_file": str(predictions_path),
        "test_used_for_training": False,
        "test_used_for_checkpoint_selection": False,
    }

    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print("TEST EVALUATION COMPLETE")
    print("=" * 70)
    print(json.dumps(metrics, indent=2))
    print("\nSaved:")
    print(predictions_path)
    print(metrics_path)
    print(audit_path)


# ============================================================
# COMMAND LINE
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="PyMETA frozen test evaluator"
    )

    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Run validation-only dry run without reading test rows",
    )

    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run final held-out test evaluation",
    )

    parser.add_argument(
        "--confirm-test-access",
        action="store_true",
        help="Explicitly authorize reading test records and labels",
    )

    args = parser.parse_args()

    if args.preflight and args.evaluate:
        parser.error("Choose only one mode")

    if args.preflight:
        if args.confirm_test_access:
            parser.error("--confirm-test-access is not needed for preflight")

        run_preflight()
        return

    if args.evaluate:
        if not args.confirm_test_access:
            parser.error(
                "Test access blocked. Add --confirm-test-access "
                "only after preflight passes and you approve."
            )

        run_evaluation()
        return

    parser.error("Specify --preflight or --evaluate")


if __name__ == "__main__":
    main()

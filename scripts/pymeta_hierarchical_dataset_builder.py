import json
import hashlib
import os
from pathlib import Path
from collections import Counter

ROOT = Path.home() / "pymeta_project"
DATA = ROOT / "data"
OUT = ROOT / "outputs" / "hierarchical_preparation"

SOURCES = {
    "train": DATA / "flat_train_overlay_cleaned_1536.json",
    "validation": DATA / "flat_validation.json",
}

EXPECTED_ROWS = {"train": 38918, "validation": 4863}

EXPECTED_LABELS = {
    "train": {
        "No Error": 19822,
        "Logic Error": 8079,
        "Explicit Error": 11017,
    },
    "validation": {
        "No Error": 1875,
        "Logic Error": 1495,
        "Explicit Error": 1493,
    },
}

OUTPUTS = {
    ("stage1", "train"): DATA / "hierarchical_stage1_train.json",
    ("stage1", "validation"): DATA / "hierarchical_stage1_validation.json",
    ("stage2", "train"): DATA / "hierarchical_stage2_train.json",
    ("stage2", "validation"): DATA / "hierarchical_stage2_validation.json",
}


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json_write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")

    try:
        with temp.open("w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, separators=(",", ":"))
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def read_and_validate(split, path):
    if not path.is_file():
        raise FileNotFoundError(f"Missing source file: {path}")

    records = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(records, list):
        raise ValueError(f"{split}: source must contain a JSON list")

    if len(records) != EXPECTED_ROWS[split]:
        raise ValueError(
            f"{split}: expected {EXPECTED_ROWS[split]} rows, got {len(records)}"
        )

    counts = Counter()
    validated = []

    for i, record in enumerate(records):
        try:
            messages = record["messages"]
            if len(messages) != 3:
                raise ValueError("expected exactly three messages")

            if [m["role"] for m in messages] != [
                "system", "user", "assistant"
            ]:
                raise ValueError("unexpected message roles")

            gold = json.loads(messages[2]["content"])["label"]

            if gold not in ("No Error", "Logic Error", "Explicit Error"):
                raise ValueError(f"unknown gold label: {gold}")

            if not isinstance(messages[1]["content"], str):
                raise ValueError("user message is not a string")

            counts[gold] += 1
            validated.append((messages[1]["content"], gold))

        except Exception as exc:
            raise ValueError(
                f"{split}: invalid record at index {i}: {exc}"
            ) from exc

    if dict(counts) != EXPECTED_LABELS[split]:
        raise ValueError(
            f"{split}: label counts mismatch.\n"
            f"Expected: {EXPECTED_LABELS[split]}\n"
            f"Actual:   {dict(counts)}"
        )

    return validated, counts


def make_record(system_prompt, user_message, target):
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
            {
                "role": "assistant",
                "content": json.dumps(
                    {"label": target},
                    ensure_ascii=False
                ),
            },
        ]
    }


STAGE1_SYSTEM = """You are a Python programming evaluator.
Classify the student's submission into exactly one category:
- Explicit Error: the code contains a detectable programming error,
  such as syntax, indentation, undefined-name, or type errors.
- Non-Explicit: the code has no explicit programming error.
  This includes both logically incorrect and correct solutions.
Use only the question, reference solution, and student code provided.
Do not invent execution results.
Return only this JSON format:
{"label": "<label>"}
Allowed labels: "Explicit Error", "Non-Explicit"."""

STAGE2_SYSTEM = """You are a Python programming evaluator.
The submission has already been classified as having no explicit
programming error. Decide whether its logic correctly solves the
programming question.
Classify into exactly one category:
- Logic Error: syntactically valid code whose logic does not correctly
  solve the question.
- No Error: code that correctly solves the question.
Use only the question, reference solution, and student code provided.
Do not invent execution results.
Return only this JSON format:
{"label": "<label>"}
Allowed labels: "Logic Error", "No Error"."""


def build_stage1(user_message, gold):
    target = "Explicit Error" if gold == "Explicit Error" else "Non-Explicit"
    return make_record(STAGE1_SYSTEM, user_message, target)


def build_stage2(user_message, gold):
    if gold == "Explicit Error":
        return None
    return make_record(STAGE2_SYSTEM, user_message, gold)


def verify_output(path, expected_count, expected_targets):
    records = json.loads(path.read_text(encoding="utf-8"))

    if len(records) != expected_count:
        raise ValueError(
            f"{path.name}: expected {expected_count}, got {len(records)}"
        )

    counts = Counter()
    for i, record in enumerate(records):
        messages = record["messages"]
        target = json.loads(messages[2]["content"])["label"]
        counts[target] += 1

        if messages[0]["content"] not in (STAGE1_SYSTEM, STAGE2_SYSTEM):
            raise ValueError(f"{path.name}: unexpected system prompt at {i}")

    if dict(counts) != expected_targets:
        raise ValueError(
            f"{path.name}: target mismatch: {dict(counts)}"
        )

    return len(records), dict(counts), sha256(path)


def main():
    print("=" * 65)
    print("PyMETA HIERARCHICAL DATASET BUILDER")
    print("=" * 65)

    # Validate both complete source splits before writing any output.
    source_data = {}
    source_hashes = {}

    for split, path in SOURCES.items():
        print(f"\nValidating source: {path.name}")
        source_data[split], counts = read_and_validate(split, path)
        source_hashes[split] = sha256(path)
        print("Rows:", len(source_data[split]))
        print("Original labels:", dict(counts))
        print("Source SHA-256:", source_hashes[split])

    generated = {}

    for split, rows in source_data.items():
        stage1_records = []
        stage2_records = []

        for user_message, gold in rows:
            stage1_records.append(build_stage1(user_message, gold))
            stage2_record = build_stage2(user_message, gold)
            if stage2_record is not None:
                stage2_records.append(stage2_record)

        generated[("stage1", split)] = stage1_records
        generated[("stage2", split)] = stage2_records

    expected_targets = {
        ("stage1", "train"): {
            "Explicit Error": 11017, "Non-Explicit": 27901
        },
        ("stage1", "validation"): {
            "Explicit Error": 1493, "Non-Explicit": 3370
        },
        ("stage2", "train"): {
            "Logic Error": 8079, "No Error": 19822
        },
        ("stage2", "validation"): {
            "Logic Error": 1495, "No Error": 1875
        },
    }

    expected_counts = {
        ("stage1", "train"): 38918,
        ("stage1", "validation"): 4863,
        ("stage2", "train"): 27901,
        ("stage2", "validation"): 3370,
    }

    print("\nWriting four datasets...")
    for key, path in OUTPUTS.items():
        atomic_json_write(path, generated[key])
        print("Written:", path.relative_to(ROOT))

    print("\nRe-reading and verifying generated datasets...")
    audit_outputs = {}

    for key, path in OUTPUTS.items():
        count, targets, digest = verify_output(
            path,
            expected_counts[key],
            expected_targets[key],
        )

        audit_outputs[f"{key[0]}_{key[1]}"] = {
            "path": str(path.relative_to(ROOT)),
            "rows": count,
            "target_counts": targets,
            "sha256": digest,
        }

        print(f"{key[0]} {key[1]}: PASS | rows={count}")
        print("Target counts:", targets)
        print("SHA-256:", digest)

    audit = {
        "experiment": "PyMETA hierarchical classification",
        "builder_version": "1.0",
        "test_data_accessed": False,
        "source_files": {
            split: {
                "path": str(path.relative_to(ROOT)),
                "sha256": source_hashes[split],
                "rows": EXPECTED_ROWS[split],
                "original_label_counts": EXPECTED_LABELS[split],
            }
            for split, path in SOURCES.items()
        },
        "generated_datasets": audit_outputs,
        "validation": {
            "all_source_counts_passed": True,
            "all_source_label_counts_passed": True,
            "all_output_counts_passed": True,
            "all_output_target_counts_passed": True,
        },
    }

    audit_path = OUT / "dataset_build_audit.json"
    atomic_json_write(audit_path, audit)

    print("\nAudit record:", audit_path.relative_to(ROOT))
    print("Audit SHA-256:", sha256(audit_path))
    print("\nFINAL RESULT: PASS")


if __name__ == "__main__":
    main()

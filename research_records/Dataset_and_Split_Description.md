# Dataset and Split Description

## 1. Dataset Source

This research uses the PyMETA dataset hosted on Hugging Face:

- **Repository:** `CircleCat/pymeta`
- **Dataset version:** v2.0
- **Revision:** `db66bd04916cdbefee9ad66774b4598e5f2a3ba0`
- **Dataset:** Python programming submissions from students.

The v2.0 revision introduced the QuestionID-based split and the associated dataset metadata.

## 2. Dataset Size and Official Split

The dataset contains 48,646 records associated with 153 unique QuestionIDs.

| Split | Records | Unique QuestionIDs |
|---|---:|---:|
| Training | 38,918 | 133 |
| Validation | 4,863 | 11 |
| Test | 4,865 | 9 |
| **Total** | **48,646** | **153** |

The official split is based on QuestionID. The three splits were previously audited and found to have no pairwise QuestionID overlap.

The dataset repository's `metadata/question_groups.csv` provides the QuestionID-to-split assignments and associated metadata.

**Audit provenance note:** These split statistics and the zero-overlap finding are results from the earlier dataset audit. The original audit output and `question_groups.csv` are not currently preserved in the workstation project's local directory. The local frozen configuration independently confirms the use of the official QuestionID-based split and explicitly prohibits random splitting, but does not independently reproduce the zero-overlap result.

## 3. Target Classification

The research uses three target classes:

1. No Error
2. Logic Error
3. Explicit Error

The gold fine-grained label is `all_errortype2`, mapped as follows:

- `No error` → No Error
- `LogicError` → Logic Error
- All other fine-grained error types → Explicit Error

The binary `error_category` field is not used as the gold target.

## 4. Class Distribution

The previously completed dataset audit reported the following class distributions:

| Class | Training | Validation | Test |
|---|---:|---:|---:|
| No Error | 19,822 | 1,875 | 1,510 |
| Logic Error | 8,079 | 1,495 | 1,813 |
| Explicit Error | 11,017 | 1,493 | 1,542 |
| **Total** | **38,918** | **4,863** | **4,865** |

These values are recorded as prior audit findings. They have not been independently recalculated from the original CSV files on the workstation.

## 5. Permitted Model Inputs

Only the following fields are permitted as model inputs:

- `question`
- `exceptedAnswer`
- `studentAnswer`

Identifiers, execution information, error labels, and other outcome-related fields are excluded from model inputs. These include `userId`, `questionId`, `attemptId`, `status`, `testOutcome`, `all_errortype2`, and `error_category`.

The `answer_preload` field is also excluded.

Student submissions are preserved as submitted. Syntax errors, indentation errors, naming errors, empty submissions, and other student-code characteristics are not repaired or normalized in a way that changes their original content.

## 6. Split Integrity and Leakage Prevention

The official QuestionID-based split is retained. Random splitting is not used.

The training and validation artifacts are processed representations of the official splits:

- Training: `data/flat_train_overlay_cleaned_1536.json`
- Validation: `data/flat_validation.json`

Their record counts and SHA-256 hashes are recorded in the frozen configuration and training reproducibility records.

The test split is not used for training, checkpoint selection, or validation. Final test evaluation must remain a separate evaluation stage after the experiment and artifact audit are complete.

## 7. Evidence and Reproducibility Status

The following local artifacts support the processing and experiment configuration:

- `config/training_config_frozen.json`
- `config/training_loop_implementation_spec_v1.json`
- `audits/step13/checkpoint_manifest.json`
- `scripts/scripts/step14_63_dataset_loading_audit.py`
- `scripts/scripts/step14_64_dataset_hash_verification.py`

The local loading audit verifies the processed training and validation record counts and message structure. The hash verification script checks the processed artifacts against their expected hashes.

These scripts do not independently establish the original dataset's QuestionID counts, pairwise split overlap, or class distributions.

**Reproducibility limitation:** The original dataset audit outputs and source metadata file should be recovered and archived if available. Until then, the original split and class-distribution statistics should be identified as previously completed audit findings rather than newly reproduced local results.

---

**Document classification:** Research methodology record
**Intended use:** Dataset and experimental methodology sections of the research paper
**Status:** Prepared from previously established audit findings and locally verified configuration evidence.
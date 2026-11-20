# Dataset and Data Splits

## 1. Dataset

This research uses the PyMETA dataset published at:

- Repository: https://huggingface.co/datasets/CircleCat/pymeta
- Dataset revision: `db66bd04916cdbefee9ad66774b4598e5f2a3ba0`

The dataset contains student Python programming submissions and is used to study three-class logic-error classification.

## 2. Classification Labels

The target task uses three classes:

1. **No Error** — the submission is classified as having no error.
2. **Logic Error** — the submission contains a logical error.
3. **Explicit Error** — the submission contains an explicit programming error, such as a syntax, name, type, or indentation error.

The three-class target is derived from `all_errortype2`:

- `No error` maps to `No Error`.
- `LogicError` maps to `Logic Error`.
- Other explicit error categories map to `Explicit Error`.

The `error_category` field is not treated as the gold target for this experiment.

## 3. Official Dataset Splits

The v2.0 dataset uses a QuestionID-based split. Questions are assigned to one split so that submissions belonging to the same question do not cross the training, validation, and test boundaries.

| Split | Submissions | Unique QuestionIDs |
|---|---:|---:|
| Train | 38,918 | 133 |
| Validation (dev) | 4,863 | 11 |
| Test | 4,865 | 9 |
| **Total** | **48,646** | **153** |

The split metadata is provided in `metadata/question_groups.csv` in the dataset repository.

The dataset revision and its split metadata are the reference for these figures. Older README statistics may describe an earlier dataset version and should not be substituted for the v2.0 split.

## 4. Class Distribution

The class distributions recorded during the dataset audit are:

| Class | Train | Validation | Test |
|---|---:|---:|---:|
| No Error | 19,822 | 1,875 | 1,510 |
| Logic Error | 8,079 | 1,495 | 1,813 |
| Explicit Error | 11,017 | 1,493 | 1,542 |
| **Total** | **38,918** | **4,863** | **4,865** |

These are dataset audit counts, not model evaluation results.

## 5. Permitted Model Inputs

The permitted input fields are:

- `question`
- `exceptedAnswer`
- `studentAnswer`

These fields provide the programming question, expected answer, and student's submitted answer.

Identifiers, execution outcomes, target labels, and other metadata must not be supplied as model input. In particular, fields such as `userId`, `questionId`, `answer_preload`, `attemptId`, `status`, `testOutcome`, and error-label fields are excluded from model inputs.

## 6. Leakage Prevention

The following rules apply throughout the experiments:

- Preserve the official QuestionID-based train, validation, and test assignments.
- Do not create a random row-level split.
- Do not move submissions between the official splits.
- Use training data for model fitting.
- Use validation data for model selection and development decisions.
- Keep test data held out for final evaluation.
- Do not use identifiers, target labels, or execution outcomes as model features.

The audit of the v2.0 split found no QuestionID overlap between train, validation, and test.

## 7. Data Quality Notes

The dataset audit identified repeated `studentAnswer` values and repeated code within questions. Repeated code is not automatically treated as leakage: the official question-level split is the primary protection against question overlap across splits.

Five training records were identified with execution status 4 and were inspected during the data audit. Any exclusion or treatment of such records must be documented in the corresponding experiment records rather than assumed from the dataset totals.

The test split contains one submission with an empty `studentAnswer`. Its presence should be considered when interpreting evaluation coverage.

## 8. Reproducibility

For reproducible dataset preparation, record the exact dataset revision and preserve the official split files and metadata. Do not silently replace the v2.0 data with an updated dataset revision.

Dataset statistics in this document describe the audited v2.0 dataset. They do not establish model performance or experimental conclusions.

# PyMETA: Logic-Error Classification in Student Python Code

**Research project:** Evaluating an Explicit-Error-Aware Hierarchical Approach for Logic-Error Classification in Student Python Code.

This repository contains the research implementation, experiment specifications, dataset documentation, and selected research records for investigating automated classification of student Python programming submissions.

The study considers two classification formulations: a flat three-class classifier and an explicit-error-aware hierarchical classifier.

## 1. Research Objective

The objective is to investigate whether decomposing student-code error classification into two stages provides a useful alternative to direct three-class classification.

The task involves three classes:

- **No Error** — submissions classified as having no error.
- **Logic Error** — submissions containing logical errors.
- **Explicit Error** — submissions containing explicit programming errors, including syntax, name, type, and indentation errors.

The research focuses on classification methodology, question-level generalization, and reliable evaluation.

## 2. Dataset

The experiments use the [PyMETA dataset](https://huggingface.co/datasets/CircleCat/pymeta).

| Property | Value |
|---|---|
| Dataset | CircleCat/pymeta |
| Dataset revision | `db66bd04916cdbefee9ad66774b4598e5f2a3ba0` |
| Total submissions | 48,646 |
| Unique questions | 153 |
| Classification classes | 3 |
| Split strategy | QuestionID-based |

### Official Dataset Splits

| Split | Submissions | QuestionIDs |
|---|---:|---:|
| Training | 38,918 | 133 |
| Validation | 4,863 | 11 |
| Test | 4,865 | 9 |
| **Total** | **48,646** | **153** |

The official question-level split is preserved to prevent submissions belonging to the same question from crossing dataset boundaries.

**No random row-level split is used.**

The permitted model inputs are `question`, `exceptedAnswer`, and `studentAnswer`. Identifiers, execution outcomes, and target-label fields are excluded from model inputs.

See [Dataset and Splits](docs/dataset_and_splits.md) for the dataset details, class distributions, and data-integrity rules.

## 3. Models and Methodology

### Flat Three-Class Baseline

The baseline directly predicts one of the three target classes:

1. No Error
2. Logic Error
3. Explicit Error

### Explicit-Error-Aware Hierarchical Classifier

The hierarchical approach decomposes classification into two stages:

- **Stage 1:** Distinguishes Explicit Error from the remaining submissions.
- **Stage 2:** Classifies the remaining submissions as No Error or Logic Error.

The final prediction belongs to the same three-class label space as the flat baseline.

### Base Model

The principal model is:

- Model: `Qwen/Qwen2.5-Coder-7B-Instruct`
- Revision: `c03e6d358207e414f1eca0bb1891e29f1db0e242`
- Fine-tuning: LoRA-based parameter-efficient fine-tuning
- Quantization: 4-bit NF4 with double quantization
- Computation precision: FP16

See [Methodology](docs/methodology.md) for the classification design.

## 4. Evaluation

The primary evaluation metric is macro-averaged F1-score.

Additional evaluation measures include:

- Per-class precision, recall, and F1-score.
- Confusion matrix.
- Overall accuracy.
- LER_over and Logic Error recall as additional experiment-specific criteria.

The validation split is used for development decisions, while the official test split is reserved for held-out evaluation.

Training completion, validation results, and test results must be treated as separate forms of evidence.

No claim of hierarchical-model superiority should be made without verified comparative evaluation results.

See [Evaluation Protocol](docs/evaluation_protocol.md) for the evaluation requirements.

## 5. Repository Structure

```text
pymeta-logic-error-classification/
│
├── config/
│   ├── hierarchical_experiment_spec_v1.json
│   ├── overflow_retry_policy_amendment_v1_1.json
│   └── training_loop_implementation_spec_v1.json
│
├── docs/
│   ├── dataset_and_splits.md
│   ├── methodology.md
│   ├── experimental_setup.md
│   ├── evaluation_protocol.md
│   ├── reproducibility.md
│   └── limitations.md
│
├── research_records/
│   ├── Dataset_and_Split_Description.md
│   └── Training_Completion_and_Validation_Record.md
│
├── scripts/
│   ├── Training and evaluation implementations
│   ├── Dataset preparation utilities
│   └── Hierarchical classification components
│
├── tests/
│   ├── test_hierarchical_cascade_evaluator.py
│   ├── test_hierarchical_checkpoint_recovery.py
│   └── test_hierarchical_pipeline_recovery.py
│
├── .gitignore
└── README.md
```

The tree above summarizes the main repository organization. Consult the actual directories for the complete file listing.

## 6. Documentation

Detailed research documentation is available in the `docs/` directory.

| Document | Description |
|---|---|
| [Dataset and Splits](docs/dataset_and_splits.md) | Dataset version, split statistics, labels, and leakage prevention |
| [Methodology](docs/methodology.md) | Flat baseline and hierarchical classification design |
| [Experimental Setup](docs/experimental_setup.md) | Model, fine-tuning, training configuration, and environment |
| [Evaluation Protocol](docs/evaluation_protocol.md) | Metrics, prediction handling, and evaluation requirements |
| [Reproducibility](docs/reproducibility.md) | Dataset, model, configuration, and environment requirements |
| [Limitations](docs/limitations.md) | Scope, methodological limitations, and future research directions |

## 7. Reproducibility

Reproducing the experiments requires:

1. Access to the specified PyMETA dataset revision.
2. Access to the pinned base-model revision.
3. The appropriate Python and machine-learning environment.
4. The experiment configuration files.
5. The corresponding training and evaluation scripts.

The repository does not distribute the dataset, model weights, LoRA adapters, checkpoints, generated predictions, or local training artifacts.

Review the configuration files and documentation before executing any experiment. Repository setup does not automatically launch model training.

See [Reproducibility](docs/reproducibility.md) for further details.

## 8. Research Integrity

The project follows these principles:

- Preserve the official QuestionID-based dataset splits.
- Prevent target-label and metadata leakage.
- Keep test data separate from training and validation.
- Pin dataset and model revisions.
- Distinguish implementation specifications from verified experiment completion.
- Report only metrics supported by saved evaluation artifacts.
- Disclose methodological limitations and evaluation constraints.

## 9. Limitations

The research is limited to the PyMETA dataset and its represented student Python programming tasks.

The hierarchical approach also introduces routing-error propagation: an incorrect first-stage decision can affect the final classification.

Further limitations and possible research extensions are documented in [Limitations](docs/limitations.md).

## 10. Research Status

This repository provides the research implementation, experiment specifications, documentation, and selected experiment records.

The presence of implementation code or training records should not independently be interpreted as proof that every planned experiment has been completed.

Reported model performance and research conclusions must be supported by their corresponding verified evaluation artifacts.

---

**Dataset:** [CircleCat/pymeta](https://huggingface.co/datasets/CircleCat/pymeta)

**Repository:** [PyMETA Logic-Error Classification](https://github.com/Sheerazkhan-8861/pymeta-logic-error-classification)
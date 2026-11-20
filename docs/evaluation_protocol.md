# Evaluation Protocol

## 1. Evaluation Objective

The evaluation measures the performance of three-class student Python error classification.

The target classes are:

- No Error
- Logic Error
- Explicit Error

The evaluation protocol is intended to provide consistent and reproducible assessment of model predictions.

## 2. Dataset Splits

The official QuestionID-based dataset splits are preserved:

- Training: model fitting.
- Validation: development and model-selection decisions.
- Test: held-out final evaluation.

The test split must not influence training, hyperparameter selection, or checkpoint selection.

## 3. Evaluation Metrics

The primary evaluation metric is macro-averaged F1-score.

Macro-F1 calculates the F1-score independently for each class and takes their unweighted average. This gives each class equal importance, regardless of its number of examples.

Additional metrics include:

- Per-class precision.
- Per-class recall.
- Per-class F1-score.
- Confusion matrix.
- Overall accuracy.

The experiment specification also identifies lower LER_over as a secondary criterion and higher Logic Error recall as a tertiary criterion.

Metric definitions and aggregation must remain consistent across compared experiments.

## 4. Prediction Format

The flat baseline uses a strict JSON prediction format.

The documented generation configuration uses greedy decoding and a maximum of 16 newly generated tokens.

Predictions are evaluated according to the expected output format. Invalid outputs are not silently repaired or retried.

## 5. Flat Baseline Evaluation

The repository includes a dedicated test evaluator for the flat three-class baseline.

The held-out test evaluation must use the official test split and the saved model checkpoint associated with the evaluated experiment.

The evaluation output should preserve:

- Model and checkpoint identity.
- Dataset revision and split.
- Class-wise metrics.
- Macro-F1.
- Confusion matrix.
- Prediction validity information.
- Evaluation configuration.

## 6. Hierarchical Evaluation

The hierarchical classifier produces a final three-class prediction through two-stage routing:

1. Stage 1 detects Explicit Error versus the remaining submissions.
2. Stage 2 distinguishes No Error from Logic Error for submissions not assigned to Explicit Error.

Evaluation of the complete hierarchy must account for the final routed three-class predictions.

Stage-wise performance may be reported separately when supported by saved evaluation artifacts.

A Stage 1 or Stage 2 metric alone must not be presented as the overall three-class hierarchical result.

## 7. Comparison Requirements

A valid comparison between the flat and hierarchical approaches requires:

- The same official evaluation split.
- Clearly identified model checkpoints.
- Consistent target-label definitions.
- Consistent metric implementations.
- Explicit identification of invalid or missing predictions.

Any difference in evaluation coverage or protocol must be disclosed.

## 8. Test-Result Integrity

The existence of a training completion record or validation metrics does not establish held-out test performance.

Test metrics must be taken from the saved test evaluation artifact and verified against the evaluated model and configuration.

The held-out test evaluation must not be rerun merely to reconstruct missing documentation when an existing evaluation artifact is intended to be used.

## 9. Reporting

Reported results must distinguish:

- Training results.
- Validation results.
- Held-out test results.
- Stage-wise hierarchical results.
- Final three-class predictions.

No performance claim should be made without a corresponding verified evaluation artifact.

In particular, the hierarchical approach must not be described as superior to the flat baseline unless a valid comparative evaluation supports that conclusion.

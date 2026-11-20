# Limitations

## 1. Dataset Scope

This study uses the PyMETA dataset, which contains student Python programming submissions.

The findings are therefore limited to the dataset, programming questions, student population, and error categories represented in PyMETA.

Results should not automatically be generalized to other programming languages, educational settings, or software-development tasks.

## 2. Question-Level Generalization

The official dataset split is based on QuestionID.

This prevents the same question from appearing across training, validation, and test splits. However, it also means that evaluation measures generalization to questions not represented in training.

Performance on this split may differ from performance on previously seen questions or other datasets.

## 3. Class Imbalance

The three target classes have different numbers of examples.

In particular, Logic Error has fewer training examples than No Error and Explicit Error.

Consequently, overall accuracy alone may not adequately describe model performance. Macro-F1 and class-wise precision, recall, and F1-score are important for interpreting results.

## 4. Label Quality

The target labels are derived from the dataset's `all_errortype2` field.

The mapping groups multiple explicit error categories into one Explicit Error class. This simplifies the task but removes distinctions between individual explicit error types.

Dataset labels may also contain annotation or classification limitations that are not resolved by the modeling approach.

## 5. Input and Execution Information

The model uses the programming question, expected answer, and student submission as input.

Execution outcomes and other metadata are excluded from model inputs to prevent leakage.

This means that the classifier relies on the information available in these permitted text fields and does not independently execute student programs as part of the classification decision.

## 6. Hierarchical Error Propagation

The hierarchical approach uses a first-stage routing decision followed by a second-stage classification decision.

An incorrect Stage 1 decision can affect the final prediction. In particular, an Explicit Error submission incorrectly routed to Stage 2 cannot be recovered by the second stage under the defined cascade.

Therefore, stage-wise performance and final three-class performance should be distinguished.

## 7. Model and Computational Constraints

The experiments use a 7-billion-parameter code model with parameter-efficient fine-tuning and quantization.

The results may depend on the selected model, quantization configuration, adapter settings, hardware, and software environment.

The study does not establish that the same results will be obtained with other model families or computational configurations.

## 8. Evaluation Evidence

Conclusions must be limited to experiments with verified evaluation artifacts.

Training completion and validation results do not establish held-out test performance.

The hierarchical approach should not be claimed to outperform the flat baseline unless a valid comparison on the same held-out evaluation split supports that conclusion.

External zero-shot model results should not be reported unless their prediction and evaluation artifacts are available and verified.

## 9. Output-Format Constraints

The documented generation protocol expects strict JSON output and does not automatically retry or repair invalid predictions.

This makes output validity part of the evaluation process. Reported results should disclose invalid or missing predictions and follow the defined metric-handling procedure.

## 10. Future Work

Potential future work includes:

- Evaluating the approach on additional programming datasets.
- Investigating generalization across programming languages.
- Examining the impact of class imbalance.
- Analyzing hierarchical routing errors.
- Comparing additional model architectures under controlled conditions.
- Studying the effect of execution-based evidence where an appropriate leakage-safe protocol can be established.

These are possible research directions, not completed experiments or established findings.

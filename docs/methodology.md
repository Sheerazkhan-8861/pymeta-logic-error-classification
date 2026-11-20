# Methodology

## 1. Research Objective

This study evaluates an explicit-error-aware hierarchical approach for classifying student Python submissions into three categories:

- No Error
- Logic Error
- Explicit Error

The central research question is whether separating explicit-error detection from the remaining classification task provides a useful approach to student-code error classification.

## 2. Overall Approach

Two classification approaches are considered:

1. A flat three-class classification baseline.
2. A two-stage hierarchical classification approach.

Both approaches use the same underlying PyMETA task definition and permitted input fields.

## 3. Input Representation

Each submission is represented using the following fields:

- `question`: the programming problem statement.
- `exceptedAnswer`: the expected answer.
- `studentAnswer`: the student's submitted answer.

These fields provide the context needed to assess the submission. Identifiers, execution outcomes, and target-label metadata are excluded from model input.

## 4. Flat Classification Baseline

The baseline formulates the task as direct three-class classification.

Given an input submission, the model predicts one of:

- No Error
- Logic Error
- Explicit Error

The baseline provides a reference point for evaluating the hierarchical formulation.

## 5. Hierarchical Classification

The proposed approach decomposes the three-class task into two stages.

### Stage 1: Explicit-Error Detection

The first stage distinguishes Explicit Error submissions from submissions belonging to the other two classes.

Its output determines whether a submission is assigned to the Explicit Error branch or passed to the second stage.

### Stage 2: No-Error versus Logic-Error Classification

Submissions routed to the second stage are classified into:

- No Error
- Logic Error

This decomposition makes explicit-error detection a separate decision rather than requiring one model decision across all three classes.

## 6. Model Initialization

The experiments use Qwen2.5-Coder-7B-Instruct as the principal model.

The hierarchical implementation initializes Stage 2 independently from the same initial adapter configuration rather than initializing it from the trained Stage 1 adapter.

This distinction is important when interpreting the training procedure and reproducing the experiment.

## 7. Parameter-Efficient Fine-Tuning

The principal model is adapted using LoRA-based parameter-efficient fine-tuning with 4-bit quantization.

The documented configuration uses:

- 4-bit NF4 quantization.
- Double quantization.
- FP16 computation.
- LoRA adapters.

Exact training parameters and implementation settings are maintained in the repository configuration and experiment records.

## 8. Prediction and Routing

The hierarchical system uses the Stage 1 prediction to determine the routing decision.

If Stage 1 predicts Explicit Error, the final output is Explicit Error.

Otherwise, the submission is passed to Stage 2, whose prediction determines No Error or Logic Error.

The final output therefore belongs to the same three-class label space as the flat baseline.

## 9. Experimental Comparison

The flat baseline and hierarchical approach are evaluated as distinct experimental formulations.

A comparison must use verified evaluation outputs and the same clearly defined evaluation split. Performance differences must not be inferred from training completion records or validation results alone.

## 10. Methodological Integrity

The following principles govern the methodology:

- Preserve the official QuestionID-based dataset splits.
- Exclude identifiers and target-label leakage from model inputs.
- Use validation data for development decisions.
- Keep test data held out for final evaluation.
- Record the exact model revision and experiment configuration.
- Report only results supported by saved evaluation artifacts.

The methodology describes the implemented experimental design. It does not, by itself, establish that the hierarchical approach performs better than the flat baseline.

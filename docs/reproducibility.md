# Reproducibility

## 1. Purpose

This document describes the information required to reproduce the PyMETA classification experiments.

Reproducibility depends on preserving the dataset version, official splits, model revision, training configuration, implementation, and evaluation protocol.

## 2. Dataset Version

The experiments use the PyMETA dataset:

- Repository: `CircleCat/pymeta`
- Revision: `db66bd04916cdbefee9ad66774b4598e5f2a3ba0`

The official QuestionID-based train, validation, and test assignments must be preserved.

Do not substitute a newer dataset revision without documenting and auditing the change.

## 3. Model Version

The principal model is:

- Model: `Qwen/Qwen2.5-Coder-7B-Instruct`
- Revision: `c03e6d358207e414f1eca0bb1891e29f1db0e242`

The exact model revision should be used when reproducing the documented experiments.

## 4. Configuration Files

The repository maintains experiment specifications and implementation settings in `config/`.

Relevant files include:

- `hierarchical_experiment_spec_v1.json`
- `training_loop_implementation_spec_v1.json`
- `overflow_retry_policy_amendment_v1_1.json`

These files should be preserved and consulted before running an experiment.

A configuration specification describes intended or defined behavior; it does not independently prove that an experiment was executed.

## 5. Software and Hardware

The recorded GPU workstation environment includes:

- Linux.
- Python 3.14.4.
- PyTorch 2.11.0+cu130.
- NVIDIA RTX 4000 Ada Generation GPU.
- Approximately 19.55 GiB GPU memory.
- Visual Studio Code.

A reproduction report should additionally record the installed versions of Transformers, Datasets, Accelerate, PEFT, BitsAndBytes, NumPy, Pandas, and scikit-learn.

Differences in hardware or software versions should be documented.

## 6. Randomness and Training State

The frozen flat-baseline configuration specifies random seed 42.

For reproducibility, preserve the random-number-generator states and data ordering used by the training implementation.

The checkpoint implementation records relevant state, including:

- Model adapter parameters.
- Tokenizer state.
- Optimizer state.
- Learning-rate scheduler state.
- Gradient-scaler state.
- Random-number-generator state.
- Data permutation and next micro-batch position.
- Training history and skipped updates.
- Relevant hashes.

Checkpoint recovery should follow the documented implementation and overflow-retry policy.

## 7. Training and Evaluation Separation

Training data is used for model fitting.

Validation data is used for development and model selection.

Test data remains held out for final evaluation.

The official QuestionID split must not be replaced with a random row-level split.

## 8. Experiment Records

Each reproduced experiment should identify:

- Experiment name and architecture.
- Dataset revision.
- Model name and revision.
- Configuration file and version.
- Software and hardware environment.
- Training completion status.
- Checkpoint identity.
- Evaluation split and protocol.
- Saved metrics and prediction artifacts.

Training completion, validation performance, and test performance must be recorded separately.

## 9. Reproduction Limitations

Exact reproduction may be affected by hardware, software, library, and numerical differences.

A reproduction should report actual environment details and observed results rather than assume that matching configuration files guarantees identical outputs.

No missing metric or experiment result should be reconstructed by guessing.

# Experimental Setup

## 1. Base Model

The principal model used in this research is:

- Model: `Qwen/Qwen2.5-Coder-7B-Instruct`
- Model revision: `c03e6d358207e414f1eca0bb1891e29f1db0e242`

The model revision is pinned to support reproducibility.

## 2. Fine-Tuning Approach

Parameter-efficient fine-tuning is performed using LoRA adapters with 4-bit quantization.

The documented configuration includes:

- Quantization: 4-bit NF4
- Double quantization: enabled
- Computation precision: FP16
- Optimizer: PagedAdamW8bit
- Learning rate: 0.0002
- Weight decay: 0.01
- Learning-rate schedule: cosine
- Warmup steps: 146
- Random seed: 42

## 3. Training Configuration

The frozen flat-baseline configuration specifies:

| Parameter | Value |
|---|---:|
| Training epochs | 2 |
| Training examples | 38,918 |
| Micro-batch size | 1 |
| Gradient accumulation steps | 16 |
| Updates per epoch | 2,433 |
| Total optimizer updates | 4,866 |
| Checkpoint interval | 250 updates |
| Maximum generated tokens | 16 |

The configuration files and implementation are authoritative if a value in this summary differs from the actual experiment configuration.

## 4. Validation

The validation split is used for development and model-selection decisions.

The official test split is not used for training, hyperparameter selection, or checkpoint selection.

## 5. Generation Configuration

The documented flat-baseline evaluation configuration uses:

- Greedy decoding.
- Maximum 16 newly generated tokens.
- Strict JSON output expectations.
- No automatic retry or output repair.

Invalid outputs must be handled according to the evaluation protocol rather than silently corrected.

## 6. Checkpointing and Recovery

The training implementation supports checkpoint recovery.

The recorded checkpoint state includes model adapter parameters, tokenizer state, optimizer state, scheduler state, gradient-scaler state, random-number-generator state, data permutation, next micro-batch position, training history, skipped updates, and relevant hashes.

The training specification also documents overflow handling with a maximum of five retries.

## 7. Training Completion Record

A training completion record dated 17 November 2026 reports:

- Two training epochs.
- 4,866 completed updates out of 4,866 planned updates.

This completion record must not be attributed to a particular architecture unless its associated experiment artifacts establish that attribution.

## 8. Hardware and Software Environment

The recorded GPU workstation environment includes:

- Operating system: Linux.
- Python: 3.14.4.
- PyTorch: 2.11.0+cu130.
- GPU: NVIDIA RTX 4000 Ada Generation.
- GPU memory: approximately 19.55 GiB.
- Development environment: Visual Studio Code.

These values describe the recorded environment and should be checked against the environment record for the specific experiment being reproduced.

## 9. Configuration Authority

The repository's configuration files and experiment records take precedence over this descriptive summary.

Relevant configuration files include:

- `config/hierarchical_experiment_spec_v1.json`
- `config/training_loop_implementation_spec_v1.json`
- `config/overflow_retry_policy_amendment_v1_1.json`

The hierarchical experiment specification must not be interpreted as proof of completed hierarchical training. Training completion and evaluation claims require their corresponding records and artifacts.

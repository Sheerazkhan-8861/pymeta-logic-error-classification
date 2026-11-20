# PyMETA Research Project
## Training Completion and Validation Record

**Research Title:** Evaluating an Explicit-Error-Aware Hierarchical Approach for Logic-Error Classification in Student Python Code

**Record Type:** Model Training Completion Record
**Project Phase:** Phase 4 — Model Training
**Status:** Training Completed
**Date:** 17 November 2026

---

### 1. Training Completion Summary

The model training process was successfully completed using the frozen training configuration established for the PyMETA research project.

Training was conducted for two epochs using the official QuestionID-based training split. The validation dataset was used for checkpoint evaluation and model selection. The test dataset was not accessed during training or validation.

The training process completed all 4,866 planned successful optimizer updates. One numerical overflow event occurred during training and was successfully recovered through the predefined overflow-retry mechanism.

### 2. Training Execution Results

| Parameter | Recorded Result |
|---|---:|
| Training epochs | 2 |
| Successful optimizer updates | 4,866 / 4,866 |
| Overflow retry events | 1 |
| Successfully recovered overflow events | 1 |
| Training completion | Successful |
| Test dataset accessed | No |

### 3. Validation Results

The model was evaluated on the official validation dataset after each training epoch.

| Metric | Epoch 1 | Epoch 2 |
|---|---:|---:|
| Macro-F1 | 94.38% | 96.72% |
| LER_over | 2.36% | 1.10% |
| Logic Error Recall | 89.23% | 92.37% |
| Invalid Output Rate | 0.00% | 0.00% |
| Invalid Predictions | 0 | 0 |

Epoch 2 achieved the strongest validation performance according to the predefined checkpoint-selection criteria.

### 4. Best Validation Checkpoint

The second epoch was selected as the best validation checkpoint.

**Best validation metrics:**

- Macro-F1: 0.967206
- LER_over: 0.011039
- Logic Error Recall: 0.923746
- Invalid Output Rate: 0.000000
- Invalid Predictions: 0

**Confusion Matrix — Epoch 2**

Rows represent actual classes; columns represent predicted classes.

| Actual / Predicted | No Error | Logic Error | Explicit Error |
|---|---:|---:|---:|
| No Error | 1,865 | 10 | 0 |
| Logic Error | 75 | 1,381 | 39 |
| Explicit Error | 6 | 25 | 1,462 |

### 5. Checkpoint Selection Method

Checkpoint selection followed the predefined evaluation criteria:

1. Higher validation Macro-F1 was preferred.
2. In the event of a Macro-F1 tie, lower LER_over was preferred.
3. If both previous criteria were tied, higher Logic Error recall was preferred.
4. An exact tie retained the earlier checkpoint.

Epoch 2 achieved improved Macro-F1, lower LER_over, and higher Logic Error recall compared with Epoch 1.

### 6. Dataset Integrity and Evaluation Separation

The official QuestionID-based dataset split was retained throughout the experiment.

- Training dataset: 38,918 examples.
- Validation dataset: 4,863 examples.
- Test dataset: 4,865 examples.

The test dataset was not accessed during training, checkpoint selection, or validation evaluation.

The reported performance values in this record are exclusively validation results. They must not be interpreted or reported as final test performance.

### 7. Research Reproducibility and Artifact Preservation

The following artifacts must be preserved and independently verified:

- Frozen training configuration.
- Training and validation data artifacts.
- Training execution logs.
- Epoch-wise validation metrics.
- Best validation checkpoint and adapter weights.
- Final recovery checkpoint.
- Optimizer, scheduler, and gradient scaler states.
- Training history and checkpoint-selection metadata.
- Software environment and GPU information.
- Independent SHA-256 artifact integrity manifest.

The artifact audit and SHA-256 manifest generation have been completed. The manifest checksum was independently verified, and all 29 inventoried files passed the recorded hash checks. The checkpoint verification report documents the scope and limitations of checkpoint integrity verification.

### 8. Important Research Qualification

The successful completion of model training and the achievement of strong validation performance do not establish final model generalization.

Final performance claims must be based exclusively on the subsequent frozen evaluation using the untouched test dataset.

No final test performance is claimed in this record.

---

**Record Status:** Training completion established; SHA-256 manifest generated and verified; checkpoint integrity audit completed with documented limitations.

**Research Integrity Note:** All reported metrics originate from the completed validation evaluations. No test results, fabricated statistics, or unverified artifact-integrity claims are included.
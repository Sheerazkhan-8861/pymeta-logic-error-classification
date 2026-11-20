# PyMETA: Logic-Error Classification in Student Python Code

Research project evaluating an explicit-error-aware hierarchical approach
against a flat three-class classifier for classifying student Python code.

## Research scope

Dataset: CircleCat/pymeta
Dataset revision: db66bd04916cdbefee9ad66774b4598e5f2a3ba0

Classes:
- No Error
- Logic Error
- Explicit Error

The experiments use the official question-level train, development, and
test splits. No random row split is used.

## Models

The main fine-tuned model is Qwen2.5-Coder-7B-Instruct.
The experiments compare a flat three-class baseline with a two-stage
hierarchical classifier.

## Repository contents

- `scripts/`: training, evaluation, data preparation, and test code
- `config/`: experiment and implementation specifications
- `research_records/`: selected dataset and training records

## Data and model artifacts

Dataset files, model weights, LoRA adapters, checkpoints, local caches,
predictions, credentials, and virtual environments are not included.
Refer to the dataset source and revision above for dataset access.

## Reproducibility

Review the experiment specifications and scripts before running.
Local dataset preparation and model setup are required. Training is not
launched by repository setup.

## Research status

This repository documents the research implementation. Consult the paper
and accompanying experiment records for reported results and limitations.

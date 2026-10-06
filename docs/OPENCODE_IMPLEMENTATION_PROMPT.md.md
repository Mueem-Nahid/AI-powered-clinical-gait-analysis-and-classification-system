# OpenCode Implementation Prompt — AI-Powered Clinical Gait Analysis & Classification System

You are the primary senior ML Engineer + MLOps Engineer responsible for implementing this repository.

## 1. Mission

Build a production-quality, reproducible, modular **AI-Powered Clinical Gait Analysis & Classification System** for MediAI Dynamics.

The system is a:

- Clinical Decision Support System (CDSS)
- Biomedical time-series analytics platform
- AutoML-style model selection pipeline
- ML experimentation and model evaluation platform
- Optional model-serving API

The system receives a gait dataset such as `gait.csv` and automatically:

1. Validates and profiles the dataset.
2. Performs automated EDA.
3. Prevents patient-level data leakage.
4. Converts raw gait measurements into suitable time-series representations.
5. Trains five competing classification approaches.
6. Automatically tunes their hyperparameters.
7. Benchmarks them using a common evaluation protocol.
8. Selects the best model using macro F1 as the primary metric.
9. Generates model-specific interpretability visualizations.
10. Generates a clinically readable PDF report.
11. Saves the winning model and its configuration.
12. Tracks experiments and metadata where practical.
13. Optionally serves the winning model through FastAPI.
14. Provides monitoring and reproducibility metadata.

The implementation must be clean enough for a senior-level ML engineering take-home assignment and simple enough to run on a standard 4-core CPU / 8 GB RAM machine within the assignment's time constraint.

---

# 2. IMPORTANT SOURCE OF TRUTH

The repository should follow the engineering philosophy of the existing Fraud Detection Inference Service architecture:

- modular application structure
- dedicated training components
- monitoring
- infrastructure
- documentation
- Docker
- CI/CD
- MLflow-style experiment/model lifecycle
- production-oriented model loading
- health checks
- metrics
- tests

Do NOT blindly copy components that do not fit gait analysis.

For example:

- Feast is NOT required for this project.
- Redis is NOT required for the core implementation.
- PostgreSQL is optional rather than mandatory.
- The central problem is AutoML/model benchmarking, not online feature retrieval.

The new architecture should therefore be an adaptation of the fraud project, not a clone.

A supplied project document also defines expected reporting around:

- statistical features
- DTW alignment/distance
- signal quality
- segmentation
- training convergence
- confusion matrices
- precision/recall/ROC
- per-class performance
- subject-specific predictions
- model explainability
- comparative benchmarking
- clinical outcome reporting
- anomaly detection

Use those concepts where they fit the assignment.

---

# 3. ORIGINAL ASSIGNMENT REQUIREMENTS

Implement the following requirements exactly.

## Dataset

Input CSV structure:

```text
subject       Patient ID
condition     Target class
replication   Trial number
leg           1=Left, 2=Right
joint         1=Hip, 2=Knee, 3=Ankle
time          Normalized gait cycle percentage, expected 0–100
angle         Joint angle in degrees
```

Expected six signals:

```text
Left Hip
Left Knee
Left Ankle
Right Hip
Right Knee
Right Ankle
```

The system must not rely on hard-coded patient IDs, class distributions, or row counts.

It may infer:

- subjects
- classes
- joints
- legs
- repetitions
- time resolution
- sequence length

from the dataset.

---

# 4. CLI REQUIREMENT

The primary entry point must be:

```bash
python run_automl_pipeline.py --data data/gait.csv
```

The user must not need to manually choose:

- model
- hyperparameters
- train/test rows
- feature columns
- winner

The pipeline must make those decisions automatically.

Provide useful optional CLI arguments, but all of them must have sensible defaults.

Suggested:

```bash
python run_automl_pipeline.py \
    --data data/gait.csv \
    --output output/ \
    --seed 42
```

Optional arguments may include:

```text
--max-runtime-minutes
--skip-serving
--skip-mlflow
--fast
--verbose
```

Do not require these for the normal workflow.

---

# 5. ARCHITECTURE

Use the following target architecture.

```text
                         gait.csv
                            |
                            v
                +-----------------------+
                |   Data Validation     |
                | Schema / Quality Gate  |
                +-----------+-----------+
                            |
                            v
                +-----------------------+
                | Automated EDA         |
                | Profiling + Charts    |
                +-----------+-----------+
                            |
                            v
                +-----------------------+
                | Preprocessing Engine  |
                | Resampling             |
                | Missing Signal Handle  |
                | Normalization          |
                +-----------+-----------+
                            |
                  +---------+---------+
                  |                   |
                  v                   v
          Sequence Representation   Feature Representation
                  |                   |
       +----------+--------+          |
       |          |        |          |
       v          v        v          v
      LSTM       CNN   Transformer   XGBoost
       |          |        |          |
       +----------+--------+          |
                  |                   |
                  v                   v
                 DTW + KNN       Statistical Features
                  |                   |
                  +---------+---------+
                            |
                            v
                  Model Benchmarking
                            |
                            v
                   Best Model Selection
                            |
            +---------------+---------------+
            |               |               |
            v               v               v
      Model Artifact     PDF Report     Metadata
            |
            v
       Model Registry
            |
            v
        FastAPI API
            |
            v
     Prometheus / Grafana
```

---

# 6. PROJECT STRUCTURE

Implement this target layout.

```text
clinical-gait-automl/
│
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── schemas.py
│   ├── model_loader.py
│   ├── inference.py
│   ├── metrics.py
│   └── health.py
│
├── automl/
│   ├── __init__.py
│   ├── orchestrator.py
│   ├── model_selector.py
│   ├── tuner.py
│   ├── benchmark.py
│   └── registry.py
│
├── data/
│   ├── __init__.py
│   ├── loader.py
│   ├── validator.py
│   ├── splitter.py
│   ├── preprocessing.py
│   ├── segmentation.py
│   └── dataset.py
│
├── models/
│   ├── __init__.py
│   ├── base.py
│   ├── lstm.py
│   ├── cnn.py
│   ├── transformer.py
│   ├── dtw_knn.py
│   └── xgboost.py
│
├── features/
│   ├── __init__.py
│   ├── statistical.py
│   ├── biomechanical.py
│   └── extractor.py
│
├── explainability/
│   ├── __init__.py
│   ├── shap_explainer.py
│   ├── attention.py
│   ├── saliency.py
│   └── dtw_visualizer.py
│
├── reporting/
│   ├── __init__.py
│   ├── report_generator.py
│   ├── plots.py
│   └── templates/
│
├── monitoring/
│   ├── prometheus.yml
│   └── grafana/
│
├── config/
│   ├── automl.yaml
│   └── models.yaml
│
├── docs/
│   ├── ARCHITECTURE.md
│   ├── AUTOML.md
│   ├── MODELING.md
│   ├── CLINICAL_INTERPRETABILITY.md
│   ├── DEPLOYMENT.md
│   ├── RUNBOOK.md
│   └── adr/
│
├── tests/
│   ├── test_data.py
│   ├── test_preprocessing.py
│   ├── test_models.py
│   ├── test_automl.py
│   ├── test_reporting.py
│   └── test_api.py
│
├── scripts/
│   ├── run_local.sh
│   ├── train.sh
│   └── deploy.sh
│
├── data/
│   └── gait.csv
│
├── output/
│   ├── models/
│   ├── reports/
│   ├── figures/
│   ├── leaderboard.csv
│   ├── metrics.json
│   ├── best_model_config.json
│   └── run_metadata.json
│
├── Dockerfile
├── docker-compose.yml
├── docker-compose.prod.yml
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
└── run_automl_pipeline.py
```

Avoid duplicate `data/` directories. Keep only one canonical data directory.

---

# 7. DESIGN PRINCIPLE: SHARED DATA ABSTRACTION

Create a central abstraction such as:

```python
class GaitDataset:
    ...
```

It should expose functionality conceptually equivalent to:

```python
dataset.subjects
dataset.conditions
dataset.signals
dataset.trials
dataset.sequence_length
dataset.to_sequences()
dataset.to_features()
dataset.subject_split()
```

Every model must consume outputs from this common data layer.

Do not implement five independent CSV parsers.

---

# 8. PHASE 0 — REPOSITORY DISCOVERY

Before writing major code:

1. Inspect the existing repository.
2. Inspect all files already present.
3. Preserve useful existing conventions.
4. Identify whether a sample dataset already exists.
5. Identify current Python version.
6. Identify existing CI/CD conventions.
7. Identify existing Docker conventions.
8. Identify whether MLflow is already configured.
9. Do not destroy unrelated files.

Create or update documentation only after understanding the current repository.

---

# 9. PHASE 1 — DATA VALIDATION

Implement:

```text
data/loader.py
data/validator.py
```

Requirements:

Validate:

- file exists
- CSV can be loaded
- required columns exist
- numeric fields are numeric
- subject exists
- condition exists
- leg values are valid
- joint values are valid
- time values are reasonable
- angle is numeric
- duplicate records
- missing values
- missing expected signals
- insufficient samples
- class distribution

Generate a validation summary.

Example:

```text
DATA QUALITY REPORT
-------------------
Rows: 12345
Subjects: 42
Conditions: 3
Replications: 126
Missing values: 0
Duplicates: 4
Invalid time values: 0
Missing signals: 2

Status: PASS WITH WARNINGS
```

The pipeline should fail early for critical errors.

Do not silently repair corrupt data.

---

# 10. PHASE 2 — AUTOMATED EDA

Generate:

## Dataset summary

- number of unique subjects
- conditions
- legs
- joints
- replications
- rows
- missing values
- duplicates

## Statistics per condition

At minimum:

```text
mean
std
min
max
```

## Statistics per joint

Same fields.

## Average gait cycle visualization

Create a six-panel figure:

```text
Left Hip       Right Hip
Left Knee      Right Knee
Left Ankle     Right Ankle
```

Each panel shows average angle versus normalized gait-cycle time for every condition.

Save:

```text
output/figures/average_gait_cycles.png
```

Also generate:

```text
condition_distribution.png
signal_quality.png
segmentation_preview.png
```

Do not use hard-coded colors unless the chart library defaults.

---

# 11. PHASE 3 — SUBJECT-WISE SPLITTING

This is mandatory.

Never split rows randomly.

Use subject-level splitting.

Preferred strategy:

```text
Train: 70%
Validation: 15%
Test: 15%
```

For small datasets, use grouped CV during tuning when appropriate.

Important invariant:

```python
train_subjects.isdisjoint(val_subjects)
train_subjects.isdisjoint(test_subjects)
val_subjects.isdisjoint(test_subjects)
```

Add explicit unit tests for this.

NEVER use test data to:

- tune hyperparameters
- select preprocessing parameters
- select features
- choose the best model

The test set is used only for final comparison.

---

# 12. PHASE 4 — Gait Cycle Processing

Input:

```text
subject
condition
replication
leg
joint
time
angle
```

Create six aligned signals per gait cycle.

Canonical ordering:

```text
[
    left_hip,
    left_knee,
    left_ankle,
    right_hip,
    right_knee,
    right_ankle
]
```

Represent one sample approximately as:

```text
(time_steps, 6)
```

For example:

```text
(101, 6)
```

But do NOT hard-code 101.

Derive or configure the sequence length.

Normalize/resample each gait cycle so that signals share the same temporal resolution.

Handle:

- missing time points
- duplicate time points
- missing channels
- irregular sampling

Use interpolation only when clinically and technically reasonable.

Do not leak information from test subjects into training preprocessing.

---

# 13. PHASE 5 — COMMON PREPROCESSING

Implement a reusable preprocessing pipeline.

Requirements:

- deterministic random seed
- training-only normalization fitting
- consistent inference preprocessing
- feature-order preservation
- serialization of preprocessing metadata

Save:

```text
output/models/preprocessor.pkl
```

or an equivalent format.

Record:

```json
{
  "sequence_length": 101,
  "num_channels": 6,
  "feature_order": [
    "left_hip",
    "left_knee",
    "left_ankle",
    "right_hip",
    "right_knee",
    "right_ankle"
  ]
}
```

---

# 14. PHASE 6 — MODEL 1: LSTM / GRU

Implement a configurable recurrent classifier.

Example:

```text
Input
↓
LSTM/GRU
↓
Dropout
↓
optional second recurrent layer
↓
Dense
↓
Softmax
```

Tune at least:

- recurrent units
- number of recurrent layers
- dropout
- learning rate

Preferred tuner:

```text
Keras Tuner Hyperband
```

Use:

- EarlyStopping
- ReduceLROnPlateau

The search must be bounded.

Do not launch an enormous search.

Persist:

- best model
- hyperparameters
- training history
- validation metrics

---

# 15. PHASE 7 — MODEL 2: 1D CNN

Implement:

```text
Input
↓
Conv1D
↓
BatchNorm
↓
Pooling
↓
Conv1D
↓
GlobalAveragePooling
↓
Dense
↓
Softmax
```

Tune:

- filters
- kernel size
- number of convolution blocks
- dropout
- learning rate

Use Hyperband.

Persist training results.

---

# 16. PHASE 8 — MODEL 3: TRANSFORMER

Implement a lightweight Transformer Encoder appropriate for the runtime constraint.

Architecture:

```text
Input projection
↓
Positional encoding
↓
Transformer Encoder
↓
Global pooling
↓
Dense
↓
Softmax
```

Tune:

- attention heads
- embedding dimension
- transformer blocks
- dropout
- learning rate

Do not create an oversized Transformer.

CPU runtime matters more than architectural novelty.

The Transformer must expose or otherwise provide access to attention information for interpretability.

---

# 17. PHASE 9 — MODEL 4: DTW + KNN

Implement DTW-based classification.

Pipeline:

```text
Sequence
↓
DTW distance
↓
distance matrix
↓
KNN
```

Tune:

- number of neighbors
- warping window
- weighting strategy

Use GridSearchCV or RandomizedSearchCV where feasible.

Performance requirements:

- cap sequence length if necessary
- constrain warping window
- avoid unnecessary O(n²) behavior
- cache repeated distances where appropriate

Generate:

```text
dtw_cost_matrix.png
dtw_alignment.png
dtw_prototype.png
```

The project should clearly document DTW computational complexity.

---

# 18. PHASE 10 — MODEL 5: FEATURE ENGINEERING + XGBOOST

Create reusable feature extraction.

For every one of the six signals, compute a useful set of statistical features.

Minimum recommended features:

```text
mean
std
min
max
median
range
skewness
kurtosis
percentiles
RMS
energy
```

Biomechanical features where robust:

```text
range of motion
peak angle
minimum angle
maximum angle
time of peak
mean velocity
mean acceleration
```

Avoid features that depend on future/test information.

Create a stable feature order.

Train XGBoost.

Tune:

- n_estimators
- max_depth
- learning_rate
- subsample
- colsample_bytree

Use RandomizedSearchCV or GridSearchCV.

Persist:

```text
feature_metadata.json
feature_importance.csv
```

---

# 19. PHASE 11 — COMMON EVALUATION FRAMEWORK

All models must be evaluated with the same test set.

Calculate:

```text
accuracy
precision
recall
F1
```

Primary:

```text
macro F1
```

Also measure:

```text
inference time per sample
```

Prefer micro-benchmarking with a warm-up phase.

Do not include model loading time in inference latency unless explicitly reported separately.

Store:

```json
{
  "accuracy": 0.91,
  "precision_macro": 0.90,
  "recall_macro": 0.89,
  "f1_macro": 0.905,
  "inference_seconds_per_sample": 0.003
}
```

---

# 20. MODEL LEADERBOARD

Generate:

```text
output/leaderboard.csv
```

Columns:

```text
model
accuracy
precision
recall
f1_score
inference_time_seconds_per_sample
training_time_seconds
parameter_count
```

Sort primarily by:

```text
f1_score DESC
```

Break ties sensibly using latency.

Also produce a visual comparison chart.

---

# 21. AUTOMATED WINNER SELECTION

Implement:

```python
select_best_model(results)
```

Primary criterion:

```text
macro F1
```

Secondary:

```text
accuracy
```

Tertiary:

```text
inference latency
```

Do not manually choose a model.

The winning model must be discovered from the benchmark.

---

# 22. MODEL INTERPRETABILITY

Implement model-specific explainability.

## Transformer wins

Generate:

```text
attention heatmap
```

showing temporal attention behavior.

## CNN wins

Generate:

```text
activation / saliency visualization
```

## LSTM/GRU wins

Do NOT falsely call arbitrary recurrent activations “attention”.

Instead generate:

```text
gradient-based temporal saliency
```

## XGBoost wins

Generate:

```text
SHAP feature importance bar chart
```

If SHAP cannot be safely installed or executed in the environment, provide a well-documented fallback feature-importance plot and clearly label the fallback.

## DTW wins

Generate:

```text
prototype gait cycle
DTW alignment visualization
```

The implementation must select the appropriate visualization automatically.

---

# 23. CONFUSION MATRIX

Generate the confusion matrix for the winning model.

Save:

```text
output/figures/confusion_matrix.png
```

Include:

- class labels
- counts
- normalized percentages if useful

---

# 24. ADDITIONAL ERROR ANALYSIS

Generate:

```text
misclassification_report.csv
```

Include:

```text
subject
true_condition
predicted_condition
confidence
```

Where appropriate.

Also identify:

- most difficult class
- class imbalance
- low-confidence samples
- repeated misclassification patterns

---

# 25. CONFIDENCE HANDLING

For models that expose probabilities:

Calculate confidence.

Example:

```text
max predicted probability
```

Define generic confidence tiers:

```text
HIGH
MEDIUM
LOW
```

Thresholds must not be presented as clinically validated.

Document them as engineering heuristics.

---

# 26. OPTIONAL OOD DETECTION

Implement a lightweight out-of-distribution warning mechanism when practical.

Possible approaches:

- distance-based feature-space detection
- Isolation Forest
- Mahalanobis distance

The objective is not to create a validated clinical OOD detector.

The objective is to flag obviously unusual inputs.

Return:

```text
OOD = true/false
```

and document limitations.

---

# 27. PDF REPORT

Automatically generate:

```text
output/reports/clinical_gait_analysis_report.pdf
```

The report must include:

## Executive Summary

Plain English.

Required structure:

> The AutoML pipeline selected [Model Name] as the winner because it achieved [X]% accuracy and [Y]ms inference time, making it suitable for [Real-time screening / Offline diagnosis].

Do not make stronger clinical claims than the data support.

## Section 1 — Dataset overview

## Section 2 — Data quality

## Section 3 — Average gait cycle

## Section 4 — Preprocessing/segmentation

## Section 5 — Model training summary

## Section 6 — Model leaderboard

## Section 7 — Best-model confusion matrix

## Section 8 — Explainability

## Section 9 — Error analysis

## Section 10 — Deployment readiness

## Section 11 — Clinical limitations

Use professional visual formatting.

The PDF should be understandable by both:

- ML engineers
- clinicians

---

# 28. CLINICAL SAFETY LANGUAGE

This system is a CDSS.

Do NOT describe model output as an autonomous diagnosis.

Use language such as:

```text
decision support
screening aid
model prediction
clinical review recommended
```

Include limitations in the PDF:

- dataset size
- cohort representativeness
- sensor variability
- measurement noise
- class imbalance
- lack of external validation
- need for clinician oversight

---

# 29. ARTIFACT MANAGEMENT

Save the following:

```text
output/
├── models/
│   ├── best_model.*
│   └── preprocessor.*
│
├── reports/
│   └── clinical_gait_analysis_report.pdf
│
├── figures/
│
├── leaderboard.csv
├── metrics.json
├── best_model_config.json
├── run_metadata.json
└── dataset_profile.json
```

The exact model extension depends on the winning model.

For example:

```text
.keras
.pkl
```

---

# 30. MODEL CONFIGURATION

Create a configuration file such as:

```text
config/automl.yaml
```

Example concepts:

```yaml
pipeline:
  random_seed: 42
  train_ratio: 0.70
  validation_ratio: 0.15
  test_ratio: 0.15
  primary_metric: f1_macro
  max_runtime_minutes: 55

deep_learning:
  tuner: hyperband
  max_epochs: 30
  early_stopping_patience: 5

classical:
  cv_folds: 3
  random_search_iterations: 15
```

Important:

Configuration may define generic search ranges.

Do NOT hard-code dataset-specific facts such as:

```text
subject = [1,2,3,...]
```

or a fixed known class distribution.

---

# 31. BEST MODEL CONFIGURATION JSON

Generate automatically:

```text
output/best_model_config.json
```

It should contain:

```json
{
  "model_name": "...",
  "hyperparameters": {},
  "input_shape": [],
  "class_labels": [],
  "feature_order": [],
  "sequence_length": 0,
  "preprocessing_version": "...",
  "metrics": {}
}
```

---

# 32. RUN METADATA

Create:

```text
output/run_metadata.json
```

Include:

```text
run_id
timestamp
git_commit
dataset_hash
python_version
package_versions
random_seed
training_environment
```

This is important for reproducibility.

---

# 33. MLFLOW

Where practical, use MLflow for experiment tracking.

Structure:

```text
experiment:
    gait-automl

runs:
    lstm/*
    cnn/*
    transformer/*
    dtw_knn/*
    xgboost/*
```

Log:

- parameters
- metrics
- training duration
- inference duration
- dataset hash
- model artifact
- best hyperparameters

If MLflow is optional in local mode, the core pipeline must still work without a remote MLflow server.

Do not make successful execution dependent on external cloud services.

---

# 34. FASTAPI SERVICE

Implement an optional inference API.

Endpoints:

```text
GET /health
GET /ready
GET /model
POST /predict
GET /metrics
```

The API should load the winning model artifact and matching preprocessing pipeline.

The API must not retrain models.

---

# 35. PREDICTION CONTRACT

Example:

```json
{
  "prediction": 2,
  "confidence": 0.93,
  "model": "transformer",
  "model_version": "1",
  "clinical_flag": "HIGH_CONFIDENCE"
}
```

For sequence input, validate:

- number of channels
- feature order
- sequence length or supported variable length
- numeric values
- missing values

Do not accept arbitrary malformed payloads.

---

# 36. MONITORING

Use Prometheus-compatible metrics.

Track at minimum:

```text
prediction_count
prediction_latency
prediction_errors
model_version
low_confidence_predictions
out_of_distribution_predictions
```

Prometheus/Grafana are optional for the take-home runtime but the architecture should support them cleanly.

---

# 37. DOCKER

Create:

```text
Dockerfile
docker-compose.yml
```

Local composition may contain:

```text
api
mlflow
```

Do not introduce unnecessary services.

A simple local setup is preferable.

---

# 38. CI/CD

Create GitHub Actions workflow.

On pull request:

```text
lint
unit tests
API tests
data-pipeline tests
small model smoke test
Docker build
```

Do NOT run the complete expensive AutoML search on every pull request.

Provide a lightweight smoke-training mode.

---

# 39. TESTING

Write tests for:

## Data

- schema validation
- malformed CSV
- missing columns
- invalid values
- duplicates

## Splitting

- no subject leakage

## Preprocessing

- deterministic transformation
- feature ordering
- shape correctness

## Models

- each model can construct
- each model can train on tiny synthetic data
- predictions have correct shape

## AutoML

- model registration
- winner selection
- leaderboard generation

## Reporting

- PDF file generated
- figures generated
- required sections populated

## API

- health endpoint
- model endpoint
- prediction validation

---

# 40. RUNTIME BUDGET

The assignment requires completion under roughly:

```text
60 minutes CPU
30 minutes GPU
```

Optimize accordingly.

Preferred strategy:

### Deep learning

```text
Hyperband
20-ish trials or otherwise bounded search
20–30 max epochs
EarlyStopping
ReduceLROnPlateau
```

### Classical

```text
RandomizedSearchCV
10–20 iterations
```

### DTW

Use:

- limited sequence resolution
- bounded warping window
- caching
- parallelization where beneficial

Never sacrifice subject-wise leakage prevention for speed.

---

# 41. OPTIONAL GPU ACCELERATION

Detect GPU automatically.

If available:

- use TensorFlow GPU
- optionally use mixed precision

If no GPU:

- continue on CPU without user intervention

Do not crash because GPU is absent.

---

# 42. DETERMINISM

Set seeds for:

```text
Python
NumPy
TensorFlow
scikit-learn
```

Where full TensorFlow determinism conflicts with performance, document the tradeoff.

---

# 43. ERROR HANDLING

The CLI should fail gracefully.

Examples:

```text
ERROR: Required column `condition` is missing.
```

rather than:

```text
KeyError: 'condition'
```

Use structured logging where practical.

Provide clear stage progress:

```text
[1/10] Loading dataset
[2/10] Validating dataset
[3/10] Running EDA
[4/10] Preparing sequences
[5/10] Training LSTM
...
[10/10] Generating report
```

---

# 44. PERFORMANCE AND CACHING

Implement caching where it materially improves runtime.

Potential cache targets:

```text
processed sequences
engineered features
DTW distance matrices
EDA outputs
```

Use dataset hash + preprocessing configuration to avoid stale-cache bugs.

Cache correctness matters more than cleverness.

---

# 45. DATASET HASHING

Calculate SHA-256 for the input dataset.

Use it in:

```text
run_metadata.json
MLflow
cache keys
reports
```

This helps prove reproducibility.

---

# 46. MODEL VERSIONING

Every final model artifact must contain metadata identifying:

```text
model name
version
dataset hash
training run
preprocessing version
```

Do not allow an API to load a model with an incompatible preprocessor.

---

# 47. DOCUMENTATION REQUIREMENTS

README must include:

1. Project overview
2. Architecture
3. Environment setup
4. Dataset format
5. Pipeline execution
6. Outputs
7. Model descriptions
8. Hyperparameter tuning strategy
9. Data leakage prevention
10. Interpretability
11. Clinical limitations
12. API usage
13. Docker usage
14. Testing
15. Troubleshooting

---

# 48. ARCHITECTURE DOCUMENT

Create a strong `docs/ARCHITECTURE.md`.

Include:

- system diagram
- data flow
- training flow
- serving flow
- model lifecycle
- observability
- design decisions
- trade-offs
- limitations

Also document why:

```text
Feast omitted
Redis optional
PostgreSQL optional
MLflow useful
FastAPI separate from training
subject-wise splitting mandatory
```

---

# 49. ADRs

Create lightweight ADR documents.

At minimum:

```text
ADR-001-subject-wise-data-splitting.md
ADR-002-gait-sequence-representation.md
ADR-003-automated-model-selection.md
ADR-004-model-serving-boundary.md
ADR-005-clinical-explainability.md
```

Each ADR should contain:

```text
Context
Decision
Alternatives
Consequences
```

---

# 50. CODING STYLE

Follow:

- PEP8
- type hints
- docstrings
- small functions
- single responsibility
- clear naming
- no giant scripts
- no unnecessary abstractions
- dependency injection where it materially helps testing

Avoid:

```python
# 1500-line run_automl_pipeline.py
```

The top-level script should be thin.

Conceptually:

```python
def main():
    pipeline = GaitAutoMLPipeline(...)
    pipeline.run()
```

---

# 51. TOP-LEVEL ORCHESTRATOR

Implement something conceptually like:

```python
class GaitAutoMLPipeline:

    def run(self):
        self.validate()
        self.profile()
        self.prepare()
        self.split()

        lstm_result = self.run_lstm()
        cnn_result = self.run_cnn()
        transformer_result = self.run_transformer()
        dtw_result = self.run_dtw()
        xgb_result = self.run_xgboost()

        leaderboard = self.benchmark(
            [
                lstm_result,
                cnn_result,
                transformer_result,
                dtw_result,
                xgb_result,
            ]
        )

        winner = self.select_winner(leaderboard)

        self.explain(winner)
        self.generate_report(leaderboard, winner)
        self.save_artifacts(winner)

        return winner
```

Do not copy this literally if your design is better; preserve the separation of concerns.

---

# 52. EXTRA ENGINEERING FEATURES TO IMPLEMENT

Prioritize these after core functionality works:

## High priority

- Early stopping
- deterministic seeds
- dataset hashing
- reproducibility metadata
- model contract
- preprocessing serialization
- structured logs
- smoke-test mode

## Medium priority

- MLflow tracking
- Prometheus metrics
- SHAP
- OOD detection
- confidence tiers
- Docker Compose

## Optional

- Pulumi
- production cloud deployment
- asynchronous training jobs
- advanced Grafana dashboards
- full champion/challenger lifecycle

Do not spend time on optional infrastructure before the core assignment is complete.

---

# 53. QUALITY GATE

The implementation is NOT finished until:

### Functional

```text
python run_automl_pipeline.py --data data/gait.csv
```

works from a clean environment.

### Data

- schema validation works
- EDA works
- subject leakage test passes

### Models

All five models can execute:

```text
LSTM/GRU
1D-CNN
Transformer
DTW-KNN
XGBoost
```

### AutoML

- hyperparameter tuning runs
- leaderboard generated
- winner selected automatically

### Reporting

PDF contains:

- executive summary
- EDA
- leaderboard
- confusion matrix
- interpretability
- limitations

### Artifacts

Exists:

```text
best model
best_model_config.json
clinical_gait_analysis_report.pdf
leaderboard.csv
run_metadata.json
```

### Testing

All core tests pass.

### Packaging

Fresh virtual environment setup works.

---

# 54. IMPORTANT: DO NOT CHEAT THE ASSIGNMENT

Do NOT:

- use random row splitting
- tune on the test set
- hard-code the winning model
- hard-code known test labels
- hard-code a >85% accuracy result
- fabricate metrics
- fabricate clinical findings
- fake attention maps for models that do not have attention
- silently drop subjects
- silently remove classes
- claim clinical validation

If a model fails to achieve >85% accuracy on the provided dataset, report the actual performance honestly.

Do not manipulate the evaluation to force the target metric.

---

# 55. IMPLEMENTATION ORDER

Follow this exact order.

## Phase 1

Repository inspection + architecture skeleton.

## Phase 2

Dataset loader + validator.

## Phase 3

Subject-wise splitter + gait sequence builder.

## Phase 4

EDA and plots.

## Phase 5

Feature engineering.

## Phase 6

XGBoost.

## Phase 7

DTW + KNN.

## Phase 8

LSTM/GRU.

## Phase 9

1D CNN.

## Phase 10

Transformer.

## Phase 11

Unified benchmarking.

## Phase 12

Winner selection.

## Phase 13

Explainability.

## Phase 14

PDF reporting.

## Phase 15

Artifact persistence.

## Phase 16

Tests.

## Phase 17

FastAPI.

## Phase 18

Docker/MLflow/monitoring.

## Phase 19

Documentation and final cleanup.

Do not jump to deployment before the core pipeline works.

---

# 56. AGENT EXECUTION RULES

You are operating as an autonomous coding agent.

Before each implementation phase:

1. Inspect relevant files.
2. Make the smallest coherent change.
3. Run tests.
4. Run a smoke test where applicable.
5. Fix errors immediately.
6. Update documentation when the implementation meaningfully changes architecture.

Do not create placeholder files just to satisfy the directory tree.

Do not leave major TODOs for core assignment functionality.

Do not stop after generating a design. Implement the system.

---

# 57. DEFINITION OF DONE

The project is considered complete when a reviewer can clone the repository and execute:

```bash
python -m venv .venv
```

then install requirements and run:

```bash
python run_automl_pipeline.py --data data/gait.csv
```

and receive:

```text
dataset validation
EDA outputs
five model runs
hyperparameter tuning
model leaderboard
winner selection
confusion matrix
interpretability visualization
PDF report
best model artifact
configuration JSON
metadata JSON
```

without manually selecting a model or hyperparameter.

The reviewer should also be able to execute:

```bash
pytest
```

and:

```bash
docker compose up
```

for the optional API/MLflow stack.

---

# 58. FINAL AGENT BEHAVIOR

Think like a senior ML platform engineer, not like a notebook author.

Prioritize:

1. Correctness
2. Leakage prevention
3. Reproducibility
4. Modular architecture
5. Runtime efficiency
6. Explainability
7. Clinical safety
8. Testing
9. Maintainability
10. Deployment readiness

Avoid unnecessary complexity unless it provides measurable value.

When there is a tradeoff between sophistication and reliability, prefer the simplest implementation that fully satisfies the requirement.

Start by inspecting the current repository and implementing Phase 1. Continue through the phases until the system reaches the Definition of Done.
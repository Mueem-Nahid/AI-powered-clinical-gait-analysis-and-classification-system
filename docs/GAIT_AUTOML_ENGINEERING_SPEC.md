# AI-Powered Clinical Gait Analysis & Classification System
## AutoML + MLOps Engineering Specification for Coding Agents

> **Purpose:** This document is the implementation blueprint for OpenCode, Codex, Claude Code, or another coding agent. The agent should use this specification as the source of truth when implementing the repository.
>
> **Primary goal:** Build a reproducible, dataset-driven AutoML pipeline for gait classification that trains and benchmarks five model families, selects the best model automatically, generates a clinically interpretable PDF report, saves deployment artifacts, and exposes the winning model through a clean inference service.

---

## 1. Project Context

### 1.1 Project name

**AI-Powered Clinical Gait Analysis & Classification System**

Project type:

- Clinical Decision Support System (CDSS)
- Biomedical time-series analytics
- AutoML / model benchmarking platform
- MLOps-ready inference service

Target stakeholders:

- Physiatrists
- Orthopedic surgeons
- Physical therapists
- Biomechanical researchers

### 1.2 Business / technical problem

The clinical AI team receives gait joint-angle time-series data from IMU sensors and motion-capture systems. Different patient cohorts may contain different movement conditions, joints, and trial characteristics. The current workflow requires data scientists to manually choose models and tune hyperparameters.

The desired system is an **AutoML-style pipeline** that takes a raw gait dataset and automatically:

1. validates and profiles the dataset;
2. prevents subject-level data leakage;
3. converts long-format gait measurements into model-ready representations;
4. trains five competing time-series classification approaches;
5. tunes their important hyperparameters automatically;
6. evaluates them on an untouched subject-wise test set;
7. benchmarks accuracy, precision, recall, F1 and inference latency;
8. selects the best model by overall macro F1;
9. generates explainability artifacts appropriate to the winning model;
10. generates a clinician-oriented PDF report;
11. persists the model and configuration;
12. optionally registers the model with MLflow;
13. serves the selected model through FastAPI.

---

# 2. Assignment Requirements

## 2.1 Required dataset schema

The expected raw dataset has the following columns:

| Column | Meaning |
|---|---|
| `subject` | Patient ID |
| `condition` | Target class / gait pathology or movement pattern |
| `replication` | Trial / replication number |
| `leg` | Left=1, Right=2 |
| `joint` | Hip=1, Knee=2, Ankle=3 |
| `time` | Normalized gait-cycle percentage |
| `angle` | Joint angle in degrees |

The provided `sample.xlsx` contains 100 populated rows, seven columns, and the expected seven-column schema. The visible sample begins with one subject/condition/trial/leg/joint signal and time values from 0 onward; therefore the sample workbook must not be assumed to represent the complete class/signal distribution of the real `gait.csv` dataset.

## 2.2 Required Task 1 - Automated EDA

Automatically produce:

- unique subject count;
- unique condition count;
- unique leg count;
- unique joint count;
- row count;
- replication count;
- missing-value summary;
- duplicate summary;
- basic statistics by condition;
- basic statistics by joint;
- condition distribution;
- signal availability by leg/joint;
- average gait-cycle visualization for all six expected signals:
  - Left Hip
  - Left Knee
  - Left Ankle
  - Right Hip
  - Right Knee
  - Right Ankle

The earlier project/report material also calls for signal-quality analysis, segmentation visualization, statistical-feature reporting, and DTW alignment/distance visualization.

## 2.3 Required Task 2 - Five competing models

The system must automatically train and evaluate:

1. **LSTM / GRU**
2. **1D-CNN**
3. **Transformer Encoder**
4. **DTW + KNN**
5. **Feature Engineering + XGBoost**

Deep-learning tuning must cover at least three hyperparameters per model. Preferred tuning mechanism:

- Keras Tuner Hyperband for LSTM/GRU, CNN and Transformer.

Classical-model tuning must use:

- RandomizedSearchCV or GridSearchCV for DTW+KNN;
- RandomizedSearchCV for XGBoost.

## 2.4 Required leakage prevention

The split must happen by subject, never by row.

Recommended baseline:

- 70% subjects -> training
- 15% subjects -> validation
- 15% subjects -> final test

For small datasets, prefer grouped stratification or GroupKFold/StratifiedGroupKFold where technically feasible.

**Hard rule:** A patient must never appear in both training and test data.

The same subject-level grouping must be respected during hyperparameter search. Never let a CV fold contain samples from the same patient in both train and validation.

## 2.5 Required Task 3 - PDF report

The generated PDF must contain:

- executive summary;
- dataset overview;
- EDA summary;
- average gait cycle plots;
- signal quality summary;
- segmentation summary;
- model leaderboard;
- confusion matrix for the best model;
- per-class metrics;
- inference latency;
- model-training summary;
- model-specific interpretability visualization;
- final model recommendation;
- deployment-readiness information;
- limitations / clinical disclaimer.

Leaderboard columns:

- Accuracy
- Precision
- Recall
- F1-score
- Inference time per sample

Primary selection metric: **macro F1** unless a documented configuration explicitly changes it.

## 2.6 Required Task 4 - deployment artifacts

Save under `./output/`:

- best model artifact (`.keras`, `.h5`, `.pkl`, `.joblib`, etc.);
- `best_model_config.json`;
- final PDF report;
- leaderboard CSV;
- metrics JSON;
- plots / explainability artifacts;
- optional MLflow run metadata.

## 2.7 Runtime constraints

The full workflow should target:

- under 60 minutes on 8 GB RAM / 4 CPU cores;
- under 30 minutes when a free GPU is available.

The implementation must therefore use bounded search spaces and early stopping rather than exhaustive hyperparameter enumeration.

## 2.8 Allowed / expected technologies

Core libraries from the assignment:

- Python 3.9+
- pandas
- numpy
- scikit-learn
- tensorflow or pytorch
- keras-tuner or optuna
- dtaidistance
- xgboost
- matplotlib
- seaborn
- weasyprint / reportlab / fpdf

Recommended engineering additions:

- MLflow
- FastAPI
- pydantic
- pytest
- prometheus-client
- Docker / Docker Compose
- PyYAML
- joblib
- scipy
- shap

Avoid unnecessary infrastructure for the take-home scope. Feast and Redis are **not required** for the core gait solution.

---

# 3. Architectural Direction

## 3.1 Architectural principle

Use the **engineering structure and MLOps philosophy of the existing Fraud Detection Inference Service**, but do not copy its domain-specific components blindly.

The fraud repository already separates application serving, training, monitoring, infrastructure, documentation and scripts, and uses FastAPI, MLflow, Prometheus/Grafana, Docker Compose, Pulumi and GitHub Actions. The GitHub repository currently exposes directories such as `app`, `training`, `monitoring`, `infrastructure`, `docs`, and `scripts`.

Reference repository:

https://github.com/Mueem-Nahid/fraud-detection-inference-service

Use that project as the structural inspiration for:

- modularity;
- model loading;
- API boundaries;
- monitoring;
- docs/ADR style;
- Dockerized local environment;
- test separation;
- CI/CD organization;
- deployment-readiness.

## 3.2 What to keep from the fraud architecture

Keep / adapt:

- `app/` for inference API;
- `training/` for model training workflows;
- `monitoring/` for Prometheus/Grafana;
- `docs/` for architecture, runbook and ADRs;
- `scripts/` for repeatable operational commands;
- Docker Compose for local integration;
- MLflow for experiment tracking/model registry;
- model-loader abstraction;
- health endpoint;
- metrics endpoint;
- automated tests;
- GitHub Actions;
- optional IaC layer.

## 3.3 What not to copy by default

Do not add these just for architectural symmetry:

- Feast;
- Redis as an online feature store;
- a full transactional PostgreSQL dependency;
- unnecessary microservices;
- Kubernetes;
- complex event streaming.

The gait workload is batch AutoML + inference rather than real-time feature retrieval over transaction events.

---

# 4. Target System Architecture

```text
                         +----------------------+
                         |      gait.csv        |
                         +----------+-----------+
                                    |
                                    v
                         +----------------------+
                         | Data Validation       |
                         | Schema + Quality Gate |
                         +----------+-----------+
                                    |
                                    v
                         +----------------------+
                         | Automated EDA          |
                         +----------+-----------+
                                    |
                                    v
                         +----------------------+
                         | Subject-wise Split     |
                         | Train / Val / Test     |
                         +----------+-----------+
                                    |
                                    v
                  +-------------------------------------------+
                  |      Gait Preprocessing Layer             |
                  | resampling | interpolation | scaling     |
                  | cycle segmentation | sequence creation   |
                  +-------------------+-----------------------+
                                      |
                +---------------------+---------------------+
                |                     |                     |
                v                     v                     v
       Sequence Representation   Feature Representation   Distance Representation
                |                     |                     |
       +--------+--------+            |                    |
       |        |        |            |                    |
       v        v        v            v                    v
     LSTM      CNN   Transformer    XGBoost             DTW + KNN
       |        |        |            |                    |
       +--------+--------+------------+--------------------+
                                |
                                v
                       +----------------------+
                       | Benchmark / Evaluate |
                       +----------+-----------+
                                  |
                                  v
                       +----------------------+
                       | Champion Selection   |
                       | Macro F1 primary     |
                       +----------+-----------+
                                  |
             +--------------------+--------------------+
             |                    |                    |
             v                    v                    v
       Best Model Artifact    Explainability      PDF Report
             |                    |                    |
             +--------------------+--------------------+
                                  |
                                  v
                         +----------------------+
                         | MLflow Registry      |
                         +----------+-----------+
                                    |
                                    v
                         +----------------------+
                         | FastAPI Inference     |
                         +----------+-----------+
                                    |
                                    v
                            Prometheus Metrics
                                    |
                                    v
                                 Grafana
```

---

# 5. Recommended Repository Structure

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
│   └── tests/
│       ├── test_health.py
│       └── test_predict.py
│
├── automl/
│   ├── __init__.py
│   ├── orchestrator.py
│   ├── model_selector.py
│   ├── benchmark.py
│   ├── tuner.py
│   └── registry.py
│
├── data/
│   ├── __init__.py
│   ├── loader.py
│   ├── validator.py
│   ├── splitter.py
│   ├── dataset.py
│   ├── preprocessing.py
│   └── segmentation.py
│
├── features/
│   ├── __init__.py
│   ├── statistical.py
│   ├── biomechanical.py
│   └── extractor.py
│
├── models/
│   ├── __init__.py
│   ├── base.py
│   ├── lstm.py
│   ├── cnn.py
│   ├── transformer.py
│   ├── dtw_knn.py
│   └── xgboost_model.py
│
├── explainability/
│   ├── __init__.py
│   ├── attention.py
│   ├── saliency.py
│   ├── activation.py
│   ├── shap_explainer.py
│   └── dtw_visualizer.py
│
├── reporting/
│   ├── __init__.py
│   ├── report_generator.py
│   ├── plots.py
│   ├── tables.py
│   └── templates/
│
├── training/
│   ├── run_automl.py
│   └── smoke_test.py
│
├── monitoring/
│   ├── prometheus.yml
│   └── grafana/
│       └── dashboards/
│
├── config/
│   ├── automl.yaml
│   ├── models.yaml
│   └── report.yaml
│
├── scripts/
│   ├── run_local.sh
│   ├── run_automl.sh
│   ├── smoke_test.sh
│   └── clean_output.sh
│
├── docs/
│   ├── ARCHITECTURE.md
│   ├── AUTOML.md
│   ├── DATA_PIPELINE.md
│   ├── MODELING.md
│   ├── EVALUATION.md
│   ├── CLINICAL_INTERPRETABILITY.md
│   ├── DEPLOYMENT.md
│   ├── RUNBOOK.md
│   └── adr/
│       ├── ADR-001-subject-wise-split.md
│       ├── ADR-002-gait-representation.md
│       ├── ADR-003-model-selection.md
│       ├── ADR-004-automl-budget.md
│       └── ADR-005-model-serving.md
│
├── tests/
│   ├── test_data_validation.py
│   ├── test_preprocessing.py
│   ├── test_segmentation.py
│   ├── test_features.py
│   ├── test_model_interfaces.py
│   ├── test_automl.py
│   └── test_reporting.py
│
├── output/
│   ├── models/
│   ├── figures/
│   ├── reports/
│   ├── leaderboard.csv
│   ├── metrics.json
│   ├── best_model_config.json
│   └── run_metadata.json
│
├── data_input/
│   └── gait.csv
│
├── run_automl_pipeline.py
├── Dockerfile
├── docker-compose.yml
├── docker-compose.prod.yml
├── requirements.txt
├── requirements-dev.txt
├── .env.example
├── .gitignore
├── README.md
└── Makefile
```

---

# 6. Core Domain Model

Create a canonical in-memory representation rather than letting each model implement its own raw CSV parsing.

## 6.1 Gait sample identity

A model sample should normally represent one gait trial / replication for one subject, with its class label.

Recommended logical identity:

```text
(subject, replication)
```

The exact grouping must be validated from the data. If the source dataset represents gait cycles differently, the implementation must infer a safe trial key rather than silently merging unrelated rows.

## 6.2 Six-channel signal representation

Expected canonical channels:

```text
left_hip
left_knee
left_ankle
right_hip
right_knee
right_ankle
```

Canonical sequence shape:

```text
(time_steps, 6)
```

The pipeline should not permanently assume a specific time-step count. Resampling should produce a configurable common sequence length, with an automatic default inferred from the observed time resolution.

Example:

```text
(101, 6)
```

is a likely representation when the normalized gait cycle has 0..100 inclusive samples.

---

# 7. Data Pipeline Design

## Phase A - Input validation

Validate:

- required columns exist;
- columns are numeric/categorical as expected;
- condition is present and non-null;
- subject IDs are present;
- time values are finite;
- angle values are numeric and finite;
- time range is sensible;
- leg values are known;
- joint values are known;
- no illegal condition values;
- duplicate records are identified;
- every logical trial has enough signal points.

The pipeline should fail fast on invalid schema and produce warnings for recoverable data-quality issues.

## Phase B - Data profiling

Automatically produce:

- shape;
- unique counts;
- class balance;
- subject balance;
- trials per condition;
- trials per subject;
- signal completeness;
- missingness;
- outlier count;
- time-grid consistency;
- angle statistics.

## Phase C - Subject-wise splitting

Never split rows randomly.

Implementation preference:

```python
train_test_split(subject_ids, ...)
```

followed by optional grouped stratification or GroupKFold/StratifiedGroupKFold for tuning.

Persist split IDs to:

```text
output/splits.json
```

This makes leakage debugging reproducible.

## Phase D - Gait-cycle normalization

For each trial/channel:

1. sort by time;
2. remove invalid time duplicates according to a documented rule;
3. handle missing time samples using interpolation only when safe;
4. resample to a common grid;
5. preserve the physical angle values;
6. optionally normalize channels using training-only statistics.

**Do not fit scalers on validation/test data.**

## Phase E - Segmentation

If the dataset already represents a single normalized gait cycle, do not arbitrarily create windows that duplicate the same cycle. If continuous sequences are provided, create windows only after defining a clinically meaningful segmentation strategy.

For this assignment, prefer whole-trial / whole-cycle samples because the input explicitly contains normalized gait cycle time.

---

# 8. Automated EDA Outputs

Generate these figures programmatically:

1. `dataset_overview.png`
2. `class_distribution.png`
3. `signal_quality.png`
4. `average_gait_cycles.png`
5. `raw_vs_resampled.png`
6. `segmentation_overview.png`
7. `condition_joint_statistics.png`

The six-channel gait-cycle figure should have six subplots and plot mean curves by condition.

The supplied project material also recommends:

- feature statistics tables and correlation visualizations;
- DTW cost matrices / warping paths;
- raw versus filtered signals;
- segmented windows by class.

---

# 9. Feature Engineering Design

For every canonical signal, compute a safe, bounded set of statistical features.

Recommended baseline features:

```text
mean
std
min
max
median
range
q05
q25
q75
q95
skewness
kurtosis
RMS
signal energy
peak index
peak value
trough index
trough value
```

Optional biomechanical features:

```text
range of motion
maximum angular velocity
minimum angular velocity
maximum angular acceleration
minimum angular acceleration
time-to-peak
time-to-trough
```

All feature transformations must be deterministic and serializable.

Feature order must be saved in the model configuration.

---

# 10. Model Specifications

## 10.1 LSTM / GRU

Implement a configurable recurrent classifier.

Preferred baseline:

```text
Input
 -> Recurrent Block(s)
 -> Dropout
 -> Dense
 -> Softmax
```

Hyperparameters to tune:

- recurrent type: LSTM or GRU;
- number of layers;
- units per layer;
- dropout;
- learning rate;
- batch size if runtime permits.

Use:

- EarlyStopping;
- ReduceLROnPlateau;
- deterministic seeds where practical.

Do not search a huge architecture space.

## 10.2 1D-CNN

Baseline:

```text
Input
 -> Conv1D
 -> BatchNorm
 -> Pooling
 -> Conv1D
 -> GlobalAveragePooling
 -> Dense
 -> Softmax
```

Tune:

- number of filters;
- kernel size;
- number of convolution blocks;
- dropout;
- learning rate.

## 10.3 Transformer Encoder

Baseline:

```text
Input
 -> Linear Projection
 -> Positional Encoding
 -> Transformer Encoder Block(s)
 -> Global Pooling
 -> Dense
 -> Softmax
```

Tune:

- embedding dimension;
- number of heads;
- number of encoder blocks;
- feed-forward dimension;
- dropout;
- learning rate.

Keep the model small enough for CPU.

## 10.4 DTW + KNN

Pipeline:

```text
Gait Sequence
 -> DTW distance
 -> Pairwise distance matrix
 -> KNN
```

Tune:

- number of neighbors;
- distance weighting;
- DTW warping-window constraint.

Because DTW is expensive:

- use bounded sequences;
- use a constrained warping window;
- cache training distances;
- avoid unnecessary repeated distance computation.

## 10.5 Feature Engineering + XGBoost

Pipeline:

```text
Raw sequence
 -> statistical / biomechanical features
 -> training-only preprocessing
 -> XGBoost
```

Tune:

- `n_estimators`;
- `max_depth`;
- `learning_rate`;
- `subsample`;
- `colsample_bytree`;
- optionally `min_child_weight`.

Use a bounded `RandomizedSearchCV`.

---

# 11. AutoML Orchestrator

Create a central orchestrator class.

Example responsibility model:

```python
class GaitAutoMLPipeline:
    def run(self):
        self.validate()
        self.profile()
        self.split()
        self.prepare()
        self.train_all_models()
        self.evaluate_all_models()
        self.select_champion()
        self.explain_champion()
        self.generate_report()
        self.persist_artifacts()
        self.register_model()
```

The top-level entry point must remain thin:

```bash
python run_automl_pipeline.py --data data_input/gait.csv
```

The command should run without interactive prompts.

Optional arguments:

```text
--data
--output
--config
--max-runtime-minutes
--resume
--skip-mlflow
--device auto|cpu|gpu
```

The default invocation must work without requiring these flags.

---

# 12. Hyperparameter Optimization Strategy

## 12.1 Runtime-first search policy

The system is not intended to find a mathematically global optimum. It must find a strong candidate under a fixed compute budget.

Recommended defaults:

```yaml
auto_ml:
  max_runtime_minutes: 55
  primary_metric: f1_macro
  random_seed: 42

deep_learning:
  tuner: hyperband
  max_epochs: 30
  early_stopping_patience: 5

classical:
  search_type: randomized
  n_iter: 10-20
  cv: 3
```

These are generic pipeline limits, not dataset-specific learned values.

## 12.2 Search budget

Preferred strategy:

- LSTM/GRU: Hyperband with small/medium search space;
- CNN: Hyperband with small/medium search space;
- Transformer: Hyperband with intentionally small search space;
- DTW+KNN: RandomizedSearchCV / small grid;
- XGBoost: RandomizedSearchCV.

Stop or reduce trials automatically when the configured time budget is close to being exhausted.

---

# 13. Evaluation Design

## 13.1 Primary metric

Primary model-selection metric:

**Macro F1 on the untouched subject-wise test set.**

Secondary metrics:

- Accuracy
- Precision (macro)
- Recall (macro)
- Inference latency

Also report:

- per-class precision;
- per-class recall;
- per-class F1;
- confusion matrix.

## 13.2 Inference latency

Measure actual prediction latency using warm execution.

Method:

1. load model;
2. warm up once or more;
3. run N repeated samples;
4. compute mean latency;
5. report seconds/sample and ms/sample.

For batch-capable deep learning models, measure both:

- one-sample latency;
- batch inference throughput if practical.

The required leaderboard should use **seconds per sample**.

## 13.3 Statistical caution

Do not claim clinical superiority from one small sample dataset.

Report:

- sample size;
- number of subjects;
- class distribution;
- confidence limitations;
- external validation not performed unless actually performed.

---

# 14. Champion Selection Policy

Selection hierarchy:

1. highest macro F1;
2. if nearly tied, prefer higher recall;
3. if still tied, prefer lower inference latency;
4. if still tied, prefer smaller/less complex model.

Do not silently optimize for an unreported hidden score.

Store the exact selection rule in `best_model_config.json`.

Example:

```json
{
  "selection_metric": "f1_macro",
  "tie_breakers": [
    "recall_macro",
    "latency_ms_per_sample",
    "model_complexity"
  ]
}
```

---

# 15. Explainability Requirements

Use a model-specific strategy.

## If Transformer wins

Generate:

- attention heatmap over time;
- optionally signal/channel aggregation;
- highlighted temporal regions.

## If LSTM / GRU wins

Do **not** claim attention weights unless an attention mechanism was actually implemented.

Preferred:

- gradient-based saliency;
- Integrated Gradients if available;
- temporal saliency overlay on gait curves.

## If CNN wins

Generate:

- activation map / saliency map;
- optional filter-response visualization.

## If XGBoost wins

Generate:

- SHAP feature-importance bar chart;
- optional SHAP beeswarm plot.

## If DTW + KNN wins

Generate:

- prototype / average gait cycle;
- DTW alignment visualization;
- nearest-neighbor examples if practical.

The supplied project/report material specifically distinguishes Transformer attention, LSTM/GRU hidden-state visualization, CNN activation maps, SHAP/LIME feature importance, and DTW prototype/alignment plots as model-appropriate explainability outputs.

---

# 16. Clinical Safety Enhancements

These are enhancements beyond the minimum assignment and should be implemented only after the required pipeline is reliable.

## 16.1 Confidence threshold

Return:

```text
HIGH_CONFIDENCE
MEDIUM_CONFIDENCE
LOW_CONFIDENCE
```

Do not present low-confidence predictions as definitive diagnoses.

## 16.2 Out-of-distribution detection

Preferred first implementation:

- standardized feature representation;
- distance-based OOD score / Mahalanobis distance or Isolation Forest;
- warning when a sample is far from the training distribution.

The OOD mechanism must be documented as a screening safeguard, not a validated clinical diagnostic test.

## 16.3 Calibration

Optional but valuable:

- temperature scaling on validation predictions;
- calibration report;
- reliability diagram.

## 16.4 Clinical disclaimer

Every report/API response intended for users should clearly state that the model is a decision-support tool and does not independently establish a medical diagnosis.

---

# 17. Model Artifact Contract

Every champion model must have an associated metadata contract.

Example:

```json
{
  "model_name": "transformer",
  "model_version": "1",
  "input_shape": [101, 6],
  "class_labels": [1, 2, 3],
  "channel_order": [
    "left_hip",
    "left_knee",
    "left_ankle",
    "right_hip",
    "right_knee",
    "right_ankle"
  ],
  "preprocessing_version": "1.0.0",
  "selection_metric": "f1_macro",
  "metrics": {
    "accuracy": 0.0,
    "f1_macro": 0.0,
    "precision_macro": 0.0,
    "recall_macro": 0.0,
    "latency_ms_per_sample": 0.0
  }
}
```

The model loader must validate this contract before serving predictions.

---

# 18. MLflow Design

MLflow should be an **optional but preferred** MLOps layer.

Experiment name:

```text
clinical-gait-automl
```

Each model/tuning trial may record:

- model family;
- hyperparameters;
- training time;
- validation metrics;
- test metrics for final selected candidates only;
- inference latency;
- dataset hash;
- Git commit SHA;
- Python version;
- library versions;
- artifact paths.

Recommended registry name:

```text
gait_classifier
```

Aliases:

```text
candidate
production
```

Only the selected champion should be promoted to `production` automatically.

Do not expose credentials or private infrastructure settings in source control.

---

# 19. FastAPI Inference Service

The API is not the primary assignment requirement, but it is the preferred architecture enhancement because it aligns with the existing fraud-detection service.

## Endpoints

### `GET /health`

Return:

- service status;
- loaded model name;
- model version;
- preprocessing version.

### `POST /predict`

Accept a model-compatible gait sample.

Return:

```json
{
  "prediction": 2,
  "confidence": 0.93,
  "model": "transformer",
  "model_version": "1",
  "confidence_band": "HIGH_CONFIDENCE",
  "explanation_available": true
}
```

### `GET /metrics`

Expose application metrics.

### `POST /reload-model`

Optional operational endpoint to reload a new champion model safely.

Protect it appropriately in production.

---

# 20. Monitoring

Use Prometheus/Grafana only as a lightweight MLOps enhancement.

Track:

- prediction count;
- request count;
- request latency;
- model version;
- inference errors;
- input validation errors;
- low-confidence prediction count;
- OOD warning count;
- predicted-class distribution.

The dashboard should distinguish operational health from model quality.

Do not pretend that production accuracy is known unless ground-truth feedback actually exists.

---

# 21. Configuration Strategy

Use YAML configuration for generic pipeline behavior.

Example `config/automl.yaml`:

```yaml
pipeline:
  random_seed: 42
  primary_metric: f1_macro
  max_runtime_minutes: 55
  output_dir: output

split:
  train_size: 0.70
  validation_size: 0.15
  test_size: 0.15

sequence:
  length: auto
  channels: auto

training:
  early_stopping_patience: 5
  max_epochs: 30
  batch_size: auto

search:
  deep_learning_engine: hyperband
  classical_engine: randomized
  classical_iterations: 15
  cv_folds: 3

report:
  include_clinical_disclaimer: true
```

Important:

- configuration must be generic;
- dataset-derived dimensions must be detected automatically;
- do not hard-code sample-specific class counts, signal counts or sequence lengths into the source;
- model search spaces must be defined centrally and reused consistently.

---

# 22. Logging Strategy

Use standard Python `logging`.

Every major stage must emit:

```text
[DATA]
[EDA]
[SPLIT]
[PREPROCESS]
[TUNING]
[TRAIN]
[EVALUATE]
[SELECT]
[EXPLAIN]
[REPORT]
[REGISTRY]
[API]
```

Example:

```text
[TRAIN] Starting Transformer Hyperband search
[TUNING] Trial 4/20
[TUNING] Validation macro-F1=0.884
[EVALUATE] Transformer test macro-F1=0.902
[SELECT] Current champion=Transformer
```

Logs should make a failed run diagnosable without opening a debugger.

---

# 23. Reproducibility

Every run should generate:

```text
output/run_metadata.json
```

Include:

- UTC timestamp;
- dataset SHA256 hash;
- Git commit SHA if available;
- random seed;
- Python version;
- package versions;
- device used;
- run duration;
- training subject IDs;
- validation subject IDs;
- test subject IDs.

Set seeds for:

- Python;
- NumPy;
- TensorFlow/PyTorch;
- scikit-learn where relevant.

Document any source of nondeterminism.

---

# 24. Testing Strategy

## Unit tests

Test independently:

- schema validation;
- subject splitting;
- no overlap between splits;
- resampling;
- missing-value handling;
- feature extraction;
- model factory interfaces;
- benchmark calculations;
- report generation inputs.

## Integration test

Use a tiny synthetic gait dataset to verify:

```bash
python run_automl_pipeline.py --data tests/fixtures/tiny_gait.csv
```

It does not need to optimize every model fully; it only needs to prove the pipeline can execute end-to-end.

## Leakage test

Mandatory test:

```python
assert train_subjects.isdisjoint(val_subjects)
assert train_subjects.isdisjoint(test_subjects)
assert val_subjects.isdisjoint(test_subjects)
```

## Artifact test

After a successful test run, verify:

- model exists;
- JSON exists;
- leaderboard exists;
- PDF exists;
- report is non-empty;
- model can be reloaded.

---

# 25. Report Design

The PDF should look like a professional technical/clinical analysis report rather than a raw notebook export.

## Recommended order

### Page 1 - Executive Summary

Include:

- dataset size;
- subject count;
- class count;
- best model;
- test accuracy;
- macro F1;
- latency;
- suitability statement;
- clinical disclaimer.

### Page 2 - Dataset & Quality

Include:

- schema;
- missing values;
- class distribution;
- subject counts;
- quality warnings.

### Page 3 - Gait Visualization

Six-panel average gait-cycle plot.

### Page 4 - Preprocessing & Segmentation

Show:

- raw/resampled examples;
- segmentation summary;
- sequence shape.

### Page 5 - Model Leaderboard

Table with:

- model;
- accuracy;
- precision;
- recall;
- F1;
- inference time.

### Page 6 - Winner Performance

Include:

- confusion matrix;
- per-class metrics;
- error analysis.

### Page 7 - Explainability

Model-specific visualization.

### Page 8 - Deployment Readiness

Include:

- artifact paths;
- model version;
- preprocessing contract;
- latency;
- model size;
- limitations.

The supplied report material also recommends misclassification analysis, subject-specific prediction summaries, severity/progression reporting where applicable, rehabilitation outcome tracking, condition-prediction logs and low-confidence/anomaly reports.

---

# 26. Clinical Language Policy

Use careful language.

Preferred:

- `predicted condition`;
- `decision-support signal`;
- `model confidence`;
- `screening support`;
- `requires clinician review`.

Avoid:

- `diagnoses the patient`;
- `confirms disease`;
- `clinically proven` unless external validation actually exists;
- `safe for diagnosis` without evidence.

The system should explicitly state that model outputs support clinical review and are not an autonomous medical diagnosis.

---

# 27. CI/CD

Adapt the existing project’s CI/CD philosophy.

## Pull request pipeline

```text
Checkout
  -> lint
  -> unit tests
  -> package/import test
  -> tiny pipeline smoke test
  -> Docker build
```

Do not perform the complete expensive AutoML search on every pull request.

## Full experiment pipeline

Triggered manually or by a dedicated workflow:

```text
Data validation
 -> full AutoML search
 -> benchmark
 -> report
 -> model artifact
 -> MLflow registration
```

## Deployment pipeline

```text
Build image
 -> integration test
 -> deploy
 -> /health validation
```

---

# 28. Docker Design

Recommended services:

```text
api
automl-worker
mlflow
prometheus
```

Optional:

```text
grafana
```

For the assignment, a single container can still run the pipeline if required. However, keep training and serving responsibilities logically separate in code.

Avoid adding PostgreSQL/Redis solely because the fraud project uses them.

---

# 29. Infrastructure Strategy

### Minimum submission

- local Python execution;
- Dockerfile;
- Docker Compose.

### Optional enhancement

Use Pulumi or Terraform for a small AWS deployment.

Potential deployment:

```text
AWS EC2
  |
  +-- FastAPI
  +-- MLflow
  +-- Prometheus
  +-- Grafana
```

Do not spend the majority of the take-home time on cloud infrastructure. The assignment's strongest scoring areas are code quality, reproducibility, model performance, report quality and useful engineering enhancements.

---

# 30. README Requirements

README must include:

1. project overview;
2. architecture diagram;
3. directory structure;
4. setup instructions;
5. Python version;
6. installation command;
7. how to run AutoML;
8. expected outputs;
9. how to run API;
10. how to run tests;
11. Docker instructions;
12. model summary;
13. limitations;
14. clinical disclaimer;
15. reproducibility information.

Primary command:

```bash
python run_automl_pipeline.py --data data_input/gait.csv
```

Docker command:

```bash
docker compose up --build
```

---

# 31. Requirements File Strategy

Use two requirements files if practical:

### `requirements.txt`

Production/runtime dependencies:

- pandas
- numpy
- scikit-learn
- tensorflow
- keras-tuner
- dtaidistance
- xgboost
- matplotlib
- seaborn
- reportlab or weasyprint
- fastapi
- uvicorn
- pydantic
- joblib
- mlflow
- prometheus-client
- pyyaml
- shap

### `requirements-dev.txt`

- pytest
- pytest-cov
- ruff or flake8
- black
- mypy if used

Pin versions only after testing the actual environment. Never claim a dependency combination is compatible without testing it.

---

# 32. Performance Engineering

Important for the 60-minute requirement.

## Deep learning

Use:

- early stopping;
- reduced epoch budgets;
- Hyperband;
- small models;
- optional mixed precision only on supported GPUs;
- GPU auto-detection;
- efficient batching.

## DTW

Use:

- fixed/common sequence length;
- constrained warping window;
- cached distances;
- parallelism where safe.

## XGBoost

Use:

- histogram tree method where appropriate;
- bounded estimators;
- randomized search.

## Report generation

Generate figures once and reuse them in PDF.

---

# 33. Innovation / Extra Enhancement Priorities

Do not implement every enhancement before the required assignment works.

Priority order:

### Tier 1 - High value, low complexity

- early stopping;
- dataset hash;
- subject split persistence;
- model metadata contract;
- clean PDF report;
- model-specific explainability;
- structured logging;
- tests;
- MLflow tracking.

### Tier 2 - Strong differentiators

- confidence bands;
- OOD detection;
- probability calibration;
- champion/challenger registry;
- inference benchmark;
- Prometheus/Grafana;
- model reload endpoint.

### Tier 3 - Nice-to-have only

- Pulumi AWS deployment;
- scheduled retraining;
- full longitudinal patient dashboard;
- online feedback loop;
- Kubernetes.

---

# 34. Phase-by-Phase Implementation Plan

## Phase 0 - Repository bootstrap

Tasks:

- create directory structure;
- create `requirements.txt`;
- create `.env.example`;
- create config files;
- configure logging;
- configure lint/test tooling;
- create README skeleton.

Definition of done:

```bash
python -m pytest
```

runs with at least one passing test.

---

## Phase 1 - Data loading and validation

Implement:

- CSV loader;
- schema validation;
- type coercion;
- duplicate detection;
- missing-data report;
- class/subject summary.

Outputs:

```text
output/data_quality.json
output/data_quality_report.txt
```

Definition of done:

- invalid schema fails clearly;
- valid sample loads successfully.

---

## Phase 2 - Subject-wise split

Implement:

- subject-level train/validation/test split;
- grouped tuning CV;
- split persistence;
- leakage tests.

Definition of done:

- no subject overlap;
- split is reproducible.

---

## Phase 3 - Sequence preprocessing

Implement:

- channel pivoting;
- trial grouping;
- interpolation;
- resampling;
- normalization using train-only statistics;
- canonical `(T, 6)` output.

Definition of done:

```text
X_sequence.shape == (N, T, 6)
```

or an automatically inferred equivalent.

---

## Phase 4 - EDA/report primitives

Implement reusable plotting functions.

Required figures:

- six-signal average gait cycles;
- class distribution;
- signal-quality plot;
- raw vs processed;
- statistics table;
- optional DTW plot.

Definition of done:

- every figure is generated without manual plot commands.

---

## Phase 5 - Feature engineering

Implement:

- statistical features;
- biomechanical features;
- feature naming/order;
- feature matrix persistence.

Definition of done:

- deterministic output;
- serializable feature metadata.

---

## Phase 6 - Model interfaces

Create a common model interface:

```python
fit()
predict()
predict_proba()
score()
save()
load()
```

Implement each model behind the same interface.

Definition of done:

- synthetic dataset can call all five models through one interface.

---

## Phase 7 - Hyperparameter tuning

Implement:

- Hyperband for DL models;
- RandomizedSearchCV / GridSearchCV for DTW and XGBoost;
- runtime budget;
- early stopping;
- trial logging.

Definition of done:

- tuning runs without interactive input;
- at least three meaningful hyperparameters are searched for every DL family.

---

## Phase 8 - Evaluation / benchmark

Implement:

- common metric calculation;
- inference latency benchmark;
- confusion matrix;
- per-class metrics;
- leaderboard.

Definition of done:

`leaderboard.csv` contains all five models.

---

## Phase 9 - Champion selection

Implement:

- macro-F1 ranking;
- tie-breaking;
- champion metadata;
- best model copy.

Definition of done:

A new dataset can automatically identify a champion without changing code.

---

## Phase 10 - Explainability

Implement model-specific explanation dispatch:

```text
Transformer -> attention
LSTM/GRU   -> saliency
CNN        -> activation/saliency
XGBoost    -> SHAP
DTW        -> alignment/prototype
```

Definition of done:

The final report always contains exactly one valid, model-appropriate interpretability visualization.

---

## Phase 11 - PDF report

Implement:

- report template;
- tables;
- plots;
- executive summary;
- recommendation paragraph;
- clinical disclaimer.

Definition of done:

A clean PDF is generated automatically from one command.

---

## Phase 12 - Artifact persistence

Implement:

- model save;
- JSON config;
- leaderboard;
- metrics;
- run metadata;
- figures;
- PDF.

Definition of done:

All required deliverables are under `output/`.

---

## Phase 13 - MLflow

Implement:

- experiment creation;
- parameter logging;
- metric logging;
- artifact logging;
- champion registration.

Definition of done:

A complete AutoML run is inspectable in MLflow.

---

## Phase 14 - FastAPI

Implement:

- `/health`;
- `/predict`;
- `/metrics`;
- model reload if safe.

Definition of done:

The persisted champion can be loaded into the API without running training again.

---

## Phase 15 - Monitoring

Implement:

- request count;
- latency;
- prediction distribution;
- low confidence count;
- OOD warnings.

Definition of done:

Prometheus can scrape API metrics.

---

## Phase 16 - Docker and CI

Implement:

- Dockerfile;
- Compose stack;
- GitHub Actions;
- smoke test;
- deployment health check.

Definition of done:

A clean environment can build and run the service without manual code modifications.

---

# 35. Definition of Done - Assignment

The implementation is complete only when all of the following are true:

- [ ] `python run_automl_pipeline.py --data data_input/gait.csv` works;
- [ ] no interactive prompts are required;
- [ ] raw input is validated automatically;
- [ ] EDA is generated automatically;
- [ ] subject-wise split is enforced;
- [ ] leakage tests pass;
- [ ] LSTM/GRU is trained and tuned;
- [ ] 1D-CNN is trained and tuned;
- [ ] Transformer is trained and tuned;
- [ ] DTW+KNN is tuned;
- [ ] XGBoost is tuned;
- [ ] all five are benchmarked;
- [ ] macro F1 is computed;
- [ ] inference latency is measured;
- [ ] best model is automatically selected;
- [ ] best model confusion matrix is generated;
- [ ] model-specific explainability plot is generated;
- [ ] PDF report is generated;
- [ ] best model artifact is saved;
- [ ] best hyperparameter config is saved;
- [ ] leaderboard is saved;
- [ ] run metadata is saved;
- [ ] test suite passes;
- [ ] README contains exact setup/run instructions;
- [ ] no secret credentials are committed.

---

# 36. Definition of Done - Stronger Submission

A stronger implementation additionally has:

- [ ] MLflow experiment tracking;
- [ ] model registry;
- [ ] FastAPI service;
- [ ] Prometheus metrics;
- [ ] Docker Compose;
- [ ] structured logging;
- [ ] OOD warning;
- [ ] confidence thresholding;
- [ ] probability calibration;
- [ ] model contract validation;
- [ ] Git commit and dataset hash in metadata;
- [ ] CI pipeline;
- [ ] ADR documentation.

---

# 37. Important Implementation Guardrails for the Coding Agent

1. **Do not use random row splitting.**
2. **Do not fit preprocessing on the test set.**
3. **Do not tune against the final test set.**
4. **Do not hard-code the dataset's class count when it can be inferred.**
5. **Do not hard-code six channels without validating available leg/joint combinations.**
6. **Do not create clinically misleading claims.**
7. **Do not add infrastructure that does not improve the assignment.**
8. **Do not create one giant Python script containing every class and function.**
9. **Do not let FastAPI own model training jobs.**
10. **Do not report test metrics from the validation set.**
11. **Do not call LSTM output an attention map unless attention is actually implemented.**
12. **Do not make SHAP mandatory for models where it is inappropriate or computationally excessive.**
13. **Do not silently skip a model because it performs poorly.**
14. **Do not fail the whole run because optional MLflow/Grafana functionality is unavailable; core AutoML must still run locally.**
15. **Do not require external cloud credentials for the base assignment.**

---

# 38. Recommended Coding-Agent Execution Strategy

The coding agent should work incrementally and verify every phase.

Recommended order:

```text
1. Inspect repository and input dataset
2. Create architecture and config
3. Implement validation
4. Implement subject splitting
5. Implement sequence preparation
6. Implement EDA
7. Implement XGBoost baseline first
8. Implement CNN
9. Implement LSTM/GRU
10. Implement Transformer
11. Implement DTW+KNN
12. Implement tuning
13. Implement common evaluation
14. Implement champion selection
15. Implement explainability
16. Implement PDF report
17. Implement artifact persistence
18. Add MLflow
19. Add FastAPI
20. Add monitoring
21. Add Docker / CI
22. Run full clean-environment verification
23. Package deliverable
```

### Why XGBoost first?

It is generally the fastest way to establish a working end-to-end benchmark and validate that the data representation is meaningful before spending compute on three deep-learning searches.

### Why build the common model interface early?

It prevents the evaluation and reporting code from becoming coupled to TensorFlow-specific or XGBoost-specific behavior.

---

# 39. Suggested Agent Prompt

Use the following text when assigning the implementation to a coding agent:

> Implement this repository strictly according to `GAIT_AUTOML_ENGINEERING_SPEC.md`.
>
> First inspect the current repository and the provided dataset. Do not assume the dataset distribution beyond what can be verified from the files.
>
> Preserve the structural style of the existing Mueem-Nahid fraud-detection inference service: separate application, training, monitoring, infrastructure, docs, scripts, tests, Docker and CI concerns.
>
> Build the gait system as an AutoML + MLOps project, not as a monolithic notebook.
>
> The mandatory functionality is:
>
> - automated data validation;
> - automated EDA;
> - subject-wise split with zero leakage;
> - normalized gait-cycle sequence construction;
> - LSTM/GRU with automated hyperparameter tuning;
> - 1D-CNN with automated hyperparameter tuning;
> - Transformer Encoder with automated hyperparameter tuning;
> - DTW + KNN with automated search;
> - feature engineering + XGBoost with automated search;
> - common evaluation and benchmark;
> - automatic champion selection by macro F1;
> - model-specific explainability;
> - automated PDF report;
> - model/config/leaderboard/metadata artifacts;
> - reproducible execution;
> - tests.
>
> Then add the MLOps enhancements in this priority order:
>
> 1. MLflow
> 2. FastAPI
> 3. Prometheus
> 4. Docker Compose
> 5. confidence thresholding / OOD detection
> 6. CI/CD
> 7. optional cloud IaC
>
> Keep the default execution path dependency-light enough to run locally without cloud services.
>
> Do not use Feast or Redis unless there is a concrete requirement discovered in the data or service architecture.
>
> Before considering the work complete, execute the full pipeline on the supplied data, verify every artifact, run tests, validate the PDF visually, confirm that train/validation/test subjects do not overlap, and document any limitation or failure honestly.

---

# 40. Final Architecture Summary

The final project should be understood as:

```text
                  Clinical Gait Dataset
                           |
                           v
                  Automated Data QA/EDA
                           |
                           v
                    Subject-wise Split
                           |
                           v
                  Gait Preprocessing Core
                           |
          +----------------+----------------+
          |                |                |
          v                v                v
     Deep Learning    Distance Model    Feature Model
          |                |                |
   +------+------+         |                |
   |      |      |         |                |
  LSTM   CNN Transformer   DTW+KNN       XGBoost
   |      |      |         |                |
   +------+------+---------+----------------+
                          |
                          v
                Automated Benchmarking
                          |
                          v
                 Champion Model Select
                          |
           +--------------+--------------+
           |              |              |
           v              v              v
      Explanation      PDF Report    Model Artifact
           |                             |
           +--------------+--------------+
                          |
                          v
                       MLflow
                          |
                          v
                       FastAPI
                          |
                    Prometheus/Grafana
```

The key architectural distinction is:

**Fraud project:** primarily production model serving + MLOps.

**Gait project:** AutoML + biomedical time-series preprocessing + model benchmarking + explainability + clinical reporting + MLOps/serving.

This keeps the engineering strengths of the existing fraud project while making the architecture appropriate to the gait-analysis problem.

---

# 41. Source / Reference Material

Existing architecture reference:

https://github.com/Mueem-Nahid/fraud-detection-inference-service

Project requirement source:

- MediAI Dynamics assignment brief supplied in the conversation.

Supplementary gait-report requirements supplied in the project PDF:

- data preparation and feature reports;
- DTW alignment and distance reporting;
- signal-quality reporting;
- segmentation reporting;
- training/convergence reporting;
- performance/error analysis;
- subject-specific prediction reporting;
- model-specific explainability;
- five-model comparative benchmark;
- clinical application reporting.

The PDF specifically emphasizes clinical trust through model-specific interpretability, including attention visualizations for Transformers, hidden-state visualization for recurrent models, activation maps for 1D-CNN, SHAP/LIME for feature-engineering ML, and DTW prototypes/alignment plots for DTW-based models.

---

# 42. Non-Goals

Unless time remains after the mandatory scope works reliably, do not prioritize:

- Kubernetes;
- complex streaming architecture;
- real-time sensor ingestion;
- multi-tenant authentication;
- full electronic health record integration;
- patient-level longitudinal database;
- advanced cloud networking;
- distributed GPU training;
- continuous online learning.

These may be future roadmap items, not take-home assignment requirements.

---

# 43. Success Criteria

A successful submission should feel like a **small production ML platform**, not a collection of five notebooks.

The reviewer should be able to:

```text
clone
  -> install
  -> provide gait.csv
  -> run one command
  -> receive:
       - model leaderboard
       - best model
       - configuration
       - explainability plot
       - clinical PDF report
       - reproducibility metadata
  -> optionally start FastAPI
  -> inspect model metrics / operational metrics
```

That is the intended end state.

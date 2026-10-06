# AI-Powered Clinical Gait Analysis & Classification System

A reproducible, dataset-driven **AutoML pipeline** for clinical gait classification. It
validates and profiles a gait joint-angle dataset, trains and tunes **five competing
time-series models** with subject-level leakage prevention, selects the best model by
macro F1, and produces a clinician-oriented PDF report plus deployment artifacts.

> **Clinical disclaimer:** This is a decision-support tool, not a diagnostic device.
> Model outputs support clinician review and do not independently establish a medical
> diagnosis.

## Overview

The pipeline turns a raw long-format gait CSV into a complete model benchmark:

```
gait.csv -> validate -> EDA -> subject-wise split -> preprocess
         -> [LSTM/GRU, 1D-CNN, Transformer, DTW+KNN, XGBoost] (auto-tuned)
         -> benchmark -> champion selection -> explainability -> PDF report
         -> artifacts (model, config, leaderboard, metrics, metadata)
```

## Dataset format

Long-format CSV with these columns:

| Column | Meaning |
|---|---|
| `subject` | Patient ID |
| `condition` | Target class / movement pattern |
| `replication` | Trial number |
| `leg` | 1=Left, 2=Right |
| `joint` | 1=Hip, 2=Knee, 3=Ankle |
| `time` | Normalized gait-cycle percentage (0–100) |
| `angle` | Joint angle in degrees |

The pipeline auto-detects subjects, classes, channels, and sequence length — no
dataset-specific values are hard-coded.

## Environment

- Python **3.11** (TensorFlow has no Python 3.14 wheel; 3.11 is the tested target)
- 4 CPU cores / 8 GB RAM (target: < 60 min full run)

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt -r requirements-dev.txt
```

## Run the AutoML pipeline

```bash
python run_automl_pipeline.py --data data/gait.csv
```

Optional flags (all have sane defaults): `--output`, `--config`, `--seed`,
`--max-runtime-minutes`, `--fast`, `--skip-mlflow`, `--verbose`, `--device`.

## Outputs

Everything lands under `output/`:

- `models/` — best model artifact + serialized preprocessor
- `figures/` — EDA and explainability plots
- `reports/clinical_gait_analysis_report.pdf`
- `leaderboard.csv`, `metrics.json`, `best_model_config.json`,
  `run_metadata.json`, `dataset_profile.json`, `splits.json`

## Run tests

```bash
pytest
```

## Project structure

See the plan and (once implemented) `docs/ARCHITECTURE.md`. Packages: `dataprep/`
(data + validation + splitting + sequences), `features/`, `models/`, `automl/`,
`explainability/`, `reporting/`, `common/`.

## Limitations

Small dataset, single cohort, no external validation; interpretability and confidence
thresholds are engineering heuristics, not clinically validated. Reported performance is
honest and measured on an untouched subject-wise test split.

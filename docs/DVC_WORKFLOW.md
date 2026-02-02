# DVC Workflow Guide

Learn how to use DVC (Data Version Control) to manage datasets and pipelines in this project.

## Table of Contents

- [Overview](#overview)
- [Common Commands](#common-commands)
  - [Getting Data](#getting-data)
  - [Working with Data](#working-with-data)
  - [Creating Data Pipelines](#creating-data-pipelines)
- [Adding a Data Processing Function](#adding-a-data-processing-function)
- [Adding a New Model Training Function](#adding-a-new-model-training-function)
- [Launching Training with Specific Parameters](#launching-training-with-specific-parameters)
  - [Option 1: Edit params.yaml Directly](#option-1-edit-paramsyaml-directly)
  - [Option 2: Override Parameters on the Command Line](#option-2-override-parameters-on-the-command-line)
  - [Option 3: Run Experiments in Parallel (Queue)](#option-3-run-experiments-in-parallel-queue)
  - [Comparing Experiments](#comparing-experiments)
  - [Applying an Experiment](#applying-an-experiment)
- [Team Collaboration](#team-collaboration)
- [Best Practices](#best-practices)
- [Common Workflows](#common-workflows)
- [Troubleshooting](#troubleshooting)
- [Quick Reference](#quick-reference)

---

## Overview

**DVC** tracks your data files and pipelines, similar to how Git tracks code. Data is stored in **OVH Object Storage** (S3-compatible) cloud storage, while Git only tracks small `.dvc` metadata files.

### Data Organization

```
data/
├── bronze/                    # 🥉 Bronze: Immutable raw data (NEVER modify!)
│   ├── 0. Airport data/       #    Flight records (CSV)
│   ├── 1. AC characteristics/ #    Aircraft specifications (XLSX)
│   ├── 2. Weather data/       #    Historical weather (CSV/XLSX)
│   └── 3. Test set/           #    Test data files
├── silver/                    # 🥈 Silver: Cleaned, validated data (Parquet)
│   ├── 0. Airport data/
│   ├── 1. AC characteristics/
│   ├── 2. Weather data/
│   └── 3. Test set/
├── gold/                      # 🥇 Gold: Feature-engineered, model-ready data
│   ├── training_set_airport_data.parquet        # With NaN (for XGBoost, LightGBM, CatBoost)
│   ├── training_set_airport_data_clean.parquet  # NaN handled (for sklearn models)
│   ├── test_set_airport_data.parquet
│   └── test_set_airport_data_clean.parquet
└── predictions/               # 📊 Model predictions for dashboard
    └── dashboard_data.parquet

src/pipelines/
├── data/                      # 📝 Data pipeline scripts (bronze→silver→gold)
│   ├── bronze_to_silver.py
│   └── silver_to_gold.py
├── models/                    # 🤖 Model training & prediction scripts
│   ├── train_models.py        #    Training orchestration
│   ├── create_predictions.py  #    Inference pipeline with SHAP
│   ├── xgboost_model.py
│   ├── lightgbm_model.py
│   ├── catboost_model.py
│   ├── random_forest.py
│   └── linear_regression.py
└── utils/                     # 🔧 Shared utilities
    ├── model_inputs_loading.py
    └── model_prediction_making.py

constants/                     # 📋 Global constants
├── paths.py                   #    Centralized path definitions
├── column_names.py            #    Dataset column name constants
└── feature_categories.py      #    SHAP feature groupings

params.yaml                    # ⚙️ All tunable parameters (tracked by DVC)
dvc.yaml                       # 🔄 Pipeline stage definitions
```

## Common Commands

### Getting Data

```bash
# Pull latest data from OVH Object Storage
dvc pull

# Check what's in sync
dvc status
```

### Working with Data

#### 1. Never Modify Raw Data

Raw data in `data/bronze/` is **immutable**. Always:
- ✅ Read from `data/bronze/`
- ✅ Write to `data/silver/` or `data/gold/`
- ❌ Never modify files in `data/bronze/`

**Tracking rules:**
- `data/bronze/`: Use `dvc add` to track raw data files
- `data/silver/` and `data/gold/`: Tracked automatically via pipeline `outs:` in `dvc.yaml` (don't use `dvc add`)

#### 2. Create New Processed Datasets

Silver and gold files are tracked automatically through the pipeline outputs in `dvc.yaml`. Don't use `dvc add` manually for these files.

```bash
# 1. Add your processing function to the pipeline script
#    (see "Adding a Data Processing Function" section)

# 2. Define the output in dvc.yaml under the appropriate stage:
#    outs:
#      - data/silver/cleaned_flights.parquet

# 3. Run the pipeline - DVC tracks the output automatically
uv run dvc repro bronze_to_silver

# 4. Commit the lock file
git add dvc.yaml dvc.lock
git commit -m "Add cleaned dataset pipeline"

# 5. Push data and code
dvc push
git push
```

#### 3. Update Existing Datasets

```bash
# Modify your processing script and re-run the pipeline
uv run dvc repro bronze_to_silver

# DVC detects the change automatically via dvc.lock
dvc status

# Commit the updated lock file
git add dvc.lock
git commit -m "Update cleaned dataset: added date parsing"

# Push to OVH and Git
dvc push
git push
```

### Creating Data Pipelines

Define reproducible pipelines in `dvc.yaml`:

```yaml
stages:
  bronze_to_silver:
    cmd: uv run python -m src.pipelines.data.bronze_to_silver
    deps:
      - src/pipelines/data/bronze_to_silver.py
      - data/bronze/raw_flights.csv
    outs:
      - data/silver/cleaned_flights.parquet

  silver_to_gold:
    cmd: uv run python -m src.pipelines.data.silver_to_gold
    deps:
      - src/pipelines/data/silver_to_gold.py
      - data/silver/cleaned_flights.parquet
    outs:
      - data/gold/flight_features.parquet

  train_models:
    cmd: uv run python -m src.pipelines.models.train_models
    deps:
      - src/pipelines/models/train_models.py
      - data/gold/flight_features.parquet
    params:
      - train_models
      - hyperparameters
```

Run the pipeline:

```bash
# Run entire pipeline
uv run dvc repro

# DVC automatically:
# - Runs stages in correct order
# - Only re-runs what changed
# - Tracks all outputs

# Visualize the pipeline
uv run dvc dag

# Push all results
dvc push
git add dvc.lock
git commit -m "Run data pipeline"
git push
```

## Adding a Data Processing Function

Follow these steps to add a new data processing step to the pipeline.

### Step 1: Write Your Processing Function

Add your function to the appropriate pipeline file:

**For Bronze → Silver transformations:** `src/pipelines/data/bronze_to_silver.py`

```python
# src/pipelines/data/bronze_to_silver.py

import pandas as pd
from constants.paths import BRONZE_DIR, SILVER_DIR
from src.utils.logger import logger


def clean_flight_data(input_filename: str, output_filename: str) -> None:
    """Clean raw flight data and save to silver.

    Args:
        input_filename: Name of the input file in bronze directory.
        output_filename: Name of the output file in silver directory.
    """
    input_path = BRONZE_DIR / input_filename
    output_path = SILVER_DIR / output_filename

    logger.info(f"Reading data from {input_path}")
    df = pd.read_csv(input_path)

    logger.info("Cleaning data: removing nulls, fixing types...")
    df = df.dropna(subset=["flight_id"])
    df["departure_time"] = pd.to_datetime(df["departure_time"])

    logger.info(f"Writing cleaned data to {output_path}")
    df.to_parquet(output_path, index=False)


def bronze_to_silver():
    """Pipeline to transform bronze data to silver data."""
    clean_flight_data("raw_flights.csv", "cleaned_flights.parquet")
```

**For Silver → Gold transformations:** `src/pipelines/data/silver_to_gold.py`

```python
# src/pipelines/data/silver_to_gold.py

import pandas as pd
from constants.paths import SILVER_DIR, GOLD_DIR
from src.utils.logger import logger


def create_features(input_filename: str, output_filename: str) -> None:
    """Create features for model training.

    Args:
        input_filename: Name of the input file in silver directory.
        output_filename: Name of the output file in gold directory.
    """
    input_path = SILVER_DIR / input_filename
    output_path = GOLD_DIR / output_filename

    logger.info(f"Reading data from {input_path}")
    df = pd.read_parquet(input_path)

    logger.info("Creating features...")
    df["hour_of_day"] = df["departure_time"].dt.hour
    df["day_of_week"] = df["departure_time"].dt.dayofweek

    logger.info(f"Writing features to {output_path}")
    df.to_parquet(output_path, index=False)


def silver_to_gold():
    """Pipeline to transform silver data to gold data."""
    create_features("cleaned_flights.parquet", "flight_features.parquet")
```

### Step 2: Update dvc.yaml

Add your data dependencies and outputs to the appropriate stage in `dvc.yaml`:

```yaml
# dvc.yaml

stages:
  bronze_to_silver:
    cmd: uv run python -m src.pipelines.data.bronze_to_silver
    deps:
      - src/pipelines/data/bronze_to_silver.py
      - data/bronze/raw_flights.csv          # Add your input files
    outs:
      - data/silver/cleaned_flights.parquet  # Add your output files

  silver_to_gold:
    cmd: uv run python -m src.pipelines.data.silver_to_gold
    deps:
      - src/pipelines/data/silver_to_gold.py
      - data/silver/cleaned_flights.parquet  # Depends on previous stage output
    outs:
      - data/gold/flight_features.parquet    # Add your output files
```

### Step 3: Run the Pipeline

```bash
# Run only your stage
uv run dvc repro bronze_to_silver

# Or run the full pipeline
uv run dvc repro

# Visualize the pipeline
uv run dvc dag
```

## Adding a New Model Training Function

Follow these steps to add a new model type to the training pipeline.

### Step 1: Create the Model Training File

Create a new file in `src/pipelines/models/` (e.g., `xgboost_model.py`):

```python
# src/pipelines/models/xgboost_model.py

from dotenv import load_dotenv
import mlflow
from mlflow.models.signature import infer_signature
import optuna
import pandas as pd
from sklearn.metrics import root_mean_squared_error
from xgboost import XGBRegressor


def optimize_xgboost_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    hyperparameter_grid: dict | None,
    n_trials: int = 50,
) -> dict:
    """Optimize an XGBoost model with Optuna.

    Args:
        X_train: Training feature matrix.
        y_train: Training target vector.
        X_test: Testing feature matrix.
        y_test: Testing target vector.
        hyperparameter_grid: Hyperparameter grid for optimization.
        n_trials: Number of Optuna trials.

    Returns:
        Best hyperparameters found by Optuna.
    """
    def objective(trial):
        if hyperparameter_grid is not None:
            params = {
                key: trial.suggest_categorical(key, values)
                for key, values in hyperparameter_grid.items()
            }
        else:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            }

        model = XGBRegressor(**params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return root_mean_squared_error(y_test, y_pred)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    return study.best_params


def train_xgboost_model(
    run_name: str,
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    hyperparameter_grid: dict | None = None,
    n_trials: int = 50,
):
    """Train and log an XGBoost model using MLflow.

    Args:
        run_name: Name of the MLflow run.
        model_name: Name to log the model under in MLflow.
        X_train: Training feature matrix.
        y_train: Training target vector.
        X_test: Testing feature matrix.
        y_test: Testing target vector.
        hyperparameter_grid: Hyperparameter grid for optimization.
        n_trials: Number of Optuna trials.
    """
    load_dotenv()

    with mlflow.start_run(run_name=run_name):
        best_params = optimize_xgboost_model(
            X_train, y_train, X_test, y_test, hyperparameter_grid, n_trials
        )

        model = XGBRegressor(**best_params)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        rmse = root_mean_squared_error(y_test, y_pred)

        mlflow.log_params(best_params)
        mlflow.log_metric("rmse", rmse)

        signature = infer_signature(X_train, model.predict(X_train))
        mlflow.sklearn.log_model(model, name=model_name, signature=signature)
```

### Step 2: Register the Model in train_models.py

Update `src/pipelines/models/train_models.py` to include your new model:

```python
# src/pipelines/models/train_models.py

from src.pipelines.models.linear_regression import train_linear_regression_model
from src.pipelines.models.xgboost_model import train_xgboost_model  # Add import
from src.pipelines.utils.model_inputs_loading import (
    load_hyperparameter_grid,
    load_nb_optimization_trials,
    load_params,
    load_training_data,
)


def train_models(model_type: str, run_name: str, model_name: str):
    """Train machine learning models for plane taxi time prediction."""
    X_train, y_train, X_test, y_test = load_training_data()

    if model_type == "linear_regression":
        hyperparameter_grid = load_hyperparameter_grid(model_type)
        nb_optimization_trials = load_nb_optimization_trials()
        train_linear_regression_model(
            run_name=run_name,
            model_name=model_name,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            hyperparameter_grid=hyperparameter_grid,
            n_trials=nb_optimization_trials,
        )

    # Add your new model type here
    elif model_type == "xgboost":
        hyperparameter_grid = load_hyperparameter_grid(model_type)
        nb_optimization_trials = load_nb_optimization_trials()
        train_xgboost_model(
            run_name=run_name,
            model_name=model_name,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            hyperparameter_grid=hyperparameter_grid,
            n_trials=nb_optimization_trials,
        )

    else:
        raise ValueError(f"Unsupported model type: {model_type}")
```

### Step 3: Add Hyperparameters to params.yaml

Add the hyperparameter grid for your new model in `params.yaml`:

```yaml
# params.yaml

train_models:
  model_type: xgboost              # Change to your new model
  run_name: xgboost_run
  model_name: xgboost_model
  n_trials: 50

hyperparameters:
  linear_regression:
    fit_intercept: [true, false]

  # Add your new model's hyperparameter grid
  xgboost:
    n_estimators: [50, 100, 200]
    max_depth: [3, 5, 7]
    learning_rate: [0.01, 0.1, 0.2]
```

### Step 4: Run Training

```bash
# Run training with current params.yaml settings
uv run dvc repro train_models

# Or run the full pipeline
uv run dvc repro
```

## Launching Training with Specific Parameters

DVC tracks parameters in `params.yaml` and lets you override them for experiments.

### Option 1: Edit params.yaml Directly

Modify `params.yaml` and run:

```bash
uv run dvc repro train_models
```

### Option 2: Override Parameters on the Command Line

Use `--set-param` to override without editing files:

```bash
# Change number of optimization trials
uv run dvc exp run --set-param train_models.n_trials=100

# Change model type
uv run dvc exp run --set-param train_models.model_type=xgboost

# Change multiple parameters at once
uv run dvc exp run \
  --set-param train_models.model_type=xgboost \
  --set-param train_models.n_trials=200 \
  --set-param train_models.run_name=xgboost_experiment_1
```

### Option 3: Run Experiments in Parallel (Queue)

Queue multiple experiments with different parameters:

```bash
# Queue experiments
uv run dvc exp run --queue --set-param train_models.n_trials=50
uv run dvc exp run --queue --set-param train_models.n_trials=100
uv run dvc exp run --queue --set-param train_models.n_trials=200

# Run all queued experiments
uv run dvc exp run --run-all
```

### Comparing Experiments

```bash
# View all experiments
uv run dvc exp show

# Compare parameters between experiments
uv run dvc params diff

# Compare specific experiments
uv run dvc exp diff exp-abc123 exp-def456
```

### Applying an Experiment

After finding the best experiment, apply it to your workspace:

```bash
# Apply experiment results to workspace
uv run dvc exp apply exp-abc123

# Commit the changes
git add params.yaml dvc.lock
git commit -m "Apply best experiment: n_trials=100"
```

## Team Collaboration

### Pulling Teammate's Data

```bash
# Teammate creates new dataset and pushes
# You pull their changes:

git pull                  # Get .dvc files
dvc pull                 # Get actual data

# Now you have their datasets!
```

### Sharing Your Data

Silver and gold outputs are tracked through the pipeline. After updating your pipeline:

```bash
# Run the pipeline to generate outputs
uv run dvc repro

# Commit the pipeline changes and lock file
git add dvc.yaml dvc.lock
git commit -m "Add new feature set"

# Push data and code
dvc push
git push

# Teammates can now pull your data!
```

### Avoiding Conflicts

**Data files:** DVC handles this automatically. Each version is stored separately in OVH Object Storage.

**.dvc files:** Treat like code:
- Pull before making changes: `git pull`
- Communicate with team about major data updates
- Use descriptive commit messages

## Best Practices

### 1. Descriptive Commit Messages

```bash
# ❌ Bad
git commit -m "Update data"

# ✅ Good
git commit -m "Add cleaned electricity data: removed outliers, filled missing values"
```

### 2. Version Your Outputs

Pipeline outputs are versioned automatically through `dvc.lock`. Each time you run the pipeline, DVC tracks the new version.

```bash
# Iterative improvements: just run the pipeline
uv run dvc repro silver_to_gold
git add dvc.lock
git commit -m "Update gold features: added rolling averages"

# For experiments with different parameters, use DVC experiments
uv run dvc exp run --set-param train_models.run_name=experiment_v1
uv run dvc exp run --set-param train_models.run_name=experiment_v2
```

### 3. Use Pipelines for Reproducibility

Instead of manual steps, define pipelines in `dvc.yaml`:

```yaml
stages:
  silver_to_gold:
    cmd: uv run python -m src.pipelines.data.silver_to_gold
    deps:
      - src/pipelines/data/silver_to_gold.py
      - data/silver/cleaned_data.parquet
    outs:
      - data/gold/features.parquet
```

Then anyone can reproduce: `uv run dvc repro`

### 4. Document Your Data

Add a `README.md` in each data directory:

```markdown
## data/gold/README.md

## flight_features.parquet
- Created: 2025-11-18
- Source: data/silver/cleaned_flights.parquet
- Processing: src/pipelines/data/silver_to_gold.py
- Features: 15 columns including time-based features
- Rows: 1,234
```

## Common Workflows

### Starting Fresh

```bash
# Remove local data
rm -rf data/bronze data/silver data/gold

# Pull everything from OVH
uv run dvc pull
```

### Checking What Changed

```bash
# See what data changed locally
dvc status

# See what changed in Git
git status

# See pipeline status
dvc status
```

### Reverting to Previous Version

```bash
# Find the commit with the data version you want
git log -- data/gold/features.parquet.dvc

# Check out that version
git checkout <commit-hash> data/gold/features.parquet.dvc

# Get the data
uv run dvc checkout data/gold/features.parquet
```

## Troubleshooting

**"Unable to locate credentials"**
- Make sure you've configured DVC remote credentials:
  ```bash
  source .env
  dvc remote modify --local ovh-storage access_key_id $OVH_ACCESS_KEY_ID
  dvc remote modify --local ovh-storage secret_access_key $OVH_SECRET_ACCESS_KEY
  ```

**"File not found in cache"**
- Run `dvc pull` to download from OVH Object Storage

**"Conflict in .dvc file"**
- Usually safe to accept both versions
- Then run `dvc checkout` to sync

**"Push takes forever"**
- DVC uploads only new/changed data
- Large files take time on first push
- Subsequent pushes are faster

## Quick Reference

```bash
# Get data
uv run dvc pull                # Download all data from OVH
uv run dvc status              # Check sync status

# Track data
dvc add <file>                 # Track bronze (raw) data only
dvc push                       # Upload to OVH
# Note: Silver/gold files are tracked via pipeline outputs in dvc.yaml

# Pipelines
uv run dvc repro               # Run full pipeline
uv run dvc repro <stage>       # Run specific stage
uv run dvc dag                 # Visualize pipeline

# Experiments
uv run dvc exp run             # Run experiment
uv run dvc exp run --set-param train_models.n_trials=100  # Override params
uv run dvc exp show            # List experiments
uv run dvc params diff         # Compare parameters

# Info
uv run dvc status              # Show data changes
uv run dvc diff                # Compare versions
```

## Next Steps

- Define your data pipeline stages in `dvc.yaml`
- Create data processing functions in `src/pipelines/data/`
- Create model training functions in `src/pipelines/models/`
- Configure parameters in `params.yaml`
- Run experiments: `uv run dvc exp run --set-param train_models.model_type=xgboost`

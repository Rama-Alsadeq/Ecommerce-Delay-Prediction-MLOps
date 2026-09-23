# Ecommerce-Delay-Prediction-MLOps 

> **Production-oriented MLOps pipeline for e-commerce delivery delay prediction**

[![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange?logo=scikit-learn)](https://scikit-learn.org/)
[![Logistic Regression](https://img.shields.io/badge/Model-Logistic%20Regression-orange)](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
[![FastAPI](https://img.shields.io/badge/FastAPI-API-009688?logo=fastapi)](https://fastapi.tiangolo.com/)
[![MLflow](https://img.shields.io/badge/MLflow-Model%20Management-0194E2?logo=mlflow)](https://mlflow.org/)
[![DVC](https://img.shields.io/badge/DVC-Data%20Versioning-13ADC7?logo=dvc)](https://dvc.org/)
[![Great Expectations](https://img.shields.io/badge/Great%20Expectations-Data%20Validation-4B8BBE)](https://greatexpectations.io/)
[![Docker](https://img.shields.io/badge/Docker-Containerization-2496ED?logo=docker)](https://www.docker.com/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-Database-4169E1?logo=postgresql)](https://www.postgresql.org/)
[![GitHub Actions](https://img.shields.io/badge/GitHub%20Actions-CI%2FCD-2088FF?logo=githubactions)](https://github.com/features/actions)
[![Pytest](https://img.shields.io/badge/Pytest-Testing-0A9EDC?logo=pytest)](https://pytest.org/)
[![Black](https://img.shields.io/badge/Black-Code%20Formatting-000000)](https://black.readthedocs.io/)
[![Ruff](https://img.shields.io/badge/Ruff-Linting-D7FF64)](https://docs.astral.sh/ruff/)

A production-oriented machine learning and MLOps project for predicting **late vs. on-time delivery** using the Brazilian E-Commerce (Olist) dataset, with an additional regression model for predicting **delivery delay in days (`delay_days`)**.

The project covers the complete machine learning lifecycle, starting from relational e-commerce data and exploratory analysis, progressing through feature engineering and model development, and continuing into data validation, experiment tracking, model registry, API serving, containerization, automated testing, CI/CD, and production monitoring.

The primary production objective is **binary classification of delivery delays**. A separate **regression task** is also included to estimate the number of delayed days.

---

## Table of Contents

* [Project Overview](#project-overview)
* [Problem Definition](#problem-definition)
* [Data Workflow](#data-workflow)
* [Notebook Workflow](#notebook-workflow)
* [Data Preparation](#data-preparation)
* [Exploratory Data Analysis](#exploratory-data-analysis)
* [Feature Engineering](#feature-engineering)
* [Model Development](#model-development)
* [Production Architecture](#production-architecture)
* [Repository Structure](#repository-structure)
* [Configuration](#configuration)
* [Data and Artifact Versioning](#data-and-artifact-versioning)
* [Data Validation](#data-validation)
* [Experiment Tracking and Model Registry](#experiment-tracking-and-model-registry)
* [Production Inference](#production-inference)
* [FastAPI](#fastapi)
* [Monitoring](#monitoring)
* [Logging](#logging)
* [Testing](#testing)
* [Code Quality](#code-quality)
* [Docker and Docker Compose](#docker-and-docker-compose)
* [CI/CD](#cicd)
* [Reproducibility](#reproducibility)
* [Known Limitations](#known-limitations)
* [Running the Project](#running-the-project)
* [Project Summary](#project-summary)

---

## Project Overview

The project uses the **Brazilian E-Commerce (Olist)** dataset to develop a machine learning system around delivery performance.

The main question is:

> **Can we predict whether an order will be delivered late or on time using information available at prediction time?**

The project also includes a second modeling objective:

> **Can we predict the number of days by which an order is delayed?**

This results in two modeling paths.

### Classification

The primary production task predicts:

```text
delayed = 0 → On time
delayed = 1 → Late
```

### Regression

The additional regression task predicts:

```text
delay_days
```

representing the number of delivery-delay days.

The classification model is currently the model exposed through the production API.

---

## Problem Definition

Delivery performance is determined from the relationship between the actual delivery date and the estimated delivery date.

The binary target is derived as:

```text
delayed = 1
```

when the actual delivery is later than the estimated delivery date.

Otherwise:

```text
delayed = 0
```

The regression target is:

```text
delay_days
```

The project follows an important modeling constraint:

> Features used for prediction must represent information that would be available at prediction time.

Variables that depend on future information or directly reveal the outcome are therefore excluded from the production feature set.

---

## Data Workflow

The original data is stored in relational Olist tables.

The first stage focuses on understanding the individual tables and their relationships before constructing the machine learning dataset.

```text
Olist relational tables
        │
        ▼
Table inspection
        │
        ▼
Relationship analysis
        │
        ▼
Aggregation of one-to-many tables
        │
        ▼
Order-level ML table
        │
        ▼
Target creation
        │
        ▼
Train / validation / test split
```

A key principle is maintaining the correct **unit of observation**.

The final modeling table contains:

> **One row per order**

This is particularly important for tables such as order items and payments, where multiple records can belong to a single order.

Aggregating these tables before joining prevents unintended row multiplication and distorted features.

---

## Notebook Workflow

The project uses a sequence of focused notebooks rather than placing the entire analysis in one notebook.

Each notebook has a specific responsibility and produces artifacts that can be consumed by subsequent stages.

```text
01 — Read and join the tables
        ↓
02 — Create the labels
        ↓
03 — Train / validation / test split
        ↓
04 — Exploratory data analysis
        ↓
05 — Feature engineering
        ↓
06 — Train / tune / evaluate
```

### 01 — Read and Join the Tables

This stage:

* Reads the Olist tables from PostgreSQL.
* Inspects row counts and table structure.
* Identifies primary and foreign keys.
* Checks duplicates.
* Determines what one row represents in each table.
* Analyzes relationships between tables.
* Aggregates one-to-many tables before joining.

The main output is an order-level ML table with one row per order.

### 02 — Create the Labels

The delivery dates are used to create:

```text
delayed
delay_days
```

The classification label is checked against real orders to verify that the target-generation logic is correct.

The class distribution is also examined to understand the balance between late and on-time orders.

### 03 — Train / Validation / Test Split

The labeled dataset is divided into:

```text
train
validation
test
```

The split is performed before detailed modeling decisions so that the test set does not influence feature engineering or model selection.

The label distribution and date ranges are also checked across the resulting datasets.

### 04 — Exploratory Data Analysis

Detailed EDA is performed primarily on the training data.

The analysis covers:

* data types
* dataset shape
* memory usage
* missing values
* numerical distributions
* skewness
* outliers
* categorical cardinality
* rare categories
* inconsistent values
* correlations
* groupby analysis
* cross-tabulations
* dates
* seasonality
* geography
* customer and seller information
* geographic distance

The results guide feature engineering and model development.

EDA outputs are saved as artifacts, including charts and a findings summary.

### 05 — Feature Engineering

The selected features are transformed into model-ready representations.

The workflow includes:

* missing-value handling
* feature transformations
* categorical transformations
* temporal features
* geographic features
* replacement rules
* value clipping
* scaling where required

Only information available at prediction time is used.

Fitted preprocessing objects are saved so they can be reused during inference.

### 06 — Train, Tune, and Evaluate

Models are trained using the prepared datasets.

The classification workflow focuses on the `delayed` target.

The regression workflow predicts `delay_days`.

The modeling process includes baseline comparison, model training, validation, and final evaluation using the test set only after development decisions have been made.

---

## Feature Engineering

The final feature engineering workflow produces model-ready datasets and fitted preprocessing artifacts.

Important saved artifacts include:

```text
feature_names.json
imputation_medians.joblib
replacement_values.joblib
scaler.joblib
geo_mean_delay_maps.joblib
global_mean_delay.joblib
preprocessing_config.json
```

The production system loads these fitted objects rather than fitting new transformations on incoming data.

This prevents training/inference preprocessing mismatch and avoids data leakage.

---

## Model Development

The project contains two modeling objectives.

### Classification Model

The primary model predicts whether an order is delayed.

Target:

```text
delayed
```

The production classification model is:

```text
Logistic Regression
```

The classification training workflow uses SMOTE to address class imbalance.

The registered production model is:

```text
Brazilian-E-Commerce-Classifier
```

with registered version:

```text
1
```

### Regression Model

The additional regression objective predicts:

```text
delay_days
```

The regression workflow includes an:

```text
XGBoost
```

model.

The regression model is an additional modeling path and is not currently exposed through the primary production API.

---

## Production Architecture

The project transforms the notebook workflow into a production-oriented inference system.

```text
                    ┌──────────────────────┐
                    │      FastAPI         │
                    │       API            │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Validation           │
                    │ + Feature Preparation│
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Saved Preprocessing   │
                    │ Artifacts             │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ MLflow Model Registry │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Classification Model  │
                    └──────────┬───────────┘
                               │
                               ▼
                    Prediction + Monitoring
```

Supporting infrastructure includes:

```text
PostgreSQL
Jupyter/Python
MLflow
FastAPI
Docker
Docker Compose
```

---

## Repository Structure

```text
Ecommerce-Delay-Prediction-MLOps/
│
├── app/
│   └── main.py
│
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── features.py
│   ├── predictor.py
│   ├── preprocessing.py
│   ├── validation.py
│   ├── logging_config.py
│   └── monitoring.py
│
├── notebooks/
│   └── analysis and modeling notebooks
│
├── data/
│   └── Olist data and ML datasets
│
├── artifacts/
│   ├── eda/
│   ├── feature_engineering/
│   ├── models/
│   ├── production/
│   └── monitoring/
│
├── config/
│   └── config.yaml
│
├── tests/
│   ├── conftest.py
│   ├── test_api.py
│   ├── test_features.py
│   ├── test_monitoring.py
│   ├── test_predictor.py
│   ├── test_preprocessing.py
│   ├── test_validation.py
│   └── fixtures/
│
├── gx/
│   └── Great Expectations configuration
│
├── requirements/
│   ├── runtime.txt
│   └── dev.txt
│
├── Dockerfile
├── Dockerfile.api
├── Dockerfile.mlflow
├── docker-compose.yml
├── pytest.ini
├── requirements.txt
└── README.md
```

---

## Configuration

Project configuration is centralized in:

```text
config/config.yaml
```

The configuration includes:

* project paths
* model paths
* preprocessing artifacts
* targets
* API configuration
* MLflow configuration
* monitoring configuration

The monitoring configuration includes:

```yaml
monitoring:
  prediction_log: logs/predictions.jsonl
  reference_distribution: artifacts/monitoring/reference_prediction_distribution.json
  thresholds:
    min_samples_for_drift: 30
    error_rate: 0.05
    latency_ms: 500
    psi_warning: 0.10
    psi_alert: 0.20
```

Centralizing these values avoids embedding operational parameters directly in the application code.

---

## Data and Artifact Versioning

DVC is used to version large datasets and machine learning artifacts.

The project tracks:

```text
data/
artifacts/feature_engineering/
artifacts/models/
```

A Google Drive DVC remote is configured for storing versioned data and artifacts.

The project therefore separates:

```text
Git
→ source code, configuration, tests, documentation

DVC
→ datasets and large ML artifacts
```

This allows the data and artifact state to be reproduced alongside the corresponding code version.

---

## Data Validation

Great Expectations is used for dataset-level data quality validation.

The validation suite contains:

```text
112 expectations
```

The final valid dataset achieved:

```text
112 / 112 expectations passed
```

The expectations cover properties such as:

* column existence
* data types
* missing values
* valid ranges
* valid categories
* structural consistency

A deliberately invalid value such as:

```text
purchase_month = 13
```

was also tested and correctly caused validation failure.

This provides a validation layer before model development and production inference.

---

## Experiment Tracking and Model Registry

MLflow is used for experiment tracking and model management.

The classification experiment is:

```text
Brazilian-E-Commerce-Classification
```

The registered production model is:

```text
Brazilian-E-Commerce-Classifier
```

Current registered version:

```text
1
```

The production API loads the model from the MLflow Model Registry rather than from a notebook-specific model path.

```text
MLflow Model Registry
        ↓
Brazilian-E-Commerce-Classifier
        ↓
Version 1
        ↓
FastAPI
```

This separates model development from production model serving.

---

## Production Inference

Production inference uses a dedicated set of fitted artifacts:

```text
artifacts/production/
├── feature_names.json
├── imputation_medians.joblib
├── replacement_values.joblib
└── scaler.joblib
```

The inference pipeline is:

```text
Incoming request
        ↓
Pydantic schema validation
        ↓
Feature validation
        ↓
Feature preparation
        ↓
Saved replacement rules
        ↓
Saved imputation values
        ↓
Feature ordering
        ↓
Saved scaler
        ↓
MLflow registered model
        ↓
Prediction
```

No preprocessing object is fitted during inference.

The same fitted preprocessing state used during model development is reused for incoming data.

---

## FastAPI

The production classification model is served through FastAPI.

### Health

```http
GET /health
```

Returns API and model status.

Example:

```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_name": "Brazilian-E-Commerce-Classifier",
  "model_version": "1"
}
```

### Model Information

```http
GET /model
```

Returns:

* model name
* model version
* model URI
* model type
* feature count

### Single Prediction

```http
POST /predict
```

Returns:

```json
{
  "prediction": 1,
  "probability": 0.6004965668449761,
  "model_version": "1"
}
```

### Batch Prediction

```http
POST /predict/batch
```

Accepts multiple orders and returns a prediction, probability, and model version for each order.

### Monitoring Metrics

```http
GET /metrics
```

Returns:

* request count
* error count
* error rate
* average latency
* prediction distribution
* prediction drift
* monitoring alerts

### API Validation

Pydantic schemas reject:

* missing required features
* unexpected features
* invalid data types
* invalid values

The request models use:

```text
extra="forbid"
```

to prevent undeclared fields from being silently accepted.

---

## Monitoring

The application includes both operational and model monitoring.

### Request Metrics

The API tracks:

```text
request count
error count
error rate
average latency
```

### Prediction Logging

Successful predictions are stored in:

```text
logs/predictions.jsonl
```

Each record contains:

```text
timestamp
prediction
probability
model_version
latency_ms
```

Example:

```json
{
  "timestamp": "2026-09-23T12:54:52.087936+00:00",
  "prediction": 1,
  "probability": 0.6004965668449761,
  "model_version": "1",
  "latency_ms": 88.33
}
```

### Prediction Distribution

The system tracks the proportion of:

```text
prediction = 0
prediction = 1
```

A reference distribution generated from validation data is used for comparison.

Current reference distribution:

```text
0 → 0.484588804422944
1 → 0.515411195577056
```

### Drift Detection

Population Stability Index (PSI) is used to compare the current prediction distribution with the reference distribution.

Configured thresholds:

```text
PSI < 0.10
→ normal

0.10 ≤ PSI < 0.20
→ warning

PSI ≥ 0.20
→ alert
```

Drift is evaluated only after at least:

```text
30 predictions
```

This prevents very small samples from producing misleading drift signals.

---

## Logging

Application logging uses Python's `logging` library.

Logs are written to:

* console
* rotating log files

The logging format contains:

```text
timestamp
log level
module
message
```

Prediction requests include operational information such as:

* prediction
* probability
* model version
* latency

Errors are logged separately with appropriate log levels.

---

## Testing

The project contains unit and integration tests using Pytest.

The current test suite contains:

```text
31 tests
```

Latest local result:

```text
31 passed
```

### API Tests

* health endpoint
* model information
* single prediction
* batch prediction
* missing feature rejection
* unexpected feature rejection
* invalid value rejection

### Feature Tests

* feature artifact loading
* replacement rules
* clipping
* imputation
* feature order
* output shape

### Predictor Tests

* model loading
* single prediction
* batch prediction
* consistency between single and batch prediction

### Preprocessing Tests

* scaler loading
* preprocessing output shape
* use of the fitted scaler

### Validation Tests

* valid data
* missing features
* unexpected features
* empty data
* non-numeric values
* negative values
* infinite values
* allowed missing values

### Monitoring Tests

* request metrics
* prediction logging
* prediction distribution
* PSI calculation
* drift status

Run the complete test suite with:

```powershell
python -m pytest -vv --timeout=30
```

---

## Code Quality

The project uses:

* **Black** for formatting
* **Ruff** for linting and static analysis
* **Pytest** for testing

Current local validation:

```text
Black
17 files would be left unchanged.

Ruff
All checks passed!

Pytest
31 passed.
```

Run the checks with:

```powershell
python -m black --check app src tests
python -m ruff check app src tests --ignore EXE002
python -m pytest -vv --timeout=30
```

---

## Docker and Docker Compose

Docker is used to provide reproducible development and production environments.

### Development Image

`Dockerfile` provides the Python/Jupyter environment used for development and notebook execution.

### Production API Image

`Dockerfile.api` builds a dedicated API image containing only the components required for production inference.

The production image includes:

```text
app/
src/
config/
production inference artifacts
monitoring reference data
runtime dependencies
```

The complete training environment and notebook datasets are not required by the API container.

### MLflow Image

`Dockerfile.mlflow` provides the MLflow server environment.

### Docker Compose

The project uses Docker Compose to coordinate:

```text
postgresql
python
mlflow
api
```

PostgreSQL provides the local database and MLflow backend storage.

The Python service provides the notebook/development environment.

MLflow provides experiment tracking, model registry, and artifact storage.

The API provides production inference and monitoring.

---

## CI/CD

GitHub Actions automates project validation and container delivery.

The workflow runs on:

```text
push
pull_request
```

The pipeline performs:

```text
Checkout
    ↓
Python setup
    ↓
Dependency installation
    ↓
Ruff
    ↓
Black
    ↓
Pytest
    ↓
Docker build
    ↓
Docker image push
```

The production image is pushed to GitHub Container Registry when the workflow is triggered by a push to the main branch and the preceding validation stages succeed.

---

## Reproducibility

Reproducibility is maintained across the complete ML lifecycle.

### Git

Tracks:

* source code
* configuration
* tests
* Docker configuration
* CI/CD
* documentation

### DVC

Tracks:

* datasets
* feature engineering artifacts
* model artifacts

### Saved Preprocessing

The exact fitted preprocessing objects are saved and reused during inference.

### MLflow

The production model is identified through the MLflow Model Registry.

### Configuration

Operational parameters are centralized in:

```text
config/config.yaml
```

### Automated Testing

The production modules are covered by automated tests.

---

## Notebook-to-Production Consistency

A key design goal is maintaining consistency between model development and production inference.

The production pipeline reuses the fitted artifacts produced during model development:

```text
feature names
imputation values
replacement rules
scaler
```

The transformation sequence is therefore consistent:

```text
Raw features
     ↓
Feature rules
     ↓
Imputation
     ↓
Feature ordering
     ↓
Scaling
     ↓
Prediction
```

The API does not refit these transformations on new data.

This avoids training/inference mismatch and prevents preprocessing leakage.

---

## Data Quality and Failure Handling

The project uses multiple validation layers.

### API Schema Validation

FastAPI/Pydantic validates incoming request structure and types.

### Feature Validation

The internal validation module checks expected features and acceptable values.

### Dataset Validation

Great Expectations validates dataset-level data quality.

### Feature Preparation

Known preprocessing rules handle expected cases such as:

* missing values
* replacement rules
* clipping
* feature ordering

Invalid input is rejected instead of being silently passed to the model.

---

## Artifacts

### EDA

```text
artifacts/eda/
```

Contains saved EDA outputs and findings.

### Feature Engineering

```text
artifacts/feature_engineering/
```

Contains feature datasets, targets, preprocessing artifacts, and feature definitions.

### Models

```text
artifacts/models/
```

Contains locally stored model artifacts used during development and testing.

### Production

```text
artifacts/production/
```

Contains only the inference artifacts required by the production API.

### Monitoring

```text
artifacts/monitoring/
```

Contains the reference prediction distribution used for drift monitoring.

---

## Example Production Prediction

A real test record was passed through the production inference pipeline.

The resulting response was:

```json
{
  "prediction": 1,
  "probability": 0.6004965668449761,
  "model_version": "1"
}
```

This verifies the complete inference path:

```text
API request
    ↓
Validation
    ↓
Feature preparation
    ↓
Preprocessing
    ↓
MLflow model
    ↓
Prediction
    ↓
Probability
    ↓
Model version
```

---

## Known Limitations

### In-Memory Monitoring Counters

The request and prediction counters are maintained in memory.

After restarting the API, the following counters start again from zero:

```text
request_count
error_count
prediction distribution
```

The prediction JSONL log is persistent, but the current monitoring implementation does not reconstruct the in-memory counters from historical logs when the API starts.

### FastAPI Startup Event

The current implementation uses FastAPI's startup event mechanism. The installed FastAPI version reports that `on_event` is deprecated in favor of lifespan handlers.

This currently produces a deprecation warning but does not prevent the application from running.

### Production Model Scope

The production API currently exposes the classification model.

The `delay_days` regression model is included as an additional modeling result but is not currently exposed through a production endpoint.

---

## Running the Project

### Start the Environment

```powershell
docker compose up -d --build
```

Check the services:

```powershell
docker compose ps
```

### API

The API is available at:

```text
http://localhost:8000
```

Interactive API documentation:

```text
http://localhost:8000/docs
```

Health:

```text
http://localhost:8000/health
```

Model information:

```text
http://localhost:8000/model
```

Monitoring:

```text
http://localhost:8000/metrics
```

### MLflow

MLflow is available at:

```text
http://localhost:5000
```

### Jupyter

The development notebook environment is available at:

```text
http://localhost:8888
```

---

## Development Validation

Before committing changes, run:

```powershell
python -m black --check app src tests
python -m ruff check app src tests --ignore EXE002
python -m pytest -vv --timeout=30
```

A successful local validation currently produces:

```text
Black: 17 files unchanged
Ruff: All checks passed
Pytest: 31 passed
```

---

## Project Summary

The complete workflow can be summarized as:

```text
Olist Relational Data
        │
        ▼
Table Understanding
        │
        ▼
Aggregation and Joining
        │
        ▼
Order-Level ML Dataset
        │
        ▼
Target Creation
        │
        ▼
Train / Validation / Test
        │
        ▼
EDA
        │
        ▼
Feature Engineering
        │
        ▼
Classification + Regression
        │
        ▼
Saved ML Artifacts
        │
        ▼
Data Validation
        │
        ▼
DVC Versioning
        │
        ▼
MLflow Tracking and Registry
        │
        ▼
FastAPI Inference
        │
        ▼
Docker
        │
        ▼
Automated Testing
        │
        ▼
CI/CD
        │
        ▼
Production Monitoring
```

The result is a complete ML system that connects data preparation, exploratory analysis, feature engineering, model development, artifact management, validation, model registration, API serving, testing, deployment, and monitoring within one reproducible workflow.

---

## Training

This project was developed as part of the **Advanced MLOps Training program by Qafza**.

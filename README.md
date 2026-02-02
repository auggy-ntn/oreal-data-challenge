# XHEC L'Oréal Data Science Challenge: Marketing Mix Modeling

<!-- Build & CI Status -->
![CI](https://github.com/auggy-ntn/oreal-data-challenge/actions/workflows/ci.yaml/badge.svg?event=push)

<!-- Code Quality & Tools -->
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)

<!-- Environment & Package Management -->
![Python Version](https://img.shields.io/badge/python-3.13+-blue.svg)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)

## Table of Contents

- [XHEC L'Oréal Data Science Challenge: Marketing Mix Modeling](#xhec-loréal-data-science-challenge-marketing-mix-modeling)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Quick Start](#quick-start)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
  - [Project Structure](#project-structure)
  - [Commands Reference](#commands-reference)
    - [Data \& Pipeline](#data--pipeline)
    - [Code Quality](#code-quality)
    - [Dashboard](#dashboard)
  - [Configuration](#configuration)
    - [Environment Variables](#environment-variables)
  - [Additional Documentation](#additional-documentation)
  - [Authors](#authors)

---

## Introduction

This repository contains the codebase for the **XHEC L'Oréal Data Science Challenge** focused on marketing mix modeling using machine learning techniques.

---

## Quick Start

### Prerequisites

- Python 3.13+
- [uv](https://github.com/astral-sh/uv) package manager
- Git

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/auggy-ntn/oreal-data-challenge
cd oreal-data-challenge

# 2. Install dependencies
uv sync

# 3. Set up pre-commit hooks
uv run pre-commit install

# 4. Configure environment variables (see docs/SETUP.md for details)
cp .env.example .env
# Edit .env with your credentials

# 5. Configure DVC remote
source .env
dvc remote modify --local ovh-storage access_key_id $OVH_ACCESS_KEY_ID
dvc remote modify --local ovh-storage secret_access_key $OVH_SECRET_ACCESS_KEY

# 6. Pull data from remote storage
uv run dvc pull

# 7. Launch the dashboard
uv run streamlit run src/streamlit/streamlit_app.py
```

For detailed setup instructions, see [docs/SETUP.md](docs/SETUP.md).

---

## Project Structure

```
TODO
```

---

## Commands Reference

### Data & Pipeline

See the `params.yaml` file for configurable parameters you can adjust, and which will affect pipeline behavior.

```bash
# Pull data from remote storage
uv run dvc pull

# Run full pipeline
uv run dvc repro

# Run individual stages
uv run dvc repro <stage_name>

# View pipeline DAG
uv run dvc dag

# Check pipeline status
uv run dvc status
```

### Code Quality

```bash
# Lint code
uv run ruff check .

# Format code
uv run ruff format .

# Run pre-commit hooks manually
uv run pre-commit run --all-files
```

### Dashboard

```bash
# Launch Streamlit dashboard
uv run streamlit run src/streamlit/streamlit_app.py
```

---

## Configuration

<!-- ### params.yaml

All tunable parameters are centralized in `params.yaml`. -->



### Environment Variables

Copy `.env.example` to `.env` and configure:

```bash
# DVC Remote Storage (OVH Object Storage)
OVH_ACCESS_KEY_ID=your_access_key
OVH_SECRET_ACCESS_KEY=your_secret_key

# MLflow Tracking (Databricks)
DATABRICKS_HOST=https://your-workspace.cloud.databricks.com
DATABRICKS_TOKEN=your_token
MLFLOW_EXPERIMENT_ID=your_experiment_id
```

---

## Additional Documentation

| Document | Description |
|----------|-------------|
| [docs/SETUP.md](docs/SETUP.md) | Complete developer setup guide |
| [docs/DVC_WORKFLOW.md](docs/DVC_WORKFLOW.md) | Data versioning workflow and best practices |
| [docs/PROJECT_OWNER_CHECKLIST.md](docs/PROJECT_OWNER_CHECKLIST.md) | Setup guide for project owners |

---

## Authors

**XHEC Data Science Challenge Team**

- William BELAIDI
- Grégoire BIDAULT
- Aymeric DE LONGEVIALLE
- Paul FILISETTI
- Augustin NATON
- Louis PERETIE

---

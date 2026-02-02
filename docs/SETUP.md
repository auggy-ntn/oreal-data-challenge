# Team Setup Guide

Complete setup instructions for new team members to get started with the project.

## Prerequisites

- Python 3.13+
- Git
- [uv](https://docs.astral.sh/uv/) package manager

## Step-by-Step Setup

### 1. Clone the Repository

```bash
git clone https://github.com/auggy-ntn/oreal-data-challenge.git
cd oreal-data-challenge
```

### 2. Install Dependencies

```bash
# Install all project dependencies (including dev tools)
uv sync
```

This installs:
- Core dependencies (pandas, scikit-learn, streamlit, dvc, etc.)
- Development tools (ruff, pre-commit, nbstripout)

### 3. Set Up Pre-commit Hooks

```bash
uv run pre-commit install
```

This ensures code quality checks run automatically before each commit.

### 4. Configure Environment Variables

Create your `.env` file from the template:

```bash
cp .env.example .env
```

Now edit `.env` and add the DVC credentials:

#### OVH Object Storage (DVC - Data Storage)

**You'll receive these from the project owner:**

```bash
OVH_ACCESS_KEY_ID=<provided_by_owner>
OVH_SECRET_ACCESS_KEY=<provided_by_owner>
```

> **Note:** The owner will share these credentials securely. Do NOT create your own OVH account.

### 5. Configure MLflow (Experiment Tracking)

We use MLflow with Databricks as the backend for tracking and sharing experiments.

The shared Databricks settings are already in `.env.example`. You only need to add the access token:

**You need to create a free Databricks account and get invited to the workspace:**

1. **Create a free Databricks account:**
   - Go to: https://www.databricks.com/try-databricks
   - Sign up for free
   - Choose "Get started with Community Edition" or free trial

2. **Wait for workspace invitation:**
   - The project owner will invite you to their workspace
   - Check your email for the invitation
   - Accept the invitation

3. **Get your workspace URL:**
   - After accepting, you'll be redirected to the workspace

4. **Generate your personal access token:**
   - In Databricks, click your username (top right) → **Settings**
   - Go to **Developer** → **Access Tokens**
   - Click **Generate New Token**
   - Give it a name (e.g., "data-challenge-local")
   - Set expiration (10 days recommended - short project)
   - Click **Generate**
   - **⚠️ COPY THE TOKEN NOW** (shown only once!)

5. **Add to `.env`:**
   ```bash
   DATABRICKS_HOST=https://dbc-XXXXXX-XXXX.cloud.databricks.com
   DATABRICKS_TOKEN=dapi...your_token_here...
   MLFLOW_EXPERIMENT_ID=<provided_by_owner>
   ```

> 💡 **Tip:** The experiment ID will be provided by the project owner. It's the same for everyone on the team.

Replace the placeholder value in your `.env` file with the token you received.

### 6. Configure DVC Remote

DVC needs your OVH credentials configured locally (won't be committed to git):

```bash
# Load environment variables
source .env

# Configure DVC remote with your credentials
dvc remote modify --local ovh-storage access_key_id $OVH_ACCESS_KEY_ID
dvc remote modify --local ovh-storage secret_access_key $OVH_SECRET_ACCESS_KEY
```

### 7. Pull Data from DVC

Download all project data from OVH Object Storage:

```bash
dvc pull
```

This downloads:
- Raw data in `data/bronze/`
- Any processed datasets that have been shared

### 8. Verify Setup

Test that everything works:

#### Test MLflow Connection:
```bash
source .env
uv run python -c "import mlflow; mlflow.set_tracking_uri('databricks'); print('MLflow connection successful!')"
```

#### Test DVC:
```bash
dvc status
# Should show: "Data and pipelines are up to date."
```

#### Run the Pipeline:
```bash
dvc repro
```

#### Launch the Dashboard:
```bash
uv run streamlit run src/streamlit/streamlit_app.py
```

## Troubleshooting

### MLflow Issues

**"DATABRICKS_HOST environment variable not set"**
- Make sure you copied `.env.example` to `.env`
- Run `source .env` before running MLflow commands

**"Invalid token" or "Authentication failed"**
- Verify you replaced the `DATABRICKS_TOKEN` placeholder with the token from the project owner
- Tokens may expire - ask for a new one if needed

### DVC Issues

**"Unable to locate credentials"**
- Make sure your `.env` file has `OVH_ACCESS_KEY_ID` and `OVH_SECRET_ACCESS_KEY`
- Verify you've configured the DVC remote:
  ```bash
  source .env
  dvc remote modify --local ovh-storage access_key_id $OVH_ACCESS_KEY_ID
  dvc remote modify --local ovh-storage secret_access_key $OVH_SECRET_ACCESS_KEY
  ```

**"dvc pull" fails**
- Verify credentials in `.env` are correct
- Make sure you've configured the DVC remote (see above)
- Ask the project owner to confirm you're using the right credentials

### General Issues

**"Command not found: uv"**
- Install uv: `curl -LsSf https://astral.sh/uv/install.sh | sh`

**"Python version mismatch"**
- This project requires Python 3.13+
- Check your version: `python --version`

**Streamlit won't start**
- Make sure you're in the project root directory
- Try: `uv run streamlit run src/streamlit/streamlit_app.py`

## Next Steps

- Read [DVC_WORKFLOW.md](DVC_WORKFLOW.md) to learn how to work with data
- Check the [README.md](../README.md) for project structure and commands
- Explore the notebooks in `notebooks/` for data analysis examples

## Getting Help

- Check existing documentation in `docs/`
- Contact the project owner for credential issues

# Project Owner Checklist

**You are receiving ownership of this Eleven Data Challenge project. This guide will help you set up the infrastructure so your team can collaborate effectively.**

---

## Overview

As the project owner, you need to:
1. Set up data versioning infrastructure (DVC remote)
2. Set up experiment tracking infrastructure (MLflow)
3. Provide credentials to team members
4. Configure repository settings

---

## Step 1: Set Up Data Versioning (DVC Remote)

### Why You Need This
- Team members need to pull/push datasets without Git (files are too large)
- Ensures everyone works with the same data versions
- Tracks data lineage and pipeline reproducibility

### Choose Your Storage Backend

#### Option A: OVH Object Storage (Original Setup)

**Setup Steps:**
1. Create OVH Public Cloud account: https://www.ovhcloud.com/
2. Create Object Storage container:
   - Go to Public Cloud → Object Storage → Create container
   - Container name: `eleven-data-challenge` (or your choice)
   - Region: Choose closest to your team
   - Container type: Private
3. Create S3 credentials:
   - Go to Public Cloud → Object Storage → S3 Users
   - Create a new S3 user
   - **SAVE THESE CREDENTIALS** (shown only once):
     - `Access Key` → This is your `OVH_ACCESS_KEY_ID`
     - `Secret Key` → This is your `OVH_SECRET_ACCESS_KEY`

4. Configure DVC remote in the project:
```bash
cd /path/to/eleven-data-challenge

# Add OVH S3 remote
dvc remote add -d ovh-storage s3://eleven-data-challenge
dvc remote modify ovh-storage endpointurl https://s3.gra.io.cloud.ovh.net
```

5. Configure your local credentials (do NOT commit):
```bash
# Add to .env
echo "OVH_ACCESS_KEY_ID=your_access_key_here" >> .env
echo "OVH_SECRET_ACCESS_KEY=your_secret_key_here" >> .env

# Configure DVC
source .env
dvc remote modify --local ovh-storage access_key_id $OVH_ACCESS_KEY_ID
dvc remote modify --local ovh-storage secret_access_key $OVH_SECRET_ACCESS_KEY
```

6. Push existing data to remote:
```bash
dvc push
```

7. **Share with team**: Give them the credentials (use secure method: 1Password, LastPass, encrypted message)

#### Option B: AWS S3 (Alternative)

**Setup Steps:**
1. Create AWS account: https://aws.amazon.com/
2. Create S3 bucket:
   - S3 Console → Create bucket
   - Bucket name: `eleven-data-challenge`
   - Region: Choose closest to your team
   - Block all public access: Yes
3. Create IAM user for DVC access:
   - IAM Console → Users → Add user
   - User name: `dvc-user`
   - Access type: Programmatic access
   - Permissions: Attach `AmazonS3FullAccess` (or custom bucket-only policy)
   - **SAVE CREDENTIALS**: Access key ID and Secret access key
4. Configure DVC remote:
```bash
dvc remote add -d s3remote s3://eleven-data-challenge/dvc-storage
dvc remote modify s3remote region eu-west-1  # your region
```


---

## Step 2: Set Up Experiment Tracking (MLflow)

### Why You Need This
- Track model training experiments (parameters, metrics, artifacts)
- Compare model performance across runs
- Share experiment results with the team
- Reproduce and deploy models

### Databricks MLflow (Recommended)

We recommend Databricks as the MLflow backend because it provides:
- Free Community Edition tier
- Managed MLflow tracking server
- Shared workspace for team collaboration
- Built-in experiment UI

> **Note:** Other MLflow backends exist (self-hosted server, Azure ML, AWS SageMaker, etc.) but are not covered here. See [MLflow documentation](https://mlflow.org/docs/latest/tracking.html) for alternatives.

**Setup Steps:**

1. **Create a Databricks account:**
   - Go to: https://www.databricks.com/try-databricks
   - Sign up for free (Community Edition or free trial)

2. **Create a workspace:**
   - After signup, you'll be assigned a workspace
   - Note your workspace URL: `https://dbc-XXXXXX-XXXX.cloud.databricks.com`

3. **Create an MLflow experiment:**
   - In Databricks, go to **Machine Learning** → **Experiments**
   - Click **Create Experiment**
   - Name it: `eleven-data-challenge` (or your choice)
   - Note the **Experiment ID** (visible in the URL or experiment details)

4. **Generate your personal access token:**
   - Click your username (top right) → **Settings**
   - Go to **Developer** → **Access Tokens**
   - Click **Generate New Token**
   - Name: `eleven-data-challenge-owner`
   - Expiration: Set appropriately (or no expiration for long projects)
   - Click **Generate** and **SAVE THE TOKEN** (shown only once!)

5. **Configure your local environment:**
```bash
# Add to .env
echo "DATABRICKS_HOST=https://dbc-XXXXXX-XXXX.cloud.databricks.com" >> .env
echo "DATABRICKS_TOKEN=dapi_your_token_here" >> .env
echo "MLFLOW_EXPERIMENT_ID=your_experiment_id" >> .env
```

6. **Test the connection:**
```bash
source .env
uv run python -c "import mlflow; mlflow.set_tracking_uri('databricks'); print('MLflow connection successful!')"
```

7. **Invite team members to the workspace:**
   - In Databricks, go to **Settings** → **Identity and access**
   - Click **Users** → **Add User**
   - Enter each team member's email
   - They will receive an invitation to join the workspace

8. **Share with team:**
   - Workspace URL (`DATABRICKS_HOST`)
   - Experiment ID (`MLFLOW_EXPERIMENT_ID`)
   - Each team member creates their own personal access token

---

## Step 3: Share Credentials with Team

### What to Share

**For DVC (Data Versioning):**
- Storage backend type (OVH, S3, etc.)
- Bucket/container name
- Access credentials (access key + secret key)
- Endpoint URL (if OVH or other S3-compatible)

**For MLflow (Experiment Tracking):**
- Databricks workspace URL (`DATABRICKS_HOST`)
- MLflow experiment ID (`MLFLOW_EXPERIMENT_ID`)
- Instructions for team members to create their own access tokens

### Credentials Document Template

Create a document (store in password manager) with this info:

```
ELEVEN Data Challenge - Team Credentials
=====================================

PROJECT REPOSITORY
URL: https://github.com/your-org/eleven-data-challenge
Main Branch: main

DVC REMOTE STORAGE (OVH Object Storage)
Container: eleven-data-challenge
Endpoint: https://s3.gra.io.cloud.ovh.net
Access Key ID: xxxxxxxxxxxxxxxxxxxxx
Secret Access Key: xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

MLFLOW TRACKING (Databricks)
Workspace URL: https://dbc-XXXXXX-XXXX.cloud.databricks.com
Experiment ID: 123456789
Note: Each team member must create their own access token in Databricks

SETUP INSTRUCTIONS
See: docs/SETUP.md
Owner Checklist: docs/PROJECT_OWNER_CHECKLIST.md
```

---

## Step 4: Update Project Documentation

### Update .env.example

Update `.env.example` with your actual endpoint (DO NOT include secrets):

```bash
# .env.example
# DVC Remote Storage (OVH Object Storage)
OVH_ACCESS_KEY_ID=your_access_key_from_owner
OVH_SECRET_ACCESS_KEY=your_secret_key_from_owner
```

### Update DVC Configuration

If you changed DVC remote setup, update `.dvc/config`:

```bash
# Commit this to Git
git add .dvc/config
git commit -m "Update DVC remote configuration"
git push
```

---

## Step 5: Onboard Team Members

1. **Provide Access:**
   - GitHub repository: Add as collaborator
   - DVC credentials: Share via secure method
   - Databricks workspace: Invite via email (they create their own token)

2. **Direct them to setup docs:**
   - [SETUP.md](SETUP.md) - Complete setup guide
   - [DVC_WORKFLOW.md](DVC_WORKFLOW.md) - Data versioning workflow

---

## Verification Checklist

Before inviting team members, verify:

### DVC Remote
- [ ] Bucket/storage created and accessible
- [ ] Credentials generated and saved securely
- [ ] Successfully ran `dvc push` from your machine
- [ ] Tested `dvc pull` on a fresh clone (or different directory)

### MLflow Tracking
- [ ] Databricks workspace created
- [ ] MLflow experiment created
- [ ] Personal access token generated
- [ ] Successfully logged a test run from your machine
- [ ] Team members invited to workspace

### Documentation
- [ ] `.env.example` updated with correct endpoint info
- [ ] `docs/SETUP.md` reviewed and accurate
- [ ] `README.md` has correct quick start instructions
- [ ] Credentials document prepared (stored securely)

### Repository
- [ ] All code committed and pushed
- [ ] `.gitignore` includes `.env`, `.dvc/config.local`
- [ ] CI/CD pipeline passing (GitHub Actions)
- [ ] Repository permissions set correctly

### Team Access
- [ ] GitHub repository access granted
- [ ] Credentials shared securely
- [ ] Onboarding instructions sent

---

## Support Resources

### For You (Project Owner)
- DVC Documentation: https://dvc.org/doc
- MLflow Documentation: https://mlflow.org/docs/latest/index.html
- Databricks Documentation: https://docs.databricks.com/
- OVH Object Storage Docs: https://docs.ovh.com/gb/en/storage/object-storage/
- AWS S3 Docs: https://docs.aws.amazon.com/s3/

### For Team Members
- `docs/SETUP.md` - Complete setup guide
- `docs/DVC_WORKFLOW.md` - Data versioning workflow
- `README.md` - Project overview and quick start

---

## You're Done!

Once you've completed this checklist:
1. Team can clone the repository
2. Team can `dvc pull` to get data
3. Team can run the pipeline (`dvc repro`)
4. Team can train models and track experiments in MLflow
5. Team can launch the dashboard
6. Team can collaborate on development

---

**Questions?** Create an issue in the repository or reach out to the original team members listed in README.md.

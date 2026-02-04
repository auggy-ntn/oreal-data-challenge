"""Path configurations for the project.

This module centralizes all path definitions to ensure consistency across the project.
"""

from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
BRONZE_DIR = DATA_DIR / "bronze"
SILVER_DIR = DATA_DIR / "silver"

# Bronze excel data file
BRONZE_DATA_FILE = BRONZE_DIR / "Dataset UK L'Oreal Paris Haircare - HEC Training.xlsx"
# Sheet names
KPI_SHEET = "KPI to model"
COMMERCIAL_SHEET = "Commercial Variables"
KPI_AND_COM_DEF_SHEET = "KPI & Com Variables Definition"
A_AND_P_SHEET = "A&P Variables"
A_AND_P_DEF_SHEET = "A&P Variables Definition"

# Silver processed data files
SILVER_TARGET_FILE = SILVER_DIR / "target.csv"
SILVER_A_AND_P_FEATURES_FILE = SILVER_DIR / "a_and_p_features.csv"
SILVER_COMMERCIAL_FEATURES_FILE = SILVER_DIR / "commercial_features.csv"

# Gold processed data files (PyMC-Marketing ready)
GOLD_DIR = DATA_DIR / "gold"
GOLD_ONLINE_DATA_FILE = GOLD_DIR / "online_mmm_data.csv"
GOLD_OFFLINE_DATA_FILE = GOLD_DIR / "offline_mmm_data.csv"

# Models directory
MODELS_DIR = PROJECT_ROOT / "models"

# Configuration directory
CONFIG_DIR = PROJECT_ROOT / "config"

# Parameters file
PARAMS_FILE = "params.yaml"

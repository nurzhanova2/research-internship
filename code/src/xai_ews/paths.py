"""Project paths. All code resolves data locations through this module."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
MANIFEST_DIR = DATA_DIR / "manifests"
DICTIONARY_DIR = DATA_DIR / "dictionaries"
REFERENCE_DIR = DATA_DIR / "reference"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
EXTERNAL_DIR = DATA_DIR / "external"

REPORTS_DIR = PROJECT_ROOT / "reports"
CONFIGS_DIR = PROJECT_ROOT / "configs"

# The field dictionaries are identical across quarters; 2008Q1 also has the leakage register.
DEFAULT_DICTIONARY = DICTIONARY_DIR / "fannie_2008q1_field_dictionary.csv"
LEAKAGE_REGISTER = DICTIONARY_DIR / "fannie_2008q1_leakage_register_v01.csv"
PANEL_DIR = PROCESSED_DIR / "panel"
# Train vintages at 100 %, used only for train observations
INTERIM_TRAIN_FULL_DIR = INTERIM_DIR / "train_full"
TRAIN_FULL_DIR = PROCESSED_DIR / "train_full"
LEAKAGE_REGISTER_V02 = DICTIONARY_DIR / "fannie_leakage_register_v02.csv"
DATA_AUDIT_DIR = REPORTS_DIR / "data_audit"
# Stage 2: feature selection
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
FEATURE_SELECTION_DIR = REPORTS_DIR / "feature_selection"
STAGE2_INTERIM_DIR = INTERIM_DIR / "stage2"   # cached model outputs of the stage-2 notebooks
# Stage 3: models and hypotheses H2-H4
MODELING_DIR = REPORTS_DIR / "modeling"   # models B0 / M1 / M2, hypotheses H2-H4, test protocol
MODELS_DIR = PROCESSED_DIR / "models"           # fitted models, Optuna studies
PREDICTIONS_DIR = PROCESSED_DIR / "predictions" # OOF / fold-average predictions (stage 4 input)
DISASTER_FLAGS = PROCESSED_DIR / "disaster_flags.parquet"      # natural-disaster flags per (zip3, m)
DEFAULT_HISTORY = PROCESSED_DIR / "default_history.parquet"    # loan -> first snapshot with 90+ DPD in t-11..t
VALIDATION_HALVES = PROCESSED_DIR / "validation_halves.parquet"  # loan -> half A (tuning) / B (calibration)
# Optuna journal; on Colab it lives on the local disk (SQLite on Google Drive does not lock reliably)
OPTUNA_DB = Path(os.environ.get("XAI_OPTUNA_DB", MODELS_DIR / "optuna.db"))
# Stage 4: explanations (SHAP, LIME, reason codes)
XAI_DIR = REPORTS_DIR / "xai"

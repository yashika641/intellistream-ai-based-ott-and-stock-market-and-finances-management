# utils/constants.py

from pathlib import Path

# === Project Directories === #
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"
DASHBOARD_DIR = BASE_DIR / "dashboard"
CONFIG_DIR = BASE_DIR / "utils"
DOCS_DIR = BASE_DIR / "docs"

# === Config Files === #
CONFIG_FILE_PATH = CONFIG_DIR / "config.yaml"

# === API Endpoints (Backend) === #
API_PREFIX = "/api/v1"
HEALTH_CHECK_ROUTE = f"{API_PREFIX}/health"
NLP_ROUTE = f"{API_PREFIX}/nlp"
STOCK_FORECAST_ROUTE = f"{API_PREFIX}/forecast"

# === Model Names / Filenames === #
NLP_MODEL_NAME = "bert_sentiment_model.pkl"
FORECAST_MODEL_NAME = "stock_lstm_model.h5"

# === Log File Path === #
LOG_FILE = BASE_DIR / "logs" / "app.log"

# === Others === #
DEFAULT_ENCODING = "utf-8"

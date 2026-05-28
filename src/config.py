import os
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent

DATA_DIR = BASE_DIR / "data"
SYNTHETIC_DATA_DIR = DATA_DIR / "synthetic"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

SRC_DIR = BASE_DIR / "src"
DATA_GEN_DIR = SRC_DIR / "data_generation"
PREPROCESSING_DIR = SRC_DIR / "preprocessing"
MODELS_DIR = SRC_DIR / "models"
GRC_DIR = SRC_DIR / "grc"
UTILS_DIR = SRC_DIR / "utils"

OUTPUT_DIR = BASE_DIR / "outputs"
LOGS_DIR = OUTPUT_DIR / "logs"
MODELS_OUTPUT_DIR = OUTPUT_DIR / "models"
RESULTS_DIR = OUTPUT_DIR / "results"

CONFIG_DIR = BASE_DIR / "config"

for directory in [SYNTHETIC_DATA_DIR, PROCESSED_DATA_DIR, LOGS_DIR, MODELS_OUTPUT_DIR, RESULTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

NUM_RECORDS = 1_000_000
XML_OUTPUT_FILE = SYNTHETIC_DATA_DIR / "synthetic_aadhar.xml"

CITIES = [
    "Bangalore", "Delhi", "Mumbai", "Hyderabad", "Pune",
    "Chennai", "Kolkata", "Ahmedabad", "Jaipur", "Lucknow",
    "Chandigarh", "Indore", "Nagpur", "Surat", "Vadodara"
]

INCOME_BRACKETS = ["<2L", "2-5L", "5-10L", "10-25L", ">25L"]

RISK_CATEGORIES = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]

CHUNK_SIZE = 100_000

NUM_WORKERS = 1

CATEGORICAL_FEATURES = [
    "city",
    "income_bracket",
    "document_status",
    "risk_category"
]

NUMERICAL_FEATURES = [
    "days_since_update",
    "days_to_expiry",
    "num_missing_fields",
    "age"
]

ENCODING_METHOD = "one_hot"

SCALING_METHOD = "standard"

TRAIN_TEST_SPLIT = 0.8

MODEL_TYPE = "classifier"

XGBOOST_PARAMS = {
    "objective": "multi:softmax",
    "num_class": 4,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 1,
    "gamma": 0,
    "random_state": 42,
    "n_estimators": 100,
}

TRAINED_MODEL_FILE = MODELS_OUTPUT_DIR / "recommender_model.pkl"

EVALUATION_METRICS = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "inference_latency",
    "memory_usage",
    "processing_time"
]

RESULTS_FILE = RESULTS_DIR / "metrics.csv"

AUDIT_LOG_FILE = LOGS_DIR / "audit.log"

LOG_RECOMMENDATION_GENERATED = True
LOG_RECOMMENDATION_ACCESSED = True

DATA_RETENTION_DAYS = 30
RETENTION_CHECK_INTERVAL_DAYS = 1

COMPLIANCE_METRICS_FILE = RESULTS_DIR / "compliance_metrics.csv"

ALLOWED_USERS = ["officer", "admin", "analyst"]

INFERENCE_BATCH_SIZE = 1000

PRIORITY_THRESHOLDS = {
    "CRITICAL": 0.75,
    "HIGH": 0.50,
    "MEDIUM": 0.25,
    "LOW": 0.0
}

LOG_LEVEL = "INFO"

VERBOSE = True

SAVE_PREPROCESSED_DATA = True
PREPROCESSED_DATA_FILE = PROCESSED_DATA_DIR / "preprocessed_data.pkl"

RANDOM_SEED = 42
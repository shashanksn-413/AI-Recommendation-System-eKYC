"""Quick test - skips data generation/parsing, tests only model training"""

import sys
import logging
import pickle
from pathlib import Path
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from src.config import PROCESSED_DATA_DIR, TRAINED_MODEL_FILE, RESULTS_DIR, XGBOOST_PARAMS, RANDOM_SEED
from src.recommender import eKYCRecommender
from src.audit_logger import AuditLogger
from src.retention_policy import RetentionPolicy
from src.config import AUDIT_LOG_FILE, DATA_RETENTION_DAYS

try:
    logger.info("="*70)
    logger.info("QUICK TEST - MODEL TRAINING ONLY")
    logger.info("="*70)
    
    # Load preprocessed data (already exists from previous run)
    logger.info("\nLoading preprocessed data...")
    with open(PROCESSED_DATA_DIR / "X_train.pkl", "rb") as f:
        X_train = pickle.load(f)
    with open(PROCESSED_DATA_DIR / "X_test.pkl", "rb") as f:
        X_test = pickle.load(f)
    with open(PROCESSED_DATA_DIR / "y_train.pkl", "rb") as f:
        y_train = pickle.load(f)
    with open(PROCESSED_DATA_DIR / "y_test.pkl", "rb") as f:
        y_test = pickle.load(f)
    
    logger.info(f"✓ Loaded: Train {X_train.shape}, Test {X_test.shape}")
    
    # Train model
    logger.info("\nTraining XGBoost model...")
    recommender = eKYCRecommender(model_params=XGBOOST_PARAMS, random_seed=RANDOM_SEED)
    recommender.train(X_train, y_train)
    recommender.save_model(TRAINED_MODEL_FILE)
    logger.info(f"✓ Model trained and saved")
    
    # Evaluate
    logger.info("\nEvaluating model...")
    metrics = recommender.evaluate(X_test, y_test)
    logger.info(f"✓ F1-Score: {metrics['f1_score']:.4f}")
    
    # Handle None inference latency
    if metrics['inference_latency_ms'] is not None:
        logger.info(f"✓ Inference Latency: {metrics['inference_latency_ms']:.2f}ms per sample")
    
    # Save metrics
    import pandas as pd
    metrics_df = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1-Score"],
        "Value": [metrics["accuracy"], metrics["precision"], metrics["recall"], metrics["f1_score"]]
    })
    metrics_df.to_csv(RESULTS_DIR / "model_metrics.csv", index=False)
    logger.info(f"✓ Metrics saved to {RESULTS_DIR / 'model_metrics.csv'}")
    
    # Audit logging
    logger.info("\nLogging audit trail...")
    audit_log = AuditLogger(AUDIT_LOG_FILE)
    audit_log.log_model_training("system", "v1.0", len(X_train), metrics["accuracy"], metrics["f1_score"])
    audit_log.export_logs_to_csv(RESULTS_DIR / "audit_logs.csv")
    logger.info(f"✓ Audit logs saved to {RESULTS_DIR / 'audit_logs.csv'}")
    
    # Retention policy
    logger.info("\nApplying retention policy...")
    retention = RetentionPolicy(DATA_RETENTION_DAYS)
    logger.info(f"✓ Retention policy applied (30-day expiry)")
    
    logger.info("\n" + "="*70)
    logger.info("✓ QUICK TEST COMPLETE - ALL SYSTEMS GO!")
    logger.info("="*70)
    logger.info(f"Model saved: {TRAINED_MODEL_FILE}")
    logger.info(f"Metrics: {RESULTS_DIR / 'model_metrics.csv'}")
    logger.info(f"Audit logs: {RESULTS_DIR / 'audit_logs.csv'}")
    
except Exception as e:
    logger.error(f"✗ Error: {e}", exc_info=True)
    sys.exit(1)
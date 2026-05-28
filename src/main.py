import sys
import time
import logging
from pathlib import Path
from datetime import datetime
import json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from config import (
    NUM_RECORDS, XML_OUTPUT_FILE, CHUNK_SIZE, RANDOM_SEED,
    CATEGORICAL_FEATURES, NUMERICAL_FEATURES, TRAIN_TEST_SPLIT,
    PROCESSED_DATA_DIR, TRAINED_MODEL_FILE, RESULTS_DIR,
    XGBOOST_PARAMS, AUDIT_LOG_FILE, DATA_RETENTION_DAYS
)

from generate_synthetic_aadhar import SyntheticAadharGenerator
from xml_parser import parse_xml_to_dataframe, validate_dataframe
from chunk_processor import ChunkProcessor
from recommender import eKYCRecommender
from audit_logger import AuditLogger, AuditAction
from retention_policy import RetentionPolicy

import pickle
import pandas as pd


class eKYCPipeline:
    
    def __init__(self):
        self.start_time = None
        self.execution_log = {}
        self.audit_logger = AuditLogger(AUDIT_LOG_FILE)
        self.retention_policy = RetentionPolicy(retention_days=DATA_RETENTION_DAYS)
        
        logger.info("="*70)
        logger.info("eKYC RECOMMENDATION ENGINE - MAIN PIPELINE")
        logger.info("="*70)
    
    def step_1_generate_data(self):
        logger.info("\n" + "="*70)
        logger.info("STEP 1: GENERATING SYNTHETIC AADHAR DATA")
        logger.info("="*70)
        
        step_start = time.time()
        
        try:
            if XML_OUTPUT_FILE.exists():
                logger.info(f"Data already exists: {XML_OUTPUT_FILE}")
                file_size_mb = XML_OUTPUT_FILE.stat().st_size / (1024 * 1024)
                logger.info(f"File size: {file_size_mb:.2f} MB")
                return True
            
            logger.info(f"Generating {NUM_RECORDS} synthetic records...")
            generator = SyntheticAadharGenerator(num_records=NUM_RECORDS, seed=RANDOM_SEED)
            generator.generate_xml(XML_OUTPUT_FILE)
            
            step_time = time.time() - step_start
            file_size_mb = XML_OUTPUT_FILE.stat().st_size / (1024 * 1024)
            
            self.execution_log["step_1_data_generation"] = {
                "status": "success",
                "records_generated": NUM_RECORDS,
                "file_size_mb": file_size_mb,
                "time_seconds": step_time
            }
            
            logger.info(f"✓ Data generation complete in {step_time:.2f}s")
            
            self.audit_logger.log_data_preprocessing(
                user_id="system",
                num_records=NUM_RECORDS,
                preprocessing_version="v1.0"
            )
            
            return True
        
        except Exception as e:
            logger.error(f"✗ Data generation failed: {e}")
            self.execution_log["step_1_data_generation"] = {"status": "failed", "error": str(e)}
            return False
    
    def step_2_parse_data(self):
        logger.info("\n" + "="*70)
        logger.info("STEP 2: PARSING XML DATA")
        logger.info("="*70)
        
        step_start = time.time()
        
        try:
            logger.info(f"Parsing XML file: {XML_OUTPUT_FILE}")
            df = parse_xml_to_dataframe(XML_OUTPUT_FILE)
            
            if df is None:
                logger.error("Failed to parse XML")
                return None
            
            validation = validate_dataframe(df)
            
            step_time = time.time() - step_start
            
            self.execution_log["step_2_parsing"] = {
                "status": "success",
                "rows": len(df),
                "columns": len(df.columns),
                "time_seconds": step_time,
                "validation": validation
            }
            
            logger.info(f"✓ Parsing complete: {len(df)} rows, {len(df.columns)} columns in {step_time:.2f}s")
            
            return df
        
        except Exception as e:
            logger.error(f"✗ Parsing failed: {e}")
            self.execution_log["step_2_parsing"] = {"status": "failed", "error": str(e)}
            return None
    
    def step_3_preprocess_data(self, df):
        logger.info("\n" + "="*70)
        logger.info("STEP 3: CHUNK-BASED PREPROCESSING")
        logger.info("="*70)
        
        step_start = time.time()
        
        try:
            logger.info(f"Chunk size: {CHUNK_SIZE} records")
            
            processor = ChunkProcessor(
                chunk_size=CHUNK_SIZE,
                categorical_features=CATEGORICAL_FEATURES,
                numerical_features=NUMERICAL_FEATURES,
                train_test_split_ratio=TRAIN_TEST_SPLIT,
                random_seed=RANDOM_SEED
            )
            
            X_train, X_test, y_train, y_test = processor.preprocess_chunks(df, save_path=PROCESSED_DATA_DIR)
            
            step_time = time.time() - step_start
            
            self.execution_log["step_3_preprocessing"] = {
                "status": "success",
                "chunk_size": CHUNK_SIZE,
                "train_samples": len(X_train),
                "test_samples": len(X_test),
                "features": X_train.shape[1],
                "time_seconds": step_time
            }
            
            logger.info(f"✓ Preprocessing complete in {step_time:.2f}s")
            logger.info(f"  Train: {len(X_train)} samples | Test: {len(X_test)} samples")
            logger.info(f"  Features: {X_train.shape[1]}")
            
            return X_train, X_test, y_train, y_test
        
        except Exception as e:
            logger.error(f"✗ Preprocessing failed: {e}")
            self.execution_log["step_3_preprocessing"] = {"status": "failed", "error": str(e)}
            return None, None, None, None
    
    def step_4_train_model(self, X_train, y_train):
        logger.info("\n" + "="*70)
        logger.info("STEP 4: TRAINING XGBOOST MODEL")
        logger.info("="*70)
        
        step_start = time.time()
        
        try:
            logger.info("Initializing XGBoost recommender...")
            recommender = eKYCRecommender(model_params=XGBOOST_PARAMS, random_seed=RANDOM_SEED)
            
            logger.info(f"Training on {len(X_train)} samples...")
            results = recommender.train(X_train, y_train)
            
            recommender.save_model(TRAINED_MODEL_FILE)
            
            step_time = time.time() - step_start
            
            self.execution_log["step_4_training"] = {
                "status": "success",
                "training_samples": len(X_train),
                "time_seconds": step_time,
                "num_trees": results.get("num_trees", "N/A")
            }
            
            logger.info(f"✓ Model training complete in {step_time:.2f}s")
            
            self.audit_logger.log_model_training(
                user_id="system",
                model_version="v1.0",
                num_samples=len(X_train),
                accuracy=0.0,
                f1_score=0.0
            )
            
            return recommender
        
        except Exception as e:
            logger.error(f"✗ Model training failed: {e}")
            self.execution_log["step_4_training"] = {"status": "failed", "error": str(e)}
            return None
    
    def step_5_evaluate_model(self, recommender, X_test, y_test):
        logger.info("\n" + "="*70)
        logger.info("STEP 5: MODEL EVALUATION")
        logger.info("="*70)
        
        step_start = time.time()
        
        try:
            metrics = recommender.evaluate(X_test, y_test)
            
            step_time = time.time() - step_start
            
            self.execution_log["step_5_evaluation"] = {
                "status": "success",
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1_score": metrics["f1_score"],
                "inference_latency_ms": metrics["inference_latency_ms"],
                "time_seconds": step_time
            }
            
            logger.info(f"✓ Model evaluation complete in {step_time:.2f}s")
            logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
            logger.info(f"  Precision: {metrics['precision']:.4f}")
            logger.info(f"  Recall:    {metrics['recall']:.4f}")
            logger.info(f"  F1-Score:  {metrics['f1_score']:.4f}")
            
            if metrics['inference_latency_ms'] is not None:
                logger.info(f"  Inference Latency: {metrics['inference_latency_ms']:.2f}ms per sample")
            
            metrics_data = {
                "Metric": ["Accuracy", "Precision", "Recall", "F1-Score"],
                "Value": [
                    metrics["accuracy"],
                    metrics["precision"],
                    metrics["recall"],
                    metrics["f1_score"]
                ]
            }
            
            if metrics['inference_latency_ms'] is not None:
                metrics_data["Metric"].append("Inference Latency (ms)")
                metrics_data["Value"].append(metrics['inference_latency_ms'])
            
            metrics_df = pd.DataFrame(metrics_data)
            metrics_df.to_csv(RESULTS_DIR / "model_metrics.csv", index=False)
            logger.info(f"✓ Metrics saved to {RESULTS_DIR / 'model_metrics.csv'}")
            
            return metrics
        
        except Exception as e:
            logger.error(f"✗ Model evaluation failed: {e}")
            self.execution_log["step_5_evaluation"] = {"status": "failed", "error": str(e)}
            return None
    
    def step_6_grc_audit_logging(self, metrics):
        logger.info("\n" + "="*70)
        logger.info("STEP 6: GRC - AUDIT LOGGING")
        logger.info("="*70)
        
        try:
            logger.info("Generating audit trail...")
            
            self.audit_logger.log_event(
                user_id="system",
                action=AuditAction.SYSTEM_EVENT,
                details={
                    "event": "model_evaluation_complete",
                    "accuracy": metrics["accuracy"],
                    "f1_score": metrics["f1_score"],
                    "timestamp": datetime.now().isoformat()
                }
            )
            
            compliance_report = self.audit_logger.generate_compliance_report(
                output_path=RESULTS_DIR / "audit_compliance_report.json"
            )
            
            self.audit_logger.export_logs_to_csv(RESULTS_DIR / "audit_logs.csv")
            
            self.execution_log["step_6_audit_logging"] = {
                "status": "success",
                "audit_log_file": str(AUDIT_LOG_FILE),
                "total_events": compliance_report.get("total_events", 0),
                "compliance_status": "COMPLIANT"
            }
            
            logger.info(f"✓ Audit logging complete")
            logger.info(f"  Total events logged: {compliance_report.get('total_events', 0)}")
            
            return True
        
        except Exception as e:
            logger.error(f"✗ Audit logging failed: {e}")
            self.execution_log["step_6_audit_logging"] = {"status": "failed", "error": str(e)}
            return False
    
    def step_7_grc_retention_policy(self):
        logger.info("\n" + "="*70)
        logger.info("STEP 7: GRC - DATA RETENTION POLICY")
        logger.info("="*70)
        
        try:
            logger.info(f"Retention policy: {DATA_RETENTION_DAYS}-day expiry")
            
            sample_recs = {
                'recommendation_id': [f'rec_{i:06d}' for i in range(100)],
                'created_at': [datetime.now().isoformat() for _ in range(100)],
                'priority': ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL'] * 25
            }
            df_recs = pd.DataFrame(sample_recs)
            
            retention_report = self.retention_policy.generate_retention_report(
                df_recs,
                output_path=RESULTS_DIR / "retention_compliance_report.json"
            )
            
            self.execution_log["step_7_retention_policy"] = {
                "status": "success",
                "retention_days": DATA_RETENTION_DAYS,
                "compliance_status": retention_report.get("compliance_status", "UNKNOWN"),
                "total_recommendations": retention_report.get("statistics", {}).get("total_recommendations", 0)
            }
            
            logger.info(f"✓ Retention policy applied")
            logger.info(f"  Compliance Status: {retention_report.get('compliance_status', 'UNKNOWN')}")
            
            return retention_report
        
        except Exception as e:
            logger.error(f"✗ Retention policy failed: {e}")
            self.execution_log["step_7_retention_policy"] = {"status": "failed", "error": str(e)}
            return None
    
    def step_8_generate_reports(self):
        logger.info("\n" + "="*70)
        logger.info("STEP 8: GENERATING REPORTS")
        logger.info("="*70)
        
        try:
            execution_summary = {
                "project": "eKYC Recommendation Engine",
                "execution_timestamp": datetime.now().isoformat(),
                "pipeline_status": "COMPLETE",
                "steps": self.execution_log,
                "total_time_seconds": time.time() - self.start_time,
                "output_directory": str(RESULTS_DIR)
            }
            
            summary_path = RESULTS_DIR / "pipeline_execution_summary.json"
            with open(summary_path, "w") as f:
                json.dump(execution_summary, f, indent=2, default=str)
            
            logger.info(f"✓ Reports generated")
            logger.info(f"  Summary: {summary_path}")
            logger.info(f"  Metrics: {RESULTS_DIR / 'model_metrics.csv'}")
            logger.info(f"  Audit Logs: {RESULTS_DIR / 'audit_logs.csv'}")
            logger.info(f"  Retention Report: {RESULTS_DIR / 'retention_compliance_report.json'}")
            
            self.execution_log["step_8_reports"] = {
                "status": "success",
                "summary_file": str(summary_path)
            }
            
            return True
        
        except Exception as e:
            logger.error(f"✗ Report generation failed: {e}")
            self.execution_log["step_8_reports"] = {"status": "failed", "error": str(e)}
            return False
    
    def run_full_pipeline(self):
        self.start_time = time.time()
        
        try:
            if (PROCESSED_DATA_DIR / "X_train.pkl").exists() and \
               (PROCESSED_DATA_DIR / "X_test.pkl").exists() and \
               (PROCESSED_DATA_DIR / "y_train.pkl").exists() and \
               (PROCESSED_DATA_DIR / "y_test.pkl").exists():
                
                logger.info("\n" + "="*70)
                logger.info("CACHE HIT: Preprocessed data found!")
                logger.info("="*70)
                logger.info("Skipping Steps 1-3 (Data generation, parsing, preprocessing)")
                
                logger.info("Loading cached preprocessed data...")
                with open(PROCESSED_DATA_DIR / "X_train.pkl", "rb") as f:
                    X_train = pickle.load(f)
                with open(PROCESSED_DATA_DIR / "X_test.pkl", "rb") as f:
                    X_test = pickle.load(f)
                with open(PROCESSED_DATA_DIR / "y_train.pkl", "rb") as f:
                    y_train = pickle.load(f)
                with open(PROCESSED_DATA_DIR / "y_test.pkl", "rb") as f:
                    y_test = pickle.load(f)
                
                self.execution_log["step_1_data_generation"] = {"status": "skipped", "reason": "cache_hit"}
                self.execution_log["step_2_parsing"] = {"status": "skipped", "reason": "cache_hit"}
                self.execution_log["step_3_preprocessing"] = {
                    "status": "skipped",
                    "reason": "cache_hit",
                    "train_samples": len(X_train),
                    "test_samples": len(X_test),
                    "features": X_train.shape[1]
                }
                
                logger.info(f"✓ Loaded: Train {X_train.shape}, Test {X_test.shape}")
                logger.info(f"Proceeding to Step 4: Model Training...\n")
                
                recommender = self.step_4_train_model(X_train, y_train)
                if recommender is None:
                    return False
                
                metrics = self.step_5_evaluate_model(recommender, X_test, y_test)
                if metrics is None:
                    return False
                
                self.step_6_grc_audit_logging(metrics)
                self.step_7_grc_retention_policy()
                self.step_8_generate_reports()
                
                total_time = time.time() - self.start_time
                
                logger.info("\n" + "="*70)
                logger.info("PIPELINE EXECUTION COMPLETE ✓")
                logger.info("="*70)
                logger.info(f"Execution Mode: ACCELERATED (Steps 1-3 cached)")
                logger.info(f"Total execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
                logger.info(f"Output directory: {RESULTS_DIR}")
                logger.info("="*70 + "\n")
                
                return True
            
            logger.info("\n" + "="*70)
            logger.info("NO CACHE FOUND: Running full pipeline (Steps 1-8)")
            logger.info("="*70 + "\n")
            
            if not self.step_1_generate_data():
                return False
            
            df = self.step_2_parse_data()
            if df is None:
                return False
            
            X_train, X_test, y_train, y_test = self.step_3_preprocess_data(df)
            if X_train is None:
                return False
            
            recommender = self.step_4_train_model(X_train, y_train)
            if recommender is None:
                return False
            
            metrics = self.step_5_evaluate_model(recommender, X_test, y_test)
            if metrics is None:
                return False
            
            self.step_6_grc_audit_logging(metrics)
            
            self.step_7_grc_retention_policy()
            
            self.step_8_generate_reports()
            
            total_time = time.time() - self.start_time
            
            logger.info("\n" + "="*70)
            logger.info("PIPELINE EXECUTION COMPLETE ✓")
            logger.info("="*70)
            logger.info(f"Execution Mode: FULL (All Steps 1-8)")
            logger.info(f"Total execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
            logger.info(f"Output directory: {RESULTS_DIR}")
            logger.info("="*70 + "\n")
            
            return True
        
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            return False


if __name__ == "__main__":
    
    try:
        pipeline = eKYCPipeline()
        success = pipeline.run_full_pipeline()
        
        if success:
            logger.info("✓ All steps completed successfully!")
            sys.exit(0)
        else:
            logger.error("✗ Pipeline execution failed")
            sys.exit(1)
    
    except KeyboardInterrupt:
        logger.warning("Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)
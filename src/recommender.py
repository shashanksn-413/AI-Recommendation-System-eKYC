import xgboost as xgb
import numpy as np
import pandas as pd
import pickle
import logging
from pathlib import Path
from datetime import datetime
import time
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class eKYCRecommender:
    
    def __init__(self, model_params=None, random_seed=42):
        self.random_seed = random_seed
        self.model = None
        self.training_history = None
        self.feature_importance = None
        
        if model_params is None:
            model_params = {
                "objective": "multi:softmax",
                "num_class": 4,
                "max_depth": 6,
                "learning_rate": 0.1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "min_child_weight": 1,
                "gamma": 0,
                "random_state": random_seed,
                "n_estimators": 100,
                "early_stopping_rounds": 10,
                "eval_metric": "mlogloss"
            }
        
        self.model_params = model_params
        logger.info(f"Initialized eKYCRecommender with params: {model_params}")
    
    def train(self, X_train, y_train, X_val=None, y_val=None, verbose=True):
        logger.info(f"Starting model training...")
        logger.info(f"Training data shape: {X_train.shape}")
        logger.info(f"Label distribution: {pd.Series(y_train).value_counts().to_dict()}")
        
        start_time = time.time()
        
        if isinstance(X_train, pd.DataFrame):
            X_train = X_train.select_dtypes(include=[np.number])
            logger.info(f"Selected {X_train.shape[1]} numeric columns for training")
        else:
            X_train = np.array(X_train, dtype=float)
        
        if isinstance(y_train, pd.Series):
            y_train = y_train.values
        
        dtrain = xgb.DMatrix(X_train, label=y_train)
        
        evals = []
        if X_val is not None and y_val is not None:
            dval = xgb.DMatrix(X_val, label=y_val)
            evals = [(dval, "validation")]
            logger.info(f"Validation data shape: {X_val.shape}")
        
        self.model = xgb.train(
            self.model_params,
            dtrain,
            evals=evals,
            verbose_eval=10 if verbose else False,
            evals_result={}
        )
        
        training_time = time.time() - start_time
        logger.info(f"Training complete in {training_time:.2f} seconds")
        
        self.feature_importance = self.model.get_score(importance_type='weight')
        
        results = {
            "training_time": training_time,
            "samples_trained": len(X_train),
            "feature_count": X_train.shape[1],
            "num_trees": self.model.num_boosted_rounds(),
        }
        
        return results
    
    def predict_proba(self, X, batch_size=1000):
        if self.model is None:
            logger.error("Model not trained. Call train() first.")
            return None
        
        predictions = []
        total_samples = len(X)
        total_batches = (total_samples + batch_size - 1) // batch_size
        
        logger.info(f"Making predictions on {total_samples} samples in {total_batches} batches")
        
        inference_times = []
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, total_samples)
            
            X_batch = X.iloc[start_idx:end_idx] if isinstance(X, pd.DataFrame) else X[start_idx:end_idx]
            
            if isinstance(X_batch, pd.DataFrame):
                X_batch = X_batch.select_dtypes(include=[np.number])
            else:
                X_batch = np.array(X_batch, dtype=float)
            
            batch_start = time.time()
            dmatrix = xgb.DMatrix(X_batch)
            batch_preds = self.model.predict(dmatrix)
            batch_time = time.time() - batch_start
            
            inference_times.append(batch_time)
            predictions.append(batch_preds)
            
            if (batch_idx + 1) % max(1, total_batches // 10) == 0:
                avg_latency = (batch_time / len(X_batch)) * 1000
                logger.info(f"Batch {batch_idx + 1}/{total_batches} - {avg_latency:.2f}ms per sample")
        
        all_predictions = np.vstack(predictions)
        
        total_inference_time = sum(inference_times)
        avg_latency_per_sample = (total_inference_time / total_samples) * 1000
        
        logger.info(f"Total inference time: {total_inference_time:.2f}s")
        logger.info(f"Average latency: {avg_latency_per_sample:.2f}ms per sample")
        
        self.last_inference_latency = avg_latency_per_sample
        
        return all_predictions
    
    def predict(self, X, batch_size=1000):
        if self.model is None:
            logger.error("Model not trained. Call train() first.")
            return None
        
        predictions = []
        total_samples = len(X)
        total_batches = (total_samples + batch_size - 1) // batch_size
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, total_samples)
            
            X_batch = X.iloc[start_idx:end_idx] if isinstance(X, pd.DataFrame) else X[start_idx:end_idx]
            
            if isinstance(X_batch, pd.DataFrame):
                X_batch = X_batch.select_dtypes(include=[np.number])
            else:
                X_batch = np.array(X_batch, dtype=float)
            
            dmatrix = xgb.DMatrix(X_batch)
            batch_preds = self.model.predict(dmatrix)
            predictions.append(batch_preds)
        
        return np.hstack(predictions)
    
    def evaluate(self, X_test, y_test):
        logger.info("Evaluating model on test set...")
        
        y_pred = self.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        cm = confusion_matrix(y_test, y_pred)
        
        class_report = classification_report(y_test, y_pred, 
                                            target_names=["LOW", "MEDIUM", "HIGH", "CRITICAL"],
                                            zero_division=0)
        
        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "confusion_matrix": cm,
            "inference_latency_ms": self.last_inference_latency if hasattr(self, 'last_inference_latency') else None,
            "test_samples": len(X_test),
        }
        
        logger.info(f"Accuracy:  {accuracy:.4f}")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall:    {recall:.4f}")
        logger.info(f"F1-Score:  {f1:.4f}")
        if metrics['inference_latency_ms'] is not None:
            logger.info(f"Inference latency: {metrics['inference_latency_ms']:.2f}ms per sample")
        logger.info(f"\n{class_report}")
        
        return metrics
    
    def save_model(self, filepath):
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        self.model.save_model(str(filepath))
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        filepath = Path(filepath)
        
        if not filepath.exists():
            logger.error(f"Model file not found: {filepath}")
            return False
        
        self.model = xgb.Booster()
        self.model.load_model(str(filepath))
        logger.info(f"Model loaded from {filepath}")
        return True
    
    def get_feature_importance(self, top_n=20):
        if self.feature_importance is None:
            logger.warning("Feature importance not available. Train model first.")
            return None
        
        sorted_features = sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)
        top_features = sorted_features[:top_n]
        
        logger.info(f"Top {top_n} important features:")
        for idx, (feature, importance) in enumerate(top_features, 1):
            logger.info(f"  {idx}. {feature}: {importance}")
        
        return dict(top_features)
    
    def get_priority_distribution(self, y_pred):
        priority_names = {0: "LOW", 1: "MEDIUM", 2: "HIGH", 3: "CRITICAL"}
        distribution = {}
        
        for label in [0, 1, 2, 3]:
            count = np.sum(y_pred == label)
            pct = (count / len(y_pred)) * 100
            priority_names_mapped = priority_names[label]
            distribution[priority_names_mapped] = {"count": int(count), "percentage": round(pct, 2)}
        
        logger.info("Predicted Priority Distribution:")
        for priority, stats in distribution.items():
            logger.info(f"  {priority}: {stats['count']} records ({stats['percentage']}%)")
        
        return distribution


def train_recommender(X_train, y_train, X_test, y_test, model_params=None, 
                      output_dir=None):
    recommender = eKYCRecommender(model_params=model_params)
    recommender.train(X_train, y_train)
    
    metrics = recommender.evaluate(X_test, y_test)
    
    importance = recommender.get_feature_importance(top_n=15)
    
    y_pred = recommender.predict(X_test)
    priority_dist = recommender.get_priority_distribution(y_pred)
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        recommender.save_model(output_dir / "recommender_model.json")
        
        metrics_df = pd.DataFrame({
            "Metric": ["Accuracy", "Precision", "Recall", "F1-Score", "Inference Latency (ms)"],
            "Value": [
                metrics["accuracy"],
                metrics["precision"],
                metrics["recall"],
                metrics["f1_score"],
                metrics["inference_latency_ms"]
            ]
        })
        metrics_df.to_csv(output_dir / "model_metrics.csv", index=False)
        logger.info(f"Metrics saved to {output_dir / 'model_metrics.csv'}")
    
    return recommender, metrics, importance


if __name__ == "__main__":
    from config import XGBOOST_PARAMS, TRAINED_MODEL_FILE, RESULTS_DIR, RANDOM_SEED
    import pickle
    
    from pathlib import Path
    preprocessed_dir = Path("outputs/processed_data")
    
    try:
        with open(preprocessed_dir / "X_train.pkl", "rb") as f:
            X_train = pickle.load(f)
        with open(preprocessed_dir / "X_test.pkl", "rb") as f:
            X_test = pickle.load(f)
        with open(preprocessed_dir / "y_train.pkl", "rb") as f:
            y_train = pickle.load(f)
        with open(preprocessed_dir / "y_test.pkl", "rb") as f:
            y_test = pickle.load(f)
        
        logger.info("Loaded preprocessed data")
        
        recommender, metrics, importance = train_recommender(
            X_train, y_train, X_test, y_test,
            model_params=XGBOOST_PARAMS,
            output_dir=RESULTS_DIR
        )
        
        logger.info("Model training and evaluation complete!")
        
    except Exception as e:
        logger.error(f"Error: {e}")
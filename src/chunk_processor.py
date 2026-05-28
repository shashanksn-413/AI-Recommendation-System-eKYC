import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
import pickle
import logging
from pathlib import Path
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChunkProcessor:
    
    def __init__(self, chunk_size, categorical_features, numerical_features, 
                 train_test_split_ratio=0.8, scaling_method="standard", 
                 encoding_method="one_hot", random_seed=42):
        self.chunk_size = chunk_size
        self.categorical_features = categorical_features
        self.numerical_features = numerical_features
        self.train_test_split_ratio = train_test_split_ratio
        self.scaling_method = scaling_method
        self.encoding_method = encoding_method
        self.random_seed = random_seed
        
        self.scalers = {}
        self.encoders = {}
        self.feature_names = None
        
        logger.info(f"Initialized ChunkProcessor with chunk_size={chunk_size}")
        logger.info(f"Categorical features: {categorical_features}")
        logger.info(f"Numerical features: {numerical_features}")
    
    def convert_data_types(self, df):
        df = df.copy()
        
        numeric_cols = ['days_since_update', 'days_to_expiry', 'missing_fields', 'age']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        logger.info("Data types converted")
        return df
    
    def engineer_features(self, df):
        df = df.copy()
        
        df = self.convert_data_types(df)
        
        logger.info("Engineering features...")
        
        risk_mapping = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}
        if "risk_category" in df.columns:
            df["target"] = df["risk_category"].map(risk_mapping)
            logger.info("Created 'target' from risk_category")
        
        if "days_since_update" in df.columns:
            df["update_recency"] = df["days_since_update"]
        
        if "days_to_expiry" in df.columns:
            df["expiry_urgency"] = df["days_to_expiry"]
        
        if "missing_fields" in df.columns:
            df["missing_fields_flag"] = (df["missing_fields"] > 0).astype(int)
        
        if "document_status" in df.columns:
            df["document_expired"] = (df["document_status"] == "Expired").astype(int)
        
        if "ekyc_status" in df.columns:
            df["ekyc_pending"] = (df["ekyc_status"] == "Pending Reverification").astype(int)
        
        logger.info(f"Feature engineering complete. Shape: {df.shape}")
        return df
    
    def handle_missing_values(self, df):
        missing_before = df.isnull().sum().sum()
        
        if "target" in df.columns:
            df = df.dropna(subset=["target"])
        
        for col in self.numerical_features:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col] = df[col].fillna(df[col].mean())
        
        for col in self.categorical_features:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else "Unknown")
        
        missing_after = df.isnull().sum().sum()
        logger.info(f"Missing values: {missing_before} → {missing_after}")
        
        return df
    
    def encode_categorical_features(self, df, fit=False):
        df = df.copy()
        
        if self.encoding_method == "one_hot":
            logger.info("Applying one-hot encoding to categorical features...")
            
            cat_cols = [col for col in self.categorical_features if col in df.columns]
            
            if fit:
                df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
                self.feature_names = df.columns.tolist()
                logger.info(f"One-hot encoding created {len(self.feature_names)} features")
            else:
                df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
        
        elif self.encoding_method == "label":
            logger.info("Applying label encoding to categorical features...")
            
            for col in self.categorical_features:
                if col in df.columns:
                    if fit:
                        if col not in self.encoders:
                            self.encoders[col] = LabelEncoder()
                            df[col] = self.encoders[col].fit_transform(df[col].astype(str))
                        else:
                            df[col] = self.encoders[col].transform(df[col].astype(str))
                    else:
                        if col in self.encoders:
                            df[col] = self.encoders[col].transform(df[col].astype(str))
        
        return df
    
    def scale_numerical_features(self, df, fit=False):
        df = df.copy()
        
        num_cols = [col for col in self.numerical_features if col in df.columns]
        
        if not num_cols:
            logger.warning("No numerical columns found for scaling")
            return df
        
        if self.scaling_method == "standard":
            logger.info(f"Applying standard scaling to {len(num_cols)} numerical features...")
            
            if fit:
                self.scalers["standard"] = StandardScaler()
                df[num_cols] = self.scalers["standard"].fit_transform(df[num_cols])
            else:
                if "standard" in self.scalers:
                    df[num_cols] = self.scalers["standard"].transform(df[num_cols])
        
        return df
    
    def process_chunk(self, chunk_df, fit=False):
        chunk_df = self.engineer_features(chunk_df)
        
        chunk_df = self.handle_missing_values(chunk_df)
        
        chunk_df = self.encode_categorical_features(chunk_df, fit=fit)
        
        chunk_df = self.scale_numerical_features(chunk_df, fit=fit)
        
        return chunk_df
    
    def preprocess_chunks(self, df, save_path=None):
        logger.info(f"Starting chunk-based preprocessing for {len(df)} records")
        logger.info(f"Chunk size: {self.chunk_size}")
        
        processed_chunks = []
        total_chunks = (len(df) + self.chunk_size - 1) // self.chunk_size
        
        for chunk_idx in range(total_chunks):
            start_idx = chunk_idx * self.chunk_size
            end_idx = min((chunk_idx + 1) * self.chunk_size, len(df))
            
            chunk = df.iloc[start_idx:end_idx].copy()
            
            is_first_chunk = (chunk_idx == 0)
            chunk = self.process_chunk(chunk, fit=is_first_chunk)
            
            processed_chunks.append(chunk)
            
            progress_pct = ((chunk_idx + 1) / total_chunks) * 100
            logger.info(f"Processed chunk {chunk_idx + 1}/{total_chunks} ({progress_pct:.1f}%)")
        
        df_processed = pd.concat(processed_chunks, ignore_index=True)
        logger.info(f"All chunks processed. Final shape: {df_processed.shape}")
        
        if "target" in df_processed.columns:
            X = df_processed.drop(columns=["target", "risk_category"], errors="ignore")
            y = df_processed["target"]
            
            logger.info(f"Features shape: {X.shape}, Target shape: {y.shape}")
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=(1 - self.train_test_split_ratio),
                random_state=self.random_seed,
                stratify=y
            )
            
            logger.info(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
            
            if save_path:
                save_path = Path(save_path)
                save_path.mkdir(parents=True, exist_ok=True)
                
                with open(save_path / "X_train.pkl", "wb") as f:
                    pickle.dump(X_train, f)
                with open(save_path / "X_test.pkl", "wb") as f:
                    pickle.dump(X_test, f)
                with open(save_path / "y_train.pkl", "wb") as f:
                    pickle.dump(y_train, f)
                with open(save_path / "y_test.pkl", "wb") as f:
                    pickle.dump(y_test, f)
                
                with open(save_path / "scalers.pkl", "wb") as f:
                    pickle.dump(self.scalers, f)
                with open(save_path / "encoders.pkl", "wb") as f:
                    pickle.dump(self.encoders, f)
                
                logger.info(f"Saved preprocessed data to {save_path}")
            
            return X_train, X_test, y_train, y_test
        else:
            logger.error("Target column not found in processed data")
            return None, None, None, None
    
    def get_preprocessing_stats(self, df_original, df_processed):
        stats = {
            "timestamp": datetime.now().isoformat(),
            "original_shape": df_original.shape,
            "processed_shape": df_processed.shape,
            "original_columns": df_original.columns.tolist(),
            "processed_columns": df_processed.columns.tolist(),
            "chunk_size": self.chunk_size,
            "num_chunks": (len(df_original) + self.chunk_size - 1) // self.chunk_size,
            "categorical_features": self.categorical_features,
            "numerical_features": self.numerical_features,
            "encoding_method": self.encoding_method,
            "scaling_method": self.scaling_method,
        }
        
        return stats


def preprocess_dataset(df, config, output_path=None):
    processor = ChunkProcessor(
        chunk_size=config.CHUNK_SIZE,
        categorical_features=config.CATEGORICAL_FEATURES,
        numerical_features=config.NUMERICAL_FEATURES,
        train_test_split_ratio=config.TRAIN_TEST_SPLIT,
        scaling_method=config.SCALING_METHOD,
        encoding_method=config.ENCODING_METHOD,
        random_seed=config.RANDOM_SEED
    )
    
    X_train, X_test, y_train, y_test = processor.preprocess_chunks(df, save_path=output_path)
    
    return X_train, X_test, y_train, y_test, processor


if __name__ == "__main__":
    from config import (
        CHUNK_SIZE, CATEGORICAL_FEATURES, NUMERICAL_FEATURES,
        TRAIN_TEST_SPLIT, SCALING_METHOD, ENCODING_METHOD, 
        RANDOM_SEED, PROCESSED_DATA_DIR
    )
    from xml_parser import parse_xml_to_dataframe
    from config import SYNTHETIC_DATA_DIR
    
    logger.info("Loading synthetic data...")
    xml_file = SYNTHETIC_DATA_DIR / "synthetic_aadhar.xml"
    
    if xml_file.exists():
        df = parse_xml_to_dataframe(xml_file)
        
        if df is not None:
            logger.info(f"Loaded data shape: {df.shape}")
            
            X_train, X_test, y_train, y_test, processor = preprocess_dataset(
                df, 
                type('Config', (), {
                    'CHUNK_SIZE': CHUNK_SIZE,
                    'CATEGORICAL_FEATURES': CATEGORICAL_FEATURES,
                    'NUMERICAL_FEATURES': NUMERICAL_FEATURES,
                    'TRAIN_TEST_SPLIT': TRAIN_TEST_SPLIT,
                    'SCALING_METHOD': SCALING_METHOD,
                    'ENCODING_METHOD': ENCODING_METHOD,
                    'RANDOM_SEED': RANDOM_SEED
                }),
                output_path=PROCESSED_DATA_DIR
            )
            
            logger.info("Preprocessing complete!")
    else:
        logger.error(f"XML file not found: {xml_file}")
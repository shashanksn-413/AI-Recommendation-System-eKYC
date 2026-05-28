import json
import logging
from datetime import datetime
from pathlib import Path
import pandas as pd
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AuditAction(Enum):
    RECOMMENDATION_GENERATED = "generated_recommendation"
    RECOMMENDATION_ACCESSED = "accessed_recommendation"
    RECOMMENDATION_EXPORTED = "exported_recommendation"
    MODEL_TRAINED = "model_trained"
    DATA_PREPROCESSED = "data_preprocessed"
    SYSTEM_EVENT = "system_event"


class AuditLogger:
    
    def __init__(self, log_file_path, enable_console=True):
        self.log_file_path = Path(log_file_path)
        self.log_file_path.parent.mkdir(parents=True, exist_ok=True)
        self.enable_console = enable_console
        
        if not self.log_file_path.exists():
            self._init_log_file()
        
        logger.info(f"Audit logger initialized: {self.log_file_path}")
    
    def _init_log_file(self):
        with open(self.log_file_path, "w") as f:
            f.write("# Audit Log - eKYC Recommendation Engine\n")
            f.write(f"# Created: {datetime.now().isoformat()}\n")
            f.write("# Format: JSON (one record per line)\n")
            f.write("# Fields: timestamp, user, action, citizen_id, recommendation_id, ")
            f.write("priority, details\n")
            f.write("\n")
    
    def log_event(self, user_id, action, citizen_id=None, recommendation_id=None, 
                  priority=None, details=None):
        if isinstance(action, AuditAction):
            action_str = action.value
        else:
            action_str = str(action)
        
        event = {
            "timestamp": datetime.now().isoformat(),
            "user_id": user_id,
            "action": action_str,
            "citizen_id": citizen_id,
            "recommendation_id": recommendation_id,
            "priority": priority,
            "details": details if details else {}
        }
        
        try:
            with open(self.log_file_path, "a") as f:
                f.write(json.dumps(event) + "\n")
            
            if self.enable_console:
                logger.info(f"Audit: {action_str} | User: {user_id} | Citizen: {citizen_id} | Priority: {priority}")
        
        except Exception as e:
            logger.error(f"Error writing to audit log: {e}")
        
        return event
    
    def log_recommendation_generated(self, user_id, citizen_id, recommendation_id, 
                                    priority, model_version=None):
        details = {
            "action_type": "recommendation_generation",
            "model_version": model_version
        }
        
        return self.log_event(
            user_id=user_id,
            action=AuditAction.RECOMMENDATION_GENERATED,
            citizen_id=citizen_id,
            recommendation_id=recommendation_id,
            priority=priority,
            details=details
        )
    
    def log_recommendation_accessed(self, user_id, citizen_id, recommendation_id, 
                                   priority, access_method="api"):
        details = {
            "action_type": "recommendation_access",
            "access_method": access_method
        }
        
        return self.log_event(
            user_id=user_id,
            action=AuditAction.RECOMMENDATION_ACCESSED,
            citizen_id=citizen_id,
            recommendation_id=recommendation_id,
            priority=priority,
            details=details
        )
    
    def log_batch_export(self, user_id, num_records, priority_filter=None):
        details = {
            "action_type": "batch_export",
            "num_records": num_records,
            "priority_filter": priority_filter
        }
        
        return self.log_event(
            user_id=user_id,
            action=AuditAction.RECOMMENDATION_EXPORTED,
            details=details
        )
    
    def log_model_training(self, user_id, model_version, num_samples, accuracy, f1_score):
        details = {
            "model_version": model_version,
            "num_samples": num_samples,
            "accuracy": accuracy,
            "f1_score": f1_score
        }
        
        return self.log_event(
            user_id=user_id,
            action=AuditAction.MODEL_TRAINED,
            details=details
        )
    
    def log_data_preprocessing(self, user_id, num_records, preprocessing_version):
        details = {
            "num_records": num_records,
            "preprocessing_version": preprocessing_version
        }
        
        return self.log_event(
            user_id=user_id,
            action=AuditAction.DATA_PREPROCESSED,
            details=details
        )
    
    def read_audit_log(self):
        if not self.log_file_path.exists():
            logger.warning(f"Audit log file not found: {self.log_file_path}")
            return pd.DataFrame()
        
        events = []
        with open(self.log_file_path, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    try:
                        event = json.loads(line)
                        events.append(event)
                    except json.JSONDecodeError:
                        logger.warning(f"Could not parse JSON: {line}")
        
        if events:
            df = pd.DataFrame(events)
            logger.info(f"Loaded {len(df)} audit log records")
            return df
        else:
            return pd.DataFrame()
    
    def get_user_activity(self, user_id=None):
        df = self.read_audit_log()
        
        if df.empty:
            return df
        
        if user_id:
            df = df[df["user_id"] == user_id]
            logger.info(f"Activity for user {user_id}: {len(df)} events")
        else:
            logger.info(f"Total activity across all users: {len(df)} events")
        
        if not df.empty:
            summary = df.groupby(["user_id", "action"]).size().reset_index(name="count")
            return summary
        
        return df
    
    def get_access_by_priority(self):
        df = self.read_audit_log()
        
        if df.empty or "priority" not in df.columns:
            return pd.DataFrame()
        
        access_events = df[df["action"].isin([
            AuditAction.RECOMMENDATION_GENERATED.value,
            AuditAction.RECOMMENDATION_ACCESSED.value
        ])]
        
        summary = access_events.groupby("priority").size().reset_index(name="count")
        logger.info(f"Access summary by priority:\n{summary}")
        
        return summary
    
    def get_time_range_activity(self, start_time=None, end_time=None):
        df = self.read_audit_log()
        
        if df.empty:
            return df
        
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        
        if start_time:
            start_dt = pd.to_datetime(start_time)
            df = df[df["timestamp"] >= start_dt]
        
        if end_time:
            end_dt = pd.to_datetime(end_time)
            df = df[df["timestamp"] <= end_dt]
        
        logger.info(f"Activity in time range: {len(df)} events")
        return df
    
    def generate_compliance_report(self, output_path=None):
        df = self.read_audit_log()
        
        if df.empty:
            logger.warning("No audit logs to generate report")
            return {}
        
        report = {
            "report_generated": datetime.now().isoformat(),
            "total_events": len(df),
            "unique_users": df["user_id"].nunique(),
            "unique_citizens": df["citizen_id"].nunique() if "citizen_id" in df.columns else 0,
            "events_by_action": df["action"].value_counts().to_dict(),
            "events_by_priority": df.groupby("priority").size().to_dict() if "priority" in df.columns else {},
            "timestamp_range": {
                "earliest": df["timestamp"].min() if "timestamp" in df.columns else None,
                "latest": df["timestamp"].max() if "timestamp" in df.columns else None
            }
        }
        
        logger.info(f"Compliance Report:\n{json.dumps(report, indent=2, default=str)}")
        
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, "w") as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"Compliance report saved to {output_path}")
        
        return report
    
    def export_logs_to_csv(self, output_path):
        df = self.read_audit_log()
        
        if df.empty:
            logger.warning("No logs to export")
            return False
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(output_path, index=False)
        logger.info(f"Exported {len(df)} audit log records to {output_path}")
        
        return True


if __name__ == "__main__":
    from config import AUDIT_LOG_FILE, RESULTS_DIR
    
    audit_log = AuditLogger(AUDIT_LOG_FILE)
    
    audit_log.log_recommendation_generated(
        user_id="officer_001",
        citizen_id="123456789012",
        recommendation_id="rec_001",
        priority="HIGH",
        model_version="v1.0"
    )
    
    audit_log.log_recommendation_accessed(
        user_id="officer_001",
        citizen_id="123456789012",
        recommendation_id="rec_001",
        priority="HIGH",
        access_method="web"
    )
    
    audit_log.log_batch_export(
        user_id="admin_001",
        num_records=500,
        priority_filter="HIGH, CRITICAL"
    )
    
    audit_log.log_model_training(
        user_id="admin_001",
        model_version="v1.0",
        num_samples=800000,
        accuracy=0.9639,
        f1_score=0.9522
    )
    
    user_activity = audit_log.get_user_activity()
    print("User Activity:\n", user_activity)
    
    priority_access = audit_log.get_access_by_priority()
    print("\nAccess by Priority:\n", priority_access)
    
    compliance_report = audit_log.generate_compliance_report(
        output_path=RESULTS_DIR / "compliance_report.json"
    )
    
    audit_log.export_logs_to_csv(RESULTS_DIR / "audit_logs.csv")
    
    print("\nAudit logging example complete!")
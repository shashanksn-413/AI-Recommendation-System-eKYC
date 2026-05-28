import pandas as pd
from datetime import datetime, timedelta
import logging
import json
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RetentionPolicy:
    
    def __init__(self, retention_days=30, enforcement_enabled=True):
        self.retention_days = retention_days
        self.enforcement_enabled = enforcement_enabled
        self.retention_log = []
        
        logger.info(f"Retention Policy initialized: {retention_days}-day retention")
        logger.info(f"Enforcement: {'ENABLED' if enforcement_enabled else 'DISABLED'}")
    
    def calculate_expiry_date(self, creation_date):
        if isinstance(creation_date, str):
            creation_dt = datetime.fromisoformat(creation_date)
        else:
            creation_dt = creation_date
        
        expiry_dt = creation_dt + timedelta(days=self.retention_days)
        return expiry_dt
    
    def is_expired(self, creation_date, check_time=None):
        if check_time is None:
            check_time = datetime.now()
        
        expiry_date = self.calculate_expiry_date(creation_date)
        
        return check_time >= expiry_date
    
    def get_days_remaining(self, creation_date, check_time=None):
        if check_time is None:
            check_time = datetime.now()
        
        expiry_date = self.calculate_expiry_date(creation_date)
        days_remaining = (expiry_date - check_time).days
        
        return days_remaining
    
    def mark_expired(self, recommendation_df):
        df = recommendation_df.copy()
        
        if 'created_at' not in df.columns:
            logger.warning("'created_at' column not found. Cannot mark expiry status.")
            return df
        
        df['is_expired'] = df['created_at'].apply(self.is_expired)
        df['expiry_date'] = df['created_at'].apply(self.calculate_expiry_date)
        df['days_remaining'] = df['created_at'].apply(self.get_days_remaining)
        
        logger.info(f"Marked expiry status for {len(df)} records")
        logger.info(f"  Expired: {df['is_expired'].sum()}")
        logger.info(f"  Active: {(~df['is_expired']).sum()}")
        
        return df
    
    def filter_active_recommendations(self, recommendation_df):
        df = self.mark_expired(recommendation_df)
        active_df = df[~df['is_expired']].copy()
        
        logger.info(f"Filtered to active recommendations: {len(active_df)}/{len(df)}")
        
        return active_df
    
    def identify_expiring_soon(self, recommendation_df, days_threshold=7):
        df = self.mark_expired(recommendation_df)
        
        expiring_soon = df[(~df['is_expired']) & (df['days_remaining'] <= days_threshold)]
        
        logger.info(f"Recommendations expiring within {days_threshold} days: {len(expiring_soon)}")
        
        return expiring_soon
    
    def enforce_retention_policy(self, recommendation_df, delete_expired=False):
        if not self.enforcement_enabled:
            logger.warning("Retention enforcement is DISABLED")
            return {"status": "disabled"}
        
        df = self.mark_expired(recommendation_df)
        
        expired_count = df['is_expired'].sum()
        active_count = len(df) - expired_count
        
        if delete_expired:
            df_cleaned = df[~df['is_expired']].copy()
            deleted_count = len(df) - len(df_cleaned)
            logger.warning(f"DELETED {deleted_count} expired recommendations")
            
            stats = {
                "action": "delete",
                "total_records": len(df),
                "active_records": len(df_cleaned),
                "expired_deleted": deleted_count,
                "enforcement_timestamp": datetime.now().isoformat()
            }
            
            self.retention_log.append(stats)
            return df_cleaned, stats
        else:
            stats = {
                "action": "mark_only",
                "total_records": len(df),
                "active_records": active_count,
                "expired_records": expired_count,
                "enforcement_timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"Retention enforcement: {active_count} active, {expired_count} expired (marked)")
            self.retention_log.append(stats)
            return df, stats
    
    def get_retention_statistics(self, recommendation_df):
        df = self.mark_expired(recommendation_df)
        
        expired_count = df['is_expired'].sum()
        active_count = len(df) - expired_count
        
        age_distribution = {
            "0-7_days": len(df[df['days_remaining'] > 23]),
            "7-14_days": len(df[(df['days_remaining'] > 16) & (df['days_remaining'] <= 23)]),
            "14-21_days": len(df[(df['days_remaining'] > 9) & (df['days_remaining'] <= 16)]),
            "21-30_days": len(df[(df['days_remaining'] > 0) & (df['days_remaining'] <= 9)]),
            "expired": expired_count
        }
        
        stats = {
            "total_recommendations": len(df),
            "active_recommendations": active_count,
            "expired_recommendations": expired_count,
            "retention_policy_days": self.retention_days,
            "enforcement_enabled": self.enforcement_enabled,
            "age_distribution": age_distribution,
            "avg_days_remaining": int(df['days_remaining'].mean()),
            "min_days_remaining": int(df['days_remaining'].min()),
            "max_days_remaining": int(df['days_remaining'].max()),
            "statistics_timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"Retention Statistics:\n{json.dumps(stats, indent=2)}")
        
        return stats
    
    def generate_retention_report(self, recommendation_df, output_path=None):
        df = self.mark_expired(recommendation_df)
        stats = self.get_retention_statistics(recommendation_df)
        
        report = {
            "report_type": "retention_compliance",
            "generated_at": datetime.now().isoformat(),
            "policy_details": {
                "retention_days": self.retention_days,
                "enforcement_enabled": self.enforcement_enabled,
                "compliance_standard": "DPDP Act (Data Protection)"
            },
            "statistics": stats,
            "enforcement_history": self.retention_log[-10:] if self.retention_log else [],
            "compliance_status": "COMPLIANT" if stats['expired_recommendations'] == 0 else "NON_COMPLIANT"
        }
        
        logger.info(f"Compliance Status: {report['compliance_status']}")
        
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, "w") as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"Retention report saved to {output_path}")
        
        return report
    
    def export_retention_dataframe(self, recommendation_df, output_path=None):
        df = self.mark_expired(recommendation_df)
        
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_path, index=False)
            logger.info(f"Exported {len(df)} recommendations with retention metadata to {output_path}")
        
        return df
    
    def get_batch_for_deletion(self, recommendation_df, batch_size=1000):
        df = self.mark_expired(recommendation_df)
        expired_df = df[df['is_expired']].head(batch_size).copy()
        
        logger.info(f"Retrieved {len(expired_df)} expired recommendations for deletion (batch_size={batch_size})")
        
        return expired_df
    
    def schedule_retention_check(self, recommendation_df, interval_days=1):
        stats = self.get_retention_statistics(recommendation_df)
        next_check = datetime.now() + timedelta(days=interval_days)
        
        schedule_info = {
            "last_check": datetime.now().isoformat(),
            "next_check": next_check.isoformat(),
            "check_interval_days": interval_days,
            "expired_awaiting_cleanup": stats['expired_recommendations'],
            "status": "scheduled"
        }
        
        logger.info(f"Retention check scheduled for {next_check.isoformat()}")
        
        return schedule_info


def apply_retention_policy_to_batch(batch_df, retention_days=30):
    policy = RetentionPolicy(retention_days=retention_days)
    return policy.mark_expired(batch_df)


def get_compliant_recommendations(recommendation_df, retention_days=30):
    policy = RetentionPolicy(retention_days=retention_days)
    return policy.filter_active_recommendations(recommendation_df)


if __name__ == "__main__":
    from config import DATA_RETENTION_DAYS, RETENTION_CHECK_INTERVAL_DAYS, RESULTS_DIR
    
    logger.info("Retention Policy Manager - Demonstration")
    
    retention_mgr = RetentionPolicy(
        retention_days=DATA_RETENTION_DAYS,
        enforcement_enabled=True
    )
    
    sample_data = {
        'recommendation_id': ['rec_001', 'rec_002', 'rec_003', 'rec_004'],
        'citizen_id': ['111111111111', '222222222222', '333333333333', '444444444444'],
        'priority': ['HIGH', 'CRITICAL', 'MEDIUM', 'LOW'],
        'created_at': [
            (datetime.now() - timedelta(days=5)).isoformat(),
            (datetime.now() - timedelta(days=15)).isoformat(),
            (datetime.now() - timedelta(days=30)).isoformat(),
            (datetime.now() - timedelta(days=35)).isoformat(),
        ]
    }
    
    df = pd.DataFrame(sample_data)
    logger.info(f"Sample data:\n{df}\n")
    
    logger.info("Testing retention policy...")
    
    df_with_expiry = retention_mgr.mark_expired(df)
    logger.info(f"With expiry status:\n{df_with_expiry}\n")
    
    active_only = retention_mgr.filter_active_recommendations(df)
    logger.info(f"Active recommendations:\n{active_only}\n")
    
    stats = retention_mgr.get_retention_statistics(df)
    
    report = retention_mgr.generate_retention_report(df, output_path=RESULTS_DIR / "retention_report.json")
    
    logger.info("Retention policy demonstration complete!")
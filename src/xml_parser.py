import xml.etree.ElementTree as ET
import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AadharXMLParser:
    
    def __init__(self, xml_file_path):
        self.xml_file_path = Path(xml_file_path)
        self.tree = None
        self.root = None
        self.data = []
        
        if not self.xml_file_path.exists():
            raise FileNotFoundError(f"XML file not found: {self.xml_file_path}")
    
    def load_xml(self):
        try:
            self.tree = ET.parse(self.xml_file_path)
            self.root = self.tree.getroot()
            logger.info(f"Successfully loaded XML file: {self.xml_file_path}")
            return True
        except ET.ParseError as e:
            logger.error(f"Error parsing XML file: {e}")
            return False
    
    def parse_records(self, chunk_size=None):
        if self.root is None:
            logger.error("XML not loaded. Call load_xml() first.")
            return None
        
        records = []
        total_records = len(self.root.findall('Record'))
        
        logger.info(f"Parsing {total_records} records from XML...")
        
        for idx, record_elem in enumerate(self.root.findall('Record')):
            record = self._extract_record_data(record_elem)
            records.append(record)
            
            if (idx + 1) % max(1, total_records // 10) == 0:
                logger.info(f"Parsed {idx + 1}/{total_records} records ({((idx + 1) / total_records * 100):.1f}%)")
        
        df = pd.DataFrame(records)
        logger.info(f"Successfully parsed {len(df)} records into DataFrame")
        
        return df
    
    def _extract_record_data(self, record_elem):
        record = {}
        
        personal_info = record_elem.find('PersonalInfo')
        if personal_info is not None:
            record['aadhar_number'] = personal_info.findtext('AadharNumber', default='')
            record['name'] = personal_info.findtext('Name', default='')
            record['date_of_birth'] = personal_info.findtext('DOB', default='')
            record['gender'] = personal_info.findtext('Gender', default='')
            record['phone'] = personal_info.findtext('Phone', default='')
            record['email'] = personal_info.findtext('Email', default='')
        
        address_info = record_elem.find('AddressInfo')
        if address_info is not None:
            record['address_line1'] = address_info.findtext('AddressLine1', default='')
            record['city'] = address_info.findtext('City', default='')
            record['state'] = address_info.findtext('State', default='')
            record['pincode'] = address_info.findtext('Pincode', default='')
            record['address_last_updated'] = address_info.findtext('LastUpdated', default='')
        
        doc_info = record_elem.find('DocumentInfo')
        if doc_info is not None:
            record['document_type'] = doc_info.findtext('DocumentType', default='')
            record['document_issue_date'] = doc_info.findtext('IssueDate', default='')
            record['document_expiry_date'] = doc_info.findtext('ExpiryDate', default='')
            record['document_status'] = doc_info.findtext('Status', default='')
        
        ekyc_info = record_elem.find('eKYCInfo')
        if ekyc_info is not None:
            record['last_ekyc_update'] = ekyc_info.findtext('LastUpdate', default='')
            record['ekyc_status'] = ekyc_info.findtext('Status', default='')
            record['missing_fields'] = ekyc_info.findtext('MissingFields', default='0')
        
        profile_info = record_elem.find('ProfileInfo')
        if profile_info is not None:
            record['income_bracket'] = profile_info.findtext('IncomeBracket', default='')
            record['occupation'] = profile_info.findtext('Occupation', default='')
            record['risk_category'] = profile_info.findtext('RiskCategory', default='')
        
        return record
    
    def parse_chunk_by_chunk(self, chunk_size):
        if self.root is None:
            logger.error("XML not loaded. Call load_xml() first.")
            return
        
        chunk = []
        total = len(self.root.findall('Record'))
        
        logger.info(f"Parsing XML in chunks of {chunk_size}...")
        
        for idx, record_elem in enumerate(self.root.findall('Record')):
            record = self._extract_record_data(record_elem)
            chunk.append(record)
            
            if len(chunk) == chunk_size:
                df_chunk = pd.DataFrame(chunk)
                chunk_num = (idx + 1) // chunk_size
                logger.info(f"Yielding chunk {chunk_num} ({len(df_chunk)} records)")
                yield df_chunk
                chunk = []
        
        if chunk:
            df_chunk = pd.DataFrame(chunk)
            logger.info(f"Yielding final chunk ({len(df_chunk)} records)")
            yield df_chunk
    
    def get_statistics(self, df):
        print("\n" + "="*60)
        print("DATA STATISTICS")
        print("="*60)
        print(f"Total Records: {len(df)}")
        print(f"Total Columns: {len(df.columns)}")
        print(f"\nColumns: {list(df.columns)}")
        print(f"\nData Types:\n{df.dtypes}")
        print(f"\nMissing Values:\n{df.isnull().sum()}")
        print(f"\nShape: {df.shape}")
        print("="*60 + "\n")


def parse_xml_to_dataframe(xml_file_path, chunk_size=None):
    parser = AadharXMLParser(xml_file_path)
    
    if not parser.load_xml():
        return None
    
    if chunk_size:
        return parser.parse_chunk_by_chunk(chunk_size)
    else:
        return parser.parse_records()


def validate_dataframe(df):
    report = {
        "total_records": len(df),
        "total_columns": len(df.columns),
        "missing_values": df.isnull().sum().to_dict(),
        "duplicate_records": len(df[df.duplicated()]),
        "empty_cells_pct": (df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100),
    }
    
    logger.info(f"Validation Report: {report}")
    return report


if __name__ == "__main__":
    from src.config import SYNTHETIC_DATA_DIR, CHUNK_SIZE
    
    xml_path = SYNTHETIC_DATA_DIR / "synthetic_aadhar.xml"
    
    parser = AadharXMLParser(xml_path)
    parser.load_xml()
    
    df = parser.parse_records()
    parser.get_statistics(df)
    
    validation = validate_dataframe(df)
    print(f"Validation: {validation}")
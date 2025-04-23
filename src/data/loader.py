import json
import logging
import os
from typing import List, Dict, Any

def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Load JSONL file (e.g., pred.jsonl)."""
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                try:
                    item = json.loads(line.strip())
                    if 'predict' in item:
                        data.append(item)
                    else:
                        logging.warning(f"Line {line_num + 1} missing 'predict'")
                except json.JSONDecodeError:
                    logging.error(f"Invalid JSON at line {line_num + 1}")
        logging.info(f"Loaded {len(data)} items from {file_path}")
        return data
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        raise

def load_json(file_path: str) -> List[Dict[str, Any]]:
    """Load JSON file (e.g., test.json)."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if not isinstance(data, list):
                raise ValueError("Expected a JSON list")
            valid_data = [item for item in data if isinstance(item, dict) and 'raw_data' in item]
            logging.info(f"Loaded {len(valid_data)} datasets from {file_path}")
            return valid_data
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        raise

def load_prompt(file_path: str) -> str:
    """Load prompt template (e.g., prompt.txt)."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            prompt = f.read()
            logging.info(f"Loaded prompt from {file_path}")
            return prompt
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        raise

def parse_variable_names(raw_data: List[Dict[str, Any]]) -> List[str]:
    """Extract variable names from raw_data, excluding '结果'."""
    if not raw_data:
        return []
    first_sample = raw_data[0]
    variable_names = [key for key in first_sample.keys() if key != "结果"]
    logging.info(f"Parsed variable names: {variable_names}")
    return variable_names

def validate_data(analysis_texts: List[Dict], raw_datasets: List[Dict]) -> None:
    """Validate data consistency."""
    if len(analysis_texts) != len(raw_datasets):
        logging.error(f"Mismatch: {len(analysis_texts)} texts vs {len(raw_datasets)} datasets")
        raise ValueError("Data length mismatch")
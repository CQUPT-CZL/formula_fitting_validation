import sys
import os
import re
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import logging
from src.data.loader import load_jsonl, load_prompt
from src.model.llm_interface import LLMInterface
from typing import Callable, Optional, List, Dict
from src.model.code_executor import extract_code_block
from src.utils.my_logging import setup_logging
import yaml


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def extract_json_data(text: str) -> Optional[List[Dict]]:

    # Search for JSON block, accounting for markdown code fences or plain JSON
    match = re.search(r"```json\n(.*?)\n```|(\{.*?\})", text, re.DOTALL)
    if not match:
        logging.error("No JSON block found")
        return None

    # Extract JSON string (from either markdown or plain JSON match group)
    json_str = match.group(1) if match.group(1) else match.group(2)
    json_str = json_str.strip()

    try:
        # Parse JSON string into Python object
        data = json.loads(json_str)
        # Verify that 'raw_data' field exists
        if "raw_data" in data and isinstance(data["raw_data"], list):
            logging.info("Successfully extracted and parsed JSON data")
            return data["raw_data"]
        else:
            logging.error("JSON does not contain 'raw_data' field")
            return None
    except json.JSONDecodeError as e:
        logging.error(f"Failed to parse JSON: {str(e)}")
        return None


def generate_raw_data_for_entry(analysis_text: str, llm_interface: 'LLMInterface', prompt_template: str) -> str | list[
    dict]:
    try:
        llm_output = llm_interface.generate_raw_data(prompt_template, analysis_text)
        raw_data = extract_json_data(llm_output)
        if raw_data:
            # logging.info(f"Generated code for analysis: {analysis_text[:50]}...")
            return raw_data
        logging.error("No raw_data block found")
        return ""
    except Exception as e:
        logging.error(f"Error generating raw_data: {e}")
        return ""


def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    config_path = os.path.join(project_root, 'config', 'config.yaml')

    if not os.path.exists(config_path):
        logging.error(f"Config file not found at: {config_path}")
        raise FileNotFoundError(f"Config file not found at: {config_path}")

    config = load_config(config_path)
    setup_logging(config)

    # Load input JSONL
    input_path = os.path.join(project_root, config['paths']['input_data'])
    if not os.path.exists(input_path):
        logging.error(f"Input JSONL not found at: {input_path}")
        raise FileNotFoundError(f"Input JSONL not found at: {input_path}")

    data = load_jsonl(input_path)

    # Load prompt
    prompt_template = load_prompt(config['paths']['prompt_raw'])

    llm_interface = LLMInterface(config)

    # Generate code
    code_dir = os.path.join(project_root, config['paths']['generated_code_dir'])
    os.makedirs(code_dir, exist_ok=True)

    for idx, entry in enumerate(data):
        if 'predict' not in entry:
            logging.warning(f"Entry {idx + 1} missing 'predict' field")
            entry['raw_data'] = ""
            continue
        raw_data = generate_raw_data_for_entry(entry['predict'], llm_interface, prompt_template)
        entry['raw_data'] = raw_data


    # Save updated JSON
    output_path = os.path.join(project_root, config['paths']['generated_code_dir'], 'data_with_raw0426.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    logging.info(f"Saved updated JSON to {output_path}")


if __name__ == "__main__":
    main()
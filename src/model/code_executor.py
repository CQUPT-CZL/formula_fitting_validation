import re
import logging
import os
from typing import Callable, Optional

def safe_exec(code: str, scope: dict) -> Optional[Callable]:
    try:
        exec(code, scope)
        if 'calculate_value' in scope and callable(scope['calculate_value']):
            logging.info("Defined calculate_value function")
            return scope['calculate_value']
        logging.error("calculate_value not defined or not callable")
        return None
    except Exception as e:
        logging.error(f"Error executing code: {e}")
        return None

def extract_code_block(text: str) -> Optional[str]:
    match = re.search(r"```python\n(.*)\n```", text, re.DOTALL)
    if match:
        code = match.group(1).strip()
        logging.info("Extracted Python code block")
        return code
    logging.error("No Python code block found")
    return None

def load_saved_code(file_path: str) -> Optional[Callable]:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        scope = {}
        exec(code, scope)
        if 'calculate_value' in scope and callable(scope['calculate_value']):
            logging.info(f"Loaded calculate_value from {file_path}")
            return scope['calculate_value']
        logging.error(f"No valid calculate_value in {file_path}")
        return None
    except Exception as e:
        logging.error(f"Error loading code from {file_path}: {e}")
        return None
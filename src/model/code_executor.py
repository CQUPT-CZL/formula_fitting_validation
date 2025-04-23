import re
import logging
from typing import Callable, Optional

def safe_exec(code: str, scope: dict) -> Optional[Callable]:
    """Safely execute Python code and return calculate_value function."""
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
    """Extract Python code block from LLM output."""
    match = re.search(r"```python\n(.*)\n```", text, re.DOTALL)
    if match:
        code = match.group(1).strip()
        logging.info("Extracted Python code block")
        return code
    logging.error("No Python code block found")
    return None
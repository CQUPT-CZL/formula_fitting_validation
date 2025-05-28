import yaml
from src.model.llm_interface import LLMInterface
from src.utils.my_logging import setup_logging
from src.data.loader import load_json, load_prompt
from typing import List, Dict, Any
import os
import logging
from .base_metric import BaseMetric

def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
config_path = '/Users/cuiziliang/Projects/formula_fitting_validation/config/config.yaml'

class CodeEq(BaseMetric):
    """Mean Absolute Error metric."""
    def compute(
            self,
            predictions: List[float],
            ground_truths: List[float],
            raw_data: List[Dict[str, Any]],
            variable_names: List[str],
            instruction: str,
            prediction: str,
            code: str,
            gt_code: str,
            idx: int
    ):
        if idx % 2 == 0:
            try:
                if not os.path.exists(config_path):
                    logging.error(f"Config file not found at: {config_path}")
                    raise FileNotFoundError(f"Config file not found at: {config_path}")

                config = load_config(config_path)
                setup_logging(config)

                llm_interface = LLMInterface(config)

                # 获取code的api代码
                code_prompt_template = load_prompt(config['paths']['prompt_code_judge'])
                scores = llm_interface.generate_code_metrics(code_prompt_template, code, gt_code)

            except Exception as e:
                logging.error(f"Error: {e}")
                raise
        else:
            scores = {}
        return scores
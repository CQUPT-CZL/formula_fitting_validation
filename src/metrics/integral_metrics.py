from typing import List, Dict, Any

from .base_metric import BaseMetric
import numpy as np
import logging

class ICE(BaseMetric):
    def compute(
        self,
        predictions: List[float],
        ground_truths: List[float],
        raw_data: List[Dict[str, Any]],
        variable_names: List[str]
    ) -> float:
        try:
            errors = [abs(p - gt) for p, gt in zip(predictions, ground_truths)]
            ice = np.mean(errors) * 10  # 假设积分范围 0-10
            logging.info(f"Computed ICE for variables {variable_names}: {ice}")
            return ice
        except Exception as e:
            logging.error(f"Error computing ICE: {e}")
            raise
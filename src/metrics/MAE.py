import numpy as np
from typing import List, Dict, Any, Union
import logging

from numpy import floating
from .base_metric import BaseMetric


class MAE(BaseMetric):
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
    ) -> floating[Any]:
        try:
            if len(predictions) != len(ground_truths):
                logging.error("Predictions and ground truths length mismatch")
                raise ValueError("Length mismatch")
            errors = [abs(pred - gt) for pred, gt in zip(predictions, ground_truths)]
            mae = np.mean(errors)
            logging.info(f"Computed MAE for variables {variable_names}: {mae}")
            return mae
        except Exception as e:
            logging.error(f"Error computing MAE: {e}")
            raise
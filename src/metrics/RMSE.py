import numpy as np
from typing import List, Dict, Any, Union
import logging

from .base_metric import BaseMetric

class RMSE(BaseMetric):
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
    ) -> float:
        try:
            if len(predictions) != len(ground_truths):
                logging.error("Predictions and ground truths length mismatch")
                raise ValueError("Length mismatch")
            errors = [(pred - gt) ** 2 for pred, gt in zip(predictions, ground_truths)]
            mse = np.mean(errors)
            rmse = np.sqrt(mse)
            logging.info(f"Computed RMSE for variables {variable_names}: {rmse}")
            return rmse
        except Exception as e:
            logging.error(f"Error computing RMSE: {e}")
            raise
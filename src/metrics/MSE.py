import numpy as np
from typing import List, Dict, Any, Union
import logging
from .base_metric import BaseMetric
from numpy import floating


class MSE(BaseMetric):
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
            errors = [(pred - gt) ** 2 for pred, gt in zip(predictions, ground_truths)]
            mse = np.mean(errors)
            logging.info(f"Computed MSE for variables {variable_names}: {mse}")
            return mse
        except Exception as e:
            logging.error(f"Error computing MSE: {e}")
            raise

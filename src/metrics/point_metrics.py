from typing import List, Dict, Any

import numpy as np
from .base_metric import BaseMetric
import logging

class MAE(BaseMetric):
    """Mean Absolute Error metric."""
    def compute(
        self,
        predictions: List[float],
        ground_truths: List[float],
        raw_data: List[Dict[str, Any]],
        variable_names: List[str]
    ) -> float:
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


class MSE(BaseMetric):
    def compute(
        self,
        predictions: List[float],
        ground_truths: List[float],
        raw_data: List[Dict[str, Any]],
        variable_names: List[str]
    ) -> float:
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


class RMSE(BaseMetric):
    def compute(
        self,
        predictions: List[float],
        ground_truths: List[float],
        raw_data: List[Dict[str, Any]],
        variable_names: List[str]
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

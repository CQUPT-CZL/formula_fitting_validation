from abc import ABC, abstractmethod
from typing import List, Dict, Any, Union


class BaseMetric(ABC):
    """Base class for metrics, supporting variable number and names of inputs."""

    @abstractmethod
    def compute(
            self,
            predictions: List[float],
            ground_truths: List[float],
            raw_data: List[Dict[str, Any]],
            variable_names: List[str]
    ) -> Union[float, Dict, str]:
        """
        Compute the metric for a set of predictions and ground truths.

        Args:
            predictions: Predicted values from calculate_value function.
            ground_truths: True values from raw_data['结果'].
            raw_data: List of dictionaries containing variable values and result.
            variable_names: List of variable names (e.g., ['x', 'y'] or ['w', 'x1', 'x2', 'z']).

        Returns:
            Union[float, Dict, str]: Metric value (float for quantitative, Dict/str for qualitative).
        """
        pass
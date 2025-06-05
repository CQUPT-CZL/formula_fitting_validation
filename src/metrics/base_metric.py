from abc import ABC, abstractmethod
from typing import List, Dict, Any, Union
from numpy import floating


class BaseMetric(ABC):
    """Base class for metrics, supporting variable number and names of inputs."""

    @abstractmethod
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
    ) -> Union[floating[any], Dict, str]:
        """
        Compute the metric for a set of predictions and ground truths.

        Args:
            predictions: 预测的数值
            ground_truths: 真实的数值
            raw_data: 原始数据
            variable_names: 变量名
            instruction: 提示词.
            prediction: 模型输出
            code: 模型生成的代码
            gt_code: 真实代码
            idx: 样本编号
        Returns:
            Union[float, Dict, str]: Metric value (float for quantitative, Dict/str for qualitative).
        """
        pass
import sys
import os

from src.metrics.base_metric import BaseMetric

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import yaml
import json
import logging
from src.data.loader import parse_variable_names, validate_data, load_json
from src.model.code_executor import safe_exec
from src.metrics.point_metrics import MAE
from src.metrics.integral_metrics import ICE  # 确保导入 ICE
from src.utils.my_logging import setup_logging
from typing import List, Dict, Any, Optional

def evaluate_pair(
    raw_data: List[Dict[str, Any]],
    code: str,
    metrics: List['BaseMetric'],
    pair_index: int
) -> Optional[Dict[str, float]]:
    variable_names = parse_variable_names(raw_data)
    logging.info(f"Evaluating pair {pair_index} with variables: {variable_names}")

    generated_function = safe_exec(code, {})

    if generated_function:
        predictions = []
        ground_truths = []
        valid_samples = 0

        for sample_idx, sample in enumerate(raw_data):
            try:
                if not all(var in sample for var in variable_names + ["结果"]):
                    logging.warning(f"Sample {sample_idx + 1} missing required keys")
                    continue
                kwargs = {var: sample[var] for var in variable_names}
                true_result = sample["结果"]
                predicted_result = generated_function(**kwargs)

                if isinstance(predicted_result, (int, float)):
                    predictions.append(predicted_result)
                    ground_truths.append(true_result)
                    valid_samples += 1
                else:
                    logging.warning(f"Non-numeric prediction in sample {sample_idx + 1}")
            except Exception as e:
                logging.error(f"Error evaluating sample {sample_idx + 1}: {e}")

        if valid_samples > 0:
            results = {}
            for metric in metrics:
                try:
                    metric_name = metric.__class__.__name__.lower()
                    results[metric_name] = metric.compute(
                        predictions, ground_truths, raw_data, variable_names
                    )
                except Exception as e:
                    logging.error(f"Error computing {metric.__class__.__name__}: {e}")
            logging.info(f"Evaluation results for pair {pair_index}: {results}")
            return results
    return None

def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    config_path = os.path.join(project_root, 'config', 'config.yaml')

    if not os.path.exists(config_path):
        logging.error(f"Config file not found at: {config_path}")
        raise FileNotFoundError(f"Config file not found at: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    setup_logging(config)

    # Load data with code
    data_with_code_path = os.path.join(project_root, config['paths']['generated_code_dir'], 'data_with_code.json')
    if not os.path.exists(data_with_code_path):
        logging.error(f"Data with code not found at: {data_with_code_path}")
        raise FileNotFoundError(f"Data with code not found at: {data_with_code_path}")

    with open(data_with_code_path, 'r', encoding='utf-8') as f:
        data = json.load(f)


    metrics = [MAE(), ICE()]
    metric_results = {metric.__class__.__name__.lower(): [] for metric in metrics}

    for idx, en in enumerate(data):
        if 'code' not in en or not en['code']:
            logging.warning(f"Pair {idx + 1} missing code")
            for metric in metrics:
                metric_results[metric.__class__.__name__.lower()].append(None)
            continue
        result = evaluate_pair(
            en['raw_data'],
            en['code'],
            metrics,
            idx + 1
        )
        if result:
            for metric in metrics:
                metric_name = metric.__class__.__name__.lower()
                metric_results[metric_name].append(result[metric_name])
        else:
            for metric in metrics:
                metric_results[metric.__class__.__name__.lower()].append(None)

    # Save results
    os.makedirs(os.path.join(project_root, config['paths']['output_dir'], 'results'), exist_ok=True)
    results_path = os.path.join(project_root, config['paths']['output_dir'], 'results', 'metrics_results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(metric_results, f, indent=2)
    logging.info(f"Saved results to {results_path}")

if __name__ == "__main__":
    main()
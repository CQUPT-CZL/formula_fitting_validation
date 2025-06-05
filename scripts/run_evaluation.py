import sys
import os

from numpy import floating
from src.metrics.MAE import MAE
from src.metrics.RMSE import RMSE
from src.metrics.MSE import MSE
from src.metrics.IAE import IAE
from src.metrics.SC import SC
from src.metrics.LC import LC
from src.metrics.CA import CA
from src.metrics.CodeEq import CodeEq
from src.metrics.base_metric import BaseMetric

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import yaml
import json
import logging
from src.data.loader import parse_variable_names, validate_data, load_json
from src.model.code_executor import safe_exec
# from src.metrics import MAE, RMSE, MSE, SC, LC, CA, IAE, CodeEq
from src.utils.my_logging import setup_logging
from typing import List, Dict, Any, Optional

def evaluate_pair(
    raw_data: List[Dict[str, Any]],
    code: str,
    gt_code: str,
    prompt: str,
    prediction: str,
    metrics: List['BaseMetric'],
    pair_index: int
) -> dict[str, floating[Any] | dict | str] | None:
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

        results = {}
        for metric in metrics:
            metric_name = metric.__class__.__name__.lower()
            if metric_name in ['mse', 'mae', 'rmse'] and valid_samples == 0:
                results[metric_name] = None
                continue
            # print(metric)
            try:
                results[metric_name] = metric.compute(
                    # predictions, ground_truths, raw_data, variable_names
                    predictions = predictions,
                    ground_truths = ground_truths,
                    raw_data = raw_data,
                    variable_names = variable_names,
                    instruction = prompt,
                    prediction = prediction,
                    code = code,
                    gt_code = gt_code,
                    idx = pair_index
                )
            except Exception as e:
                logging.error(f"Error computing {metric.__class__.__name__}: {e}")
            logging.info(f"Evaluation results for pair {pair_index}: {results}")
            # print(results)
            # api 主观测评
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

    # input_path
    input_path = os.path.join(project_root, config['paths']['llm_with_data_code'])
    if not os.path.exists(input_path):
        logging.error(f"Data not found at: {input_path}")
        raise FileNotFoundError(f"Data not found at: {input_path}")

    file_name = os.path.splitext(os.path.basename(input_path))[0]
    logging.info(file_name)  # 输出: 两轮微调0426的*version*结果

    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 动态获取评价指标
    metric_classes = {
        'mae': MAE,
        'rmse': RMSE,
        'mse': MSE,
        'iae': IAE,
        'sc': SC,
        'lc': LC,
        'ca': CA,
        'codeeq': CodeEq,
    }

    try:
        metric_names = config.get('metrics', ['MAE', 'RMSE', 'MSE', 'IAE'])  # 默认值可选
        metrics = [metric_classes[name]() for name in metric_names if name in metric_classes]

    except KeyError as e:
        logging.error(f"Metric class not found for: {e}")
        raise

    metric_results = {metric.__class__.__name__.lower(): [] for metric in metrics}
    for idx, en in enumerate(data):
        if 'code' not in en or not en['code']:
            logging.warning(f"Pair {idx + 1} missing code")
            for metric in metrics:
                metric_results[metric.__class__.__name__.lower()].append(None)
            continue
        result = evaluate_pair(
            raw_data = en['raw_data'],
            code = en['code'],
            gt_code = en['gt_code'],
            prompt = en['prompt'],
            prediction = en['predict'],
            metrics = metrics,
            pair_index = idx + 1
        )
        if result:
            for metric in metrics:
                metric_name = metric.__class__.__name__.lower()
                metric_results[metric_name].append(result[metric_name])
        else:
            for metric in metrics:
                metric_results[metric.__class__.__name__.lower()].append(None)

    # Save results
    os.makedirs(os.path.join(project_root, config['paths']['output_dir']), exist_ok=True)
    results_path = os.path.join(project_root, config['paths']['output_dir'], file_name + '_results.json')

    # 如果已有文件，读取旧结果
    if os.path.exists(results_path):
        with open(results_path, 'r', encoding='utf-8') as f:
            old_results = json.load(f)
    else:
        old_results = {}

    # 合并指标
    # 遍历每个指标（例如 mae、rmse...）
    for metric_name, new_list in metric_results.items():
        if metric_name not in old_results:
            old_results[metric_name] = [None] * len(data)

        for i in range(len(data)):
            if i >= len(old_results[metric_name]):
                old_results[metric_name].append(None)

            old_results[metric_name][i] = new_list[i]
            # 否则保留旧值

    # 保存合并后的结果
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(old_results, f, indent=2, ensure_ascii=False)


    logging.info(f"Saved results to {results_path}")

if __name__ == "__main__":
    main()
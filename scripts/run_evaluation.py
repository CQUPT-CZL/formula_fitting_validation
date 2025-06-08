import sys
import os
import json
import logging
import yaml
from tqdm import tqdm  # 导入tqdm用于显示进度条

from numpy import floating
from src.metrics.MAE import MAE
from src.metrics.RMSE import RMSE
from src.metrics.MSE import MSE
from src.metrics.IAE import IAE
from src.metrics.SC import SC
from src.metrics.LC import LC
from src.metrics.CA import CA
from src.metrics.base_metric import BaseMetric

# 添加项目根目录到sys.path，方便模块导入
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data.loader import parse_variable_names, load_json
from src.model.code_executor import safe_exec
from src.utils.my_logging import setup_logging
from typing import List, Dict, Any

# 评估单个样本对的函数
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

    # 动态执行生成的代码，获取函数
    generated_function = safe_exec(code, {})

    if generated_function:
        predictions = []
        ground_truths = []
        valid_samples = 0

        for sample_idx, sample in enumerate(raw_data):
            try:
                # 检查样本是否包含所有变量和结果
                if not all(var in sample for var in variable_names + ["结果"]):
                    logging.warning(f"Sample {sample_idx + 1} missing required keys")
                    continue
                kwargs = {var: sample[var] for var in variable_names}
                true_result = sample["结果"]
                predicted_result = generated_function(**kwargs)

                # 只保留数值型预测结果
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
            # 计算各项指标
            try:
                results[metric_name] = metric.compute(
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
        return results
    return None

# 加载配置文件
def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

# 处理单个输入文件，进行评估并保存结果
def process_file(input_path, output_dir, metrics):
    file_name = os.path.splitext(os.path.basename(input_path))[0]
    logging.info(f"Processing file: {file_name}")
    data = load_json(input_path)
    total_entries = len(data)
    logging.info(f"Total entries to process: {total_entries}")
    
    metric_results = {metric.__class__.__name__.lower(): [] for metric in metrics}
    
    # 遍历每条数据，进行评估
    for idx, en in enumerate(tqdm(data, desc=f"Processing {file_name}", unit="entry")):
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
        logging.info(f"Processed entry {idx + 1}/{total_entries}")
    
    # 保存评估结果
    results_path = os.path.join(output_dir, file_name + '_results.json')
    
    # 如果已有文件，读取旧结果
    if os.path.exists(results_path):
        with open(results_path, 'r', encoding='utf-8') as f:
            old_results = json.load(f)
    else:
        old_results = {}
    
    # 合并新旧指标结果
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

# 递归查找指定目录下所有json/jsonl文件
def find_json_files(root_dir):
    json_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname.endswith('.json'):
                json_files.append(os.path.join(dirpath, fname))
    return json_files

# 主函数，批量处理所有数据文件
def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    config_path = os.path.join(project_root, 'config', 'config.yaml')
    
    if not os.path.exists(config_path):
        logging.error(f"Config file not found at: {config_path}")
        raise FileNotFoundError(f"Config file not found at: {config_path}")
    
    config = load_config(config_path)
    setup_logging(config)
    
    # 设置输入输出目录
    input_root = os.path.join(project_root, 'data', 'llm_with_data_code', config['paths']['version'])
    output_root = os.path.join(project_root, config['paths']['output_dir'], config['paths']['version'])
    
    # 动态获取评价指标
    metric_classes = {
        'mae': MAE,
        'rmse': RMSE,
        'mse': MSE,
        'iae': IAE,
        'sc': SC,
        'lc': LC,
        'ca': CA,
    }
    
    try:
        # 从配置文件获取需要的指标，若无则用默认值
        metric_names = config.get('metrics', ['mae', 'rmse', 'mse', 'iae'])  # 默认值可选
        metrics = [metric_classes[name.lower()]() for name in metric_names if name.lower() in metric_classes]
    except KeyError as e:
        logging.error(f"Metric class not found for: {e}")
        raise
    
    # 查找所有待处理的json/jsonl文件
    json_files = find_json_files(input_root)
    os.makedirs(output_root, exist_ok=True)
    
    # 依次处理每个文件
    for input_path in json_files:
        # print(input_path)
        # check = input()
        # if check == 'y':
        process_file(input_path, output_root, metrics)
        # else:
        #     continue

if __name__ == "__main__":
    main()
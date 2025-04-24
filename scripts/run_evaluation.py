import sys
import os

from src.metrics.base_metric import BaseMetric

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import yaml
import json
import logging
from src.data.loader import load_jsonl, load_json, load_prompt, parse_variable_names, validate_data
from src.model.llm_interface import LLMInterface
from src.model.code_executor import extract_code_block, safe_exec, load_saved_code
from src.metrics.point_metrics import MAE
from src.utils.my_logging import setup_logging
from typing import List, Dict, Any, Optional


def evaluate_pair(
        analysis_text: str,
        raw_data: List[Dict[str, Any]],
        llm_interface: 'LLMInterface',
        prompt_template: str,
        metrics: List['BaseMetric'],
        code_dir: str,
        pair_index: int
) -> Optional[Dict[str, Any]]:
    variable_names = parse_variable_names(raw_data)
    logging.info(f"Evaluating pair {pair_index} with variables: {variable_names}")

    # Check for saved code
    code_file = os.path.join(code_dir, f"code_{pair_index}.py")
    generated_function = load_saved_code(code_file)

    # Generate code if not found
    if not generated_function:
        try:
            llm_output = llm_interface.generate_code(prompt_template, analysis_text)
            code = extract_code_block(llm_output)
            if code:
                # Save code
                os.makedirs(code_dir, exist_ok=True)
                with open(code_file, 'w', encoding='utf-8') as f:
                    f.write(code)
                logging.info(f"Saved code to {code_file}")
                generated_function = safe_exec(code, {})
        except Exception as e:
            logging.error(f"Failed to generate or execute code for pair {pair_index}: {e}")
            return None

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
            results = {"valid_samples": valid_samples, "variables": variable_names}
            for metric in metrics:
                try:
                    results[metric.__class__.__name__.lower()] = metric.compute(
                        predictions, ground_truths, raw_data, variable_names
                    )
                except Exception as e:
                    logging.error(f"Error computing {metric.__class__.__name__}: {e}")
            logging.info(f"Evaluation results for pair {pair_index}: {results}")
            return results
    return None


def save_code_index(code_dir: str, index_data: List[Dict]):
    index_file = os.path.join(code_dir, "index.json")
    with open(index_file, 'w', encoding='utf-8') as f:
        json.dump(index_data, f, indent=2)
    logging.info(f"Saved code index to {index_file}")


def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    print(f"Project root: {project_root}")
    config_path = os.path.join(project_root, 'config', 'config.yaml')
    # config_path = r'/Users/cuiziliang/Projects/formula_fitting_validation/config/config.yaml'
    if not os.path.exists(config_path):
        logging.error(f"Config file not found at: {config_path}")
        raise FileNotFoundError(f"Config file not found at: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    llm_output_path = os.path.join(project_root, config['paths']['llm_output'])
    print(f"Looking for pred.jsonl at: {llm_output_path}")
    print(f"Looking for test.json at: {os.path.join(project_root, config['paths']['raw_data'])}")
    print(f"Looking for prompt.txt at: {os.path.join(project_root, config['paths']['prompt'])}")

    setup_logging(config)
    analysis_texts = load_jsonl(config['paths']['llm_output'])
    raw_datasets = load_json(config['paths']['raw_data'])
    prompt_template = load_prompt(config['paths']['prompt'])
    validate_data(analysis_texts, raw_datasets)

    llm_interface = LLMInterface(config)
    metrics = [MAE()]
    code_dir = os.path.join(project_root, config['paths']['generated_code_dir'])

    results = []
    code_index = []
    for idx, (analysis_text, dataset) in enumerate(zip(analysis_texts, raw_datasets)):
        result = evaluate_pair(
            analysis_text['predict'],
            dataset['raw_data'],
            llm_interface,
            prompt_template,
            metrics,
            code_dir,
            idx + 1
        )
        if result:
            results.append(result)
            code_index.append({
                "pair_index": idx + 1,
                "analysis_text": analysis_text['predict'],
                "code_file": f"code_{idx + 1}.py"
            })

    # Save results and code index
    os.makedirs(os.path.join(config['paths']['output_dir'], 'results'), exist_ok=True)
    results_path = os.path.join(config['paths']['output_dir'], 'results', 'metrics_results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    logging.info(f"Saved results to {results_path}")

    save_code_index(code_dir, code_index)


if __name__ == "__main__":
    main()
import yaml
import os
import json
import logging
from src.data.loader import load_jsonl, load_json, load_prompt, parse_variable_names, validate_data
from src.metrics.base_metric import BaseMetric
from src.model.llm_interface import LLMInterface
from src.model.code_executor import extract_code_block, safe_exec
from src.metrics.point_metrics import MAE
from src.utils.my_logging import setup_logging
from typing import List, Dict, Any, Optional


def evaluate_pair(
        analysis_text: str,
        raw_data: List[Dict[str, Any]],
        llm_interface: 'LLMInterface',
        prompt_template: str,
        metrics: List['BaseMetric']
) -> Optional[Dict[str, Any]]:
    """Evaluate a single pair of analysis text and raw data."""
    variable_names = parse_variable_names(raw_data)
    logging.info(f"Evaluating pair with variables: {variable_names}")

    generated_function = None
    try:
        llm_output = llm_interface.generate_code(prompt_template, analysis_text)
        code = extract_code_block(llm_output)
        if code:
            generated_function = safe_exec(code, {})
    except Exception as e:
        logging.error(f"Failed to generate or execute code: {e}")
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
            logging.info(f"Evaluation results: {results}")
            return results
    return None


def main(config_path: str):
    """Run MAE evaluation."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    setup_logging(config)

    analysis_texts = load_jsonl(config['paths']['llm_output'])
    raw_datasets = load_json(config['paths']['raw_data'])
    prompt_template = load_prompt(config['paths']['prompt'])
    validate_data(analysis_texts, raw_datasets)

    llm_interface = LLMInterface(config)
    metrics = [MAE()]  # Only MAE

    results = []
    for idx, (analysis_text, dataset) in enumerate(zip(analysis_texts, raw_datasets)):
        logging.info(f"Processing pair {idx + 1}/{len(analysis_texts)}")
        result = evaluate_pair(
            analysis_text['predict'],
            dataset['raw_data'],
            llm_interface,
            prompt_template,
            metrics
        )
        if result:
            results.append(result)

    os.makedirs(os.path.join(config['paths']['output_dir'], 'results'), exist_ok=True)
    results_path = os.path.join(config['paths']['output_dir'], 'results', 'metrics_results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    logging.info(f"Saved results to {results_path}")


if __name__ == "__main__":
    main('../config/config.yaml')
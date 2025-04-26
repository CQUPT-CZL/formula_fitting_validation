import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import logging
from src.data.loader import load_json, load_prompt
from src.model.llm_interface import LLMInterface
from src.model.code_executor import extract_code_block
from src.utils.my_logging import setup_logging
import yaml


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def generate_code_for_entry(analysis_text: str, llm_interface: 'LLMInterface', prompt_template: str) -> str:
    try:
        llm_output = llm_interface.generate_code(prompt_template, analysis_text)
        code = extract_code_block(llm_output)
        if code:
            # logging.info(f"Generated code for analysis: {analysis_text[:50]}...")
            return code
        logging.error("No Python code block found")
        return ""
    except Exception as e:
        logging.error(f"Error generating code: {e}")
        return ""


def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    config_path = os.path.join(project_root, 'config', 'config.yaml')

    if not os.path.exists(config_path):
        logging.error(f"Config file not found at: {config_path}")
        raise FileNotFoundError(f"Config file not found at: {config_path}")

    config = load_config(config_path)
    setup_logging(config)

    # Load input JSONL
    input_path = os.path.join(project_root, config['paths']['llm_with_data'])
    if not os.path.exists(input_path):
        logging.error(f"Input JSONL not found at: {input_path}")
        raise FileNotFoundError(f"Input JSONL not found at: {input_path}")

    file_name = os.path.splitext(os.path.basename(input_path))[0]
    logging.info(file_name)  # 输出: 两轮微调0426的结果

    data = load_json(input_path)

    # Load prompt
    prompt_template = load_prompt(config['paths']['prompt'])
    llm_interface = LLMInterface(config)

    # Generate code
    code_dir = os.path.join(project_root, config['paths']['generated_code_dir'])
    os.makedirs(code_dir, exist_ok=True)

    for idx, entry in enumerate(data):
        if 'predict' not in entry:
            logging.warning(f"Entry {idx + 1} missing 'pred' field")
            entry['code'] = ""
            continue
        code = generate_code_for_entry(entry['predict'], llm_interface, prompt_template)
        entry['code'] = code

    # Save updated JSON
    output_path = os.path.join(project_root, config['paths']['generated_code_dir'], file_name + '_code.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logging.info(f"Saved updated JSON to {output_path}")


if __name__ == "__main__":
    main()
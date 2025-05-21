import sys
import os
import json
import logging
import yaml
from tqdm import tqdm  # 导入tqdm用于显示进度条

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data.loader import load_json, load_prompt
from src.model.llm_interface import LLMInterface
from src.model.code_executor import extract_code_block
from src.utils.my_logging import setup_logging


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def generate_code_for_entry(analysis_text: str, llm_interface: 'LLMInterface', prompt_template: str) -> str:
    max_attempts = 3
    last_output = ""

    for attempt in range(max_attempts):
        try:
            llm_output = llm_interface.generate_code(prompt_template, analysis_text)
            code = extract_code_block(llm_output)
            if code:
                logging.info(f"Generated code for analysis (attempt {attempt + 1})")
                return code
            logging.warning(f"No Python code block found in attempt {attempt + 1}")
            last_output = llm_output
        except Exception as e:
            logging.error(f"Error generating code in attempt {attempt + 1}: {e}")
            last_output = str(e)

        if attempt < max_attempts - 1:
            logging.info(f"Retrying... ({attempt + 2}/{max_attempts})")

    logging.error(f"Failed to generate valid code after {max_attempts} attempts")
    return last_output


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
    logging.info(f"Processing file: {file_name}")

    data = load_json(input_path)
    total_entries = len(data)
    logging.info(f"Total entries to process: {total_entries}")

    # Load prompt
    prompt_template = load_prompt(config['paths']['prompt'])
    llm_interface = LLMInterface(config)

    # Generate code with progress bar
    code_dir = os.path.join(project_root, config['paths']['generated_code_dir'])
    os.makedirs(code_dir, exist_ok=True)

    # 使用tqdm显示进度条
    for idx, entry in enumerate(tqdm(data, desc="Processing entries", unit="entry")):
        if 'predict' not in entry:
            logging.warning(f"Entry {idx + 1} missing 'predict' field")
            entry['code'] = ""
            continue

        # 生成预测的代码
        code = generate_code_for_entry(entry['predict'], llm_interface, prompt_template)
        entry['code'] = code

        # 生成gt的代码
        code = generate_code_for_entry(entry['label'], llm_interface, prompt_template)
        entry['gt_code'] = code
        # 记录进度
        logging.info(f"Processed entry {idx + 1}/{total_entries}")

    # Save updated JSON
    output_path = os.path.join(project_root, config['paths']['generated_code_dir'], file_name + '_code.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logging.info(f"Saved updated JSON to {output_path}")


if __name__ == "__main__":
    main()
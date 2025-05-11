import sys
import os
import re
import json
import logging
import yaml
from typing import Callable, Optional, List, Dict
from tqdm import tqdm  # 导入tqdm用于显示进度条

# 假设这些模块存在于项目中
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data.loader import load_jsonl, load_prompt
from src.model.llm_interface import LLMInterface
from src.utils.my_logging import setup_logging


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def generate_api_pred_for_entry(llm_interface: 'LLMInterface', prompt: str) -> str | List[
    Dict]:
    max_attempts = 3
    last_output = ""

    for attempt in range(max_attempts):
        try:
            pred = llm_interface.generate_pred(prompt)
            if pred:
                logging.info(f"Generated pred for analysis (attempt {attempt + 1})")
                return pred
            logging.warning(f"No pred block found in attempt {attempt + 1}")
            last_output = pred
        except Exception as e:
            logging.error(f"Error generating raw_data in attempt {attempt + 1}: {e}")
            last_output = str(e)

        if attempt < max_attempts - 1:
            logging.info(f"Retrying... ({attempt + 2}/{max_attempts})")

    logging.error(f"Failed to generate valid pred after {max_attempts} attempts")
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
    input_path = os.path.join(project_root, config['paths']['raw_llm'])
    if not os.path.exists(input_path):
        logging.error(f"Input JSONL not found at: {input_path}")
        raise FileNotFoundError(f"Input JSONL not found at: {input_path}")

    file_name = os.path.splitext(os.path.basename(input_path))[0]
    logging.info(f"Processing file: {file_name}")

    data = load_jsonl(input_path)
    total_entries = len(data)
    logging.info(f"Total entries to process: {total_entries}")

    llm_interface = LLMInterface(config)

    # 使用tqdm显示进度条
    cnt = 0
    for idx, entry in enumerate(tqdm(data, desc="Processing entries", unit="entry")):
        pred = generate_api_pred_for_entry(llm_interface, prompt=entry['prompt'])
        entry['predict'] = pred
        # 记录进度
        logging.info(f"Processed entry {idx + 1}/{total_entries}")
        # print(entry)
        cnt += 1
        if cnt > 20:
            break

    # Save updated JSON
    output_path = os.path.join(project_root, config['paths']['generated_api_pred_dir'], 'ds_api_gen_pred.jsonl')
    # 写入JSONL文件
    with open(output_path, 'w', encoding='utf-8') as output_file:
        for item in data:
            # 将每个字典转换为JSON字符串，并添加换行符
            json_line = json.dumps(item, ensure_ascii=False) + '\n'
            output_file.write(json_line)

    logging.info(f"Saved updated JSON to {output_path}")


if __name__ == "__main__":
    main()
import sys
import os
import json
import logging
import yaml
from tqdm import tqdm  # 导入tqdm用于显示进度条

# 添加项目根目录到sys.path，方便模块导入
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data.loader import load_json, load_prompt
from src.model.llm_interface import LLMInterface
from src.model.code_executor import extract_code_block
from src.utils.my_logging import setup_logging

# 加载配置文件
def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

# 针对单条数据生成代码，最多尝试3次
def generate_code_for_entry(analysis_text: str, llm_interface: 'LLMInterface', prompt_template: str) -> str:
    max_attempts = 3
    last_output = ""
    for attempt in range(max_attempts):
        try:
            # 调用大模型接口生成代码
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

# 处理单个输入文件，生成代码并保存结果
def process_file(input_path, output_dir, llm_interface, prompt_template):
    file_name = os.path.splitext(os.path.basename(input_path))[0]
    logging.info(f"Processing file: {file_name}")
    data = load_json(input_path)
    total_entries = len(data)
    logging.info(f"Total entries to process: {total_entries}")
    # 遍历每条数据，生成代码
    for idx, entry in enumerate(tqdm(data, desc=f"Processing {file_name}", unit="entry")):
        if 'predict' not in entry:
            logging.warning(f"Entry {idx + 1} missing 'predict' field")
            entry['code'] = ""
            continue
        # 生成预测的代码
        code = generate_code_for_entry(entry['predict'], llm_interface, prompt_template)
        entry['code'] = code
        logging.info(f"Processed entry {idx + 1}/{total_entries}")
    # 保存带代码的新json文件
    output_path = os.path.join(output_dir, file_name + '_code.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logging.info(f"Saved updated JSON to {output_path}")

# 递归查找指定目录下所有json/jsonl文件
def find_json_files(root_dir):
    json_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname.endswith('.json') or fname.endswith('.jsonl'):
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
    input_root = os.path.join(project_root, 'data', 'llm_with_data', config['paths']['version'])
    output_root = os.path.join(project_root, 'data', 'llm_with_data_code', config['paths']['version'])
    prompt_template = load_prompt(config['paths']['prompt_code'])
    llm_interface = LLMInterface(config)
    # 查找所有待处理的json/jsonl文件
    json_files = find_json_files(input_root)
    os.makedirs(output_root, exist_ok=True)
    # 依次处理每个文件
    for input_path in json_files:
        # print(input_path)
        # check = input()
        # if check == 'y':
        process_file(input_path, output_root, llm_interface, prompt_template)
        # else:
        #     continue

if __name__ == "__main__":
    main()
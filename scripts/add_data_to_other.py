import json
import os
import yaml


def add_data_to_jsonl(json_file_path, jsonl_file_path, output_dir):
    """
    将JSON文件中的数据添加到JSONL文件中，并保存为新的JSON文件
    
    参数:
    json_file_path: JSON文件路径，包含raw_data和gt_code数据
    jsonl_file_path: JSONL文件路径，需要添加数据的文件
    output_dir: 输出目录
    """
    # 读取 JSON 文件
    with open(json_file_path, "r", encoding='utf-8') as f:
        json_data = json.load(f)  # 解析为列表，每个元素是 {"raw_data": "..."}

    # 读取 JSON Lines 文件并修改
    new_json = []
    with open(jsonl_file_path, "r", encoding='utf-8') as f_in:
        for i, obj in enumerate(f_in):
            # 解析 JSONL 行
            obj = json.loads(obj)
            # 获取对应的 raw_data（确保 JSON 文件和 JSONL 文件行数相同）
            if i < len(json_data):
                raw_data = json_data[i].get("raw_data", "")
                obj["raw_data"] = raw_data  # 添加 raw_data 字段
                gt_code = json_data[i].get("gt_code", "")
                obj["gt_code"] = gt_code  # 添加 gt_code 字段
            new_json.append(obj)

    # 生成输出文件名：基于jsonl文件名添加后缀
    jsonl_filename = os.path.basename(jsonl_file_path)
    name_without_ext = os.path.splitext(jsonl_filename)[0]
    output_filename = f"{name_without_ext}_with_data.json"
    output_path = os.path.join(output_dir, output_filename)
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 写入新文件
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(new_json, f, indent=2, ensure_ascii=False)

    print(f"处理完成！结果已保存到: {output_path}")
    return output_path


# 示例用法
if __name__ == "__main__":

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    config_path = os.path.join(project_root, 'config', 'config.yaml')

    # 导入yaml配置文件读取函数
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 设置输入输出目录
    input_root = os.path.join(project_root, 'data', 'raw_llm', config['paths']['version'])
    output_directory = os.path.join(project_root, 'data', 'llm_with_data', config['paths']['version'])
    
    # JSON文件路径（包含raw_data和gt_code）
    json_file = os.path.join(project_root, 'data', 'llm_with_data_code', 'v3', '1.5b_no_sft_with_data_code.json')
    
    # 检查input_root目录是否存在
    if not os.path.exists(input_root):
        print(f"❌ 输入目录不存在: {input_root}")
        exit(1)
    
    # 检查JSON文件是否存在
    if not os.path.exists(json_file):
        print(f"❌ JSON文件不存在: {json_file}")
        exit(1)
    
    # 遍历input_root目录中的所有jsonl文件
    jsonl_files = [f for f in os.listdir(input_root) if f.endswith('.jsonl')]
    
    if not jsonl_files:
        print(f"⚠️ 在目录 {input_root} 中没有找到任何 .jsonl 文件")
        exit(1)
    
    print(f"🔍 找到 {len(jsonl_files)} 个 JSONL 文件需要处理")
    
    # 处理每个jsonl文件
    for jsonl_filename in jsonl_files:
        jsonl_file_path = os.path.join(input_root, jsonl_filename)
        print(f"\n📝 正在处理: {jsonl_filename}")
        
        try:
            output_file = add_data_to_jsonl(json_file, jsonl_file_path, output_directory)
            print(f"✅ 成功处理: {jsonl_filename} -> {os.path.basename(output_file)}")
        except Exception as e:
            print(f"❌ 处理 {jsonl_filename} 时出错: {str(e)}")
    
    print(f"\n🎉 所有文件处理完成！输出目录: {output_directory}")
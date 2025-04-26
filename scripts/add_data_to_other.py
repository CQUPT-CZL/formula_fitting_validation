import json

# 读取 JSON 文件
with open("../data/llm_with_data/两轮微调0426的结果_with_data.json", "r") as f:
    json_data = json.load(f)  # 解析为列表，每个元素是 {"raw_data": "..."}

# 读取 JSON Lines 文件并修改
new_json = []
with open("../data/raw_llm/0微调qwen结果.jsonl", "r") as f_in:
    for i, line in enumerate(f_in):
        # 解析 JSONL 行
        jsonl_obj = json.loads(line.strip())

        # 获取对应的 raw_data（确保 JSON 文件和 JSONL 文件行数相同）
        if i < len(json_data):
            raw_data = json_data[i].get("raw_data", "")
            jsonl_obj["raw_data"] = raw_data  # 添加 raw_data 字段
        new_json.append(jsonl_obj)

        # 写入新文件

output_path = '../data/llm_with_data/0微调qwen结果_with_data.json'
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(new_json, f, indent=2, ensure_ascii=False)


print("处理完成！结果已保存")
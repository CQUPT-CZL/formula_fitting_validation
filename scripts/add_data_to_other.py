import json

# 读取 JSON 文件
with open("../data/llm_with_data/v2/1.5b_all_sft_with_data.json", "r", encoding='utf-8') as f:
    json_data = json.load(f)  # 解析为列表，每个元素是 {"raw_data": "..."}

# 读取 JSON Lines 文件并修改
new_json = []
with open("../data/raw_llm/v2/1.5b_all_sft.jsonl", "r", encoding='utf-8') as f_in:
    # json_data_1 = json.load(f_in)
    for i, obj in enumerate(f_in):
        # 解析 JSONL 行
        obj = json.loads(obj)
        # 获取对应的 raw_data（确保 JSON 文件和 JSONL 文件行数相同）
        if i < len(json_data):
            raw_data = json_data[i].get("raw_data", "")
            obj["raw_data"] = raw_data  # 添加 raw_data 字段
        new_json.append(obj)

        # 写入新文件

output_path = '../data/llm_with_data/v2/1.5b_all_sft_with_data.json'
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(new_json, f, indent=2, ensure_ascii=False)

print("处理完成！结果已保存")
import json

# 读取 JSON 文件
with open("../data/raw_data/raw_data_test.json", "r") as f:
    json_data = json.load(f)  # 解析为列表，每个元素是 {"raw_data": "..."}

# 读取 JSON Lines 文件并修改
with open("../data/llm_output/pred.jsonl", "r") as f_in, \
        open("../data/llm_output/pred_with_raw.jsonl", "w") as f_out:
    for i, line in enumerate(f_in):
        # 解析 JSONL 行
        jsonl_obj = json.loads(line.strip())

        # 获取对应的 raw_data（确保 JSON 文件和 JSONL 文件行数相同）
        if i < len(json_data):
            raw_data = json_data[i].get("raw_data", "")
            jsonl_obj["raw_data"] = raw_data  # 添加 raw_data 字段

        # 写入新文件
        f_out.write(json.dumps(jsonl_obj) + "\n")

print("处理完成！结果已保存")
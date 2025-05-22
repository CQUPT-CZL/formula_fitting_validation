import json

file = "../data/llm_with_data_code/v3/1.5b_all_sft_with_data_code.json"

# 读取 JSON Lines 文件并修改
with open(file, 'r', encoding='utf-8') as f:
    data = json.load(f)

for item in data:
    print('----->>')
    print('gt_code:')
    print(item['gt_code'])
    print('code:')
    print(item['code'])
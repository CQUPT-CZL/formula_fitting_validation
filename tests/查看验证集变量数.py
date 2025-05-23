import json

file = '../data/llm_with_data_code/v3/1.5b_no_sft_with_data_code.json'

with open(file, encoding='utf-8') as f:
    data = json.load(f)

for item in data:
    print(len(item['raw_data'][0]) - 1)
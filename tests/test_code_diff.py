import json
import inspect
import random
import numpy as np

file = r'../data/llm_with_data_code/v3/1.5b_no_sft_with_data_code.json'

with open(file, encoding='utf-8') as f:
    data = json.load(f)

def get_func_from_code(code_str):
    # 在 exec 中导入需要的库（math、numpy等）
    local_vars = {"math": __import__("math"), "np": np}
    exec(code_str, {}, local_vars)
    return local_vars.get("calculate_value")

def generate_inputs(func, n_samples=20, input_range=(-20, 20)):
    sig = inspect.signature(func)
    param_count = len(sig.parameters)
    inputs = []
    for _ in range(n_samples):
        args = tuple(random.randint(*input_range) for _ in range(param_count))
        # print(args)
        inputs.append(args)
    return inputs

def compare_pair(code_str, gt_code_str, n_samples=20):
    try:
        func_pred = get_func_from_code(code_str)
        func_gt = get_func_from_code(gt_code_str)
        inputs = generate_inputs(func_gt, n_samples)

        diffs = []
        for args in inputs:
            try:
                y_pred = func_pred(*args)
                y_true = func_gt(*args)
                diffs.append(abs(y_pred - y_true))
            except Exception as e:
                # 不可比较，记录为无穷大误差
                print(e)
                # diffs.append(float("inf"))

        diffs = np.array(diffs)
        return {
            "mae": np.mean(diffs),
            "max_error": np.max(diffs),
            "mse": np.mean(diffs ** 2),
            "valid_count": np.sum(np.isfinite(diffs))
        }
    except Exception as e:
        return {"error": str(e)}

def batch_compare(pairs, n_samples=20):
    results = []
    for i, (code, gt_code) in enumerate(pairs):
        if i < 49:
            continue
        result = compare_pair(code, gt_code, n_samples)

        results.append(result)
    return results

results = []
i = 0
for item in data:
    # if i < 49:
    #     i += 1
    #     continue

    result = compare_pair(item['code'], item['gt_code'])
    results.append(result)

print(len(results))
import pandas as pd
df = pd.DataFrame(results)
print(df.head())

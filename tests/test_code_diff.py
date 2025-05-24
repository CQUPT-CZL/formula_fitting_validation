import json
import inspect
import random
import numpy as np
import os
from src.metrics.integral_metrics import IAE
import pandas as pd



# === 主逻辑：遍历目录下所有 json 文件，计算每个的平均 MAE ===
def evaluate_folder(folder_path):
    results_summary = []

    for file_name in os.listdir(folder_path):
        print(file_name)
        if file_name.endswith(".json"):
            full_path = os.path.join(folder_path, file_name)
            try:
                with open(full_path, encoding='utf-8') as f:
                    data = json.load(f)
                maes = []
                for item in data:
                    mae = IAE().compute(item['code'], item['gt_code'])
                    if mae is not None:
                        maes.append(np.round(mae, 4))
                    else:
                        print('0---')
                        maes.append(None)
                print(len(maes), maes)
                file_name = '../output/results/v3/' + file_name.replace('.json', "_results.json")
                with open (file_name, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                data['iae'] = maes

                with open(file_name, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)

                # results_summary.append({
                #     "file": file_name,
                #     "mae_mean": mean_mae,
                #     "valid_samples": len(maes),
                #     "total_samples": len(data)
                # })
            except Exception as e:
                # print(e)
                results_summary.append({
                    "file": file_name,
                    "mae_mean": None,
                    "error": str(e)
                })
        # break

    df_result = pd.DataFrame(results_summary)
    return df_result

# === 调用示例（修改成你的文件夹路径） ===
folder = "../data/llm_with_data_code/v3"  # 你要评估的文件夹路径
df_summary = evaluate_folder(folder)

# === 可选：保存为 CSV ===
# df_summary.to_csv("model_mae_summary.csv", index=False)

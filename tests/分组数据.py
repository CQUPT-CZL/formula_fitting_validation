import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
import logging
import os

file = '../data/llm_with_data_code/v3/1.5b_no_sft_with_data_code.json'

with open(file, encoding='utf-8') as f:
    data = json.load(f)

# 提取每个样本的变量数
var_nums = [len(item['raw_data'][0]) - 1 for item in data]


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
config_path = os.path.join(project_root, 'config', 'config.yaml')

if not os.path.exists(config_path):
    logging.error(f"Config file not found at: {config_path}")
    raise FileNotFoundError(f"Config file not found at: {config_path}")

config = load_config(config_path)

# 设置文件夹路径
folder_path = os.path.join(project_root, config['paths']['output_dir'])
output_dir = os.path.join(folder_path, "plots")
os.makedirs(output_dir, exist_ok=True)

# 你所有要画的指标列表（可以自行添加）
metrics_list = ['mse', 'mae', 'rmse', 'SC', 'LC', 'CA', 'CodeEq']

# 读取所有 JSON 文件
model_metrics = {}
for file_name in os.listdir(folder_path):
    if file_name.endswith(".json"):
        file_path = os.path.join(folder_path, file_name)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            model_name = Path(file_name).stem
            # 强调文件名的后缀
            suffix = "_with_data_code_results"
            model_name = model_name.replace(suffix, "")
            model_metrics[model_name] = {str(k): {metric: [] for metric in metrics_list} for k in [2, 3, 4]}

            # 采集各指标
            for metric in metrics_list:
                if metric in ['SC', 'LC', 'CA', 'CodeEq']:
                    # 从 api 部分抽取
                    if 'api' in data:
                        for i, item in enumerate(data['api']):
                            if isinstance(item, dict) and metric in item:
                                score = item[metric].get('score', None)
                            else:
                                score = None
                            if i < len(var_nums):
                                model_metrics[model_name][str(var_nums[i])][metric].append(score)
                else:
                    if metric in data:
                        for i, item_val in enumerate(data[metric]):
                            if i < len(var_nums):
                                model_metrics[model_name][str(var_nums[i])][metric].append(item_val)
        except json.JSONDecodeError as e:
            print(f"Error: Failed to parse {file_name}: {e}")
            continue


# 异常值移除 + 平均值计算
def remove_outliers_and_average(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    q1, q3 = np.percentile(values, 25), np.percentile(values, 75)
    iqr = q3 - q1
    filtered = [v for v in values if q1 - 1.5 * iqr <= v <= q3 + 1.5 * iqr]
    return np.mean(filtered) if filtered else None


# 依次为每个指标作图
for metric in metrics_list:
    # 构造 DataFrame：行为模型，列为变量数（2/3/4），值为均值
    results = {}
    for model, tasks in model_metrics.items():
        results[model] = {}
        for task, metrics in tasks.items():
            vals = metrics.get(metric, [])
            avg = remove_outliers_and_average(vals)
            results[model][task] = avg
    df = pd.DataFrame(results).T
    # 只保留有数据的列
    df = df.loc[:, (df.notnull().any())]
    # 可视化
    ax = df.plot(kind='bar', figsize=(10, 6))
    ax.set_title(f"Average {metric.upper()} per Variable Count (Outliers Removed)")
    ax.set_ylabel(metric.upper())
    ax.set_xlabel("Model")
    plt.xticks(rotation=45, ha='right')
    plt.legend(title="Variable Count")
    plt.tight_layout()
    # plt.savefig(os.path.join(output_dir, f"{metric}_barplot_by_var_count.png"))
    plt.show()
    plt.close()


print(f"All plots saved to {output_dir}")


def df_to_latex_booktabs(df, caption="Model metrics comparison", label="tab:metrics"):
    colnames = " & ".join([str(col) for col in df.columns])
    header = "\\toprule\nModel & " + colnames + " \\\\\n\\midrule"
    rows = []
    for idx, row in df.iterrows():
        row_str = " & ".join(f"{v:.2f}" if pd.notnull(v) else "---" for v in row)
        rows.append(f"{idx} & {row_str} \\\\")
    table = (
        "\\begin{table}[ht]\n"
        "\\centering\n"
        "\\begin{tabular}{l" + "c" * len(df.columns) + "}\n"
        f"{header}\n"
        + "\n".join(rows) +
        "\n\\bottomrule\n"
        "\\end{tabular}\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        "\\end{table}"
    )
    return table


results_full = {}
for model, tasks in model_metrics.items():
    results_full[model] = {}
    for task, metrics in tasks.items():  # task: '2', '3', '4'
        for metric in metrics_list:
            key = f"{metric.upper()}-{task}"
            vals = metrics.get(metric, [])
            avg = remove_outliers_and_average(vals)
            results_full[model][key] = avg

df_full = pd.DataFrame(results_full).T

latex_code = df_to_latex_booktabs(df_full)
with open(os.path.join(output_dir, "all_metrics_table.tex"), "w", encoding="utf-8") as f:
    f.write(latex_code)

print("Latex三线表代码已保存：", os.path.join(output_dir, "all_metrics_table.tex"))
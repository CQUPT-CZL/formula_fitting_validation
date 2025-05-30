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

# 要的指标列表
metrics_list = ['mse', 'mae', 'rmse', 'sc', 'lc', 'ca', 'iae']

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
            model_metrics[model_name] = {str(k): {metric: [] for metric in metrics_list} for k in [2, 3, 4, 'all']}

            # 采集各指标
            for metric in metrics_list:
                if metric in ['sc', 'lc', 'ca', 'CodeEq']:
                    # 从 api 部分抽取
                    for i, item in enumerate(data[metric]):
                        if isinstance(item, dict):
                            score = item.get('score', None)
                        else:
                            score = None
                            continue
                        if i < len(var_nums):
                            model_metrics[model_name][str(var_nums[i])][metric].append(score)
                            model_metrics[model_name]['all'][metric].append(score)
                else:
                    if metric in data:
                        for i, item_val in enumerate(data[metric]):
                            if i < len(var_nums):
                                model_metrics[model_name][str(var_nums[i])][metric].append(item_val)
                                model_metrics[model_name]['all'][metric].append(item_val)
        except json.JSONDecodeError as e:
            print(f"Error: Failed to parse {file_name}: {e}")
            continue


print(model_metrics)

# 异常值移除 + 平均值计算
def remove_outliers_and_average(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    q1, q3 = np.percentile(values, 25), np.percentile(values, 75)
    iqr = q3 - q1
    filtered = [v for v in values if q1 - 1.5 * iqr <= v <= q3 + 1.5 * iqr]
    return np.mean(filtered) if filtered else None


def escape_latex(s):
    return str(s).replace('_', r'\_')

# 指标是否是“越小越好”
minimize_metrics = {'mse', 'mae', 'rmse', 'iae'}

def df_to_latex_booktabs_single_metric_highlighted(df, metric, caption_prefix="Model performance on", label_prefix="tab:"):
    colnames = " & ".join([str(col) for col in df.columns])
    header = "\\toprule\nModel & " + colnames + " \\\\\n\\midrule"
    rows = []

    # 每列找最优值和次优值（跳过 None）
    col_best_idx = {}
    col_second_idx = {}
    for col in df.columns:
        col_vals = df[col].dropna()
        if col_vals.empty:
            continue
        sorted_vals = col_vals.sort_values(ascending=(metric.lower() in minimize_metrics))
        best_val = sorted_vals.iloc[0]
        second_val = sorted_vals.iloc[1] if len(sorted_vals) > 1 else None
        col_best_idx[col] = (col_vals == best_val)
        col_second_idx[col] = (col_vals == second_val) if second_val is not None else pd.Series(False, index=col_vals.index)

    for idx, row in df.iterrows():
        row_name = escape_latex(idx)
        row_cells = []
        for col in df.columns:
            val = row[col]
            if pd.isnull(val):
                cell = "---"
            else:
                is_best = col_best_idx[col].get(idx, False)
                is_second = col_second_idx[col].get(idx, False)
                formatted = f"{val:.2f}"
                if is_best:
                    formatted = f"\\textbf{{{formatted}}}"
                elif is_second:
                    formatted = f"\\underline{{{formatted}}}"
                cell = formatted
            row_cells.append(cell)
        rows.append(f"{row_name} & {' & '.join(row_cells)} \\\\")

    table = (
        "\\begin{table}[ht]\n"
        "\\centering\n"
        "\\begin{tabular}{l" + "c" * len(df.columns) + "}\n"
        f"{header}\n"
        + "\n".join(rows) +
        "\n\\bottomrule\n"
        "\\end{tabular}\n"
        f"\\caption{{{caption_prefix} {metric.upper()}}}\n"
        f"\\label{{{label_prefix}{metric.lower()}}}\n"
        "\\end{table}\n\n"
    )
    return table



# 汇总写入一个 .tex 文件
all_tables = ""

for metric in metrics_list:
    results = {}
    for model, tasks in model_metrics.items():
        results[model] = {}
        for var_count in ['2', '3', '4', 'all']:
            vals = tasks.get(var_count, {}).get(metric, [])
            avg = remove_outliers_and_average(vals)
            results[model][var_count] = avg

    df_metric = pd.DataFrame(results).T
    # 添加 "All" 列，表示总体平均
    # df_metric['All'] = df_metric.mean(axis=1)
    #
    # # 保留有数据的列（包括 All）
    # df_metric = df_metric.loc[:, (df_metric.notnull().any())]

    all_tables += df_to_latex_booktabs_single_metric_highlighted(df_metric, metric)


# 最终写入一个文件
tex_path = os.path.join(output_dir, "all_metrics_table.tex")
with open(tex_path, "w", encoding="utf-8") as f:
    f.write(all_tables)

print(f"LaTeX 三线表合集已保存：{tex_path}")

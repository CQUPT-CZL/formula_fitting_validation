import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 设置文件夹路径（请替换为你的 JSON 文件夹路径）
folder_path = r'/Users/cuiziliang/Projects/formula_fitting_validation/output/results'
output_dir = os.path.join(folder_path, "plots")
os.makedirs(output_dir, exist_ok=True)

## 函数：清理 None 值
def clean_data(data):
    return [x for x in data if x is not None and not np.isnan(x)]

# 函数：使用 IQR 方法移除异常值
def remove_outliers(data):
    cleaned_data = clean_data(data)
    if len(cleaned_data) < 4:  # 需要足够数据计算四分位数
        print(f"Warning: Insufficient valid data ({len(cleaned_data)} values) for outlier removal.")
        return cleaned_data
    q1 = np.percentile(cleaned_data, 25)
    q3 = np.percentile(cleaned_data, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return [x for x in cleaned_data if lower_bound <= x <= upper_bound]

# 读取所有 JSON 文件
model_metrics = {}
for file_name in os.listdir(folder_path):
    if file_name.endswith(".json"):
        file_path = os.path.join(folder_path, file_name)
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            model_name = Path(file_name).stem  # 使用文件名（无扩展名）作为模型名
            model_metrics[model_name] = data
        except json.JSONDecodeError as e:
            print(f"Error: Failed to parse {file_name}: {e}")
            continue

# 提取所有指标
all_metrics = set()
for model_data in model_metrics.values():
    all_metrics.update(model_data.keys())
all_metrics = sorted(all_metrics)  # 排序以保持一致

# 处理数据：移除异常值并计算均值和中位数
stats_data = {metric: {'mean': {}, 'median': {}} for metric in all_metrics}
cleaned_data = {metric: {} for metric in all_metrics}

for metric in all_metrics:
    for model_name, model_data in model_metrics.items():
        if metric in model_data and model_data[metric]:  # 确保指标存在且非空
            # 移除异常值
            clean_values = remove_outliers(model_data[metric])
            cleaned_data[metric][model_name] = clean_values
            # 计算均值和中位数
            if clean_values:  # 确保清理后仍有数据
                stats_data[metric]['mean'][model_name] = np.mean(clean_values)
                stats_data[metric]['median'][model_name] = np.median(clean_values)
            else:
                print(f"Warning: No valid data for {metric} in {model_name} after cleaning.")
        else:
            print(f"Warning: Metric {metric} missing or empty in {model_name}.")

# 可视化 1：箱线图
n_metrics = len(all_metrics)
if n_metrics > 0:
    fig, axes = plt.subplots(n_metrics, 1, figsize=(10, 4 * n_metrics), sharex=True)
    if n_metrics == 1:
        axes = [axes]  # 确保单指标时 axes 可迭代
    for i, metric in enumerate(all_metrics):
        # 准备箱线图数据
        plot_data = pd.DataFrame({
            model_name: pd.Series(values) for model_name, values in cleaned_data[metric].items()
        })
        if not plot_data.empty:
            sns.boxplot(data=plot_data, ax=axes[i])
            axes[i].set_title(f'Boxplot of {metric} (Outliers Removed)')
            axes[i].set_ylabel(metric)
            axes[i].set_xlabel('Model')
            axes[i].grid(True)
            axes[i].tick_params(axis='x', rotation=45)
        else:
            print(f"Warning: No data to plot for {metric}.")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'boxplot_metrics.png'))
    plt.close()
else:
    print("Error: No metrics found to plot.")

# 可视化 2：条形图（均值和中位数）
if n_metrics > 0:
    fig, axes = plt.subplots(n_metrics, 2, figsize=(14, 4 * n_metrics), sharex=True)
    if n_metrics == 1:
        axes = [[axes[0], axes[1]]]  # 调整单指标情况
    for i, metric in enumerate(all_metrics):
        # 均值条形图
        means = stats_data[metric]['mean']
        if means:
            sns.barplot(x=list(means.keys()), y=list(means.values()), ax=axes[i][0])
            axes[i][0].set_title(f'Mean of {metric} (Outliers Removed)')
            axes[i][0].set_ylabel('Mean')
            axes[i][0].set_xlabel('Model')
            axes[i][0].grid(True)
            axes[i][0].tick_params(axis='x', rotation=45)
        # 中位数条形图
        medians = stats_data[metric]['median']
        if medians:
            sns.barplot(x=list(medians.keys()), y=list(medians.values()), ax=axes[i][1])
            axes[i][1].set_title(f'Median of {metric} (Outliers Removed)')
            axes[i][1].set_ylabel('Median')
            axes[i][1].set_xlabel('Model')
            axes[i][1].grid(True)
            axes[i][1].tick_params(axis='x', rotation=45)
        else:
            print(f"Warning: No data to plot for {metric}.")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'barplot_metrics.png'))
    plt.close()
else:
    print("Error: No metrics found to plot.")

print(f"Plots saved to {output_dir}: boxplot_metrics.png, barplot_metrics.png")
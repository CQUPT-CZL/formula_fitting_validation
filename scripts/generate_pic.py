import json
import os
import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
from pathlib import Path

def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
config_path = os.path.join(project_root, 'config', 'config.yaml')

if not os.path.exists(config_path):
    logging.error(f"Config file not found at: {config_path}")
    raise FileNotFoundError(f"Config file not found at: {config_path}")

config = load_config(config_path)

# 设置文件夹路径（请替换为你的 JSON 文件夹路径）
folder_path = os.path.join(project_root, config['paths']['output_dir'])
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

            # 强调文件名的后缀
            suffix = "_with_data_code_results"
            model_name = model_name.replace(suffix, "")
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

# 箱线图
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

# 条形图（均值和中位数）
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

import numpy as np
from math import pi

# 雷达图
if n_metrics >= 3:  # 雷达图适合 3 个及以上指标
    # 准备数据（均值）
    labels = all_metrics
    n_metrics = len(labels)
    angles = [n / float(n_metrics) * 2 * pi for n in range(n_metrics)]
    angles += angles[:1]  # 闭合图形

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    for model_name in stats_data[all_metrics[0]]['mean']:
        values = [stats_data[metric]['mean'].get(model_name, 0) for metric in all_metrics]
        # 归一化值以适应雷达图
        max_val = max(max(values), 1)  # 避免除以 0
        values = [v / max_val for v in values]
        values += values[:1]  # 闭合图形
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=model_name)
        ax.fill(angles, values, alpha=0.1)

    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids([a * 180 / pi for a in angles[:-1]], labels)
    ax.set_title("Radar Chart of Mean Metrics (Outliers Removed)", pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
    plt.savefig(os.path.join(output_dir, 'radarplot_metrics.png'))
    plt.close()
else:
    print("Warning: Radar chart requires at least 3 metrics.")


# 小提琴图
if n_metrics > 0:
    fig, axes = plt.subplots(n_metrics, 1, figsize=(10, 4 * n_metrics), sharex=True)
    if n_metrics == 1:
        axes = [axes]
    for i, metric in enumerate(all_metrics):
        plot_data = pd.DataFrame({
            model_name: pd.Series(values) for model_name, values in cleaned_data[metric].items()
        })
        if not plot_data.empty:
            sns.violinplot(data=plot_data, ax=axes[i])
            axes[i].set_title(f'Violin Plot of {metric} (Outliers Removed)')
            axes[i].set_ylabel(metric)
            axes[i].set_xlabel('Model')
            axes[i].grid(True)
            axes[i].tick_params(axis='x', rotation=45)
            # 可选：添加对数刻度
            if plot_data.max().max() > 1000:  # 如果值范围大
                axes[i].set_yscale('log')
        else:
            print(f"Warning: No data to plot for {metric}.")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'violinplot_metrics.png'))
    plt.close()
else:
    print("Error: No metrics found to plot.")


# 异常值比例热力图
outlier_proportions = {metric: {} for metric in all_metrics}
for metric in all_metrics:
    for model_name, model_data in model_metrics.items():
        if metric in model_data and model_data[metric]:
            total_count = len(model_data[metric])  # 原始数据点数
            valid_count = len(cleaned_data[metric].get(model_name, []))  # 清理后数据点数
            if total_count > 0:
                outlier_proportions[metric][model_name] = (total_count - valid_count) / total_count
            else:
                outlier_proportions[metric][model_name] = 0.0
        else:
            outlier_proportions[metric][model_name] = 0.0

# 转换为 DataFrame 并绘制热力图
outlier_df = pd.DataFrame(outlier_proportions).T  # 行是指标，列是模型
if not outlier_df.empty:
    plt.figure(figsize=(12, 6))
    sns.heatmap(outlier_df, annot=True, cmap="Reds", fmt=".2f", cbar_kws={'label': 'Outlier Proportion'})
    plt.title("Proportion of Outliers per Metric and Model")
    plt.xlabel("Model")
    plt.ylabel("Metric")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'outlier_proportion_heatmap.png'))
    plt.close()
else:
    print("Warning: No data for outlier proportion heatmap.")

# 更新打印语句
print(f"Plots saved to {output_dir}: boxplot_metrics.png, barplot_metrics.png, outlier_proportion_heatmap.png")


print(f"Plots saved to {output_dir}: boxplot_metrics.png, barplot_metrics.png")
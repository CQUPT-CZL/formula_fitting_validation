
# FitJudge

<p align="center">
    <img src="data/others/logo.png"/>
</p>

<p align="center">
    <img src="https://img.shields.io/badge/python-%E2%89%A53.10-5be.svg">
</p>

## 项目简介
FitJudge 是一个用于自动化评估和验证公式拟合效果的工具，支持多种指标和自动化流程，适用于机器学习和数据分析场景。

## 安装与环境要求
- Python >= 3.10
- 推荐使用虚拟环境（如conda或venv）
- 依赖包请参考 `requirements.txt`

## 快速开始 🚀
1. 生成原始数据（raw_data）：
   - 通过 prompt 生成 raw_data（同一版本只需生成一次，可复用）。
2. 生成预测代码：
   - 使用 pred 生成代码。
3. 运行评估：
   - 执行评估脚本，查看结果。

## 目录结构
```
├── config/                # 配置文件
├── data/                  # 数据目录
│   ├── raw_llm/           # llama-factory 拷贝的原始预测结果
│   ├── llm_with_data/     # 由 raw_llm 生成的数据 json
│   └── llm_with_data_code/# 由 llm_with_data 生成的样本代码
├── output/                # 输出目录
│   ├── logs/              # 日志
│   └── results/           # 评估结果
├── scripts/               # 脚本
├── src/                   # 源码
└── tests/                 # 测试
```

## 主要指标说明
- `MSE`：均方误差
- `MAE`：平均绝对误差
- `RMSE`：均方根误差
- `SC`（步骤完整性）：Step Completeness
- `LC`（逻辑一致性）：Logic Consistency
- `CA`（分析清晰度）：Clarity of Analysis
- `CodeEq`（公式代码一致性）：Code Equation Consistency
- `WIE`（空间平均绝对误差乘体积）：Weighted Integral Error

## 联系方式
如有问题或建议，欢迎提交 issue 或联系作者。

---

让公式拟合评测更智能高效！✨
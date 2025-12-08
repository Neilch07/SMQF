# SMQF 使用指南

## 快速开始

### 1. 测试单个因子

```bash
# 测试 vol_std20 因子
python test_vol_std20.py
```

### 2. 训练模型

#### 基础训练（使用所有因子库）

```bash
# 默认会使用 alpha101 + gtja191 + chenhan_factor 所有因子
python model_train.py \
    --engine xgb \
    --label cumulative \
    --horizon 5
```

#### 使用 Rank IC 筛选因子（推荐）

```bash
# Rank IC 对离群值更稳健
python model_train.py \
    --rank-ic \
    --ic-metric rank_ic \
    --ic-threshold 0.02 \
    --top-n 30 \
    --engine xgb
```

#### 使用 IC 筛选因子

```bash
# IC 对线性关系更敏感
python model_train.py \
    --rank-ic \
    --ic-metric ic \
    --ic-threshold 0.005 \
    --top-n 30 \
    --engine xgb
```

---

## 参数说明

### 因子模块

```bash
--modules alpha101 gtja191 chenhan_factor  # 默认使用全部三个
--modules alpha101                         # 只使用 alpha101
--modules chenhan_factor                   # 只使用自定义因子
```

### IC 筛选方式

```bash
--ic-metric rank_ic     # 使用 Rank IC（默认，推荐）
--ic-metric ic          # 使用 IC
```

**Rank IC vs IC:**
- **Rank IC**: 基于排序的相关性，对离群值更稳健，推荐使用
- **IC**: 基于值的相关性，对线性关系更敏感

### IC 阈值设置

```bash
# Rank IC 阈值建议
--ic-threshold 0.01    # 宽松筛选（更多因子）
--ic-threshold 0.02    # 适中筛选（推荐）
--ic-threshold 0.03    # 严格筛选（更少但更强的因子）

# IC 阈值建议
--ic-threshold 0.003   # 宽松筛选
--ic-threshold 0.005   # 适中筛选（推荐）
--ic-threshold 0.01    # 严格筛选
```

### 预测标签类型

```bash
--label cumulative --horizon 5   # 累计收益 [t+1, t+5]
--label "ret#2"                  # 收益比率 t+2/t+1
--label "ret#5"                  # 收益比率 t+5/t+1
--label "ret#20"                 # 收益比率 t+20/t+1
```

### 模型引擎

```bash
--engine xgb        # XGBoost（推荐，快速）
--engine lgbm       # LightGBM（内存友好）
--engine catboost   # CatBoost（准确性高）
```

---

## 完整示例

### 示例 1: 快速训练

```bash
# 使用所有因子，XGBoost，预测5日累计收益
python model_train.py
```

### 示例 2: Rank IC 筛选 + 多标签

```bash
# 使用 Rank IC 筛选前30个因子，预测 t+5/t+1 收益比
python model_train.py \
    --rank-ic \
    --ic-metric rank_ic \
    --ic-threshold 0.02 \
    --top-n 30 \
    --label "ret#5" \
    --engine xgb \
    --run-name rankic_ret5
```

### 示例 3: IC 筛选 + 自定义时间段

```bash
# 使用 IC 筛选，自定义训练/验证/测试时间
python model_train.py \
    --rank-ic \
    --ic-metric ic \
    --ic-threshold 0.005 \
    --top-n 50 \
    --start-date 2015-01-01 \
    --end-date 2023-12-31 \
    --train-end 2020-12-31 \
    --val-end 2022-06-30 \
    --engine xgb \
    --run-name ic_custom_dates
```

### 示例 4: 只使用自定义因子

```bash
# 只使用 chenhan_factor 模块的因子
python model_train.py \
    --modules chenhan_factor \
    --engine xgb
```

---

## 结果查看

训练完成后，结果保存在 `runs/run_<engine>_<name>/` 目录下：

```
runs/run_xgb_20250101_123456/
├── run_config.json          # 配置信息
├── metrics.json             # 性能指标
├── xgb_model.json           # 训练好的模型
├── test_preds.parquet       # 测试集预测
├── equity_curve.png         # 净值曲线图
└── factor_ic_rank.json      # IC排序结果（如使用 --rank-ic）
```

### 查看结果

```bash
# 查看性能指标
cat runs/run_*/metrics.json

# 查看 IC 排序
cat runs/run_*/factor_ic_rank.json

# 查看净值曲线
open runs/run_*/equity_curve.png
```

---

## 常见问题

### Q: 应该用 IC 还是 Rank IC？

**A: 推荐使用 Rank IC**
- Rank IC 对离群值更稳健
- 在量化金融中，排序比绝对值更重要
- 大多数情况下表现更好

### Q: IC 阈值设多少合适？

**A: 取决于筛选严格程度**
- Rank IC: 0.01-0.03（推荐 0.02）
- IC: 0.003-0.01（推荐 0.005）

### Q: top-n 设多少合适？

**A: 30-50 个因子**
- 太少：信息不足
- 太多：可能包含弱因子，增加过拟合风险
- 推荐：30-50

### Q: 哪个引擎最好？

**A: 推荐 XGBoost**
- XGBoost: 快速，稳定，适合大多数场景
- LightGBM: 内存占用少，大数据集
- CatBoost: 准确性高，但较慢

---

## 性能优化

系统自动使用全部 CPU 核心进行并行计算，无需手动设置。

如果内存不足：
```bash
# 减少因子数量
--top-n 20

# 或减少数据时间范围
--start-date 2020-01-01
```

---

*SAIF Quantitative Research Team*

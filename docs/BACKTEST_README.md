# 回测策略使用说明

## 📖 概述

`backtest_strategy.py` 模块用于基于模型预测生成交易仓位并计算策略绩效，**确保无未来信息泄露**。

## 🎯 核心逻辑（无未来信息）

### 时序关系
```
t 日收盘前：
  - 获得因子值 X(t)
  - 生成预测信号 pred(t)
  
t+1 日开盘：
  - 根据 pred(t) 构建仓位
  - 做多 top 20% 股票
  - 做空 bottom 20% 股票
  
t+1 日收盘：
  - 仓位结算
  - 获得收益 ret(t+1)
```

### 不同标签的处理

#### 1️⃣ **Cumulative 标签** (`--label cumulative`)
- **预测内容**: [t+1, t+h] 累计收益
- **交易策略**: t+1 建仓，持有 h 天
- **策略收益**: position(t) × ret[t+1:t+h]
- **数据使用**: 直接使用预测文件中的 `y` 列

#### 2️⃣ **ret#2 标签** (`--label ret#2`)
- **预测内容**: ret(t+2) / ret(t+1) 比率
- **交易策略**: t+1 开盘建仓，t+1 收盘平仓
- **策略收益**: position(t) × ret(t+1)
- **⚠️ 关键**: 需要提供原始收益数据（`--returns-file`）
- **时序说明**: 
  - 在 t 日只有预测信号
  - 在 t+1 日开盘执行交易
  - 获得 t+1 日的单期收益

#### 3️⃣ **ret#5 标签** (`--label ret#5`)
- **预测内容**: ret(t+5) / ret(t+1) 比率
- **交易策略**: 同 ret#2
- **策略收益**: position(t) × ret(t+1)

#### 4️⃣ **ret#20 标签** (`--label ret#20`)
- **预测内容**: ret(t+20) / ret(t+1) 比率
- **交易策略**: 同 ret#2
- **策略收益**: position(t) × ret(t+1)

## 🚀 使用方法

### 基本用法

```bash
# 1. 先训练模型（生成预测）
python model_train.py --label ret#5 --engine xgb --run-name test_ret5

# 2. 运行回测
python backtest_strategy.py \
  --run-dir runs/run_xgb_test_ret5 \
  --returns-file path/to/raw_returns.parquet \
  --long-quantile 0.2 \
  --short-quantile 0.2 \
  --method equal_weight
```

### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--run-dir` | 模型运行目录（必需） | - |
| `--preds-file` | 预测文件名 | `test_preds.parquet` |
| `--long-quantile` | 做多分位数（top x%） | 0.2 |
| `--short-quantile` | 做空分位数（bottom x%） | 0.2 |
| `--method` | 仓位分配方法 | `equal_weight` |
| `--returns-file` | 原始收益文件路径（ret#N 必需） | None |
| `--output-name` | 输出文件前缀 | `backtest_{method}` |

### 仓位分配方法

#### `equal_weight` (等权重)
- 多头：均分 100% 权重
- 空头：均分 -100% 权重
- 适合大多数场景

#### `pred_weight` (预测值加权)
- 多头：按预测值大小分配权重
- 空头：按预测值大小分配权重
- 适合预测值有明确含义的场景

## 📊 输出文件

运行后会在 `--run-dir` 目录下生成：

```
runs/run_xgb_test_ret5/
├── backtest_equal_weight_metrics.json      # 绩效指标
├── backtest_equal_weight_positions.parquet # 仓位数据
├── backtest_equal_weight_returns.parquet   # 策略收益
└── backtest_equal_weight_performance.png   # 绩效图表
```

### 绩效指标包括

- `cumulative_return`: 累计收益
- `annual_return`: 年化收益
- `annual_volatility`: 年化波动率
- `sharpe_ratio`: 夏普比率
- `max_drawdown`: 最大回撤
- `calmar_ratio`: Calmar 比率
- `win_rate`: 胜率
- `profit_loss_ratio`: 盈亏比

## 🔍 完整示例

### 示例 1: 使用 ret#5 标签

```bash
# 训练模型
python model_train.py \
  --label ret#5 \
  --engine xgb \
  --rank-ic \
  --top-n 30 \
  --run-name production_ret5

# 回测（需要原始收益数据）
python backtest_strategy.py \
  --run-dir runs/run_xgb_production_ret5 \
  --returns-file data/daily_returns.parquet \
  --long-quantile 0.2 \
  --short-quantile 0.2 \
  --method equal_weight \
  --output-name final_strategy
```

### 示例 2: 使用 cumulative 标签

```bash
# 训练模型
python model_train.py \
  --label cumulative \
  --horizon 10 \
  --engine lgbm \
  --run-name cumulative_10d

# 回测（不需要原始收益数据）
python backtest_strategy.py \
  --run-dir runs/run_lgbm_cumulative_10d \
  --long-quantile 0.3 \
  --short-quantile 0.3 \
  --method pred_weight
```

### 示例 3: 批量测试不同参数

```bash
#!/bin/bash
RUN_DIR="runs/run_xgb_test_ret2"

for long_q in 0.1 0.2 0.3; do
  for short_q in 0.1 0.2 0.3; do
    python backtest_strategy.py \
      --run-dir $RUN_DIR \
      --returns-file data/returns.parquet \
      --long-quantile $long_q \
      --short-quantile $short_q \
      --output-name "backtest_L${long_q}_S${short_q}"
  done
done
```

## ⚠️ 重要提醒

### 1. ret#N 标签必须提供原始收益
```bash
# ❌ 错误：ret#N 标签缺少原始收益
python backtest_strategy.py --run-dir runs/run_xgb_ret5

# ✅ 正确
python backtest_strategy.py \
  --run-dir runs/run_xgb_ret5 \
  --returns-file data/daily_returns.parquet
```

### 2. 原始收益文件格式
- **必须是 (date x ticker) 格式**
- **date 为索引，ticker 为列名**
- **值为单期收益率（日收益）**

示例：
```python
import pandas as pd

# 正确的格式
df_returns = pd.DataFrame({
    '000001.SZ': [0.01, -0.02, 0.015, ...],
    '000002.SZ': [0.005, 0.01, -0.01, ...],
    # ... 更多股票
}, index=pd.date_range('2020-01-01', periods=1000))

# 保存
df_returns.to_parquet('daily_returns.parquet')
```

### 3. 时序验证清单

确保你的回测满足以下条件：

- [ ] t 日的预测信号只使用 ≤ t 日的数据
- [ ] t+1 日的仓位只基于 t 日的信号
- [ ] t+1 日的收益在 t+1 日收盘后才能获得
- [ ] 没有使用任何未来信息

## 📈 结果解读

### 查看绩效指标

```bash
# 查看 JSON 格式的指标
cat runs/run_xgb_test/backtest_equal_weight_metrics.json
```

### 加载仓位数据

```python
import pandas as pd

# 读取仓位
df_positions = pd.read_parquet('runs/run_xgb_test/backtest_equal_weight_positions.parquet')

# 查看每日持仓
for date, group in df_positions.groupby(level=0):
    long_stocks = group[group['position'] > 0].index.get_level_values(1).tolist()
    short_stocks = group[group['position'] < 0].index.get_level_values(1).tolist()
    print(f"{date}: Long {len(long_stocks)}, Short {len(short_stocks)}")
```

### 自定义分析

```python
import pandas as pd
import numpy as np

# 读取策略收益
df_strategy = pd.read_parquet('runs/run_xgb_test/backtest_equal_weight_returns.parquet')

# 按日期聚合
daily_returns = df_strategy.groupby(level=0)['strategy_return'].sum()

# 计算月度收益
monthly_returns = daily_returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
print(monthly_returns)

# 计算滚动夏普
rolling_sharpe = (daily_returns.rolling(60).mean() / daily_returns.rolling(60).std()) * np.sqrt(243)
print(rolling_sharpe)
```

## 🐛 常见问题

### Q1: 提示 "No raw returns provided"
**A**: 使用 ret#N 标签时必须提供 `--returns-file` 参数。

### Q2: 策略收益为 NaN
**A**: 检查原始收益文件的日期和股票代码是否与预测文件匹配。

### Q3: 夏普比率为负
**A**: 这是正常的，说明策略在测试期表现不佳。需要：
- 检查模型是否过拟合
- 尝试不同的因子组合
- 调整仓位分配参数

### Q4: 如何优化策略
**A**: 可以尝试：
- 调整 `--long-quantile` 和 `--short-quantile`
- 使用 `--method pred_weight` 加权
- 增加更多有效因子（提高 IC）
- 调整训练/验证/测试集划分

## 📚 进阶用法

### 组合多个模型的预测

```python
import pandas as pd

# 加载多个模型的预测
pred1 = pd.read_parquet('runs/run_xgb_ret5/test_preds.parquet')
pred2 = pd.read_parquet('runs/run_lgbm_ret5/test_preds.parquet')
pred3 = pd.read_parquet('runs/run_catboost_ret5/test_preds.parquet')

# 集成预测（简单平均）
pred_ensemble = pred1.copy()
pred_ensemble['pred'] = (pred1['pred'] + pred2['pred'] + pred3['pred']) / 3

# 保存集成预测
pred_ensemble.to_parquet('runs/ensemble/test_preds.parquet')

# 回测集成模型
# python backtest_strategy.py --run-dir runs/ensemble --returns-file ...
```

### 动态调整仓位

修改 `generate_positions()` 函数以实现更复杂的仓位管理策略。

---

**祝你回测顺利！📈**

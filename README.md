# SMQF - Systematic Multi-factor Quantitative Finance

A professional quantitative trading system with multi-factor models and ML-based predictions.

---

## Quick Start

> 📖 详细使用说明请查看 **[USAGE.md](USAGE.md)**

### Test a Single Factor

```bash
# Test vol_std20 factor
python test_vol_std20.py
```

### Train a Model

```bash
# 使用所有因子库（alpha101, gtja191, chenhan_factor）
# 默认就会使用全部三个因子库
python model_train.py \
    --engine xgb \
    --label cumulative \
    --horizon 5

# 使用 Rank IC 筛选最佳因子（推荐）
python model_train.py \
    --rank-ic \
    --ic-metric rank_ic \
    --ic-threshold 0.02 \
    --top-n 30 \
    --engine xgb

# 使用 IC 筛选因子
python model_train.py \
    --rank-ic \
    --ic-metric ic \
    --ic-threshold 0.005 \
    --top-n 30 \
    --engine xgb
```

> 💡 更多示例和参数说明，请参考 **[USAGE.md](USAGE.md)**

---

## Project Structure

```
smqf/
├── README.md                # 项目说明
├── USAGE.md                 # 详细使用指南
├── test_vol_std20.py        # 因子测试示例
├── model_train.py           # 模型训练
│
├── quant_lib/               # Factor library
│   ├── factor.py
│   ├── alpha101.py          # 101 alpha factors
│   ├── gtja191.py           # 191 GTJA factors
│   ├── chenhan_factor.py    # Custom factors
│   └── analysis.py
│
├── scripts/
│   ├── list_factors.py
│   └── create_factors.py
│
├── tests/
│   ├── test_vol_std20.py        # Test vol_std20 factor
│   ├── test_all_factors_simple.py
│   └── test_single_factor.py
│
├── data/cn/equity/          # Market data
├── runs/                    # Training outputs
└── archive/                 # Old files
```

---

## Features

### Model Training
- **Fast Training** - Multi-core parallel processing (自动使用全部CPU)
- **IC/Rank IC Ranking** - 支持 IC 或 Rank IC 自动选择最佳因子
- **Multiple Engines** - XGBoost, LightGBM, CatBoost
- **No Data Leakage** - Strict time-series validation
- **All Factor Libraries** - 默认使用所有因子库（alpha101 + gtja191 + chenhan_factor）

### Factor Library
- **101 Alpha Factors** (WorldQuant alpha101)
- **191 GTJA Factors** (国泰君安)
- **Custom Factors** (chenhan_factor)
- Easy to extend

---

## Usage Examples

### Different Labels

```bash
# Cumulative return [t+1, t+5]
python model_train.py --label cumulative --horizon 5

# Return ratio t+2/t+1
python model_train.py --label "ret#2"

# Return ratio t+5/t+1
python model_train.py --label "ret#5"

# Return ratio t+20/t+1
python model_train.py --label "ret#20"
```

### Different Engines

```bash
# XGBoost (推荐)
python model_train.py --engine xgb

# LightGBM
python model_train.py --engine lgbm

# CatBoost
python model_train.py --engine catboost
```

### IC Metric Selection

```bash
# 使用 Rank IC 筛选（默认，推荐）
python model_train.py \
    --rank-ic \
    --ic-metric rank_ic \
    --ic-threshold 0.02 \
    --top-n 50

# 使用 IC 筛选
python model_train.py \
    --rank-ic \
    --ic-metric ic \
    --ic-threshold 0.005 \
    --top-n 50
```

### List Factors

```bash
python scripts/list_factors.py
```

---

## Workflow

1. **Test a factor** (optional)
   ```bash
   python test_vol_std20.py
   ```

2. **List available factors**
   ```bash
   python scripts/list_factors.py
   ```

3. **Train model with Rank IC ranking**
   ```bash
   python model_train.py \
       --rank-ic \
       --ic-metric rank_ic \
       --ic-threshold 0.02 \
       --top-n 30 \
       --engine xgb
   ```

4. **Check results**
   - Metrics: `runs/run_*/metrics.json`
   - Equity curve: `runs/run_*/equity_curve.png`

---

## Important Notes

### DO NOT MODIFY
- `quant_lib/` - Core factor library
- `data/` - Market data

### Safe to Modify
- Training parameters
- Custom factors in `quant_lib/chenhan_factor.py`

---

*SAIF Quantitative Research Team*

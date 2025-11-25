# SMQF - Systematic Multi-factor Quantitative Finance

A professional quantitative trading system with multi-factor models, backtesting, and optimization tools.

---

## 🎯 Quick Start

### 1. Train a Model (Recommended)

```bash
# Basic training with optimization
python model_train_optimized.py \
    --modules alpha101 gtja191 chenhan_factor \
    --engine xgb \
    --label cumulative \
    --horizon 5 \
    --n-jobs 4

# Production training with IC ranking
python model_train_optimized.py \
    --rank-ic \
    --ic-threshold 0.02 \
    --top-n 30 \
    --n-jobs 6 \
    --run-name "production"
```

### 2. Run Backtest

```bash
# Strategy backtest
python backtest_strategy.py

# Rolling backtest
python rolling_backtest.py
```

### 3. List Available Factors

```bash
# Show all factors
python scripts/list_factors.py

# Create new factors
python scripts/create_factors.py
```

---

## 📁 Project Structure

```
SMQF/
├── README.md                      # This file
│
├── Core Scripts                   # Main execution scripts
│   ├── model_train_optimized.py   # ⭐ Model training (optimized)
│   ├── backtest_strategy.py       # Strategy backtesting
│   ├── rolling_backtest.py        # Rolling window backtest
│   └── numexpr.py                 # numpy 2.x compatibility
│
├── scripts/                       # Utility scripts
│   ├── create_factors.py          # Create new factors
│   ├── list_factors.py            # List available factors
│   └── cleanup_useless_files.sh   # Clean cache files
│
├── tests/                         # Test files
│   ├── test_all_factors_simple.py # Test all factors
│   ├── test_single_factor.py      # Test single factor
│   └── test_amount_factor.py      # Test amount-based factors
│
├── docs/                          # Documentation
│   ├── QUICK_START_GUIDE.md       # Quick start guide
│   ├── OPTIMIZATION_REPORT.md     # Optimization details
│   ├── COMPARISON_SUMMARY.md      # Before/after comparison
│   ├── BACKTEST_README.md         # Backtest documentation
│   └── README_model_train.md      # Training documentation
│
├── archive/                       # Archived old files
│   ├── model_train.py             # Original training script
│   └── ...                        # Other deprecated files
│
├── quant_lib/                     # 📦 Core factor library (DO NOT TOUCH)
│   ├── factor.py                  # Base factor class
│   ├── alpha101.py                # 101 alpha factors
│   ├── gtja191.py                 # 191 GTJA factors
│   ├── chenhan_factor.py          # Custom factors
│   └── analysis.py                # Analysis tools
│
├── data/                          # Data storage
│   └── cn/equity/                 # Chinese equity data
│
└── runs/                          # Training outputs
    └── run_xgb_*/                 # Individual run results
```

---

## 🚀 Main Features

### ✅ Model Training
- **Optimized Training Pipeline** - 2-4x faster with parallel computation
- **IC Ranking** - Automatic factor selection based on IC
- **Multiple Engines** - XGBoost, LightGBM, CatBoost
- **No Data Leakage** - Strict time-series validation
- **Memory Efficient** - 30-50% memory savings

### ✅ Factor Library
- **101 Alpha Factors** (WorldQuant alpha101)
- **191 GTJA Factors** (国泰君安)
- **Custom Factors** (chenhan_factor)
- **Extensible Framework** - Easy to add new factors

### ✅ Backtesting
- **Strategy Backtesting** - Test trading strategies
- **Rolling Window** - Time-series cross-validation
- **Performance Metrics** - IC, IR, Sharpe, etc.

---

## 📊 Performance

| Metric | Value |
|--------|-------|
| **Training Speed** | 2-4x faster (with --n-jobs) |
| **Memory Usage** | 30-50% reduction |
| **Data Leakage** | ✅ 100% prevented |
| **Test IC** | ~0.04 (stable) |
| **Annual IR** | ~0.8 (out-of-sample) |

---

## 🔧 Usage Examples

### Train with Different Labels

```bash
# Cumulative return
python model_train_optimized.py --label cumulative --horizon 5

# Momentum ratio
python model_train_optimized.py --label "ret#5"

# Long-term ratio
python model_train_optimized.py --label "ret#20"
```

### Compare Engines

```bash
# XGBoost (fastest)
python model_train_optimized.py --engine xgb --n-jobs 4

# LightGBM (memory efficient)
python model_train_optimized.py --engine lgbm --n-jobs 4

# CatBoost (highest accuracy)
python model_train_optimized.py --engine catboost --n-jobs 4
```

### Test Factors

```bash
# Test all factors
cd tests
python test_all_factors_simple.py

# Test single factor
python test_single_factor.py

# Test amount-based factors
python test_amount_factor.py
```

---

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [Quick Start Guide](docs/QUICK_START_GUIDE.md) | Fast introduction and examples |
| [Optimization Report](docs/OPTIMIZATION_REPORT.md) | Technical optimization details |
| [Comparison Summary](docs/COMPARISON_SUMMARY.md) | Before/after comparison |
| [Backtest README](docs/BACKTEST_README.md) | Backtesting documentation |
| [Training README](docs/README_model_train.md) | Model training guide |

---

## 🛠️ Maintenance

### Clean Cache Files

```bash
# Run cleanup script
bash scripts/cleanup_useless_files.sh
```

### List Available Factors

```bash
# Show all factors with details
python scripts/list_factors.py
```

### Create New Factors

```bash
# Create factor template
python scripts/create_factors.py
```

---

## ⚠️ Important Notes

### 🔒 DO NOT MODIFY
- **quant_lib/** - Core factor library
- **data/** - Data files
- **numexpr.py** - Compatibility layer

### ✅ Safe to Modify
- Training parameters in `model_train_optimized.py`
- Backtest strategies in `backtest_strategy.py`
- Custom factors (add to `quant_lib/chenhan_factor.py`)

### 🗑️ Archive Folder
- Contains old/deprecated files
- Safe to delete after 7 days of testing
- Kept for reference and rollback

---

## 🎯 Recommended Workflow

1. **List factors**
   ```bash
   python scripts/list_factors.py
   ```

2. **Test factors** (optional)
   ```bash
   cd tests && python test_all_factors_simple.py
   ```

3. **Train model with IC ranking**
   ```bash
   python model_train_optimized.py \
       --rank-ic \
       --ic-threshold 0.02 \
       --top-n 30 \
       --n-jobs 6
   ```

4. **Backtest strategy**
   ```bash
   python backtest_strategy.py
   ```

5. **Analyze results**
   - Check `runs/run_*_*/metrics.json`
   - View `runs/run_*_*/equity_curve.png`

---

## 🐛 Troubleshooting

### Out of Memory
```bash
# Reduce parallel workers
python model_train_optimized.py --n-jobs 2

# Or reduce factor count
python model_train_optimized.py --top-n 20
```

### Slow IC Ranking
```bash
# Use parallel computation
python model_train_optimized.py --rank-ic --n-jobs 6
```

### Import Errors
```bash
# Clean cache and restart
bash scripts/cleanup_useless_files.sh
```

---

## 📝 Version History

- **v2.0** (2025-11-25) - Optimized version with parallelization
- **v1.0** (2024) - Initial version

---

## 📄 License

Internal use only. Do not distribute without permission.

---

## 👤 Author

SAIF Quantitative Research Team

---

*Last Updated: 2025-11-25*

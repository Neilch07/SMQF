# SMQF 文件夹整理计划

## 📋 当前文件分析

### ✅ 核心文件（保留）
- `model_train_optimized.py` (39K) - **主训练脚本**（优化版）
- `backtest_strategy.py` (16K) - 回测策略
- `rolling_backtest.py` (11K) - 滚动回测
- `numexpr.py` (367B) - numpy 2.x 兼容性处理（重要！）

### 📦 工具脚本（保留，移到 scripts/）
- `create_factors.py` (4.7K) - 因子创建工具
- `list_factors.py` (1.6K) - 因子列表工具
- `cleanup_useless_files.sh` (2K) - 清理脚本

### 🧪 测试文件（保留，移到 tests/）
- `test_all_factors_simple.py` (1.7K) - 测试所有因子
- `test_single_factor.py` (1.3K) - 测试单个因子
- `test_amount_factor.py` (1.3K) - 测试amount字段因子

### ❌ 删除文件（重复/过时）
1. **model_train.py** (33K) - 已被 model_train_optimized.py 替代
2. **test_factor.py** (1.2K) - 与 test_single_factor.py 重复
3. **test_factors.py** (1.2K) - 旧版测试，导入方式过时
4. **chenhan_factor_test.py** (702B) - 简单测试，功能重复
5. **compare_leakage_fix.py** (9.0K) - 对比脚本，已完成任务

### 📁 保持不动
- `quant_lib/` - **核心因子库，完全不动！**
- `data/` - 数据文件夹
- `.git/` - Git仓库

### 📄 文档文件（保留）
- `OPTIMIZATION_REPORT.md`
- `QUICK_START_GUIDE.md`
- `COMPARISON_SUMMARY.md`
- `BACKTEST_README.md`
- `README_model_train.md`

---

## 🎯 整理后的目录结构

```
SMQF/
├── README.md                      # 主README（新建）
├── model_train_optimized.py       # 主训练脚本
├── backtest_strategy.py           # 回测策略
├── rolling_backtest.py            # 滚动回测
├── numexpr.py                     # numpy兼容性
│
├── scripts/                       # 工具脚本
│   ├── create_factors.py
│   ├── list_factors.py
│   └── cleanup_useless_files.sh
│
├── tests/                         # 测试文件
│   ├── test_all_factors_simple.py
│   ├── test_single_factor.py
│   └── test_amount_factor.py
│
├── docs/                          # 文档
│   ├── OPTIMIZATION_REPORT.md
│   ├── QUICK_START_GUIDE.md
│   ├── COMPARISON_SUMMARY.md
│   ├── BACKTEST_README.md
│   └── README_model_train.md
│
├── archive/                       # 存档（旧文件）
│   ├── model_train.py            # 原版训练脚本
│   ├── compare_leakage_fix.py    # 对比脚本
│   ├── test_factor.py            # 旧测试
│   ├── test_factors.py           # 旧测试
│   └── chenhan_factor_test.py    # 旧测试
│
├── quant_lib/                     # 因子库（不动）
│   └── ...
│
├── data/                          # 数据
│   └── ...
│
└── runs/                          # 运行结果
    └── ...
```

---

## 🚀 执行步骤

1. ✅ 创建新目录结构
2. ✅ 移动文件到对应位置
3. ✅ 删除重复/过时文件
4. ✅ 创建主README
5. ✅ 验证整理结果

---

## ⚠️ 安全措施

- 所有删除的文件先移到 `archive/` 存档
- 保留7天后确认无误再永久删除
- `quant_lib/` 完全不动
- Git历史记录保持完整

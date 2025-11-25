# XGBoost 因子训练脚本使用说明

## 环境准备
建议使用 conda 按 `environment.yml` 创建环境，或使用下面最小指令：

```bash
conda create -n smqf python=3.11.0 numpy=1.26.4 numexpr=2.8.7 pandas attrs xgboost=2.1.1 -y
conda activate smqf
```

## 运行示例
最小测试（4 个因子，短时间窗口）：
```bash
python Python/SAIF/SMQF/model_train.py --include alpha101_003 alpha101_004 alpha191_003 alpha191_008 \
  --start-date 2023-01-01 --end-date 2023-06-30 --train-end 2023-03-31 --val-end 2023-05-31 --horizon 3 --no-zscore
```

全量使用两个模块全部因子（耗时较长）：
```bash
python Python/SAIF/SMQF/model_train.py --modules alpha101 gtja191 --horizon 5
```

## 重要参数
| 参数 | 说明 |
|------|------|
| --modules | 搜索因子所在模块列表（默认 alpha101 gtja191） |
| --include | 只使用指定因子类名（覆盖 modules 中的子集） |
| --exclude | 排除指定因子类名 |
| --start-date / --end-date | 限定数据区间，否则使用因子内部默认全样本 |
| --train-end / --val-end | 按日期切分 train/val/test 三段 |
| --horizon | 目标为未来 horizon 日累计收益 |
| --universe | 传入因子框架的股票筛选，如 top75pct |
| --no-zscore | 关闭截面 z-score 标准化 |
| --model-out / --preds-out | 模型与测试集预测输出路径 |

## 输出
运行结束后将在 `artifacts/metrics.json` 中保存评估指标：
- rmse：回归误差
- ic_mean / ic_ir：日度 Spearman IC 均值与 IR
- ls_ann_ret / ls_ann_ir：基于预测构建多空组合的年化收益与 IR（简化估算）

测试集预测（MultiIndex: date,ticker）保存在 `artifacts/test_preds.parquet`。

## 扩展建议
1. 增加特征：可读取历史收益波动、成交额中位数等做二级特征。  
2. 特征选择：加入 IC 或 SHAP 排序筛除低效因子。  
3. 时序验证：改为滚动窗口 walk-forward，减少信息泄漏。  
4. 目标改进：使用收益的 rank 或 sign 分类模型提升稳定性。  
5. 交易层：将预测转化为持仓权重并接入因子框架的交易成本评估。

## 常见问题
- 报错 `factor xxx failed: 'close'`：该因子需要的数据字段未在当前日期区间或 universe 下提供；尝试扩大日期或排除该因子。  
- NumPy 兼容问题：请确保使用 `numpy<1.27` 与当前已编译的科学库匹配。  
- 速度慢：先用 `--include` 少量因子做验证，再扩展。

## 后续可做
- 增加 LightGBM / CatBoost 对比。
- 加入仓位风险约束（行业中性、市值中性）。
- 输出因子重要性与稳定性报告。

---

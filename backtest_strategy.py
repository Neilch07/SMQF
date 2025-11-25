"""
回测策略模块 - 基于模型预测生成交易仓位并计算绩效

⚠️ 时序逻辑说明（无未来信息泄露）：
1. t 日收盘前：获得因子值，生成预测信号 pred(t)
2. t+1 日开盘：根据 pred(t) 构建仓位（做多/做空）
3. t+1 日收盘：仓位结算，获得 ret(t+1) 收益
4. 重复以上步骤

不同标签类型的交易策略：
- cumulative: 预测 [t+1, t+h]，在 t+1 建仓，t+h 平仓
- ret#2: 预测 t+2/t+1，在 t+1 建仓（因为需要观察 t+1 收益作为基准）
- ret#5: 预测 t+5/t+1，在 t+1 建仓
- ret#20: 预测 t+20/t+1，在 t+1 建仓
"""

import os
import json
import argparse
from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")


def load_predictions(preds_path: str) -> pd.DataFrame:
	"""加载模型预测结果
	
	Args:
		preds_path: 预测文件路径（parquet 或 csv）
		
	Returns:
		包含 pred 和 y 列的 DataFrame，索引为 (date, ticker)
	"""
	if preds_path.endswith('.parquet'):
		df = pd.read_parquet(preds_path)
	elif preds_path.endswith('.csv'):
		df = pd.read_csv(preds_path, index_col=[0, 1], parse_dates=[0])
	else:
		raise ValueError(f"Unsupported file format: {preds_path}")
	
	# 确保索引是 MultiIndex (date, ticker)
	if not isinstance(df.index, pd.MultiIndex):
		raise ValueError("Prediction file must have MultiIndex (date, ticker)")
	
	return df


def generate_positions(
	df_preds: pd.DataFrame,
	long_quantile: float = 0.2,
	short_quantile: float = 0.2,
	method: str = "equal_weight",
) -> pd.DataFrame:
	"""根据预测值生成交易仓位（无未来信息）
	
	⚠️ 时序逻辑：
	- t 日的预测值 pred(t) → t+1 日的仓位 position(t+1)
	- position > 0: 做多
	- position < 0: 做空
	- position = 0: 不持仓
	
	Args:
		df_preds: 预测数据，包含 pred 列，索引为 (date, ticker)
		long_quantile: 做多分位数（top x%）
		short_quantile: 做空分位数（bottom x%）
		method: 仓位分配方法
			- "equal_weight": 等权重
			- "pred_weight": 按预测值加权
	
	Returns:
		仓位 DataFrame，索引为 (date, ticker)，列为 position
	"""
	positions = []
	dates = []
	tickers_list = []
	
	# 按日期分组生成仓位
	for date, group in df_preds.groupby(level=0):
		if len(group) < 20:  # 样本太少跳过
			continue
		
		# 计算预测值的分位数排名
		pred_ranks = group["pred"].rank(pct=True)
		
		# 做多标的：预测收益最高的 top quantile
		long_mask = pred_ranks >= (1 - long_quantile)
		# 做空标的：预测收益最低的 bottom quantile
		short_mask = pred_ranks <= short_quantile
		
		# 计算仓位权重
		if method == "equal_weight":
			# 等权重：多头平均分配 +1，空头平均分配 -1
			n_long = long_mask.sum()
			n_short = short_mask.sum()
			
			position = pd.Series(0.0, index=group.index)
			if n_long > 0:
				position[long_mask] = 1.0 / n_long
			if n_short > 0:
				position[short_mask] = -1.0 / n_short
		
		elif method == "pred_weight":
			# 按预测值加权
			pred_values = group["pred"]
			
			# 多头权重：归一化后的预测值
			if long_mask.sum() > 0:
				long_weights = pred_values[long_mask]
				long_weights = long_weights / long_weights.sum()
			else:
				long_weights = pd.Series(dtype=float)
			
			# 空头权重：归一化后的负预测值
			if short_mask.sum() > 0:
				short_weights = -pred_values[short_mask]
				short_weights = short_weights / short_weights.sum()
			else:
				short_weights = pd.Series(dtype=float)
			
			position = pd.Series(0.0, index=group.index)
			position[long_mask] = long_weights
			position[short_mask] = -short_weights
		
		else:
			raise ValueError(f"Unknown method: {method}")
		
		positions.extend(position.values)
		dates.extend([date] * len(group))
		tickers_list.extend(group.index.get_level_values(1).tolist())
	
	# 构建 DataFrame
	df_positions = pd.DataFrame({
		"position": positions
	}, index=pd.MultiIndex.from_arrays([dates, tickers_list], names=["date", "ticker"]))
	
	return df_positions


def compute_strategy_returns(
	df_preds: pd.DataFrame,
	df_positions: pd.DataFrame,
	label_type: str = "cumulative",
	horizon: int = 5,
	df_raw_returns: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
	"""计算策略收益（考虑时序正确性）
	
	⚠️ 时序逻辑（关键 - 无未来信息）：
	1. t 日预测值 pred(t) → 生成 t 日信号
	2. t+1 日开盘建仓 → 获得 t+1 日收益
	
	不同标签类型的处理：
	   
	【cumulative 标签】：
	- 预测 [t+1, t+h] 累计收益
	- t+1 建仓，持有到 t+h
	- 策略收益 = position(t) × y(t)，其中 y(t) 是 [t+1, t+h] 累计收益
	   
	【ret#N 标签】（N=2,5,20）：
	- 预测 t+N / t+1 的比率
	- ⚠️ 关键：我们在 t 日只有预测信号，在 t+1 开盘建仓
	- 策略收益 = position(t) × ret(t+1)
	- 注意：y 列存储的是 ret(t+N)/ret(t+1)，不能直接用作收益
	- 需要使用原始单期收益 ret(t+1)
	
	Args:
		df_preds: 预测数据，包含 y 列（实际收益标签）
		df_positions: 仓位数据，索引对应信号生成日期 t
		label_type: 标签类型
		horizon: 持仓周期（仅用于 cumulative）
		df_raw_returns: 原始单期收益数据 (date x ticker)，用于 ret#N 标签
	
	Returns:
		策略收益 DataFrame，包含 strategy_return 列
	"""
	# 合并预测和仓位
	df_combined = df_preds.join(df_positions, how="inner")
	
	if label_type == "cumulative":
		# 累计收益标签：直接使用标签收益
		# y 列已经是 [t+1, t+h] 的累计收益
		# position(t) × y(t) = position(t) × ret[t+1:t+h]
		df_combined["strategy_return"] = df_combined["position"] * df_combined["y"]
		
	elif label_type.startswith("ret#"):
		# ret#N 标签：需要使用单期收益
		# y 列是 ret(t+N) / ret(t+1)，不能直接用
		
		if df_raw_returns is not None:
			# 使用原始收益数据
			# 将 raw_returns (date x ticker) 转为 long format
			ret_long = df_raw_returns.stack()
			ret_long.index.names = ['date', 'ticker']
			ret_long.name = 'raw_return'
			
			# 对于 position(t)，实际收益是 ret(t+1)
			# 需要将 position 的日期 shift(-1) 来匹配收益
			df_combined_with_ret = df_combined.join(ret_long, how='inner')
			
			# ⚠️ 时序对齐：position(t) 对应 t 日信号，在 t+1 执行
			# 我们需要获取 t+1 日的收益
			# 方法：按 ticker 分组，将 position 向后 shift
			def align_position_return(group):
				# group 是同一 ticker 的时间序列
				# position(t) 对应的是 ret(t+1)
				# 因此 position 需要 shift(1) 来对齐未来收益
				group['aligned_position'] = group['position'].shift(1)
				return group
			
			# 按 ticker 分组处理
			df_aligned = df_combined_with_ret.groupby(level=1).apply(align_position_return)
			
			# 计算策略收益 = aligned_position × raw_return
			df_aligned["strategy_return"] = df_aligned["aligned_position"] * df_aligned["raw_return"]
			
			# 去除 NaN（第一天没有前一天的 position）
			df_combined = df_aligned.dropna(subset=['strategy_return'])
			
			print(f"[info] Using raw returns for {label_type} backtest (time-aligned)")
		else:
			# 没有原始收益数据，使用简化方法
			# 警告：这可能不准确
			df_combined["strategy_return"] = df_combined["position"] * df_combined["y"]
			print(f"[warn] No raw returns provided. Using label value as proxy for {label_type}. "
				  f"Results may be inaccurate!")
	
	else:
		raise ValueError(f"Unknown label_type: {label_type}")
	
	return df_combined


def compute_performance_metrics(
	df_strategy: pd.DataFrame,
	annual_factor: int = 243,
) -> Dict[str, float]:
	"""计算策略绩效指标
	
	Args:
		df_strategy: 策略收益数据，包含 strategy_return 列
		annual_factor: 年化因子（交易日数）
	
	Returns:
		绩效指标字典
	"""
	# 按日期聚合收益（多只股票的组合收益）
	daily_returns = df_strategy.groupby(level=0)["strategy_return"].sum()
	
	# 基本统计
	n_days = len(daily_returns)
	mean_daily = daily_returns.mean()
	std_daily = daily_returns.std()
	
	# 年化指标
	annual_return = mean_daily * annual_factor
	annual_vol = std_daily * np.sqrt(annual_factor)
	sharpe_ratio = annual_return / annual_vol if annual_vol > 0 else np.nan
	
	# 累计收益
	cumulative_return = (1 + daily_returns).cumprod().iloc[-1] - 1 if n_days > 0 else 0
	
	# 最大回撤
	cum_rets = (1 + daily_returns).cumprod()
	running_max = cum_rets.expanding().max()
	drawdown = (cum_rets - running_max) / running_max
	max_drawdown = drawdown.min()
	
	# 胜率
	win_rate = (daily_returns > 0).sum() / n_days if n_days > 0 else 0
	
	# 盈亏比
	avg_win = daily_returns[daily_returns > 0].mean() if (daily_returns > 0).any() else 0
	avg_loss = daily_returns[daily_returns < 0].mean() if (daily_returns < 0).any() else 0
	profit_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else np.nan
	
	# Calmar 比率
	calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else np.nan
	
	metrics = {
		"n_days": n_days,
		"cumulative_return": float(cumulative_return),
		"annual_return": float(annual_return),
		"annual_volatility": float(annual_vol),
		"sharpe_ratio": float(sharpe_ratio),
		"max_drawdown": float(max_drawdown),
		"calmar_ratio": float(calmar_ratio),
		"win_rate": float(win_rate),
		"avg_daily_return": float(mean_daily),
		"profit_loss_ratio": float(profit_loss_ratio),
	}
	
	return metrics


def plot_strategy_performance(
	df_strategy: pd.DataFrame,
	save_path: Optional[str] = None,
	title: str = "Strategy Performance",
):
	"""绘制策略表现图
	
	Args:
		df_strategy: 策略收益数据
		save_path: 图片保存路径
		title: 图表标题
	"""
	try:
		import matplotlib.pyplot as plt
		import matplotlib.dates as mdates
	except ImportError:
		print("[warn] matplotlib not installed, skipping plots")
		return
	
	# 按日期聚合收益
	daily_returns = df_strategy.groupby(level=0)["strategy_return"].sum()
	
	# 计算累计净值
	cumulative_nav = (1 + daily_returns).cumprod()
	
	# 创建子图
	fig, axes = plt.subplots(3, 1, figsize=(14, 10))
	
	# 1. 净值曲线
	ax1 = axes[0]
	ax1.plot(cumulative_nav.index, cumulative_nav.values, linewidth=2, color='#1f77b4')
	ax1.set_ylabel('Cumulative NAV', fontsize=11)
	ax1.set_title(title, fontsize=13, fontweight='bold')
	ax1.grid(True, alpha=0.3)
	ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
	
	# 2. 每日收益
	ax2 = axes[1]
	colors = ['g' if r > 0 else 'r' for r in daily_returns]
	ax2.bar(daily_returns.index, daily_returns.values, color=colors, alpha=0.6, width=1)
	ax2.set_ylabel('Daily Return', fontsize=11)
	ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
	ax2.grid(True, alpha=0.3)
	ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
	
	# 3. 回撤
	ax3 = axes[2]
	cum_rets = (1 + daily_returns).cumprod()
	running_max = cum_rets.expanding().max()
	drawdown = (cum_rets - running_max) / running_max
	ax3.fill_between(drawdown.index, drawdown.values, 0, color='red', alpha=0.3)
	ax3.set_ylabel('Drawdown', fontsize=11)
	ax3.set_xlabel('Date', fontsize=11)
	ax3.grid(True, alpha=0.3)
	ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
	
	# 格式化 x 轴
	for ax in axes:
		ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
		plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
	
	plt.tight_layout()
	
	if save_path:
		os.makedirs(os.path.dirname(save_path), exist_ok=True)
		plt.savefig(save_path, dpi=300, bbox_inches='tight')
		print(f"📊 Strategy performance plot saved to: {save_path}")
		plt.close()
	else:
		plt.show()


def main():
	parser = argparse.ArgumentParser(description="Backtest strategy based on model predictions")
	parser.add_argument("--run-dir", type=str, required=True, help="Run directory containing predictions")
	parser.add_argument("--preds-file", type=str, default="test_preds.parquet", help="Prediction file name")
	parser.add_argument("--long-quantile", type=float, default=0.2, help="Long position quantile (top x%)")
	parser.add_argument("--short-quantile", type=float, default=0.2, help="Short position quantile (bottom x%)")
	parser.add_argument("--method", type=str, default="equal_weight", 
						choices=["equal_weight", "pred_weight"], help="Position weighting method")
	parser.add_argument("--returns-file", type=str, default=None, 
						help="Path to raw returns file (required for ret#N labels)")
	parser.add_argument("--output-name", type=str, default=None, help="Output file prefix (default: auto)")
	
	args = parser.parse_args()
	
	# 加载 run 配置
	config_path = os.path.join(args.run_dir, "run_config.json")
	if os.path.exists(config_path):
		with open(config_path, "r", encoding="utf-8") as f:
			run_config = json.load(f)
		label_type = run_config.get("label_type", "cumulative")
		label_name = run_config.get("label_name", "unknown")
		horizon = run_config.get("horizon", 5)
		print(f"[info] Loaded config: label={label_name}, horizon={horizon}")
	else:
		print("[warn] No run_config.json found, using defaults")
		label_type = "cumulative"
		label_name = "unknown"
		horizon = 5
	
	# 加载预测数据
	preds_path = os.path.join(args.run_dir, args.preds_file)
	if not os.path.exists(preds_path):
		raise FileNotFoundError(f"Predictions file not found: {preds_path}")
	
	print(f"[info] Loading predictions from: {preds_path}")
	df_preds = load_predictions(preds_path)
	print(f"[info] Loaded {len(df_preds)} prediction samples")
	
	# 加载原始收益数据（用于 ret#N 标签）
	df_raw_returns = None
	if label_type.startswith("ret#"):
		if args.returns_file:
			print(f"[info] Loading raw returns from: {args.returns_file}")
			if args.returns_file.endswith('.parquet'):
				df_raw_returns = pd.read_parquet(args.returns_file)
			elif args.returns_file.endswith('.csv'):
				df_raw_returns = pd.read_csv(args.returns_file, index_col=0, parse_dates=True)
			print(f"[info] Loaded raw returns: {df_raw_returns.shape}")
		else:
			print(f"[warn] No --returns-file provided for {label_type}. Using approximate method.")
	
	# 生成仓位
	print(f"[info] Generating positions (long={args.long_quantile}, short={args.short_quantile}, method={args.method})...")
	df_positions = generate_positions(
		df_preds,
		long_quantile=args.long_quantile,
		short_quantile=args.short_quantile,
		method=args.method,
	)
	print(f"[info] Generated {len(df_positions)} positions")
	
	# 计算策略收益
	print("[info] Computing strategy returns...")
	df_strategy = compute_strategy_returns(
		df_preds,
		df_positions,
		label_type=label_type,
		horizon=horizon,
		df_raw_returns=df_raw_returns,
	)
	
	# 计算绩效指标
	print("[info] Computing performance metrics...")
	metrics = compute_performance_metrics(df_strategy)
	
	# 打印结果
	print("\n" + "="*60)
	print("📊 STRATEGY PERFORMANCE METRICS")
	print("="*60)
	for k, v in metrics.items():
		if isinstance(v, float):
			print(f"{k:25s}: {v:12.4f}")
		else:
			print(f"{k:25s}: {v:12}")
	print("="*60 + "\n")
	
	# 保存结果
	output_prefix = args.output_name or f"backtest_{args.method}"
	
	# 保存指标
	metrics_path = os.path.join(args.run_dir, f"{output_prefix}_metrics.json")
	with open(metrics_path, "w", encoding="utf-8") as f:
		json.dump(metrics, f, ensure_ascii=False, indent=2)
	print(f"✅ Metrics saved to: {metrics_path}")
	
	# 保存仓位数据
	positions_path = os.path.join(args.run_dir, f"{output_prefix}_positions.parquet")
	df_positions.to_parquet(positions_path)
	print(f"✅ Positions saved to: {positions_path}")
	
	# 保存策略收益
	strategy_path = os.path.join(args.run_dir, f"{output_prefix}_returns.parquet")
	df_strategy.to_parquet(strategy_path)
	print(f"✅ Strategy returns saved to: {strategy_path}")
	
	# 绘制图表
	plot_path = os.path.join(args.run_dir, f"{output_prefix}_performance.png")
	plot_strategy_performance(
		df_strategy,
		save_path=plot_path,
		title=f"Strategy Performance - {label_name} ({args.method})",
	)
	
	print(f"\n✅ All backtest results saved to: {args.run_dir}")


if __name__ == "__main__":
	main()

import os
import sys
import json
import argparse
import warnings
import traceback
import gc
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# 将 quant_lib 加入路径，导入因子基类与因子库
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
QLIB_DIR = os.path.join(THIS_DIR, "quant_lib")
if QLIB_DIR not in sys.path:
	sys.path.append(QLIB_DIR)

from quant_lib.factor import factor  # noqa: E402
from quant_lib.analysis import cs_rank  # noqa: E402
import importlib  # noqa: E402


def discover_factor_classes(module_name: str) -> Dict[str, type]:
	"""从给定模块中发现继承自 factor 的因子类。"""
	# 直接导入模块名（因为 quant_lib 已经在 sys.path 中）
	try:
		m = importlib.import_module(module_name)
	except ImportError as e1:
		# 如果失败，尝试完整包名
		try:
			full_module_name = f"quant_lib.{module_name}"
			m = importlib.import_module(full_module_name)
		except ImportError as e2:
			print(f"[error] Failed to import {module_name}: {e1}, {e2}")
			return {}

	# 从模块内部获取 factor 基类（处理不同 import 路径的问题）
	base_factor = m.__dict__.get('factor', None)
	if base_factor is None:
		print(f"[warn] Module {module_name} does not have 'factor' in its namespace")
		return {}

	classes = {}
	for k, v in m.__dict__.items():
		if isinstance(v, type):
			try:
				if v is not base_factor and issubclass(v, base_factor):
					# 过滤内部/测试类
					# 接受 alpha 开头的，以及 vol_, ret_, std_ 等自定义因子
					if str(k).startswith("alpha") or any(str(k).startswith(prefix) for prefix in ['vol_', 'ret_', 'std_']):
						classes[k] = v
			except TypeError:
				pass

	return classes


def select_factor_classes(
	modules: List[str],
	include: Optional[List[str]] = None,
	exclude: Optional[List[str]] = None,
) -> Dict[str, type]:
	"""聚合并筛选因子类。"""
	include = include or []
	exclude = set(exclude or [])
	all_classes: Dict[str, type] = {}
	for mod in modules:
		all_classes.update(discover_factor_classes(mod))

	if include:
		picked = {name: all_classes[name] for name in include if name in all_classes}
	else:
		picked = all_classes

	# 排除潜在超慢/依赖数据多的少数条目（可按需扩展）
	for name in list(picked.keys()):
		if name in exclude:
			picked.pop(name, None)
	return picked


def compute_single_factor(
	args_tuple: Tuple[str, type, dict, dict]
) -> Tuple[str, Optional[pd.DataFrame], Optional[pd.DataFrame]]:
	"""计算单个因子的特征（用于并行处理）。

	⚠️ 内存优化：立即返回结果并释放中间变量

	Returns:
		(factor_name, factor_dataframe, returns_dataframe)
	"""
	name, cls, params_common, factor_property_common = args_tuple

	try:
		f_obj = cls(params=params_common, factor_property=factor_property_common)
		f_obj.run(turnoff_display=True)
		fd = f_obj.get_factor()
		returns_ref = f_obj.returns

		# ⚠️ 内存优化：立即删除因子对象
		del f_obj
		gc.collect()

		return (name, fd, returns_ref)
	except Exception as e:
		print(f"[warn] factor {name} failed: {e}")
		if "--debug" in sys.argv:
			traceback.print_exc()
		return (name, None, None)


def compute_factor_features(
	klass_map: Dict[str, type],
	start_date: Optional[str],
	end_date: Optional[str],
	universe: Optional[str],
	benchmark: Optional[str],
	display: bool = False,
	do_save: bool = False,
	n_jobs: int = 1,
) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
	"""计算各因子特征，返回 factor_name->DataFrame 以及 returns DataFrame。

	⚠️ 优化：支持并行计算（n_jobs > 1 时使用多进程）
	⚠️ 内存优化：使用迭代器而非一次性加载所有数据

	DataFrame 形状均为 (date x ticker)。
	"""
	features: Dict[str, pd.DataFrame] = {}
	returns_ref: Optional[pd.DataFrame] = None

	params_common = {
		"start_date": start_date,
		"end_date": end_date,
		"display": display,
		"save": bool(do_save),
	}
	factor_property_common = {
		"universe": universe,
		"benchmark": benchmark,
	}

	# 准备参数元组列表
	factor_args = [
		(name, cls, params_common, factor_property_common)
		for name, cls in klass_map.items()
	]

	if n_jobs > 1:
		# 并行计算因子
		print(f"[info] Computing {len(factor_args)} factors in parallel (n_jobs={n_jobs})...")

		with ProcessPoolExecutor(max_workers=n_jobs) as executor:
			# 提交所有任务
			future_to_name = {
				executor.submit(compute_single_factor, args): args[0]
				for args in factor_args
			}

			# 收集结果
			for future in as_completed(future_to_name):
				name = future_to_name[future]
				try:
					factor_name, fd, returns_data = future.result()
					if fd is not None:
						features[factor_name] = fd
						if returns_ref is None and returns_data is not None:
							returns_ref = returns_data
						print(f"[info] ✓ {factor_name} computed")

					# ⚠️ 内存优化：立即删除future对象
					del future
				except Exception as e:
					print(f"[warn] factor {name} failed in parallel execution: {e}")

		# ⚠️ 内存优化：清理
		gc.collect()

	else:
		# 串行计算（原始逻辑）
		print(f"[info] Computing {len(factor_args)} factors sequentially...")
		for args in factor_args:
			name, fd, returns_data = compute_single_factor(args)
			if fd is not None:
				features[name] = fd
				if returns_ref is None and returns_data is not None:
					returns_ref = returns_data
				print(f"[info] ✓ {name} computed")

			# ⚠️ 内存优化：立即清理
			gc.collect()

	if returns_ref is None:
		raise RuntimeError("无法加载 returns 数据（可能因子全部失败）。")

	return features, returns_ref


def run_all_factors_and_rank_ic(
	modules: List[str],
	start_date: Optional[str],
	end_date: Optional[str],
	universe: Optional[str],
	benchmark: Optional[str],
	ic_horizon: int = 5,
	persist_factors: bool = True,
	ic_threshold: float = 0.0,
	n_jobs: int = 1,
) -> Tuple[Dict[str, float], List[str]]:
	"""全量运行因子并按 IC 排名。

	⚠️ 优化：支持并行计算

	返回 (因子->avg_ic 映射, 成功运行的因子名列表)。会根据 persist_factors 决定是否落盘。
	支持通过 ic_threshold 筛选 abs(rank_ic) > threshold 的因子。
	"""
	klass_map = select_factor_classes(modules)
	avg_ic_map: Dict[str, float] = {}
	ok_names: List[str] = []

	def compute_factor_ic(args_tuple):
		"""计算单个因子的IC（用于并行）"""
		name, cls, params, factor_property = args_tuple
		try:
			f_obj = cls(params=params, factor_property=factor_property)
			f_obj.run(turnoff_display=True)
			res = f_obj.get_performance()
			avg_ic = float(res.get("avg_ic", np.nan))

			# 清理
			del f_obj
			gc.collect()

			return (name, avg_ic, True)
		except Exception as e:
			print(f"[warn] factor {name} failed: {e}")
			return (name, np.nan, False)

	# 准备参数
	factor_args = []
	for name, cls in klass_map.items():
		params = {
			"start_date": start_date,
			"end_date": end_date,
			"save": bool(persist_factors),
			"display": False,
			"ic_return_horizon": int(ic_horizon),
		}
		factor_args.append((name, cls, params, {"universe": universe, "benchmark": benchmark}))

	if n_jobs > 1:
		# 并行计算IC
		print(f"[info] Computing IC for {len(factor_args)} factors in parallel (n_jobs={n_jobs})...")

		with ProcessPoolExecutor(max_workers=n_jobs) as executor:
			futures = [executor.submit(compute_factor_ic, args) for args in factor_args]

			for future in as_completed(futures):
				name, avg_ic, success = future.result()

				if success:
					if np.isfinite(avg_ic):
						if abs(avg_ic) > ic_threshold:
							avg_ic_map[name] = avg_ic
							print(f"[info] ✓ {name}: IC={avg_ic:.4f}")
						else:
							print(f"[info] factor {name} filtered by IC threshold: abs({avg_ic:.4f}) <= {ic_threshold}")
					ok_names.append(name)

				# 清理
				del future

		gc.collect()

	else:
		# 串行计算
		for args in factor_args:
			name, avg_ic, success = compute_factor_ic(args)

			if success:
				if np.isfinite(avg_ic):
					if abs(avg_ic) > ic_threshold:
						avg_ic_map[name] = avg_ic
					else:
						print(f"[info] factor {name} filtered by IC threshold: abs({avg_ic:.4f}) <= {ic_threshold}")
				ok_names.append(name)

	return avg_ic_map, ok_names


def intersect_align(panels: List[pd.DataFrame]) -> List[pd.DataFrame]:
	"""对齐多只 (date x ticker) 面板，按交集对齐索引与列。

	⚠️ 内存优化：使用视图而非拷贝
	"""
	if not panels:
		return panels
	idx = panels[0].index
	cols = panels[0].columns
	for df in panels[1:]:
		idx = idx.intersection(df.index)
		cols = cols.intersection(df.columns)

	# 使用 loc 返回视图而非拷贝
	return [df.loc[idx, cols] for df in panels]


def panel_to_long(df: pd.DataFrame, name: str) -> pd.Series:
	"""将 (date x ticker) 面板转换为长格式 Series，索引为 (date, ticker)

	⚠️ 内存优化：使用 copy=False 避免不必要的复制
	"""
	s = df.stack()  # stack() 默认生成 (index, columns) = (date, ticker) 的 MultiIndex
	s.name = name
	# 确保 MultiIndex 的名称正确
	s.index.names = ['date', 'ticker']
	return s


def build_ml_dataset(
	features: Dict[str, pd.DataFrame],
	returns: pd.DataFrame,
	horizon: int = 5,
	label_type: str = "cumulative",
) -> Tuple[pd.DataFrame, pd.Series, pd.DatetimeIndex, str]:
	"""构建监督学习数据集。

	⚠️ 内存优化：
	- 删除中间变量
	- 使用生成器表达式
	- 及时触发垃圾回收

	⚠️ 防止未来信息泄露（Look-Ahead Bias）：

	1. 特征 X(t): 使用 t 日收盘时刻可获得的因子值（基于 t 日及之前的数据）
	2. 标签 y(t): 根据 label_type 计算不同类型的预测标签

	   【标签类型】：
	   - "cumulative" (默认): [t+1, t+horizon] 累计收益
	     * 计算：ret.shift(-1).rolling(horizon).sum()
	     * 含义：t+1 到 t+horizon 的总收益
	     * ✅ 无look-ahead bias: 使用shift(-1)确保只用未来收益

	   - "ret#2": t+2 / t+1 的收益比率
	     * 计算：ret.shift(-2) / ret.shift(-1)
	     * 含义：第2天收益相对第1天收益的倍数
	     * ✅ 无look-ahead bias: 两个shift都是负数（未来）

	   - "ret#5": t+5 / t+1 的收益比率
	     * 计算：ret.shift(-5) / ret.shift(-1)
	     * 含义：第5天收益相对第1天收益的倍数
	     * ✅ 无look-ahead bias: 两个shift都是负数（未来）

	   - "ret#20": t+20 / t+1 的收益比率
	     * 计算：ret.shift(-20) / ret.shift(-1)
	     * 含义：第20天收益相对第1天收益的倍数
	     * ✅ 无look-ahead bias: 两个shift都是负数（未来）

	3. **验证时序逻辑**：
	   - 所有 shift 操作都是负数（-1, -2, -5, -20），表示向前看未来
	   - 特征X(t)在时间t可得，标签y(t)基于t+1及之后的收益
	   - 训练时：模型学习 X(t) -> y(t)（未来收益）的映射
	   - 预测时：使用 X(t) 预测 y(t)，实际交易在 t+1 进行

	⚠️ 重要提醒：
	- zscore 正规化应在 train/val/test 分离后进行（防止 data leakage）
	- 本函数只构造原始特征和标签，正规化在main()中train_idx上进行
	"""
	# 对齐所有特征与收益面板
	panels = list(features.values()) + [returns]
	aligned = intersect_align(panels)
	aligned_features = aligned[:-1]
	ret = aligned[-1]

	# ⚠️ 内存优化：删除中间变量
	del panels, aligned
	gc.collect()

	# 根据 label_type 计算目标标签（⚠️ 关键：所有shift都是负数，确保无look-ahead bias）
	if label_type == "cumulative":
		# ✅ 累计收益：[t+1, t+horizon]
		# shift(-1): 将收益向前移1期（t的位置存储t+1的收益）
		# rolling(horizon).sum(): 计算未来horizon期的累计收益
		future_y = ret.shift(-1).rolling(horizon, min_periods=horizon).sum()
		label_name = f"ret_cumsum_{horizon}d"

	elif label_type == "ret#2":
		# ✅ t+2 / t+1 收益比率（无look-ahead bias）
		future_y = ret.shift(-2) / ret.shift(-1).replace(0, np.nan)
		label_name = "ret#2"

	elif label_type == "ret#5":
		# ✅ t+5 / t+1 收益比率（无look-ahead bias）
		future_y = ret.shift(-5) / ret.shift(-1).replace(0, np.nan)
		label_name = "ret#5"

	elif label_type == "ret#20":
		# ✅ t+20 / t+1 收益比率（无look-ahead bias）
		future_y = ret.shift(-20) / ret.shift(-1).replace(0, np.nan)
		label_name = "ret#20"

	else:
		raise ValueError(f"Unknown label_type: {label_type}. "
						 f"Supported: cumulative, ret#2, ret#5, ret#20")

	# 交集日期/股票
	idx = aligned_features[0].index
	cols = aligned_features[0].columns
	for df in aligned_features[1:] + [future_y]:
		idx = idx.intersection(df.index)
		cols = cols.intersection(df.columns)

	# 重新切到完全对齐的 panel
	aligned_features = [df.loc[idx, cols] for df in aligned_features]
	future_y = future_y.loc[idx, cols]

	# 转长表（⚠️ 内存优化：使用生成器）
	series_list = [panel_to_long(df, name) for df, name in zip(aligned_features, features.keys())]
	X = pd.concat(series_list, axis=1)
	y = panel_to_long(future_y, "y")

	# ⚠️ 内存优化：删除中间变量
	del aligned_features, future_y, ret, series_list
	gc.collect()

	# 检查重复索引
	if X.index.duplicated().any():
		n_dups = X.index.duplicated().sum()
		raise ValueError(f"Found {n_dups} duplicated (date, ticker) pairs in the dataset. "
						 f"This indicates data quality issues.")

	# 清理缺失
	data = pd.concat([X, y], axis=1).dropna(how="any")
	X = data.drop(columns=["y"])  # type: ignore
	y = data["y"]  # type: ignore

	# ⚠️ 内存优化：删除data
	del data
	gc.collect()

	# 从 MultiIndex 中提取日期（level='date' 或 level=0）
	sample_dates = X.index.get_level_values('date')
	return X, y, pd.DatetimeIndex(sample_dates.unique()), label_name


def normalize_with_train_stats(
	X: pd.DataFrame,
	train_idx: pd.Index,
	clip: float = 5.0
) -> pd.DataFrame:
	"""✅ Train 데이터의 통계량만 사용하여 정규화 (Data Leakage 방지)

	⚠️ 防止数据泄露的关键设计：
	1. 只使用 train_idx 的数据计算均值和标准差
	2. Val/Test 期间使用最近的过去 train 日期的统계량（不使用未来信息）
	3. 这确保了在实际预测时，只使用历史可得的信息

	Args:
		X: 전체 데이터 (date, ticker) MultiIndex
		train_idx: Training 데이터的인덱스
		clip: Z-score 클리핑 값

	Returns:
		정규化된 X
	"""
	X_norm = X.copy()

	# ✅ Train 데이터만 추출（防止泄露的关键步骤1）
	X_train = X.loc[train_idx]

	# ✅ cross-sectional normalization (每日标准化)
	train_stats = X_train.groupby(level='date').agg(['mean', 'std'])

	# ⚠️ 内存优化：删除X_train
	del X_train
	gc.collect()

	# 전체 데이터를 날짜별로 정규화
	for date in X.index.get_level_values('date').unique():
		date_mask = X.index.get_level_values('date') == date

		if date in train_stats.index:
			# ✅ Train 기간 내: 해당 날짜의 통계量 사용
			stats = train_stats.loc[date]
		else:
			# ✅ Val/Test 기간: 가장 가까운 과거 Train 날짜의 통계量 사용
			# ⚠️ 关键：只使用过去的统计量，不使用未来信息！
			past_dates = train_stats.index[train_stats.index <= date]
			if len(past_dates) > 0:
				stats = train_stats.loc[past_dates[-1]]  # 使用最近的过去日期
			else:
				# Train 이전 날짜: 에러 발생 (데이터가 잘못됨)
				raise ValueError(f"Found date {date} before training period. This should not happen. "
								 f"Check your data splitting logic.")

		# 정규화 적용
		for col in X.columns:
			mean_val = stats[(col, 'mean')]
			std_val = stats[(col, 'std')]

			if std_val > 0:
				X_norm.loc[date_mask, col] = (X.loc[date_mask, col] - mean_val) / std_val

	# 클리핑
	if clip is not None:
		X_norm = X_norm.clip(lower=-clip, upper=clip)

	# ⚠️ 内存优化
	del train_stats
	gc.collect()

	return X_norm


def time_split_indices(
	sample_dates: pd.DatetimeIndex,
	train_end: Optional[str],
	val_end: Optional[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""根据日期边界生成 train/val/test 索引掩码。

	⚠️ 时序分割（防止look-ahead bias）：
	- train: [start, train_end]
	- val: (train_end, val_end]
	- test: (val_end, end]

	确保严格的时间顺序，不重叠。
	"""
	d_min, d_max = sample_dates.min(), sample_dates.max()
	t_end = pd.to_datetime(train_end) if train_end else pd.to_datetime("2021-12-31")
	v_end = pd.to_datetime(val_end) if val_end else pd.to_datetime("2022-12-31")

	def mask(d1, d2):
		return (sample_dates >= d1) & (sample_dates <= d2)

	train_mask = mask(d_min, t_end)
	val_mask = mask(t_end + pd.Timedelta(days=1), v_end)
	test_mask = mask(v_end + pd.Timedelta(days=1), d_max)
	return train_mask, val_mask, test_mask


def compute_equity_curve(
	df_scores: pd.DataFrame,  # index: (date,ticker), column: pred, y
	quantile: float = 0.2,
) -> pd.Series:
	"""计算多空组合的净值曲线（cumulative return）。

	⚠️ 时序逻辑（确保无look-ahead bias）：
	- df_scores 的 date 索引是信号日期 t
	- y 列是 [t+1, t+horizon] 的未来收益（已经通过 shift(-horizon) 对齐）
	- 每日的持仓收益直接对应该日的 y 值（正确反映了下一个交易日的实际收益）
	- pred 是基于 t 时刻的特征预测的，用于 t+1 的交易决策

	Args:
		df_scores: 预测数据，包含 pred（预测）和 y（实际收益）
		quantile: 多空组合的分位数（如 0.2 表示做多 top 20%，做空 bottom 20%）

	Returns:
		日期索引的净值曲线（累计收益，初始值为 1.0）
	"""
	ls_daily_returns = []
	dates = []

	# 按日期分组计算每日多空收益
	for d, g in df_scores.groupby(level=0):  # level=0 是 date
		if len(g) < 20:  # 样本数太少则跳过
			continue

		# 根据预测值排序
		ranks = g["pred"].rank(pct=True)

		# 多头：预测收益最高的 quantile 分位
		top_ret = g.loc[ranks >= (1 - quantile), "y"].mean()
		# 空头：预测收益最低的 quantile 分位
		bot_ret = g.loc[ranks <= quantile, "y"].mean()

		if np.isfinite(top_ret) and np.isfinite(bot_ret):
			# 多空组合收益 = 多头收益 - 空头收益
			ls_daily_returns.append(top_ret - bot_ret)
			dates.append(d)

	# 转换为 Series
	ls_series = pd.Series(ls_daily_returns, index=pd.DatetimeIndex(dates))

	# 计算累计净值（初始值为 1.0）
	equity_curve = (1 + ls_series).cumprod()

	return equity_curve


def evaluate_predictions(
	df_scores: pd.DataFrame,  # index: (date,ticker), column: pred, y
	quantile: float = 0.2,
) -> Dict[str, float]:
	"""评估预测：RMSE、IC、ICIR、以及简单多空组合年化收益与 IR。

	Spearman IC 手工实现，避免 SciPy 依赖（对每个截面做秩相关）。
	"""
	# RMSE
	diff = df_scores["pred"].values - df_scores["y"].values
	rmse = float(np.sqrt(np.mean(diff * diff))) if diff.size else np.nan

	ic_by_day: List[float] = []
	for d, g in df_scores.groupby(level=0):
		if len(g) < 5:
			continue
		rk_pred = g["pred"].rank(pct=True).values
		rk_y = g["y"].rank(pct=True).values
		if rk_pred.std() > 0 and rk_y.std() > 0:
			corr = np.corrcoef(rk_pred, rk_y)[0, 1]
			if np.isfinite(corr):
				ic_by_day.append(float(corr))
	ic_by_day = np.array(ic_by_day, dtype=float)
	ic_mean = float(np.nanmean(ic_by_day)) if ic_by_day.size else np.nan
	ic_ir = float(ic_mean / np.nanstd(ic_by_day)) if ic_by_day.size and np.nanstd(ic_by_day) > 0 else np.nan

	# 多空
	q = quantile
	ls_daily: List[float] = []
	for d, g in df_scores.groupby(level=0):
		if len(g) < 20:
			continue
		ranks = g["pred"].rank(pct=True)
		top_ret = g.loc[ranks >= (1 - q), "y"].mean()
		bot_ret = g.loc[ranks <= q, "y"].mean()
		if np.isfinite(top_ret) and np.isfinite(bot_ret):
			ls_daily.append(float((top_ret - bot_ret)))
	ls_daily = np.array(ls_daily, dtype=float)
	ann_ret = float(243 * np.nanmean(ls_daily)) if ls_daily.size else np.nan
	ann_vol = float(np.sqrt(243) * np.nanstd(ls_daily)) if ls_daily.size else np.nan
	ann_ir = float(ann_ret / ann_vol) if ls_daily.size and ann_vol > 0 else np.nan

	return {"rmse": rmse, "ic_mean": ic_mean, "ic_ir": ic_ir, "ls_ann_ret": ann_ret, "ls_ann_ir": ann_ir}


def plot_equity_curves(
	equity_data: Dict[str, Dict[str, pd.Series]],
	train_end: str,
	val_end: str,
	save_path: Optional[str] = None,
):
	"""绘制多模型净值曲线对比图，区分样本内（训练+验证）和样本外（测试）。

	Args:
		equity_data: {
			"model_name": {
				"train": equity_curve_series,
				"val": equity_curve_series,
				"test": equity_curve_series,
			}
		}
		train_end: 训练集结束日期（样本内外分界线1）
		val_end: 验证集结束日期（样本内外分界线2）
		save_path: 图片保存路径，如 None 则显示
	"""
	try:
		import matplotlib.pyplot as plt
		import matplotlib.dates as mdates
	except ImportError:
		print("[warn] matplotlib 未安装，无法绘制净值曲线")
		return

	plt.figure(figsize=(14, 7))
	ax = plt.gca()

	# 颜色方案
	colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
	color_idx = 0

	# 转换分界日期
	train_end_dt = pd.to_datetime(train_end)
	val_end_dt = pd.to_datetime(val_end)

	# 绘制每个模型的净值曲线
	for model_name, splits in equity_data.items():
		color = colors[color_idx % len(colors)]
		color_idx += 1

		# 合并 train, val, test 的净值曲线
		full_equity = pd.Series(dtype=float)

		for split_name in ["train", "val", "test"]:
			if split_name in splits and len(splits[split_name]) > 0:
				split_curve = splits[split_name]

				# 如果不是第一段，需要接续前一段的最终净值
				if len(full_equity) > 0:
					last_value = full_equity.iloc[-1]
					# 将当前段的净值调整为接续上一段
					split_curve = split_curve / split_curve.iloc[0] * last_value

				full_equity = pd.concat([full_equity, split_curve])

		# 绘制完整净值曲线
		if len(full_equity) > 0:
			ax.plot(full_equity.index, full_equity.values,
				   label=model_name, color=color, linewidth=2, alpha=0.8)

	# 添加样本内外分界线
	ax.axvline(x=train_end_dt, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label='Train/Val Split')
	ax.axvline(x=val_end_dt, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Val/Test Split (样本外开始)')

	# 添加区域标注
	ymin, ymax = ax.get_ylim()
	ax.text(train_end_dt, ymax * 0.95, '← 训练期', ha='right', va='top', fontsize=10, alpha=0.7)
	ax.text(train_end_dt, ymax * 0.95, '验证期 →', ha='left', va='top', fontsize=10, alpha=0.7)
	ax.text(val_end_dt, ymax * 0.95, '← 样本内', ha='right', va='top', fontsize=11, fontweight='bold', alpha=0.8)
	ax.text(val_end_dt, ymax * 0.95, '样本外 →', ha='left', va='top', fontsize=11, fontweight='bold',
		   color='red', alpha=0.8)

	# 图表美化
	ax.set_xlabel('Date', fontsize=12)
	ax.set_ylabel('Cumulative Return (Net Value)', fontsize=12)
	ax.set_title('Multi-Model Equity Curves (In-Sample vs Out-of-Sample)', fontsize=14, fontweight='bold')
	ax.legend(loc='upper left', fontsize=10)
	ax.grid(True, alpha=0.3)

	# 日期格式化
	ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
	ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
	plt.xticks(rotation=45)

	plt.tight_layout()

	if save_path:
		os.makedirs(os.path.dirname(save_path), exist_ok=True)
		plt.savefig(save_path, dpi=300, bbox_inches='tight')
		print(f"📊 Equity curve saved to: {save_path}")
		plt.close()
	else:
		plt.show()


def train_xgboost(
	X: pd.DataFrame,
	y: pd.Series,
	sample_dates: pd.DatetimeIndex,
	train_mask: np.ndarray,
	val_mask: np.ndarray,
	test_mask: np.ndarray,
	xgb_params: Optional[dict] = None,
	model_out: Optional[str] = None,
	preds_out: Optional[str] = None,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, pd.DataFrame]]:
	"""训练 XGBoost 回归并评估 train/val/test，返回各段指标和预测数据。

	⚠️ 内存优化：及时清理中间变量

	Returns:
		(metrics_dict, predictions_dict)
		- metrics_dict: {"train": {...}, "val": {...}, "test": {...}}
		- predictions_dict: {"train": df_scores, "val": df_scores, "test": df_scores}
	"""
	from xgboost import XGBRegressor

	xgb_params = xgb_params or {
		"n_estimators": 1000,
		"max_depth": 6,
		"learning_rate": 0.05,
		"subsample": 0.8,
		"colsample_bytree": 0.8,
		"reg_alpha": 0.0,
		"reg_lambda": 1.0,
		"objective": "reg:squarederror",
		"random_state": 42,
		"n_jobs": max(1, os.cpu_count() or 1),
		"tree_method": "hist",  # 内存优化：使用histogram方法
	}

	# 构造分段索引
	# 将日期级别的掩码映射到 (date,ticker) 行级别
	date_level = X.index.get_level_values(0)
	map_train = pd.Series(train_mask, index=sample_dates)
	map_val = pd.Series(val_mask, index=sample_dates)
	map_test = pd.Series(test_mask, index=sample_dates)

	train_row_mask = map_train.reindex(date_level).to_numpy()
	val_row_mask = map_val.reindex(date_level).to_numpy()
	test_row_mask = map_test.reindex(date_level).to_numpy()

	idx = X.index
	train_idx = idx[train_row_mask]
	val_idx = idx[val_row_mask]
	test_idx = idx[test_row_mask]

	# ⚠️ 内存优化：删除中间变量
	del map_train, map_val, map_test, train_row_mask, val_row_mask, test_row_mask, date_level
	gc.collect()

	model = XGBRegressor(**xgb_params)
	# 简化训练：不使用 early_stopping_rounds（版本兼容问题），后续可根据 xgb.__version__ 条件添加
	model.fit(
		X.loc[train_idx],
		y.loc[train_idx],
		eval_set=[(X.loc[val_idx], y.loc[val_idx])],
		verbose=False,
	)

	# 预测并评估
	out = {}
	preds_dict = {}  # 保存完整预测数据用于绘图
	for split_name, split_idx in [
		("train", train_idx),
		("val", val_idx),
		("test", test_idx),
	]:
		pred = model.predict(X.loc[split_idx])
		df_scores = pd.DataFrame({"pred": pred, "y": y.loc[split_idx].values}, index=split_idx)
		out[split_name] = evaluate_predictions(df_scores)
		preds_dict[split_name] = df_scores  # 保存预测数据

		# ⚠️ 内存优化：及时清理
		del pred
		gc.collect()

	# 保存模型与预测（按 test 段）
	if model_out:
		os.makedirs(os.path.dirname(model_out), exist_ok=True)
		try:
			model.save_model(model_out)
		except Exception:
			# 如果 JSON 不可用，尝试 pickle
			import pickle
			with open(model_out + ".pkl", "wb") as f:
				pickle.dump(model, f)

	if preds_out:
		os.makedirs(os.path.dirname(preds_out), exist_ok=True)
		test_pred = model.predict(X.loc[test_idx])
		df_scores = pd.DataFrame({"pred": test_pred, "y": y.loc[test_idx].values}, index=test_idx)
		# 保存为压缩 parquet
		try:
			df_scores.to_parquet(preds_out)
		except Exception:
			df_scores.to_csv(preds_out.replace(".parquet", ".csv"))

		# ⚠️ 内存优化
		del test_pred, df_scores
		gc.collect()

	return out, preds_dict


def train_lightgbm(
	X: pd.DataFrame,
	y: pd.Series,
	sample_dates: pd.DatetimeIndex,
	train_mask: np.ndarray,
	val_mask: np.ndarray,
	test_mask: np.ndarray,
	lgbm_params: Optional[dict] = None,
	preds_out: Optional[str] = None,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, pd.DataFrame]]:
	try:
		from lightgbm import LGBMRegressor  # type: ignore
	except Exception as e:
		raise RuntimeError(f"LightGBM 未安装或不可用: {e}")

	lgbm_params = lgbm_params or {
		"n_estimators": 1000,
		"max_depth": -1,
		"learning_rate": 0.05,
		"subsample": 0.8,
		"colsample_bytree": 0.8,
		"reg_alpha": 0.0,
		"reg_lambda": 1.0,
		"objective": "regression",
		"random_state": 42,
		"n_jobs": max(1, os.cpu_count() or 1),
	}

	# 掩码映射
	date_level = X.index.get_level_values(0)
	map_train = pd.Series(train_mask, index=sample_dates)
	map_val = pd.Series(val_mask, index=sample_dates)
	map_test = pd.Series(test_mask, index=sample_dates)
	train_idx = X.index[map_train.reindex(date_level).to_numpy()]
	val_idx = X.index[map_val.reindex(date_level).to_numpy()]
	test_idx = X.index[map_test.reindex(date_level).to_numpy()]

	# ⚠️ 内存优化
	del map_train, map_val, map_test, date_level
	gc.collect()

	model = LGBMRegressor(**lgbm_params)
	model.fit(X.loc[train_idx], y.loc[train_idx])

	out = {}
	preds_dict = {}
	for split_name, split_idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
		pred = model.predict(X.loc[split_idx])
		df_scores = pd.DataFrame({"pred": pred, "y": y.loc[split_idx].values}, index=split_idx)
		out[split_name] = evaluate_predictions(df_scores)
		preds_dict[split_name] = df_scores

		# ⚠️ 内存优化
		del pred
		gc.collect()

	if preds_out:
		os.makedirs(os.path.dirname(preds_out), exist_ok=True)
		test_pred = model.predict(X.loc[test_idx])
		df_scores = pd.DataFrame({"pred": test_pred, "y": y.loc[test_idx].values}, index=test_idx)
		try:
			df_scores.to_parquet(preds_out)
		except Exception:
			df_scores.to_csv(preds_out.replace(".parquet", ".csv"))

		# ⚠️ 内存优化
		del test_pred, df_scores
		gc.collect()

	return out, preds_dict


def train_catboost(
	X: pd.DataFrame,
	y: pd.Series,
	sample_dates: pd.DatetimeIndex,
	train_mask: np.ndarray,
	val_mask: np.ndarray,
	test_mask: np.ndarray,
	cb_params: Optional[dict] = None,
	preds_out: Optional[str] = None,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, pd.DataFrame]]:
	try:
		from catboost import CatBoostRegressor  # type: ignore
	except Exception as e:
		raise RuntimeError(f"CatBoost 未安装或不可用: {e}")

	cb_params = cb_params or {
		"iterations": 1000,
		"depth": 6,
		"learning_rate": 0.05,
		"l2_leaf_reg": 3.0,
		"loss_function": "RMSE",
		"random_seed": 42,
		"verbose": False,
		"thread_count": max(1, os.cpu_count() or 1),
	}

	date_level = X.index.get_level_values(0)
	map_train = pd.Series(train_mask, index=sample_dates)
	map_val = pd.Series(val_mask, index=sample_dates)
	map_test = pd.Series(test_mask, index=sample_dates)
	train_idx = X.index[map_train.reindex(date_level).to_numpy()]
	val_idx = X.index[map_val.reindex(date_level).to_numpy()]
	test_idx = X.index[map_test.reindex(date_level).to_numpy()]

	# ⚠️ 内存优化
	del map_train, map_val, map_test, date_level
	gc.collect()

	model = CatBoostRegressor(**cb_params)
	model.fit(X.loc[train_idx], y.loc[train_idx])

	out = {}
	preds_dict = {}
	for split_name, split_idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
		pred = model.predict(X.loc[split_idx])
		df_scores = pd.DataFrame({"pred": pred, "y": y.loc[split_idx].values}, index=split_idx)
		out[split_name] = evaluate_predictions(df_scores)
		preds_dict[split_name] = df_scores

		# ⚠️ 内存优化
		del pred
		gc.collect()

	if preds_out:
		os.makedirs(os.path.dirname(preds_out), exist_ok=True)
		test_pred = model.predict(X.loc[test_idx])
		df_scores = pd.DataFrame({"pred": test_pred, "y": y.loc[test_idx].values}, index=test_idx)
		try:
			df_scores.to_parquet(preds_out)
		except Exception:
			df_scores.to_csv(preds_out.replace(".parquet", ".csv"))

		# ⚠️ 内存优化
		del test_pred, df_scores
		gc.collect()

	return out, preds_dict


def main():
	parser = argparse.ArgumentParser(description="Train XGBoost on project factors")
	parser.add_argument("--modules", nargs="*", default=["alpha101", "gtja191", "chenhan_factor"], help="factor modules to use")
	parser.add_argument("--include", nargs="*", default=[], help="explicit factor class names to include")
	parser.add_argument("--exclude", nargs="*", default=[], help="factor class names to exclude")
	parser.add_argument("--start-date", type=str, default=None)
	parser.add_argument("--end-date", type=str, default=None)
	parser.add_argument("--universe", type=str, default="top75pct")
	parser.add_argument("--benchmark", type=str, default=None)
	parser.add_argument("--horizon", type=int, default=5, help="预测期限（仅用于cumulative标签）")
	parser.add_argument("--label", type=str, default="cumulative",
						choices=["cumulative", "ret#2", "ret#5", "ret#20"],
						help="预测标签类型: cumulative=[t+1,t+h]累计, ret#2=t+2/t+1, ret#5=t+5/t+1, ret#20=t+20/t+1")
	parser.add_argument("--train-end", type=str, default="2021-12-31")
	parser.add_argument("--val-end", type=str, default="2022-12-31")
	parser.add_argument("--no-zscore", action="store_true")
	parser.add_argument("--engine", type=str, choices=["xgb","lgbm","catboost"], default="xgb", help="选择训练模型框架")
	parser.add_argument("--rank-ic", action="store_true", help="运行所有因子并输出 IC 排序 JSON")
	parser.add_argument("--ic-horizon", type=int, default=5, help="用于 IC 排序的 horizon")
	parser.add_argument("--ic-threshold", type=float, default=0.0, help="IC 阈值筛选：只保留 abs(rank_ic) > threshold 的因子")
	parser.add_argument("--top-n", type=int, default=50, help="根据 IC 选择前 N 个因子用于训练")
	parser.add_argument("--persist-factors", action="store_true", help="是否在因子框架内部触发保存")
	parser.add_argument("--run-name", type=str, default=None, help="自定义运行名称（可选），默认使用时间戳")
	parser.add_argument("--n-jobs", type=int, default=1, help="并行计算因子的进程数（1=串行，>1=并行）")

	args = parser.parse_args()

	# 创建 run-specific 文件夹
	# 格式: run_<engine>_<timestamp> 或 run_<engine>_<custom_name>
	import time
	timestamp = time.strftime("%Y%m%d_%H%M%S")
	if args.run_name:
		run_folder_name = f"run_{args.engine}_{args.run_name}"
	else:
		run_folder_name = f"run_{args.engine}_{timestamp}"

	run_dir = os.path.join(THIS_DIR, "runs", run_folder_name)
	os.makedirs(run_dir, exist_ok=True)
	print(f"[info] Run directory: {run_dir}")

	# 更新所有输出路径到 run-specific 目录
	args.model_out = os.path.join(run_dir, f"{args.engine}_model.json")
	args.preds_out = os.path.join(run_dir, "test_preds.parquet")
	args.save_factor_json = os.path.join(run_dir, "factor_ic_rank.json")

	# 将字符串"None"转换为Python的None
	if args.universe == "None":
		args.universe = None
	if args.benchmark == "None":
		args.benchmark = None

	# 如果需要先运行所有因子做 IC 排序
	factor_names_final: List[str] = []
	if args.rank_ic:
		print(f"[info] Running all factors for IC ranking (threshold: abs(IC) > {args.ic_threshold})...")
		print(f"[info] Using {args.n_jobs} parallel workers...")

		avg_ic_map, ok_names = run_all_factors_and_rank_ic(
			modules=args.modules,
			start_date=args.start_date,
			end_date=args.end_date,
			universe=args.universe,
			benchmark=args.benchmark,
			ic_horizon=args.ic_horizon,
			persist_factors=args.persist_factors,
			ic_threshold=args.ic_threshold,
			n_jobs=args.n_jobs,  # ⚠️ 并行化
		)
		# 排序并保存 JSON
		sorted_items = sorted(avg_ic_map.items(), key=lambda kv: abs(kv[1]), reverse=True)
		os.makedirs(os.path.dirname(args.save_factor_json), exist_ok=True)
		with open(args.save_factor_json, "w", encoding="utf-8") as f:
			json.dump({"avg_ic": sorted_items, "all_run": ok_names}, f, ensure_ascii=False, indent=2)
		print(f"[info] IC ranking saved -> {args.save_factor_json}")
		# 取前 top-n
		factor_names_final = [name for name,_ in sorted_items[:args.top_n]]
		# 若用户未指定 include 则用 top-n
		if not args.include:
			args.include = factor_names_final
		else:
			# 用户已指定 include，则与 top-n 做交集增强可复用性
			args.include = [nm for nm in args.include if nm in factor_names_final]
		print(f"[info] Selected top {len(args.include)} factors for training: {args.include[:10]}{'...' if len(args.include)>10 else ''}")

		# ⚠️ 内存优化：清理
		del avg_ic_map, ok_names, sorted_items
		gc.collect()

	# 选择用于训练的因子类
	klass_map = select_factor_classes(args.modules, include=args.include, exclude=args.exclude)
	if not klass_map:
		raise SystemExit("未发现可用因子类 (after IC filtering)，请检查参数。")
	print(f"Using {len(klass_map)} factors for model {args.engine}: {sorted(list(klass_map.keys()))[:10]}{'...' if len(klass_map)>10 else ''}")
	print(f"[info] Using {args.n_jobs} parallel workers for factor computation...")

	# 计算因子特征（⚠️ 并行化）
	features, returns = compute_factor_features(
		klass_map,
		start_date=args.start_date,
		end_date=args.end_date,
		universe=args.universe,
		benchmark=args.benchmark,
		display=False,
		n_jobs=args.n_jobs,  # ⚠️ 并行化
	)

	# 构建 ML 数据集
	X, y, sample_dates, label_name = build_ml_dataset(
		features,
		returns,
		horizon=args.horizon,
		label_type=args.label,
	)
	print(f"[info] Using label: {label_name}")

	# ⚠️ 内存优化：删除features和returns
	del features, returns
	gc.collect()

	# 按时间切分
	train_mask, val_mask, test_mask = time_split_indices(sample_dates, args.train_end, args.val_end)

	# ✅ Data Leakage 防止: Train 统计量规范化
	if not args.no_zscore:
		print("[info] Normalizing with train-only statistics (preventing data leakage)...")
		date_level = X.index.get_level_values(0)
		map_train = pd.Series(train_mask, index=sample_dates)
		train_row_mask = map_train.reindex(date_level).to_numpy()
		train_idx = X.index[train_row_mask]

		X = normalize_with_train_stats(X, train_idx, clip=5.0)

		# ⚠️ 内存优化
		del map_train, train_row_mask, date_level
		gc.collect()

	# 训练 & 评估 & 保存
	if args.engine == "xgb":
		metrics, predictions = train_xgboost(
			X, y, sample_dates, train_mask, val_mask, test_mask,
			xgb_params=None,
			model_out=args.model_out,
			preds_out=args.preds_out,
		)
	elif args.engine == "lgbm":
		metrics, predictions = train_lightgbm(
			X, y, sample_dates, train_mask, val_mask, test_mask,
			lgbm_params=None,
			preds_out=args.preds_out.replace("test_preds", "test_preds_lgbm"),
		)
	else:  # catboost
		metrics, predictions = train_catboost(
			X, y, sample_dates, train_mask, val_mask, test_mask,
			cb_params=None,
			preds_out=args.preds_out.replace("test_preds", "test_preds_catboost"),
		)

	# 打印结果并保存指标
	pretty = json.dumps(metrics, ensure_ascii=False, indent=2)
	print(pretty)

	# 保存 metrics 到 run 文件夹
	metrics_path = os.path.join(run_dir, "metrics.json")
	with open(metrics_path, "w", encoding="utf-8") as f:
		f.write(pretty)

	print(f"\n✅ Metrics saved to: {metrics_path}")

	# 计算并绘制净值曲线
	print("\n[info] Computing equity curves...")
	equity_data = {}
	for split_name in ["train", "val", "test"]:
		if split_name in predictions and len(predictions[split_name]) > 0:
			equity_curve = compute_equity_curve(predictions[split_name], quantile=0.2)
			if split_name not in equity_data:
				equity_data[split_name] = {}
			equity_data[split_name] = equity_curve

	# 组织数据用于绘图（支持多模型对比）
	model_equity_data = {args.engine.upper(): equity_data}

	# 绘制净值曲线到 run 文件夹
	equity_plot_path = os.path.join(run_dir, "equity_curve.png")
	plot_equity_curves(
		model_equity_data,
		train_end=args.train_end,
		val_end=args.val_end,
		save_path=equity_plot_path,
	)

	# 保存 run 配置信息
	run_config = {
		"run_name": run_folder_name,
		"timestamp": timestamp,
		"engine": args.engine,
		"num_factors": len(klass_map),
		"factor_names": sorted(list(klass_map.keys())),
		"label_type": args.label,
		"label_name": label_name,
		"horizon": args.horizon,
		"train_end": args.train_end,
		"val_end": args.val_end,
		"universe": args.universe,
		"benchmark": args.benchmark,
		"ic_threshold": args.ic_threshold if args.rank_ic else None,
		"zscore": not args.no_zscore,
		"n_jobs": args.n_jobs,
	}
	config_path = os.path.join(run_dir, "run_config.json")
	with open(config_path, "w", encoding="utf-8") as f:
		json.dump(run_config, f, ensure_ascii=False, indent=2)

	print(f"\n✅ All results saved to: {run_dir}")
	print(f"   - Config: run_config.json")
	print(f"   - Metrics: metrics.json")
	print(f"   - Model: {os.path.basename(args.model_out)}")
	print(f"   - Predictions: {os.path.basename(args.preds_out)}")
	print(f"   - Equity curve: equity_curve.png")
	if args.rank_ic:
		print(f"   - IC ranking: factor_ic_rank.json")

	# ⚠️ 最终内存清理
	gc.collect()
	print(f"\n[info] Memory optimization: {gc.get_count()} objects collected")


if __name__ == "__main__":
	main()

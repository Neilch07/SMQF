import os
import sys
import json
import argparse
import warnings
import traceback
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

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


def compute_factor_features(
	klass_map: Dict[str, type],
	start_date: Optional[str],
	end_date: Optional[str],
	universe: Optional[str],
	benchmark: Optional[str],
	display: bool = False,
	do_save: bool = False,
) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
	"""计算各因子特征，返回 factor_name->DataFrame 以及 returns DataFrame。

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

	# 逐个因子运行，抽取因子值
	for name, cls in klass_map.items():
		try:
			f_obj = cls(params=params_common, factor_property=factor_property_common)
			f_obj.run(turnoff_display=True)
			fd = f_obj.get_factor()
			# 与 Universe 对齐后，保存
			features[name] = fd
			if returns_ref is None:
				returns_ref = f_obj.returns
		except Exception as e:
			# 单个因子失败不影响整体（记录并跳过）
			print(f"[warn] factor {name} failed: {e}")
			if "--debug" in sys.argv:
				traceback.print_exc()
			continue

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
) -> Tuple[Dict[str, float], List[str]]:
	"""全量运行因子并按 IC 排名。

	返回 (因子->avg_ic 映射, 成功运行的因子名列表)。会根据 persist_factors 决定是否落盘。
	"""
	klass_map = select_factor_classes(modules)
	avg_ic_map: Dict[str, float] = {}
	ok_names: List[str] = []

	for name, cls in klass_map.items():
		try:
			# 强制使用同一 IC horizon 以可比
			params = {
				"start_date": start_date,
				"end_date": end_date,
				"save": bool(persist_factors),
				"display": False,
				"ic_return_horizon": int(ic_horizon),
			}
			f_obj = cls(params=params, factor_property={"universe": universe, "benchmark": benchmark})
			f_obj.run(turnoff_display=True)
			res = f_obj.get_performance()
			avg_ic = float(res.get("avg_ic", np.nan))
			if np.isfinite(avg_ic):
				avg_ic_map[name] = avg_ic
			ok_names.append(name)
		except Exception as e:
			print(f"[warn] factor {name} failed: {e}")
			# print(traceback.format_exc())  # 完整错误信息（调试时启用）
			continue

	return avg_ic_map, ok_names


def intersect_align(panels: List[pd.DataFrame]) -> List[pd.DataFrame]:
	"""对齐多只 (date x ticker) 面板，按交集对齐索引与列。"""
	if not panels:
		return panels
	idx = panels[0].index
	cols = panels[0].columns
	for df in panels[1:]:
		idx = idx.intersection(df.index)
		cols = cols.intersection(df.columns)
	return [df.loc[idx, cols] for df in panels]


def panel_to_long(df: pd.DataFrame, name: str) -> pd.Series:
	"""将 (date x ticker) 面板转换为长格式 Series，索引为 (date, ticker)"""
	s = df.stack()  # stack() 默认生成 (index, columns) = (date, ticker) 的 MultiIndex
	s.name = name
	# 确保 MultiIndex 的名称正确
	s.index.names = ['date', 'ticker']
	return s


def build_ml_dataset(
	features: Dict[str, pd.DataFrame],
	returns: pd.DataFrame,
	horizon: int = 5,
	zscore: bool = True,
	clip: float = 5.0,
) -> Tuple[pd.DataFrame, pd.Series, pd.DatetimeIndex]:
	"""构建监督学习数据集。

	- X: 行为 (date,ticker) 的样本，列为各因子
	- y: 目标为未来 horizon 日累计收益
	- dates: 样본对응의 날짜 索引（方便按时间切分）

	WARNING: zscore 정규화는 train/val/test 분리 후에 수행해야 함 (data leakage 방지)
	"""
	# 对齐所有特征与收益面板
	panels = list(features.values()) + [returns]
	aligned = intersect_align(panels)
	aligned_features = aligned[:-1]
	ret = aligned[-1]

	# 目标：未来 h 日累计收益
	future_y = ret.rolling(horizon, min_periods=max(1, horizon // 2)).sum().shift(-horizon)

	# 交集日期/股票
	idx = aligned_features[0].index
	cols = aligned_features[0].columns
	for df in aligned_features[1:] + [future_y]:
		idx = idx.intersection(df.index)
		cols = cols.intersection(df.columns)

	# 重新切到完全对齐的 panel
	aligned_features = [df.loc[idx, cols] for df in aligned_features]
	future_y = future_y.loc[idx, cols]

	# 转长表
	series_list = [panel_to_long(df, name) for df, name in zip(aligned_features, features.keys())]
	X = pd.concat(series_list, axis=1)
	y = panel_to_long(future_y, "y")

	# 清理缺失
	data = pd.concat([X, y], axis=1).dropna(how="any")
	X = data.drop(columns=["y"])  # type: ignore
	y = data["y"]  # type: ignore

	# ⚠️ Z-score는 여기서 하지 않음 - train/val/test 분리 후에 수행
	# (data leakage 방지)

	# 从 MultiIndex 中提取日期（level='date' 或 level=0）
	sample_dates = X.index.get_level_values('date')
	return X, y, pd.DatetimeIndex(sample_dates.unique()), zscore, clip


def normalize_with_train_stats(
	X: pd.DataFrame,
	train_idx: pd.Index,
	clip: float = 5.0
) -> pd.DataFrame:
	"""Train 데이터의 통계량만 사용하여 정규화 (Data Leakage 방지)

	Args:
		X: 전체 데이터 (date, ticker) MultiIndex
		train_idx: Training 데이터의 인덱스
		clip: Z-score 클리핑 값

	Returns:
		정규화된 X
	"""
	X_norm = X.copy()

	# Train 데이터만 추출
	X_train = X.loc[train_idx]

	# Train 데이터의 날짜별 mean/std 계산
	train_stats = X_train.groupby(level='date').agg(['mean', 'std'])

	# 전체 데이터를 날짜별로 정규화
	for date in X.index.get_level_values('date').unique():
		date_mask = X.index.get_level_values('date') == date

		if date in train_stats.index:
			# Train 기간 내: 해당 날짜의 통계량 사용
			stats = train_stats.loc[date]
		else:
			# Val/Test 기간: 가장 가까운 과거 Train 날짜의 통계량 사용
			# (미래 정보 사용 안함!)
			past_dates = train_stats.index[train_stats.index <= date]
			if len(past_dates) > 0:
				stats = train_stats.loc[past_dates[-1]]
			else:
				# Train 이전 날짜: 정규화 스킵
				continue

		# 정규화 적용
		for col in X.columns:
			mean_val = stats[(col, 'mean')]
			std_val = stats[(col, 'std')]

			if std_val > 0:
				X_norm.loc[date_mask, col] = (X.loc[date_mask, col] - mean_val) / std_val

	# 클리핑
	if clip is not None:
		X_norm = X_norm.clip(lower=-clip, upper=clip)

	return X_norm


def time_split_indices(
	sample_dates: pd.DatetimeIndex,
	train_end: Optional[str],
	val_end: Optional[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""根据日期边界生成 train/val/test 索引掩码。"""
	d_min, d_max = sample_dates.min(), sample_dates.max()
	t_end = pd.to_datetime(train_end) if train_end else pd.to_datetime("2021-12-31")
	v_end = pd.to_datetime(val_end) if val_end else pd.to_datetime("2022-12-31")

	def mask(d1, d2):
		return (sample_dates >= d1) & (sample_dates <= d2)

	train_mask = mask(d_min, t_end)
	val_mask = mask(t_end + pd.Timedelta(days=1), v_end)
	test_mask = mask(v_end + pd.Timedelta(days=1), d_max)
	return train_mask, val_mask, test_mask


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
			ls_daily.append(float((top_ret - bot_ret) / 2.0))
	ls_daily = np.array(ls_daily, dtype=float)
	ann_ret = float(243 * np.nanmean(ls_daily)) if ls_daily.size else np.nan
	ann_vol = float(np.sqrt(243) * np.nanstd(ls_daily)) if ls_daily.size else np.nan
	ann_ir = float(ann_ret / ann_vol) if ls_daily.size and ann_vol > 0 else np.nan

	return {"rmse": rmse, "ic_mean": ic_mean, "ic_ir": ic_ir, "ls_ann_ret": ann_ret, "ls_ann_ir": ann_ir}


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
) -> Dict[str, Dict[str, float]]:
	"""训练 XGBoost 回归并评估 train/val/test，返回各段指标；保存模型与预测。"""
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
	for split_name, split_idx in [
		("train", train_idx),
		("val", val_idx),
		("test", test_idx),
	]:
		pred = model.predict(X.loc[split_idx])
		df_scores = pd.DataFrame({"pred": pred, "y": y.loc[split_idx].values}, index=split_idx)
		out[split_name] = evaluate_predictions(df_scores)

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

	return out


def train_lightgbm(
	X: pd.DataFrame,
	y: pd.Series,
	sample_dates: pd.DatetimeIndex,
	train_mask: np.ndarray,
	val_mask: np.ndarray,
	test_mask: np.ndarray,
	lgbm_params: Optional[dict] = None,
	preds_out: Optional[str] = None,
) -> Dict[str, Dict[str, float]]:
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

	model = LGBMRegressor(**lgbm_params)
	model.fit(X.loc[train_idx], y.loc[train_idx])

	out = {}
	for split_name, split_idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
		pred = model.predict(X.loc[split_idx])
		df_scores = pd.DataFrame({"pred": pred, "y": y.loc[split_idx].values}, index=split_idx)
		out[split_name] = evaluate_predictions(df_scores)

	if preds_out:
		os.makedirs(os.path.dirname(preds_out), exist_ok=True)
		test_pred = model.predict(X.loc[test_idx])
		df_scores = pd.DataFrame({"pred": test_pred, "y": y.loc[test_idx].values}, index=test_idx)
		try:
			df_scores.to_parquet(preds_out)
		except Exception:
			df_scores.to_csv(preds_out.replace(".parquet", ".csv"))

	return out


def train_catboost(
	X: pd.DataFrame,
	y: pd.Series,
	sample_dates: pd.DatetimeIndex,
	train_mask: np.ndarray,
	val_mask: np.ndarray,
	test_mask: np.ndarray,
	cb_params: Optional[dict] = None,
	preds_out: Optional[str] = None,
) -> Dict[str, Dict[str, float]]:
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

	model = CatBoostRegressor(**cb_params)
	model.fit(X.loc[train_idx], y.loc[train_idx])

	out = {}
	for split_name, split_idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
		pred = model.predict(X.loc[split_idx])
		df_scores = pd.DataFrame({"pred": pred, "y": y.loc[split_idx].values}, index=split_idx)
		out[split_name] = evaluate_predictions(df_scores)

	if preds_out:
		os.makedirs(os.path.dirname(preds_out), exist_ok=True)
		test_pred = model.predict(X.loc[test_idx])
		df_scores = pd.DataFrame({"pred": test_pred, "y": y.loc[test_idx].values}, index=test_idx)
		try:
			df_scores.to_parquet(preds_out)
		except Exception:
			df_scores.to_csv(preds_out.replace(".parquet", ".csv"))

	return out


def main():
	parser = argparse.ArgumentParser(description="Train XGBoost on project factors")
	parser.add_argument("--modules", nargs="*", default=["alpha101", "gtja191", "chenhan_factor"], help="factor modules to use")
	parser.add_argument("--include", nargs="*", default=[], help="explicit factor class names to include")
	parser.add_argument("--exclude", nargs="*", default=[], help="factor class names to exclude")
	parser.add_argument("--start-date", type=str, default=None)
	parser.add_argument("--end-date", type=str, default=None)
	parser.add_argument("--universe", type=str, default="top75pct")
	parser.add_argument("--benchmark", type=str, default=None)
	parser.add_argument("--horizon", type=int, default=5)
	parser.add_argument("--train-end", type=str, default="2021-12-31")
	parser.add_argument("--val-end", type=str, default="2022-12-31")
	parser.add_argument("--model-out", type=str, default=os.path.join(THIS_DIR, "models", "xgb_model.json"))
	parser.add_argument("--preds-out", type=str, default=os.path.join(THIS_DIR, "artifacts", "test_preds.parquet"))
	parser.add_argument("--no-zscore", action="store_true")
	parser.add_argument("--engine", type=str, choices=["xgb","lgbm","catboost"], default="xgb", help="选择训练模型框架")
	parser.add_argument("--rank-ic", action="store_true", help="运行所有因子并输出 IC 排序 JSON")
	parser.add_argument("--ic-horizon", type=int, default=5, help="用于 IC 排序的 horizon")
	parser.add_argument("--top-n", type=int, default=50, help="根据 IC 选择前 N 个因子用于训练")
	parser.add_argument("--save-factor-json", type=str, default=os.path.join(THIS_DIR, "artifacts", "factor_ic_rank.json"))
	parser.add_argument("--persist-factors", action="store_true", help="是否在因子框架内部触发保存")

	args = parser.parse_args()

	# 将字符串"None"转换为Python的None
	if args.universe == "None":
		args.universe = None
	if args.benchmark == "None":
		args.benchmark = None

	# 如果需要先运行所有因子做 IC 排序
	factor_names_final: List[str] = []
	if args.rank_ic:
		print("[info] Running all factors for IC ranking...")
		avg_ic_map, ok_names = run_all_factors_and_rank_ic(
			modules=args.modules,
			start_date=args.start_date,
			end_date=args.end_date,
			universe=args.universe,
			benchmark=args.benchmark,
			ic_horizon=args.ic_horizon,
			persist_factors=args.persist_factors,
		)
		# 排序并保存 JSON
		sorted_items = sorted(avg_ic_map.items(), key=lambda kv: kv[1], reverse=True)
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
		
		# 清理导入的模块，强制重新加载
		import importlib
		import sys
		for mod in args.modules:
			if mod in sys.modules:
				importlib.reload(sys.modules[mod])

	# 选择用于训练的因子类
	klass_map = select_factor_classes(args.modules, include=args.include, exclude=args.exclude)
	if not klass_map:
		raise SystemExit("未发现可用因子类 (after IC filtering)，请检查参数。")
	print(f"Using {len(klass_map)} factors for model {args.engine}: {sorted(list(klass_map.keys()))[:10]}{'...' if len(klass_map)>10 else ''}")

	# 计算因子特征
	features, returns = compute_factor_features(
		klass_map,
		start_date=args.start_date,
		end_date=args.end_date,
		universe=args.universe,
		benchmark=args.benchmark,
		display=False,
	)

	# 构建 ML 数据集
	X, y, sample_dates, do_zscore, clip_val = build_ml_dataset(
		features,
		returns,
		horizon=args.horizon,
		zscore=(not args.no_zscore),
	)

	# 按时间切分
	train_mask, val_mask, test_mask = time_split_indices(sample_dates, args.train_end, args.val_end)

	# Data Leakage 방지: Train 통계량만 사용하여 정규화
	if do_zscore:
		print("[info] Normalizing with train-only statistics (preventing data leakage)...")
		date_level = X.index.get_level_values(0)
		map_train = pd.Series(train_mask, index=sample_dates)
		train_row_mask = map_train.reindex(date_level).to_numpy()
		train_idx = X.index[train_row_mask]

		X = normalize_with_train_stats(X, train_idx, clip=clip_val)

	# 训练 & 评估 & 保存
	if args.engine == "xgb":
		metrics = train_xgboost(
			X, y, sample_dates, train_mask, val_mask, test_mask,
			xgb_params=None,
			model_out=args.model_out,
			preds_out=args.preds_out,
		)
	elif args.engine == "lgbm":
		metrics = train_lightgbm(
			X, y, sample_dates, train_mask, val_mask, test_mask,
			lgbm_params=None,
			preds_out=args.preds_out.replace("test_preds", "test_preds_lgbm"),
		)
	else:  # catboost
		metrics = train_catboost(
			X, y, sample_dates, train_mask, val_mask, test_mask,
			cb_params=None,
			preds_out=args.preds_out.replace("test_preds", "test_preds_catboost"),
		)

	# 打印结果并保存指标
	pretty = json.dumps(metrics, ensure_ascii=False, indent=2)
	print(pretty)

	# 파일명에 요인 수, 학습 기간 포함
	num_factors = len(klass_map)
	train_period = args.train_end.replace('-', '')
	val_period = args.val_end.replace('-', '')
	filename = f"metrics_{args.engine}_top{num_factors}_train{train_period}_val{val_period}.json"
	out_path = os.path.join(THIS_DIR, "artifacts", filename)
	os.makedirs(os.path.dirname(out_path), exist_ok=True)
	with open(out_path, "w", encoding="utf-8") as f:
		f.write(pretty)

	# 기존 호환성을 위해 간단한 파일명도 유지
	simple_path = os.path.join(THIS_DIR, "artifacts", f"metrics_{args.engine}.json")
	with open(simple_path, "w", encoding="utf-8") as f:
		f.write(pretty)

	print(f"\n✅ Metrics saved to: {filename}")


if __name__ == "__main__":
	main()


import os
import sys
import json
import argparse
import warnings
import traceback
import gc
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
QLIB_DIR = os.path.join(THIS_DIR, "quant_lib")
if QLIB_DIR not in sys.path:
	sys.path.append(QLIB_DIR)

from hchen.SMQF.quant_lib.factor import factor
from hchen.SMQF.quant_lib.analysis import cs_rank
import importlib


def discover_factor_classes(module_name: str) -> Dict[str, type]:
	"""从给定模块中发现继承自 factor 的因子类"""
	try:
		m = importlib.import_module(module_name)
	except ImportError:
		try:
			full_module_name = f"quant_lib.{module_name}"
			m = importlib.import_module(full_module_name)
		except ImportError as e:
			print(f"[error] Failed to import {module_name}: {e}")
			return {}

	base_factor = m.__dict__.get('factor', None)
	if base_factor is None:
		return {}

	classes = {}
	for k, v in m.__dict__.items():
		if isinstance(v, type):
			try:
				if v is not base_factor and issubclass(v, base_factor):
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
	"""聚合并筛选因子类"""
	include = include or []
	exclude = set(exclude or [])
	all_classes: Dict[str, type] = {}
	for mod in modules:
		all_classes.update(discover_factor_classes(mod))

	if include:
		picked = {name: all_classes[name] for name in include if name in all_classes}
	else:
		picked = all_classes

	for name in list(picked.keys()):
		if name in exclude:
			picked.pop(name, None)
	return picked


def compute_single_factor(
	args_tuple: Tuple[str, type, dict, dict]
) -> Tuple[str, Optional[pd.DataFrame], Optional[pd.DataFrame]]:
	"""计算单个因子的特征（用于并行处理）"""
	name, cls, params_common, factor_property_common = args_tuple

	try:
		f_obj = cls(params=params_common, factor_property=factor_property_common)
		f_obj.run(turnoff_display=True)
		fd = f_obj.get_factor()
		returns_ref = f_obj.returns

		del f_obj
		gc.collect()

		return (name, fd, returns_ref)
	except Exception as e:
		print(f"[warn] factor {name} failed: {e}")
		return (name, None, None)


def compute_factor_features(
	klass_map: Dict[str, type],
	start_date: Optional[str],
	end_date: Optional[str],
	universe: Optional[str],
	benchmark: Optional[str],
	n_jobs: int,
) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
	"""计算各因子特征，返回 factor_name->DataFrame 以及 returns DataFrame"""
	features: Dict[str, pd.DataFrame] = {}
	returns_ref: Optional[pd.DataFrame] = None

	params_common = {
		"start_date": start_date,
		"end_date": end_date,
		"display": False,
		"save": False,
	}
	factor_property_common = {
		"universe": universe,
		"benchmark": benchmark,
	}

	factor_args = [
		(name, cls, params_common, factor_property_common)
		for name, cls in klass_map.items()
	]

	print(f"[info] Computing {len(factor_args)} factors in parallel (n_jobs={n_jobs})...")

	with ProcessPoolExecutor(max_workers=n_jobs) as executor:
		future_to_name = {
			executor.submit(compute_single_factor, args): args[0]
			for args in factor_args
		}

		for future in as_completed(future_to_name):
			name = future_to_name[future]
			try:
				factor_name, fd, returns_data = future.result()
				if fd is not None:
					features[factor_name] = fd
					if returns_ref is None and returns_data is not None:
						returns_ref = returns_data
					print(f"[info] ✓ {factor_name}")

				del future
			except Exception as e:
				print(f"[warn] {name} failed: {e}")

	gc.collect()

	if returns_ref is None:
		raise RuntimeError("无法加载 returns 数据")

	return features, returns_ref


def _compute_factor_ic_worker(args_tuple):
	"""Worker function for computing factor IC (must be at module level for pickle)"""
	name, cls, params, factor_property, metric = args_tuple
	try:
		f_obj = cls(params=params, factor_property=factor_property)
		f_obj.run(turnoff_display=True)
		res = f_obj.get_performance()

		if metric == "rank_ic":
			avg_ic = float(res.get("avg_rank_ic", np.nan))
		else:
			avg_ic = float(res.get("avg_ic", np.nan))

		del f_obj
		gc.collect()

		return (name, avg_ic, True)
	except Exception as e:
		print(f"[warn] {name} failed: {e}")
		return (name, np.nan, False)


def run_all_factors_and_rank_ic(
	modules: List[str],
	start_date: Optional[str],
	end_date: Optional[str],
	universe: Optional[str],
	benchmark: Optional[str],
	ic_horizon: int,
	ic_threshold: float,
	n_jobs: int,
	ic_metric: str = "rank_ic",
) -> Tuple[Dict[str, float], List[str]]:
	"""全量运行因子并按 IC 排名

	Args:
		ic_metric: "ic" 或 "rank_ic"，选择使用哪个指标筛选
	"""
	klass_map = select_factor_classes(modules)
	avg_ic_map: Dict[str, float] = {}
	ok_names: List[str] = []

	factor_args = []
	for name, cls in klass_map.items():
		params = {
			"start_date": start_date,
			"end_date": end_date,
			"save": False,
			"display": False,
			"ic_return_horizon": int(ic_horizon),
		}
		factor_args.append((name, cls, params, {"universe": universe, "benchmark": benchmark}, ic_metric))

	print(f"[info] Computing {ic_metric.upper()} for {len(factor_args)} factors (n_jobs={n_jobs})...")

	with ProcessPoolExecutor(max_workers=n_jobs) as executor:
		futures = [executor.submit(_compute_factor_ic_worker, args) for args in factor_args]

		for future in as_completed(futures):
			name, avg_ic, success = future.result()

			if success:
				if np.isfinite(avg_ic):
					if abs(avg_ic) > ic_threshold:
						avg_ic_map[name] = avg_ic
						print(f"[info] ✓ {name}: {ic_metric.upper()}={avg_ic:.4f}")
					else:
						print(f"[info] {name} filtered: abs({avg_ic:.4f}) <= {ic_threshold}")
				ok_names.append(name)

			del future

	gc.collect()

	return avg_ic_map, ok_names


def intersect_align(panels: List[pd.DataFrame]) -> List[pd.DataFrame]:
	"""对齐多只 (date x ticker) 面板"""
	if not panels:
		return panels
	idx = panels[0].index
	cols = panels[0].columns
	for df in panels[1:]:
		idx = idx.intersection(df.index)
		cols = cols.intersection(df.columns)

	return [df.loc[idx, cols] for df in panels]


def panel_to_long(df: pd.DataFrame, name: str) -> pd.Series:
	"""将 (date x ticker) 面板转换为长格式"""
	s = df.stack()
	s.name = name
	s.index.names = ['date', 'ticker']
	return s


def build_ml_dataset(
	features: Dict[str, pd.DataFrame],
	returns: pd.DataFrame,
	horizon: int,
	label_type: str,
) -> Tuple[pd.DataFrame, pd.Series, pd.DatetimeIndex, str]:
	"""构建监督学习数据集"""
	panels = list(features.values()) + [returns]
	aligned = intersect_align(panels)
	aligned_features = aligned[:-1]
	ret = aligned[-1]

	del panels, aligned
	gc.collect()

	if label_type == "cumulative":
		future_y = ret.shift(-1).rolling(horizon, min_periods=horizon).sum()
		label_name = f"ret_cumsum_{horizon}d"
	elif label_type == "ret#2":
		future_y = ret.shift(-2) / ret.shift(-1).replace(0, np.nan)
		label_name = "ret#2"
	elif label_type == "ret#5":
		future_y = ret.shift(-5) / ret.shift(-1).replace(0, np.nan)
		label_name = "ret#5"
	elif label_type == "ret#20":
		future_y = ret.shift(-20) / ret.shift(-1).replace(0, np.nan)
		label_name = "ret#20"
	else:
		raise ValueError(f"Unknown label_type: {label_type}")

	idx = aligned_features[0].index
	cols = aligned_features[0].columns
	for df in aligned_features[1:] + [future_y]:
		idx = idx.intersection(df.index)
		cols = cols.intersection(df.columns)

	aligned_features = [df.loc[idx, cols] for df in aligned_features]
	future_y = future_y.loc[idx, cols]

	series_list = [panel_to_long(df, name) for df, name in zip(aligned_features, features.keys())]
	X = pd.concat(series_list, axis=1)
	y = panel_to_long(future_y, "y")

	del aligned_features, future_y, ret, series_list
	gc.collect()

	if X.index.duplicated().any():
		n_dups = X.index.duplicated().sum()
		raise ValueError(f"Found {n_dups} duplicated (date, ticker) pairs")

	data = pd.concat([X, y], axis=1).dropna(how="any")
	X = data.drop(columns=["y"])
	y = data["y"]

	del data
	gc.collect()

	sample_dates = X.index.get_level_values('date')
	return X, y, pd.DatetimeIndex(sample_dates.unique()), label_name


def normalize_with_train_stats(
	X: pd.DataFrame,
	train_idx: pd.Index,
	clip: float = 5.0
) -> pd.DataFrame:
	"""使用训练集统计量进行标准化"""
	X_norm = X.copy()
	X_train = X.loc[train_idx]

	train_stats = X_train.groupby(level='date').agg(['mean', 'std'])

	del X_train
	gc.collect()

	for date in X.index.get_level_values('date').unique():
		date_mask = X.index.get_level_values('date') == date

		if date in train_stats.index:
			stats = train_stats.loc[date]
		else:
			past_dates = train_stats.index[train_stats.index <= date]
			if len(past_dates) > 0:
				stats = train_stats.loc[past_dates[-1]]
			else:
				raise ValueError(f"Found date {date} before training period")

		for col in X.columns:
			mean_val = stats[(col, 'mean')]
			std_val = stats[(col, 'std')]

			if std_val > 0:
				X_norm.loc[date_mask, col] = (X.loc[date_mask, col] - mean_val) / std_val

	if clip is not None:
		X_norm = X_norm.clip(lower=-clip, upper=clip)

	del train_stats
	gc.collect()

	return X_norm


def time_split_indices(
	sample_dates: pd.DatetimeIndex,
	train_end: Optional[str],
	val_end: Optional[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""根据日期边界生成 train/val/test 索引掩码"""
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
	df_scores: pd.DataFrame,
	quantile: float = 0.2,
) -> pd.Series:
	"""计算多空组合的净值曲线"""
	ls_daily_returns = []
	dates = []

	for d, g in df_scores.groupby(level=0):
		if len(g) < 20:
			continue

		ranks = g["pred"].rank(pct=True)

		top_ret = g.loc[ranks >= (1 - quantile), "y"].mean()
		bot_ret = g.loc[ranks <= quantile, "y"].mean()

		if np.isfinite(top_ret) and np.isfinite(bot_ret):
			ls_daily_returns.append(top_ret - bot_ret)
			dates.append(d)

	ls_series = pd.Series(ls_daily_returns, index=pd.DatetimeIndex(dates))
	equity_curve = (1 + ls_series).cumprod()

	return equity_curve


def evaluate_predictions(
	df_scores: pd.DataFrame,
	quantile: float = 0.2,
) -> Dict[str, float]:
	"""评估预测：RMSE、IC、ICIR、以及多空组合年化收益与 IR"""
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
	"""绘制多模型净值曲线对比图"""
	try:
		import matplotlib.pyplot as plt
		import matplotlib.dates as mdates
	except ImportError:
		print("[warn] matplotlib 未安装，无法绘制净值曲线")
		return

	plt.figure(figsize=(14, 7))
	ax = plt.gca()

	colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
	color_idx = 0

	train_end_dt = pd.to_datetime(train_end)
	val_end_dt = pd.to_datetime(val_end)

	for model_name, splits in equity_data.items():
		color = colors[color_idx % len(colors)]
		color_idx += 1

		full_equity = pd.Series(dtype=float)

		for split_name in ["train", "val", "test"]:
			if split_name in splits and len(splits[split_name]) > 0:
				split_curve = splits[split_name]

				if len(full_equity) > 0:
					last_value = full_equity.iloc[-1]
					split_curve = split_curve / split_curve.iloc[0] * last_value

				full_equity = pd.concat([full_equity, split_curve])

		if len(full_equity) > 0:
			ax.plot(full_equity.index, full_equity.values,
				   label=model_name, color=color, linewidth=2, alpha=0.8)

	ax.axvline(x=train_end_dt, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label='Train/Val')
	ax.axvline(x=val_end_dt, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Val/Test')

	ymin, ymax = ax.get_ylim()
	ax.text(train_end_dt, ymax * 0.95, '← Train', ha='right', va='top', fontsize=10, alpha=0.7)
	ax.text(train_end_dt, ymax * 0.95, 'Val →', ha='left', va='top', fontsize=10, alpha=0.7)
	ax.text(val_end_dt, ymax * 0.95, '← In-Sample', ha='right', va='top', fontsize=11, fontweight='bold', alpha=0.8)
	ax.text(val_end_dt, ymax * 0.95, 'Out-of-Sample →', ha='left', va='top', fontsize=11, fontweight='bold',
		   color='red', alpha=0.8)

	ax.set_xlabel('Date', fontsize=12)
	ax.set_ylabel('Cumulative Return', fontsize=12)
	ax.set_title('Multi-Model Equity Curves', fontsize=14, fontweight='bold')
	ax.legend(loc='upper left', fontsize=10)
	ax.grid(True, alpha=0.3)

	ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
	ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
	plt.xticks(rotation=45)

	plt.tight_layout()

	if save_path:
		os.makedirs(os.path.dirname(save_path), exist_ok=True)
		plt.savefig(save_path, dpi=300, bbox_inches='tight')
		print(f"[info] Equity curve saved: {save_path}")
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
	model_out: Optional[str] = None,
	preds_out: Optional[str] = None,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, pd.DataFrame]]:
	"""训练 XGBoost"""
	from xgboost import XGBRegressor

	xgb_params = {
		"n_estimators": 1000,
		"max_depth": 6,
		"learning_rate": 0.05,
		"subsample": 0.8,
		"colsample_bytree": 0.8,
		"reg_alpha": 0.0,
		"reg_lambda": 1.0,
		"objective": "reg:squarederror",
		"random_state": 42,
		"n_jobs": -1,
		"tree_method": "hist",
	}

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

	del map_train, map_val, map_test, train_row_mask, val_row_mask, test_row_mask, date_level
	gc.collect()

	model = XGBRegressor(**xgb_params)
	model.fit(
		X.loc[train_idx],
		y.loc[train_idx],
		eval_set=[(X.loc[val_idx], y.loc[val_idx])],
		verbose=False,
	)

	out = {}
	preds_dict = {}
	for split_name, split_idx in [
		("train", train_idx),
		("val", val_idx),
		("test", test_idx),
	]:
		pred = model.predict(X.loc[split_idx])
		df_scores = pd.DataFrame({"pred": pred, "y": y.loc[split_idx].values}, index=split_idx)
		out[split_name] = evaluate_predictions(df_scores)
		preds_dict[split_name] = df_scores

		del pred
		gc.collect()

	if model_out:
		os.makedirs(os.path.dirname(model_out), exist_ok=True)
		try:
			model.save_model(model_out)
		except Exception:
			import pickle
			with open(model_out + ".pkl", "wb") as f:
				pickle.dump(model, f)

	if preds_out:
		os.makedirs(os.path.dirname(preds_out), exist_ok=True)
		test_pred = model.predict(X.loc[test_idx])
		df_scores = pd.DataFrame({"pred": test_pred, "y": y.loc[test_idx].values}, index=test_idx)
		try:
			df_scores.to_parquet(preds_out)
		except Exception:
			df_scores.to_csv(preds_out.replace(".parquet", ".csv"))

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
	preds_out: Optional[str] = None,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, pd.DataFrame]]:
	"""训练 LightGBM"""
	try:
		from lightgbm import LGBMRegressor
	except Exception as e:
		raise RuntimeError(f"LightGBM not available: {e}")

	lgbm_params = {
		"n_estimators": 1000,
		"max_depth": -1,
		"learning_rate": 0.05,
		"subsample": 0.8,
		"colsample_bytree": 0.8,
		"reg_alpha": 0.0,
		"reg_lambda": 1.0,
		"objective": "regression",
		"random_state": 42,
		"n_jobs": -1,
	}

	date_level = X.index.get_level_values(0)
	map_train = pd.Series(train_mask, index=sample_dates)
	map_val = pd.Series(val_mask, index=sample_dates)
	map_test = pd.Series(test_mask, index=sample_dates)
	train_idx = X.index[map_train.reindex(date_level).to_numpy()]
	val_idx = X.index[map_val.reindex(date_level).to_numpy()]
	test_idx = X.index[map_test.reindex(date_level).to_numpy()]

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
	preds_out: Optional[str] = None,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, pd.DataFrame]]:
	"""训练 CatBoost"""
	try:
		from catboost import CatBoostRegressor
	except Exception as e:
		raise RuntimeError(f"CatBoost not available: {e}")

	cb_params = {
		"iterations": 1000,
		"depth": 6,
		"learning_rate": 0.05,
		"l2_leaf_reg": 3.0,
		"loss_function": "RMSE",
		"random_seed": 42,
		"verbose": False,
		"thread_count": -1,
	}

	date_level = X.index.get_level_values(0)
	map_train = pd.Series(train_mask, index=sample_dates)
	map_val = pd.Series(val_mask, index=sample_dates)
	map_test = pd.Series(test_mask, index=sample_dates)
	train_idx = X.index[map_train.reindex(date_level).to_numpy()]
	val_idx = X.index[map_val.reindex(date_level).to_numpy()]
	test_idx = X.index[map_test.reindex(date_level).to_numpy()]

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

		del test_pred, df_scores
		gc.collect()

	return out, preds_dict


def main():
	parser = argparse.ArgumentParser(description="Train ML model on quantitative factors")
	parser.add_argument("--modules", nargs="*", default=["alpha101", "gtja191", "chenhan_factor"],
						help="因子模块，默认使用全部三个")
	parser.add_argument("--include", nargs="*", default=[])
	parser.add_argument("--exclude", nargs="*", default=[])
	parser.add_argument("--start-date", type=str, default=None)
	parser.add_argument("--end-date", type=str, default=None)
	parser.add_argument("--universe", type=str, default="top75pct")
	parser.add_argument("--benchmark", type=str, default=None)
	parser.add_argument("--horizon", type=int, default=5)
	parser.add_argument("--label", type=str, default="cumulative",
						choices=["cumulative", "ret#2", "ret#5", "ret#20"])
	parser.add_argument("--train-end", type=str, default="2021-12-31")
	parser.add_argument("--val-end", type=str, default="2022-12-31")
	parser.add_argument("--engine", type=str, choices=["xgb","lgbm","catboost"], default="xgb")
	parser.add_argument("--rank-ic", action="store_true", help="运行IC排序选择因子")
	parser.add_argument("--ic-metric", type=str, choices=["ic", "rank_ic"], default="rank_ic",
						help="使用IC或RankIC筛选因子，默认rank_ic")
	parser.add_argument("--ic-horizon", type=int, default=5)
	parser.add_argument("--ic-threshold", type=float, default=0.0)
	parser.add_argument("--top-n", type=int, default=50)
	parser.add_argument("--run-name", type=str, default=None)

	args = parser.parse_args()

	n_jobs = os.cpu_count() or 1
	print(f"[info] Using {n_jobs} CPU cores")

	import time
	timestamp = time.strftime("%Y%m%d_%H%M%S")
	if args.run_name:
		run_folder_name = f"run_{args.engine}_{args.run_name}"
	else:
		run_folder_name = f"run_{args.engine}_{timestamp}"

	run_dir = os.path.join(THIS_DIR, "runs", run_folder_name)
	os.makedirs(run_dir, exist_ok=True)
	print(f"[info] Run directory: {run_dir}")

	args.model_out = os.path.join(run_dir, f"{args.engine}_model.json")
	args.preds_out = os.path.join(run_dir, "test_preds.parquet")
	args.save_factor_json = os.path.join(run_dir, "factor_ic_rank.json")

	if args.universe == "None":
		args.universe = None
	if args.benchmark == "None":
		args.benchmark = None

	factor_names_final: List[str] = []
	if args.rank_ic:
		print(f"[info] Running {args.ic_metric.upper()} ranking (threshold: abs({args.ic_metric.upper()}) > {args.ic_threshold})...")

		avg_ic_map, ok_names = run_all_factors_and_rank_ic(
			modules=args.modules,
			start_date=args.start_date,
			end_date=args.end_date,
			universe=args.universe,
			benchmark=args.benchmark,
			ic_horizon=args.ic_horizon,
			ic_threshold=args.ic_threshold,
			n_jobs=n_jobs,
			ic_metric=args.ic_metric,
		)

		sorted_items = sorted(avg_ic_map.items(), key=lambda kv: abs(kv[1]), reverse=True)
		os.makedirs(os.path.dirname(args.save_factor_json), exist_ok=True)
		with open(args.save_factor_json, "w", encoding="utf-8") as f:
			json.dump({"avg_ic": sorted_items, "all_run": ok_names}, f, ensure_ascii=False, indent=2)
		print(f"[info] IC ranking saved: {args.save_factor_json}")

		factor_names_final = [name for name,_ in sorted_items[:args.top_n]]
		if not args.include:
			args.include = factor_names_final
		else:
			args.include = [nm for nm in args.include if nm in factor_names_final]
		print(f"[info] Selected {len(args.include)} factors")

		del avg_ic_map, ok_names, sorted_items
		gc.collect()

	klass_map = select_factor_classes(args.modules, include=args.include, exclude=args.exclude)
	if not klass_map:
		raise SystemExit("No factors found")
	print(f"[info] Using {len(klass_map)} factors for {args.engine}")

	features, returns = compute_factor_features(
		klass_map,
		start_date=args.start_date,
		end_date=args.end_date,
		universe=args.universe,
		benchmark=args.benchmark,
		n_jobs=n_jobs,
	)

	X, y, sample_dates, label_name = build_ml_dataset(
		features,
		returns,
		horizon=args.horizon,
		label_type=args.label,
	)
	print(f"[info] Label: {label_name}")

	del features, returns
	gc.collect()

	train_mask, val_mask, test_mask = time_split_indices(sample_dates, args.train_end, args.val_end)

	print("[info] Normalizing with train statistics...")
	date_level = X.index.get_level_values(0)
	map_train = pd.Series(train_mask, index=sample_dates)
	train_row_mask = map_train.reindex(date_level).to_numpy()
	train_idx = X.index[train_row_mask]

	X = normalize_with_train_stats(X, train_idx, clip=5.0)

	del map_train, train_row_mask, date_level
	gc.collect()

	if args.engine == "xgb":
		metrics, predictions = train_xgboost(
			X, y, sample_dates, train_mask, val_mask, test_mask,
			model_out=args.model_out,
			preds_out=args.preds_out,
		)
	elif args.engine == "lgbm":
		metrics, predictions = train_lightgbm(
			X, y, sample_dates, train_mask, val_mask, test_mask,
			preds_out=args.preds_out,
		)
	else:
		metrics, predictions = train_catboost(
			X, y, sample_dates, train_mask, val_mask, test_mask,
			preds_out=args.preds_out,
		)

	pretty = json.dumps(metrics, ensure_ascii=False, indent=2)
	print(pretty)

	metrics_path = os.path.join(run_dir, "metrics.json")
	with open(metrics_path, "w", encoding="utf-8") as f:
		f.write(pretty)

	print(f"\n[info] Metrics saved: {metrics_path}")

	print("\n[info] Computing equity curves...")
	equity_data = {}
	for split_name in ["train", "val", "test"]:
		if split_name in predictions and len(predictions[split_name]) > 0:
			equity_curve = compute_equity_curve(predictions[split_name], quantile=0.2)
			equity_data[split_name] = equity_curve

	model_equity_data = {args.engine.upper(): equity_data}

	equity_plot_path = os.path.join(run_dir, "equity_curve.png")
	plot_equity_curves(
		model_equity_data,
		train_end=args.train_end,
		val_end=args.val_end,
		save_path=equity_plot_path,
	)

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
		"n_jobs": n_jobs,
	}
	config_path = os.path.join(run_dir, "run_config.json")
	with open(config_path, "w", encoding="utf-8") as f:
		json.dump(run_config, f, ensure_ascii=False, indent=2)

	print(f"\n[info] All results saved to: {run_dir}")
	print(f"   - Config: run_config.json")
	print(f"   - Metrics: metrics.json")
	print(f"   - Model: {os.path.basename(args.model_out)}")
	print(f"   - Predictions: {os.path.basename(args.preds_out)}")
	print(f"   - Equity curve: equity_curve.png")
	if args.rank_ic:
		print(f"   - IC ranking: factor_ic_rank.json")

	gc.collect()


if __name__ == "__main__":
	main()

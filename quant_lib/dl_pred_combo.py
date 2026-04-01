"""
DL Model Prediction Combo — 将深度学习模型预测值接入 factor_combo 回测框架。

Usage:
    # CLI 方式 — 一键换模型
    cd /home/ch/hchen/SMQF/quant_lib
    python3 dl_pred_combo.py --label ret#l#5 --experiment resmlp.f95.lml --seed 111
    python3 dl_pred_combo.py --label ret#l#22 --experiment resmlp.f95.hybrid --seed 111
    python3 dl_pred_combo.py --list                    # 列出所有可用模型
    python3 dl_pred_combo.py --list --label ret#l#22   # 列出某个 label 下的模型

    # Python API 方式
    from dl_pred_combo import DLPredCombo, quick_backtest
    f = quick_backtest("ret#l#5", "resmlp.f95.lml", seed=111)
"""

import os
import json
import time
import argparse
import numpy as np
import pandas as pd
from factor_combo_base import factor_combo_base

# ── 默认路径 ──────────────────────────────────────────────────────────
MODELS_ROOT = "/home/ch/data/result/models/ashare"
DATA_FILE = "/home/ch/data/Ashare/features/ashare_stock_daily.parquet"


class DLPredCombo(factor_combo_base):
    """将 all_pred.parquet 中的模型预测值作为因子，接入 factor_combo 回测框架。"""

    def __init__(
        self,
        model_dir: str = None,
        label: str = None,
        experiment: str = None,
        seed: int = None,
        split: str = None,
        models_root: str = MODELS_ROOT,
        data_file: str = DATA_FILE,
        **kwargs,
    ):
        # ── 解析模型路径 ─────────────────────────────────────────────
        if model_dir is not None:
            self._model_dir = model_dir
        elif label and experiment and seed is not None:
            self._model_dir = os.path.join(models_root, label, experiment, f"sd{seed}")
        else:
            raise ValueError("需要指定 model_dir 或 (label, experiment, seed)")

        pred_file = os.path.join(self._model_dir, "all_pred.parquet")
        if not os.path.isfile(pred_file):
            raise FileNotFoundError(f"找不到预测文件: {pred_file}")

        self._pred_file = pred_file
        self._split = split
        self._data_file = data_file

        # ── 从 args.json 读取 label（用于加载对应 returns 列）────────
        self._ret_col = "ret#l#1"
        args_file = os.path.join(self._model_dir, "args.json")
        if os.path.isfile(args_file):
            with open(args_file, "r") as fh:
                model_args = json.load(fh)
            if "label" in model_args:
                self._ret_col = model_args["label"]

        # ── 构造 combo 名 ───────────────────────────────────────────
        exp_name = os.path.basename(os.path.dirname(self._model_dir))
        seed_name = os.path.basename(self._model_dir)
        combo_name = f"{exp_name}.{seed_name}"
        if self._split:
            combo_name += f".{self._split}"

        # ── 设置默认 combo_property ──────────────────────────────────
        cp = kwargs.get("combo_property") or {}
        cp.setdefault("combo_name", combo_name)
        cp.setdefault("factors_needed", ["dl_pred"])
        cp.setdefault("lag_periods", 0)
        kwargs["combo_property"] = cp

        super().__init__(**kwargs)

    # ── 重写 load_data: 从 parquet 加载 returns 和 DL prediction ─────
    def load_data(self):
        # 加载 returns
        df = pd.read_parquet(self._data_file, columns=["Time", "Ticker", self._ret_col])
        returns_wide = df.pivot(index="Time", columns="Ticker", values=self._ret_col)
        returns_wide.index = pd.to_datetime(returns_wide.index)
        returns_wide.sort_index(inplace=True)
        self.returns = returns_wide
        del df

        # 加载 DL prediction
        pred = pd.read_parquet(self._pred_file)
        if self._split is not None:
            pred = pred[pred["split"] == self._split]
        pred["date"] = pd.to_datetime(pred["date"], format="%Y%m%d")
        factor_wide = pred.pivot(index="date", columns="symbol", values="pred")
        factor_wide.sort_index(inplace=True)
        del pred

        # 对齐 dates 和 tickers
        common_dates = self.returns.index.intersection(factor_wide.index)
        common_tickers = self.returns.columns.intersection(factor_wide.columns)
        self.returns = self.returns.loc[common_dates, common_tickers]
        self.factors_data["dl_pred"] = factor_wide.loc[common_dates, common_tickers].astype(float)
        self.factors_property["dl_pred"] = {"lag_periods": 0}

        # universe (如果需要)
        if self.combo_property.get("universe") is not None:
            universe_name = self.combo_property["universe"]
            try:
                udf = pd.read_parquet(self._data_file, columns=["Time", "Ticker", universe_name])
                self.universe = udf.pivot(index="Time", columns="Ticker", values=universe_name)
                self.universe.index = pd.to_datetime(self.universe.index)
                self.universe = self.universe.reindex(index=common_dates, columns=common_tickers)
            except Exception:
                self.universe = None
                self.combo_property["universe"] = None

        self.align_data()

    # ── 重写 align_data: 适配 factors_data 字典 ─────────────────────
    def align_data(self):
        if self.params["start_date"] is None:
            start_date = self.returns.index[0]
        else:
            start_date = pd.to_datetime(self.params["start_date"])
        if self.params["end_date"] is None:
            end_date = self.returns.index[-1]
        else:
            end_date = pd.to_datetime(self.params["end_date"])

        if self.params["start_date"] is not None or self.params["end_date"] is not None:
            for fld in self.factors_data:
                self.factors_data[fld] = self.factors_data[fld][start_date:end_date]
            self.returns = self.returns[start_date:end_date]
            if self.universe is not None:
                self.universe = self.universe[start_date:end_date]
            if self.index_returns is not None:
                self.index_returns = self.index_returns[start_date:end_date]

        self.params["start_date"] = start_date
        self.params["end_date"] = end_date

    # ── 实现 calc_subperiod_combo_positions ──────────────────────────
    def calc_subperiod_combo_positions(self, i0, i1, i2):
        factor_data = self.factors_data["dl_pred"]
        lag_periods = self.combo_property["lag_periods"]

        training_data = factor_data.iloc[i0:i1, :]
        testing_data = factor_data.iloc[i1:i2, :]

        training_pos = self.calc_rank_pos(lag_periods, training_data)
        testing_pos = self.calc_rank_pos(lag_periods, testing_data)

        return training_pos, testing_pos

    def set_combo_property(self):
        return {}


# ══════════════════════════════════════════════════════════════════════
# 便捷函数
# ══════════════════════════════════════════════════════════════════════

def list_available_models(label: str = None, models_root: str = MODELS_ROOT):
    """列出所有可用模型。"""
    results = []
    labels = [label] if label else sorted(os.listdir(models_root))
    for lb in labels:
        lb_dir = os.path.join(models_root, lb)
        if not os.path.isdir(lb_dir) or lb == "experiment_summary":
            continue
        for exp in sorted(os.listdir(lb_dir)):
            exp_dir = os.path.join(lb_dir, exp)
            if not os.path.isdir(exp_dir) or exp == "experiment_summary":
                continue
            for sd in sorted(os.listdir(exp_dir)):
                pred_file = os.path.join(exp_dir, sd, "all_pred.parquet")
                if os.path.isfile(pred_file):
                    results.append({
                        "label": lb,
                        "experiment": exp,
                        "seed": sd,
                        "path": os.path.join(exp_dir, sd),
                    })
    return pd.DataFrame(results)


def quick_backtest(
    label: str,
    experiment: str,
    seed: int = 111,
    split: str = None,
    testing_periods: int = 1,
    testing_period_length: int = 500,
):
    """一键回测: 指定 label/experiment/seed 即可。"""
    params = {
        "ir_details": True,
        "testing_periods": testing_periods,
        "testing_period_length": testing_period_length,
    }

    f = DLPredCombo(
        label=label,
        experiment=experiment,
        seed=seed,
        split=split,
        params=params,
    )
    f.run()
    return f


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DL Model Prediction Backtest via SMQF factor_combo")
    parser.add_argument("--label", type=str, default="ret#l#5")
    parser.add_argument("--experiment", type=str, default="resmlp.f95.lml")
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--split", type=str, default=None, choices=["train", "valid", "test"])
    parser.add_argument("--testing-periods", type=str, default="1",
                        help="0=全量, 1=单次OOS, rolling=滚动")
    parser.add_argument("--testing-period-length", type=int, default=500,
                        help="OOS 天数 (默认 500 约 2 年)")
    parser.add_argument("--quantile", type=float, default=0.2)
    parser.add_argument("--model-dir", type=str, default=None)
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    if args.list:
        df = list_available_models(args.label)
        print(df.to_string(index=False))
    else:
        tp = args.testing_periods
        if tp == "rolling":
            pass
        elif tp.isdigit():
            tp = int(tp)
        params = {
            "ir_details": True,
            "testing_periods": tp,
            "testing_period_length": args.testing_period_length,
            "quantile": args.quantile,
        }

        f = DLPredCombo(
            model_dir=args.model_dir,
            label=args.label,
            experiment=args.experiment,
            seed=args.seed,
            split=args.split,
            params=params,
        )
        f.run()

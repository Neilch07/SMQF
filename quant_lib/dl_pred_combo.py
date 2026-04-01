"""
DL Model Prediction — factor_combo_base 子类，接入 SMQF 回测框架。

继承 factor_combo_base，仅覆写 load_data / align_data / calc_subperiod_combo_positions。
无重复代码——所有回测逻辑复用基类。

Usage:
    # 单模型回测
    python3 dl_pred_combo.py --label ret#l#5 --experiment resmlp.f95.lml --seed 111

    # 批量回测所有 ashare 模型
    python3 dl_pred_combo.py --batch

    # 批量回测某个 label
    python3 dl_pred_combo.py --batch --label ret#l#22

    # 训练后自动回测（由 cs_model/train_ashare.py 调用）
    python3 dl_pred_combo.py --model-dir /path/to/sd111

    # 列出可用模型
    python3 dl_pred_combo.py --list
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from factor_combo_base import factor_combo_base

MODELS_ROOT = os.environ.get("CS_MODEL_ASHARE_MODELS",
                              "/home/ch/data/result/models/ashare")
DATA_FILE = os.environ.get("CS_MODEL_ASHARE_DATA_FILE",
                            "/home/ch/data/Ashare/features/ashare_stock_daily.parquet")


class DLPredCombo(factor_combo_base):
    """factor_combo_base 子类: 从 all_pred.parquet 加载 DL 预测作为因子。"""

    def __init__(self, model_dir, data_file=DATA_FILE, split=None, **kwargs):
        pred_file = os.path.join(model_dir, "all_pred.parquet")
        if not os.path.isfile(pred_file):
            raise FileNotFoundError(f"找不到: {pred_file}")

        self._model_dir = model_dir
        self._pred_file = pred_file
        self._split = split
        self._data_file = data_file

        # 从 args.json 读 label 列名
        self._ret_col = "ret#l#1"
        args_file = os.path.join(model_dir, "args.json")
        if os.path.isfile(args_file):
            with open(args_file) as fh:
                a = json.load(fh)
            self._ret_col = a.get("label", self._ret_col)

        # combo 名 = experiment.seed
        exp = os.path.basename(os.path.dirname(model_dir))
        sd = os.path.basename(model_dir)
        name = f"{exp}.{sd}" + (f".{split}" if split else "")

        cp = kwargs.pop("combo_property", None) or {}
        cp.setdefault("combo_name", name)
        cp.setdefault("factors_needed", ["dl_pred"])
        cp.setdefault("lag_periods", 0)
        super().__init__(combo_property=cp, **kwargs)

    # ── load_data: parquet 替代 pkl ──────────────────────────────────
    def load_data(self):
        # returns
        df = pd.read_parquet(self._data_file, columns=["Time", "Ticker", self._ret_col])
        ret = df.pivot(index="Time", columns="Ticker", values=self._ret_col)
        ret.index = pd.to_datetime(ret.index)
        ret.sort_index(inplace=True)
        self.returns = ret
        del df

        # DL prediction → wide
        pred = pd.read_parquet(self._pred_file)
        if self._split:
            pred = pred[pred["split"] == self._split]
        pred["date"] = pd.to_datetime(pred["date"], format="%Y%m%d")
        fac = pred.pivot(index="date", columns="symbol", values="pred")
        fac.sort_index(inplace=True)
        del pred

        # 对齐
        idx = self.returns.index.intersection(fac.index)
        col = self.returns.columns.intersection(fac.columns)
        self.returns = self.returns.loc[idx, col]
        self.factors_data["dl_pred"] = fac.loc[idx, col].astype(float)
        self.factors_property["dl_pred"] = {"lag_periods": 0}
        self.align_data()

    # ── align_data: 适配 factors_data dict ───────────────────────────
    def align_data(self):
        s = pd.to_datetime(self.params["start_date"]) if self.params["start_date"] else self.returns.index[0]
        e = pd.to_datetime(self.params["end_date"]) if self.params["end_date"] else self.returns.index[-1]
        if self.params["start_date"] or self.params["end_date"]:
            for k in self.factors_data:
                self.factors_data[k] = self.factors_data[k][s:e]
            self.returns = self.returns[s:e]
            if self.index_returns is not None:
                self.index_returns = self.index_returns[s:e]
        self.params["start_date"] = s
        self.params["end_date"] = e

    # ── 抽象方法实现 ─────────────────────────────────────────────────
    def calc_subperiod_combo_positions(self, i0, i1, i2):
        fac = self.factors_data["dl_pred"]
        lag = self.combo_property["lag_periods"]
        return self.calc_rank_pos(lag, fac.iloc[i0:i1]), self.calc_rank_pos(lag, fac.iloc[i1:i2])

    def set_combo_property(self):
        return {}


# ══════════════════════════════════════════════════════════════════════
# CLI 工具函数
# ══════════════════════════════════════════════════════════════════════

def _resolve_model_dir(args):
    """从 CLI 参数解析 model_dir。"""
    if args.model_dir:
        return args.model_dir
    return os.path.join(MODELS_ROOT, args.label, args.experiment, f"sd{args.seed}")


def _discover_models(label=None):
    """扫描 MODELS_ROOT 下所有有 all_pred.parquet 的模型目录。"""
    rows = []
    labels = [label] if label else sorted(os.listdir(MODELS_ROOT))
    for lb in labels:
        lb_dir = os.path.join(MODELS_ROOT, lb)
        if not os.path.isdir(lb_dir) or lb.startswith(".") or lb == "experiment_summary":
            continue
        for exp in sorted(os.listdir(lb_dir)):
            exp_dir = os.path.join(lb_dir, exp)
            if not os.path.isdir(exp_dir) or exp == "experiment_summary":
                continue
            for sd in sorted(os.listdir(exp_dir)):
                if os.path.isfile(os.path.join(exp_dir, sd, "all_pred.parquet")):
                    rows.append({"label": lb, "experiment": exp, "seed": sd,
                                 "path": os.path.join(exp_dir, sd)})
    return rows


def run_single(model_dir, params, split=None):
    """回测单个模型，返回 results dict。"""
    f = DLPredCombo(model_dir=model_dir, split=split, params=params)
    f.run()
    return f.results


def run_batch(label=None, params=None, split=None):
    """批量回测所有模型，输出汇总表。"""
    if params is None:
        params = {"ir_details": True, "testing_periods": 1,
                  "testing_period_length": 500, "display": False}

    models = _discover_models(label)
    if not models:
        print("[warn] 没有找到可用模型")
        return

    rows = []
    for m in models:
        tag = f"{m['label']}/{m['experiment']}/{m['seed']}"
        try:
            f = DLPredCombo(model_dir=m["path"], split=split, params=dict(params))
            f.run(turnoff_display=True)
            r = f.results
            rows.append({
                "label": m["label"], "experiment": m["experiment"], "seed": m["seed"],
                "train_ret": r.get("Annualized Training Return", np.nan),
                "train_ir": r.get("Annualized Training IR", np.nan),
                "test_ret": r.get("Annualized Testing Return", np.nan),
                "test_ir": r.get("Annualized Testing IR", np.nan),
                "test_net_ir": r.get("Annualized Testing Net IR", np.nan),
                "test_mdd": r.get("Maximum Testing Drawdown", np.nan),
            })
            print(f"[ok] {tag}: test_ir={rows[-1]['test_ir']:.4f}")
        except Exception as ex:
            print(f"[fail] {tag}: {ex}")

    if not rows:
        return

    df = pd.DataFrame(rows)

    # 按 experiment 聚合 (均值)
    agg = df.groupby(["label", "experiment"]).agg(
        n_seeds=("seed", "count"),
        test_ret_mean=("test_ret", "mean"),
        test_ir_mean=("test_ir", "mean"),
        test_net_ir_mean=("test_net_ir", "mean"),
        test_mdd_mean=("test_mdd", "mean"),
    ).sort_values(["label", "test_ir_mean"], ascending=[True, False])

    print("\n" + "=" * 100)
    print("SMQF Combo Backtest Summary (mean across seeds)")
    print("=" * 100)
    print(agg.to_string(float_format="%.4f"))
    return df, agg


def main():
    parser = argparse.ArgumentParser(description="DL Prediction → SMQF factor_combo backtest")
    parser.add_argument("--label", type=str, default="ret#l#5")
    parser.add_argument("--experiment", type=str, default="resmlp.f95.lml")
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--split", type=str, default=None, choices=["train", "valid", "test"])
    parser.add_argument("--model-dir", type=str, default=None)
    parser.add_argument("--testing-periods", type=str, default="1",
                        help="0=全量, 1=单次OOS, rolling=滚动")
    parser.add_argument("--testing-period-length", type=int, default=500)
    parser.add_argument("--quantile", type=float, default=0.2)
    parser.add_argument("--list", action="store_true", help="列出所有可用模型")
    parser.add_argument("--batch", action="store_true", help="批量回测所有模型")
    args = parser.parse_args()

    if args.list:
        for m in _discover_models(args.label if args.label != "ret#l#5" else None):
            print(f"{m['label']:>10s}  {m['experiment']:<35s}  {m['seed']}")
        return

    tp = args.testing_periods
    tp = tp if tp == "rolling" else int(tp)
    params = {"ir_details": True, "testing_periods": tp,
              "testing_period_length": args.testing_period_length,
              "quantile": args.quantile}

    if args.batch:
        lbl = args.label if args.label != "ret#l#5" else None
        run_batch(label=lbl, params={**params, "display": False}, split=args.split)
    else:
        model_dir = _resolve_model_dir(args)
        run_single(model_dir, params, split=args.split)


if __name__ == "__main__":
    main()

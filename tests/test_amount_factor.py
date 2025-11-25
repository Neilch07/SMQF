#!/usr/bin/env python
"""测试带amount字段的因子"""
import os
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
QLIB_DIR = os.path.join(THIS_DIR, "quant_lib")
if QLIB_DIR not in sys.path:
    sys.path.append(QLIB_DIR)

from quant_lib.alpha101 import alpha101_005

# 测试加载alpha101_005 (需要 amount 字段)
params = {
    "start_date": "2021-01-01",
    "end_date": "2023-12-31",
    "save": False,
    "display": False,
}
factor_property = {
    "universe": "top75pct",
    "benchmark": None,
}

print("Creating factor instance...")
f_obj = alpha101_005(params=params, factor_property=factor_property)

print("Factor property:", f_obj.factor_property)
print("Data needed:", f_obj.data_needed)

print("\nLoading data...")
try:
    f_obj.load_data()
    print("Data loaded successfully!")
    print("Data keys:", list(f_obj.data.keys()))
    for key in f_obj.data.keys():
        print(f"  {key}: shape = {f_obj.data[key].shape}")
except Exception as e:
    import traceback
    print(f"Error loading data: {e}")
    traceback.print_exc()

print("\nCalculating factor...")
try:
    f_obj.calc_factor()
    print("Factor calculated successfully!")
    print("Factor data shape:", f_obj.factor_data.shape)
except Exception as e:
    import traceback
    print(f"Error calculating factor: {e}")
    traceback.print_exc()

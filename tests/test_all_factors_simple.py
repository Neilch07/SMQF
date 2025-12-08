#!/usr/bin/env python
"""简单测试单个因子"""
import os
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
QLIB_DIR = os.path.join(THIS_DIR, "..", "quant_lib")
if QLIB_DIR not in sys.path:
    sys.path.append(QLIB_DIR)

from quant_lib.alpha101 import alpha101_001

def main():
    params = {
        "start_date": "2022-01-01",
        "end_date": "2023-12-31",
        "save": False,
        "display": False,
    }
    factor_property = {
        "universe": "top75pct",
        "benchmark": None,
    }

    print("Testing alpha101_001...")
    try:
        f_obj = alpha101_001(params=params, factor_property=factor_property)
        f_obj.run(turnoff_display=True)
        print("✓ alpha101_001 测试成功")
    except Exception as e:
        print(f"✗ alpha101_001 测试失败: {e}")

if __name__ == "__main__":
    main()

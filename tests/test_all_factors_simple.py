#!/usr/bin/env python
"""简单测试所有因子"""
import os
import sys
import importlib

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
QLIB_DIR = os.path.join(THIS_DIR, "quant_lib")
if QLIB_DIR not in sys.path:
    sys.path.append(QLIB_DIR)

from quant_lib.factor import factor

def discover_factor_classes(module_name):
    full_module_name = f"quant_lib.{module_name}"
    m = importlib.import_module(full_module_name)
    classes = {}
    for k, v in m.__dict__.items():
        if isinstance(v, type) and v is not factor and issubclass(v, factor):
            if str(k).startswith("alpha"):
                classes[k] = v
    return classes

def main():
    modules = ["alpha101", "gtja191"]
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
    
    all_factors = {}
    for mod in modules:
        all_factors.update(discover_factor_classes(mod))
    
    print(f"总共发现 {len(all_factors)} 个因子")
    
    success_count = 0
    failed_factors = []
    
    for name, cls in sorted(all_factors.items()):
        try:
            f_obj = cls(params=params, factor_property=factor_property)
            f_obj.run(turnoff_display=True)
            success_count += 1
            print(f"✓ {name}")
        except Exception as e:
            failed_factors.append((name, str(e)))
            print(f"✗ {name}: {e}")
    
    print(f"\n成功: {success_count}/{len(all_factors)}")
    print(f"失败: {len(failed_factors)}/{len(all_factors)}")
    
    if success_count >= 50:
        print(f"\n前50个成功的因子可以用于训练!")

if __name__ == "__main__":
    main()

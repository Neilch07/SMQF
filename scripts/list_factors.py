#!/usr/bin/env python
"""列出所有可用的因子类"""
import os
import sys
import importlib

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
QLIB_DIR = os.path.join(THIS_DIR, "quant_lib")
if QLIB_DIR not in sys.path:
    sys.path.append(QLIB_DIR)

from quant_lib.factor import factor

def discover_factor_classes(module_name: str):
    """从给定模块中发现继承自 factor 的因子类。"""
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
    
    all_factors = {}
    for mod in modules:
        try:
            factors = discover_factor_classes(mod)
            all_factors.update(factors)
            print(f"\n模块 {mod}: 发现 {len(factors)} 个因子")
        except Exception as e:
            print(f"模块 {mod} 加载失败: {e}")
    
    print(f"\n总共发现 {len(all_factors)} 个因子:")
    for i, (name, cls) in enumerate(sorted(all_factors.items()), 1):
        # 尝试获取因子需要的数据字段
        try:
            obj = cls.__new__(cls)
            obj.set_factor_property()
            factor_prop = obj.set_factor_property()
            data_needed = factor_prop.get('data_needed', [])
            print(f"{i:3d}. {name:30s} - 需要字段: {data_needed}")
        except Exception as e:
            print(f"{i:3d}. {name:30s} - 无法获取属性: {e}")

if __name__ == "__main__":
    main()

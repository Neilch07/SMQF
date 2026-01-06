"""
因子测试脚本
包含因子质量诊断功能
"""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "quant_lib")))

from factor import factor
from analysis import *
from chenhan_factor import gp_factor_7
import numpy as np

import warnings

warnings.filterwarnings("ignore")


def quick_check_factor_quality(factor_data):
    """
    快速检查因子质量
    Returns: (is_valid, message)
    """
    if factor_data is None or factor_data.empty:
        return False, "因子数据为空"

    # 检查是否为常数
    all_values = factor_data.values.flatten()
    all_values = all_values[~np.isnan(all_values)]

    if len(all_values) == 0:
        return False, "因子全是NaN"

    unique_count = len(np.unique(all_values))
    if unique_count <= 1:
        return (
            False,
            f"因子是常数（唯一值: {all_values[0] if len(all_values) > 0 else 'NaN'}）",
        )

    # 检查横截面标准差
    cs_std = factor_data.std(axis=1).mean()
    if cs_std < 1e-6:
        return False, f"横截面标准差接近0 ({cs_std:.2e})，无区分度"

    return True, f"因子有效（{unique_count}个唯一值，横截面标准差={cs_std:.4f}）"


# 获取可用数据列表
f = factor()
print("可用数据字段:")
print(f.available_data())
print()

# 测试参数
params = {"ir_details": True}
factor_property = {"universe": "top75pct"}

# 测试 GP Factor 1
print("=" * 80)
print("测试 GP Factor 1")
print("表达式: ts_min_5(sign(add(sign(abs(sign(low))), cs_rank(abs(low)))))")
print("=" * 80)

f = gp_factor_7(params=params, factor_property=factor_property)

# 加载数据
print("\n[1/3] 加载数据...")
f.load_data()

# 计算因子
print("[2/3] 计算因子...")
f.calc_factor()

# 质量检查
print("[3/3] 质量检查...")
is_valid, message = quick_check_factor_quality(f.factor_data)

print(f"\n{'✅' if is_valid else '❌'} {message}")

if not is_valid:
    print("\n" + "=" * 80)
    print("⚠️  警告：因子质量检查未通过")
    print("=" * 80)
    print(
        """
这个因子是常数或缺乏区分度，无法用于量化交易。

可能原因：
1. 表达式中有过多的 sign() 嵌套
2. 表达式退化成了常数（如 sign(正数) = 1）
3. 数据处理有误

建议：
1. 检查 deap_factor_mining/FIX_SUMMARY.md 了解详细问题
2. 重新运行遗传算法生成新因子
3. 使用 test_factors_improved.py 进行完整诊断
    """
    )

    # 显示样本数据帮助调试
    print("\n因子数据样本（前5天，前5只股票）:")
    print(f.factor_data.iloc[:5, :5])

else:
    # 如果因子有效，继续完整评价
    print("\n继续进行完整性能评价...")
    print("-" * 80)
    f.calc_factor_performance()
    f.display()

print("\n" + "=" * 80)

"""
测试手工因子（非GP生成）
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'quant_lib')))

from chenhan_factor import vol_std20, ret_sum60_rinv, ret_20, std_20
import numpy as np
import warnings
warnings.filterwarnings("ignore")

def quick_check_factor_quality(factor_data):
    """快速检查因子质量"""
    if factor_data is None or factor_data.empty:
        return False, "因子数据为空"

    all_values = factor_data.values.flatten()
    all_values = all_values[~np.isnan(all_values)]

    if len(all_values) == 0:
        return False, "因子全是NaN"

    unique_count = len(np.unique(all_values))
    if unique_count <= 1:
        return False, f"因子是常数（唯一值: {all_values[0] if len(all_values) > 0 else 'NaN'}）"

    cs_std = factor_data.std(axis=1).mean()
    if cs_std < 1e-6:
        return False, f"横截面标准差接近0 ({cs_std:.2e})，无区分度"

    return True, f"因子有效（{unique_count}个唯一值，横截面标准差={cs_std:.4f}）"


def test_factor(factor_class, factor_name):
    """测试单个因子"""
    print("\n" + "="*80)
    print(f"测试: {factor_name}")
    print("="*80)

    params = {'ir_details': True}
    factor_property = {'universe': 'top75pct'}

    f = factor_class(params=params, factor_property=factor_property)

    print("加载数据...")
    f.load_data()

    print("计算因子...")
    f.calc_factor()

    print("质量检查...")
    is_valid, message = quick_check_factor_quality(f.factor_data)

    print(f"\n{'✅' if is_valid else '❌'} {message}")

    if is_valid:
        print("\n继续完整评价...")
        f.calc_factor_performance()
        results = f.get_performance()

        # 显示关键指标
        print(f"\n关键指标:")
        print(f"  IC: {results.get('daily_avg_ic', np.nan):.6f}")
        print(f"  ICIR: {results.get('anual_icir', np.nan):.4f}")
        print(f"  年化收益: {results.get('Annualized Return', np.nan):.4f}")
        print(f"  换手率: {results.get('avg_turn_over', np.nan):.4f}")

        return True
    else:
        print("\n因子样本数据（前3天，前3股票）:")
        print(f.factor_data.iloc[:3, :3])
        return False


if __name__ == '__main__':
    print("\n" + "="*80)
    print(" "*25 + "测试手工因子")
    print("="*80)

    # 测试手工因子
    manual_factors = [
        (vol_std20, "vol_std20 (成交量标准差)"),
        (ret_20, "ret_20 (20日动量)"),
        (std_20, "std_20 (20日波动率)"),
    ]

    results = {}
    for factor_class, factor_name in manual_factors:
        try:
            success = test_factor(factor_class, factor_name)
            results[factor_name] = success
        except Exception as e:
            print(f"\n❌ 错误: {str(e)}")
            results[factor_name] = False

    # 总结
    print("\n" + "="*80)
    print("总结")
    print("="*80)
    for name, success in results.items():
        status = "✅ 有效" if success else "❌ 无效"
        print(f"{name}: {status}")

    print("\n" + "="*80)
    print("\n💡 提示:")
    print("   - GP生成的因子(gp_factor_1~10)都是常数，需要重新挖掘")
    print("   - 手工因子应该是有效的")
    print("   - 运行 'python deap_factor_mining/run_mining.py' 生成新的GP因子")
    print("="*80)

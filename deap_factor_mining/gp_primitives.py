"""
遗传编程原语定义
定义因子表达式的构建块，包括数据字段、分析函数和数学运算
实现量纲对齐机制，确保生成的因子表达式在量纲上是合理的
"""

import numpy as np
import pandas as pd
from deap import gp
import sys
import os
import keyword

# 添加父目录到路径以导入quant_lib
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 添加quant_lib到路径
quant_lib_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'quant_lib'))
if quant_lib_dir not in sys.path:
    sys.path.insert(0, quant_lib_dir)

from analysis import *


# ============================================================================
# Python保留关键字处理
# ============================================================================

# 创建保留关键字到安全名称的映射
KEYWORD_MAPPING = {
    'return': 'ret',
    'class': 'cls',
    'def': 'def_',
    'import': 'import_',
    'from': 'from_',
    'for': 'for_',
    'while': 'while_',
    'if': 'if_',
    'else': 'else_',
    'elif': 'elif_',
    'try': 'try_',
    'except': 'except_',
    'finally': 'finally_',
    'with': 'with_',
    'as': 'as_',
    'pass': 'pass_',
    'break': 'break_',
    'continue': 'continue_',
    'yield': 'yield_',
    'lambda': 'lambda_',
    'global': 'global_',
    'nonlocal': 'nonlocal_',
    'assert': 'assert_',
    'raise': 'raise_',
    'del': 'del_',
    'in': 'in_',
    'is': 'is_',
    'not': 'not_',
    'and': 'and_',
    'or': 'or_',
}

# 反向映射：从安全名称到原始名称
REVERSE_KEYWORD_MAPPING = {v: k for k, v in KEYWORD_MAPPING.items()}


def sanitize_field_name(field_name):
    """
    将数据字段名转换为安全的Python标识符

    Args:
        field_name: 原始字段名

    Returns:
        safe_name: 安全的字段名
    """
    if keyword.iskeyword(field_name):
        return KEYWORD_MAPPING.get(field_name, field_name + '_')
    return field_name


def get_original_field_name(safe_name):
    """
    将安全的字段名转换回原始字段名

    Args:
        safe_name: 安全的字段名

    Returns:
        original_name: 原始字段名
    """
    return REVERSE_KEYWORD_MAPPING.get(safe_name, safe_name)


# ============================================================================
# 量纲类型定义
# ============================================================================

class Dimension:
    """量纲类，用于跟踪数据的单位类型"""
    PRICE = 'price'       # 价格类（close, open, high, low）
    VOLUME = 'volume'     # 成交量类
    RETURN = 'return'     # 收益率类
    RATIO = 'ratio'       # 无量纲比率（0-1之间，如排序结果）
    FACTOR = 'factor'     # 因子值（最终输出）
    ANY = 'any'           # 任意类型（用于通用函数）


# ============================================================================
# 安全包装函数：处理可能的数值异常
# ============================================================================

def safe_div(x, y):
    """安全除法：避免除零和产生inf"""
    # 将y中接近0的值替换为NaN，避免除零
    y_safe = y.where(np.abs(y) > 1e-8, np.nan)
    result = x / y_safe
    # 将inf和过大的值替换为NaN
    result = result.where(np.abs(result) < 1e8, np.nan)
    return result


def safe_neg(x):
    """安全取负：保持量纲"""
    return -x


def safe_add(x, y):
    """安全加法：保持量纲"""
    return x + y


def safe_sub(x, y):
    """安全减法：保持量纲"""
    return x - y


def safe_mul(x, y):
    """安全乘法：处理可能的数值溢出"""
    result = x * y
    # 将过大的值替换为NaN
    result = result.where(np.abs(result) < 1e8, np.nan)
    return result


# ============================================================================
# 时序函数包装器（保持量纲）
# ============================================================================

def prim_ts_delay(df, n):
    """时序延迟：量纲不变"""
    return ts_delay(df, int(n))


def prim_ts_delta(df, n):
    """时序差分：量纲不变"""
    return ts_delta(df, int(n))


def prim_ts_sum(df, n):
    """时序求和：量纲不变（返回累积值）"""
    return ts_sum(df, int(n))


def prim_ts_mean(df, n):
    """时序均值：量纲不变"""
    return ts_mean(df, int(n))


def prim_ts_std(df, n):
    """时序标准差：量纲不变"""
    return ts_std(df, int(n))


def prim_ts_rank(df, n):
    """时序排序：输出比率（0-1）"""
    return ts_rank(df, int(n), pct=True)


def prim_ts_max(df, n):
    """时序最大值：量纲不变"""
    return df.rolling(int(n), min_periods=int(n/2)).max()


def prim_ts_min(df, n):
    """时序最小值：量纲不变"""
    return df.rolling(int(n), min_periods=int(n/2)).min()


def prim_ts_argmax(df, n):
    """时序最大值位置：返回位置索引"""
    return ts_argmax(df, int(n))


def prim_ts_argmin(df, n):
    """时序最小值位置：返回位置索引"""
    return ts_argmin(df, int(n))


# ============================================================================
# 横截面函数包装器
# ============================================================================

def prim_cs_rank(df):
    """横截面排序：输出比率（0-1）"""
    return cs_rank(df, pct=True)


def prim_cs_zscore(df):
    """横截面Z-score标准化：量纲转为标准分数"""
    return cs_zscore(df)


# ============================================================================
# 数学函数包装器
# ============================================================================

def prim_abs(df):
    """绝对值：量纲不变"""
    return df.abs()


def prim_log(df):
    """对数：只对正值有效，量纲转换"""
    result = np.log(df.where(df > 0, np.nan))
    return result.replace([np.inf, -np.inf], np.nan)


def prim_sqrt(df):
    """平方根：只对非负值有效"""
    return np.sqrt(df.where(df >= 0, np.nan))


def prim_square(df):
    """平方"""
    result = df ** 2
    return result.where(np.abs(result) < 1e8, np.nan)


def prim_sign(df):
    """符号函数：返回-1, 0, 1"""
    return np.sign(df)


# ============================================================================
# 滚动窗口高级函数
# ============================================================================

def prim_rolling_kurt(df, n):
    """滚动窗口峰度"""
    return df.rolling(int(n), min_periods=int(n/2)).kurt()


def prim_rolling_skew(df, n):
    """滚动窗口偏度"""
    return df.rolling(int(n), min_periods=int(n/2)).skew()


def prim_ema(df, n):
    """指数移动平均：量纲不变"""
    return df.ewm(span=int(n), min_periods=int(n/2)).mean()


# ============================================================================
# 复合函数（结合价格和成交量等）
# ============================================================================

def prim_ts_corr(df1, df2, n):
    """时序相关系数：返回相关系数（-1到1）"""
    return ts_corr1(df1, df2, int(n))


# ============================================================================
# GP原语集构建函数
# ============================================================================

def create_primitive_set(window_sizes=[5, 10, 20, 40, 60],
                         data_fields=['close', 'volume', 'return'],
                         enable_division=True,
                         enable_high_complexity=False):
    """
    创建GP原语集

    Args:
        window_sizes: 时间窗口参数列表
        data_fields: 可用的数据字段
        enable_division: 是否启用除法
        enable_high_complexity: 是否启用高复杂度函数

    Returns:
        pset: DEAP原语集对象
    """

    # 创建原语集，使用命名参数（一个参数对应每个数据字段）
    # 输出类型为pd.DataFrame
    pset = gp.PrimitiveSet("MAIN", len(data_fields))

    # ========================================================================
    # 1. 重命名参数为数据字段名
    # ========================================================================

    # 将ARG0, ARG1, ... 重命名为 close, open, high, low, volume等
    # 注意：自动处理Python保留关键字（如 'return' -> 'ret'）
    for i, field in enumerate(data_fields):
        safe_field = sanitize_field_name(field)
        pset.renameArguments(**{f'ARG{i}': safe_field})

    # 保存原始字段名映射，供后续使用
    pset._original_field_names = data_fields
    pset._safe_field_names = [sanitize_field_name(f) for f in data_fields]

    # ========================================================================
    # 2. 添加算术运算符
    # ========================================================================

    pset.addPrimitive(safe_add, 2, name='add')
    pset.addPrimitive(safe_sub, 2, name='sub')
    pset.addPrimitive(safe_mul, 2, name='mul')
    if enable_division:
        pset.addPrimitive(safe_div, 2, name='div')
    pset.addPrimitive(safe_neg, 1, name='neg')

    # ========================================================================
    # 3. 添加时序函数
    # ========================================================================

    # 为每个窗口大小添加时序函数
    # 注意：函数名不能以数字开头或包含会导致Python语法错误的字符
    for win in window_sizes:
        # 基础统计函数
        pset.addPrimitive(lambda df, w=win: prim_ts_mean(df, w), 1,
                         name=f'ts_mean_{win}')
        pset.addPrimitive(lambda df, w=win: prim_ts_std(df, w), 1,
                         name=f'ts_std_{win}')
        pset.addPrimitive(lambda df, w=win: prim_ts_sum(df, w), 1,
                         name=f'ts_sum_{win}')

        # 延迟和差分
        pset.addPrimitive(lambda df, w=win: prim_ts_delay(df, w), 1,
                         name=f'ts_delay_{win}')
        pset.addPrimitive(lambda df, w=win: prim_ts_delta(df, w), 1,
                         name=f'ts_delta_{win}')

        # 排序
        pset.addPrimitive(lambda df, w=win: prim_ts_rank(df, w), 1,
                         name=f'ts_rank_{win}')

        # 最值
        pset.addPrimitive(lambda df, w=win: prim_ts_max(df, w), 1,
                         name=f'ts_max_{win}')
        pset.addPrimitive(lambda df, w=win: prim_ts_min(df, w), 1,
                         name=f'ts_min_{win}')

        # EMA
        pset.addPrimitive(lambda df, w=win: prim_ema(df, w), 1,
                         name=f'ema_{win}')

    # ========================================================================
    # 4. 添加横截面函数
    # ========================================================================

    pset.addPrimitive(prim_cs_rank, 1, name='cs_rank')
    pset.addPrimitive(prim_cs_zscore, 1, name='cs_zscore')

    # ========================================================================
    # 5. 添加数学函数
    # ========================================================================

    pset.addPrimitive(prim_abs, 1, name='abs')
    pset.addPrimitive(prim_sign, 1, name='sign')

    # ========================================================================
    # 6. 添加高复杂度函数（可选）
    # ========================================================================

    if enable_high_complexity:
        for win in window_sizes:
            pset.addPrimitive(lambda df, w=win: prim_rolling_kurt(df, w), 1,
                            name=f'kurt_{win}')
            pset.addPrimitive(lambda df, w=win: prim_rolling_skew(df, w), 1,
                            name=f'skew_{win}')

            # 相关系数（需要两个输入）
            pset.addPrimitive(lambda df1, df2, w=win: prim_ts_corr(df1, df2, w), 2,
                            name=f'ts_corr_{win}')

    return pset


# ============================================================================
# 辅助函数：表达式树深度限制
# ============================================================================

def get_tree_depth(individual):
    """获取个体的树深度"""
    if len(individual) == 0:
        return 0

    max_depth = 0
    stack = [(0, 0)]  # (节点索引, 深度)

    while stack:
        node_idx, depth = stack.pop()
        if node_idx >= len(individual):
            continue

        max_depth = max(max_depth, depth)

        node = individual[node_idx]
        if hasattr(node, 'arity'):  # 函数节点
            # 将子节点加入栈
            for i in range(node.arity):
                stack.append((node_idx + i + 1, depth + 1))

    return max_depth


def limit_depth(individual, max_depth):
    """限制个体的树深度，如果超过则裁剪"""
    current_depth = get_tree_depth(individual)
    if current_depth <= max_depth:
        return individual

    # 如果超过最大深度，替换为简单的终端节点
    # 这里简化处理：直接截断
    return individual[:min(len(individual), max_depth * 3)]


# ============================================================================
# 测试代码
# ============================================================================

if __name__ == '__main__':
    # 创建原语集
    pset = create_primitive_set(
        window_sizes=[20, 60],
        data_fields=['close', 'volume', 'return'],
        enable_division=True,
        enable_high_complexity=False
    )

    print("原语集创建成功！")
    print(f"原语数量: {len(pset.primitives[object])}")
    print(f"终端数量: {len(pset.terminals[object])}")
    print("\n终端节点:")
    for term in pset.terminals[object]:
        print(f"  - {term.name}")
    print("\n函数节点（前20个）:")
    for i, prim in enumerate(pset.primitives[object][:20]):
        print(f"  - {prim.name} (参数数量: {prim.arity})")

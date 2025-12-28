"""
基于遗传编程的因子类
继承自SMQF的factor基类，用于评估GP生成的因子表达式
"""

import sys
import os
import numpy as np
import pandas as pd
from deap import gp

# 添加父目录到路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 添加quant_lib到路径
quant_lib_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'quant_lib'))
if quant_lib_dir not in sys.path:
    sys.path.insert(0, quant_lib_dir)

from factor import factor


class GPFactor(factor):
    """
    基于遗传编程的因子类

    这个类接收一个GP个体（表达式树），并将其作为因子计算函数
    继承自factor基类，自动获得完整的因子评价功能
    """

    def __init__(self, individual, pset, factor_name='gp_factor', **kwargs):
        """
        初始化GP因子

        Args:
            individual: DEAP GP个体（表达式树）
            pset: DEAP原语集
            factor_name: 因子名称
            **kwargs: 传递给factor基类的其他参数
        """
        self.individual = individual
        self.pset = pset
        self._factor_name = factor_name
        self._func = None  # 编译后的函数

        # 调用父类初始化
        super().__init__(**kwargs)

    def set_factor_property(self):
        """
        设置因子属性

        分析GP表达式树，确定需要哪些数据字段
        """
        # 提取表达式中使用的所有数据字段
        data_needed = self._extract_data_fields()

        factor_property = {
            'factor_name': self._factor_name,
            'data_needed': data_needed,
            'factor_type': 'market'
        }
        return factor_property

    def _extract_data_fields(self):
        """
        从GP个体中提取需要的数据字段

        遍历表达式树，找出所有的终端节点（数据字段）
        """
        data_fields = set()

        # 遍历个体的所有节点
        for node in self.individual:
            # 检查是否是终端节点
            if isinstance(node, gp.Terminal):
                field_name = node.value
                # 确保是有效的数据字段名
                if isinstance(field_name, str) and field_name in [
                    'close', 'open', 'high', 'low', 'volume', 'return',
                    'amount', 'pct_change'
                ]:
                    data_fields.add(field_name)

        # 如果没有找到数据字段，使用默认的close
        if not data_fields:
            data_fields = {'close'}

        return list(data_fields)

    def calc_factor(self):
        """
        计算因子值

        真正执行GP表达式来生成因子值
        """
        try:
            # 编译GP表达式为可执行函数
            func = gp.compile(expr=self.individual, pset=self.pset)

            # 准备参数列表，按照pset中参数的顺序
            # pset.arguments是参数名列表（安全名称），按照它们在pset中定义的顺序
            # 需要将安全名称映射回原始字段名来查找数据
            args = []
            for arg_name in self.pset.arguments:
                # 尝试获取原始字段名（如果pset保存了映射的话）
                if hasattr(self.pset, '_safe_field_names') and hasattr(self.pset, '_original_field_names'):
                    # 找到对应的原始字段名
                    try:
                        idx = self.pset._safe_field_names.index(arg_name)
                        original_field = self.pset._original_field_names[idx]
                    except (ValueError, IndexError):
                        # 如果找不到，尝试使用反向映射
                        from gp_primitives import get_original_field_name
                        original_field = get_original_field_name(arg_name)
                else:
                    # 如果pset没有保存映射，尝试使用反向映射函数
                    from gp_primitives import get_original_field_name
                    original_field = get_original_field_name(arg_name)

                # 使用原始字段名查找数据
                if original_field in self.data:
                    args.append(self.data[original_field])
                elif arg_name in self.data:
                    # 回退：尝试使用安全字段名
                    args.append(self.data[arg_name])
                else:
                    # 如果数据中没有这个字段，使用close作为默认值
                    # 或者创建一个全零的DataFrame
                    if 'close' in self.data:
                        args.append(pd.DataFrame(
                            0.0,
                            index=self.data['close'].index,
                            columns=self.data['close'].columns
                        ))
                    else:
                        raise ValueError(f"Required data field '{arg_name}' (original: '{original_field}') not found in data")

            # 执行GP函数（传入位置参数）
            result = func(*args)

            # 确保结果是DataFrame
            if not isinstance(result, pd.DataFrame):
                # 如果是Series，转为DataFrame
                if isinstance(result, pd.Series):
                    result = result.to_frame()
                # 如果是标量，广播到DataFrame
                elif isinstance(result, (int, float)):
                    if 'close' in self.data:
                        result = pd.DataFrame(
                            result,
                            index=self.data['close'].index,
                            columns=self.data['close'].columns
                        )
                    else:
                        result = pd.DataFrame()
                else:
                    raise ValueError(f"GP expression returned unexpected type: {type(result)}")

            # 处理NaN和Inf
            result = result.replace([np.inf, -np.inf], np.nan)

            # 填充NaN：使用前向填充，然后用0填充剩余的NaN
            result = result.ffill().fillna(0)

            # 保存因子数据
            self.factor_data = result

        except Exception as e:
            # 如果计算失败，返回全零的DataFrame
            print(f"Error in calc_factor: {e}")
            print(f"Expression: {str(self.individual)}")
            import traceback
            traceback.print_exc()

            # 使用close数据的形状创建全零DataFrame
            if 'close' in self.data and not self.data['close'].empty:
                self.factor_data = pd.DataFrame(
                    0.0,
                    index=self.data['close'].index,
                    columns=self.data['close'].columns
                )
            else:
                self.factor_data = pd.DataFrame()

    def get_expression_string(self):
        """
        获取因子表达式的字符串表示

        Returns:
            str: 因子表达式字符串
        """
        return str(self.individual)

    def get_expression_tree(self):
        """
        获取表达式树的可视化表示

        Returns:
            str: 表达式树的图形表示
        """
        nodes, edges, labels = gp.graph(self.individual)

        # 创建简单的文本树表示
        tree_str = f"Expression: {self.get_expression_string()}\n"
        tree_str += f"Nodes: {len(nodes)}\n"
        tree_str += f"Depth: {self.individual.height}\n"

        return tree_str


def create_gp_factor_from_individual(individual, pset, factor_name='gp_factor',
                                     params=None, factor_property=None):
    """
    便捷函数：从GP个体创建因子对象

    Args:
        individual: DEAP GP个体
        pset: DEAP原语集
        factor_name: 因子名称
        params: 因子参数字典
        factor_property: 因子属性字典

    Returns:
        GPFactor: 因子对象
    """
    return GPFactor(
        individual=individual,
        pset=pset,
        factor_name=factor_name,
        params=params,
        factor_property=factor_property
    )


# ============================================================================
# 测试代码
# ============================================================================

if __name__ == '__main__':
    from deap import creator, base, tools
    from gp_primitives import create_primitive_set

    print("测试GP因子类...")

    # 创建原语集
    pset = create_primitive_set(
        window_sizes=[20],
        data_fields=['close', 'volume'],
        enable_division=False,
        enable_high_complexity=False
    )

    # 创建GP类型
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

    # 创建工具箱
    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)

    # 生成一个随机个体
    individual = toolbox.individual()

    print(f"生成的表达式: {individual}")
    print(f"表达式深度: {individual.height}")

    # 创建GP因子（注意：这里不会实际加载数据和计算）
    gp_factor = GPFactor(
        individual=individual,
        pset=pset,
        factor_name='test_gp_factor'
    )

    print(f"\n因子名称: {gp_factor.factor_property['factor_name']}")
    print(f"需要的数据: {gp_factor.factor_property['data_needed']}")
    print(f"因子类型: {gp_factor.factor_property['factor_type']}")

    print("\nGP因子类测试完成！")

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

        临时实现：使用简单的随机因子组合来测试整个框架
        这样可以验证GA、评价、保存等所有其他功能是否正常
        """
        try:
            # 根据表达式生成不同的简单因子
            # 这里使用表达式的字符串哈希来生成"伪随机"但可重复的因子

            expr_str = str(self.individual)
            hash_val = abs(hash(expr_str)) % 10

            # 根据哈希值选择不同的简单因子
            if 'close' in self.data:
                if hash_val == 0:
                    result = -self.data['close'].rolling(20).std()
                elif hash_val == 1:
                    result = -self.data['close'].rolling(10).std()
                elif hash_val == 2:
                    result = self.data['close'].pct_change(20, fill_method=None)
                elif hash_val == 3:
                    result = -self.data['close'].rolling(5).mean() / self.data['close'].rolling(20).mean()
                elif hash_val == 4 and 'volume' in self.data:
                    result = -self.data['volume'].rolling(20).std()
                elif hash_val == 5 and 'volume' in self.data:
                    result = self.data['volume'].pct_change(10, fill_method=None)
                elif hash_val == 6:
                    result = -(self.data['close'] / self.data['close'].shift(20))
                elif hash_val == 7:
                    result = self.data['close'].rolling(20).rank(pct=True)
                elif hash_val == 8:
                    result = -self.data['close'].ewm(span=20).std()
                else:
                    result = -self.data['close'].rolling(60).std()
            else:
                result = pd.DataFrame()

            # 确保结果是DataFrame
            if not isinstance(result, pd.DataFrame):
                raise ValueError(f"GP expression returned non-DataFrame type: {type(result)}")

            # 处理NaN和Inf
            result = result.replace([np.inf, -np.inf], np.nan)

            # 简单填充NaN（使用前向填充）
            result = result.ffill().fillna(0)

            # 保存因子数据
            self.factor_data = result

        except Exception as e:
            # 如果计算失败，返回全零的DataFrame（而不是NaN）
            print(f"Error in calc_factor: {e}")
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

"""
工具函数模块
提供各种辅助功能
"""

import numpy as np
import pandas as pd
from deap import gp
import pickle
import json
import os
from typing import List, Dict, Any


def expression_to_string(individual, pset):
    """
    将GP个体转换为人类可读的字符串

    Args:
        individual: DEAP GP个体
        pset: DEAP原语集

    Returns:
        str: 可读的表达式字符串
    """
    return str(individual)


def expression_to_python_code(individual, pset):
    """
    将GP个体转换为Python代码

    Args:
        individual: DEAP GP个体
        pset: DEAP原语集

    Returns:
        str: Python代码字符串
    """
    func = gp.compile(individual, pset)
    import inspect
    try:
        code = inspect.getsource(func)
        return code
    except:
        return str(individual)


def visualize_tree(individual, filename=None):
    """
    可视化GP表达式树

    Args:
        individual: DEAP GP个体
        filename: 保存图片的文件名（如果为None则显示）

    Returns:
        None
    """
    try:
        import matplotlib.pyplot as plt
        import networkx as nx
        from networkx.drawing.nx_pydot import graphviz_layout
    except ImportError:
        print("需要安装matplotlib, networkx和pydot来可视化树")
        return

    nodes, edges, labels = gp.graph(individual)

    # 创建有向图
    g = nx.DiGraph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)

    # 使用graphviz布局
    try:
        pos = graphviz_layout(g, prog="dot")
    except:
        pos = nx.spring_layout(g)

    # 绘图
    fig, ax = plt.subplots(figsize=(12, 8))

    # 绘制节点
    nx.draw_networkx_nodes(g, pos, node_color='lightblue',
                          node_size=1000, ax=ax)

    # 绘制边
    nx.draw_networkx_edges(g, pos, edge_color='gray',
                          arrows=True, ax=ax)

    # 绘制标签
    nx.draw_networkx_labels(g, pos, labels, font_size=10, ax=ax)

    ax.set_title(f"Expression Tree (Depth: {individual.height}, Size: {len(individual)})")
    ax.axis('off')

    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"树结构已保存到: {filename}")
    else:
        plt.show()

    plt.close()


def compare_factors(factor_results: List[Dict], top_n=10):
    """
    比较多个因子的性能

    Args:
        factor_results: 因子结果列表，每个元素是包含性能指标的字典
        top_n: 显示前N个因子

    Returns:
        pd.DataFrame: 比较结果表格
    """
    # 提取关键指标
    comparison_data = []

    for i, result in enumerate(factor_results[:top_n]):
        perf = result.get('performance', {})
        row = {
            'Rank': result.get('rank', i+1),
            'Fitness': result.get('fitness', np.nan),
            'ICIR': perf.get('icir', np.nan),
            'IR': perf.get('ir', np.nan),
            'IC': perf.get('ic', np.nan),
            'Annual Return': perf.get('annual_return', np.nan),
            'Max Drawdown': perf.get('max_drawdown', np.nan),
            'Turnover': perf.get('turnover', np.nan),
            'Depth': result.get('depth', np.nan),
            'Size': result.get('size', np.nan)
        }
        comparison_data.append(row)

    df = pd.DataFrame(comparison_data)

    # 格式化输出
    pd.options.display.float_format = '{:.4f}'.format

    return df


def load_best_factors(run_dir: str):
    """
    从运行目录加载最佳因子

    Args:
        run_dir: 运行目录路径

    Returns:
        list: 最佳因子列表
    """
    best_factors_path = os.path.join(run_dir, 'best_factors.json')

    if not os.path.exists(best_factors_path):
        raise FileNotFoundError(f"未找到文件: {best_factors_path}")

    with open(best_factors_path, 'r', encoding='utf-8') as f:
        best_factors = json.load(f)

    return best_factors


def load_factor_data(run_dir: str, rank: int):
    """
    加载特定排名因子的数据

    Args:
        run_dir: 运行目录路径
        rank: 因子排名（1开始）

    Returns:
        pd.DataFrame: 因子数据
    """
    factor_data_path = os.path.join(run_dir, f'factor_data_{rank}.pkl')

    if not os.path.exists(factor_data_path):
        raise FileNotFoundError(f"未找到文件: {factor_data_path}")

    with open(factor_data_path, 'rb') as f:
        factor_data = pickle.load(f)

    return factor_data


def create_factor_report(run_dir: str, output_file: str = None):
    """
    创建因子挖掘报告

    Args:
        run_dir: 运行目录路径
        output_file: 输出文件路径（HTML格式）

    Returns:
        str: 报告HTML内容
    """
    # 加载数据
    best_factors = load_best_factors(run_dir)
    config_path = os.path.join(run_dir, 'config.json')

    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # 创建HTML报告
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>DEAP因子挖掘报告</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            h1 { color: #333; }
            h2 { color: #666; margin-top: 30px; }
            table { border-collapse: collapse; width: 100%; margin-top: 10px; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #4CAF50; color: white; }
            tr:nth-child(even) { background-color: #f2f2f2; }
            .config { background-color: #f9f9f9; padding: 10px; border-radius: 5px; }
            .expression { font-family: monospace; background-color: #f0f0f0; padding: 5px; }
        </style>
    </head>
    <body>
        <h1>DEAP 遗传算法因子挖掘报告</h1>
        <p>运行目录: {run_dir}</p>

        <h2>配置信息</h2>
        <div class="config">
            <p><strong>种群大小:</strong> {population_size}</p>
            <p><strong>进化代数:</strong> {n_generations}</p>
            <p><strong>最大深度:</strong> {max_depth}</p>
            <p><strong>数据范围:</strong> {start_date} ~ {end_date}</p>
        </div>

        <h2>最佳因子 Top {top_n}</h2>
        <table>
            <tr>
                <th>排名</th>
                <th>适应度</th>
                <th>ICIR</th>
                <th>IR</th>
                <th>年化收益</th>
                <th>最大回撤</th>
                <th>深度</th>
                <th>表达式</th>
            </tr>
    """.format(
        run_dir=run_dir,
        population_size=config['ga']['population_size'],
        n_generations=config['ga']['n_generations'],
        max_depth=config['ga']['max_depth'],
        start_date=config['factor']['start_date'],
        end_date=config['factor']['end_date'],
        top_n=min(10, len(best_factors))
    )

    # 添加因子行
    for factor in best_factors[:10]:
        perf = factor.get('performance', {})
        html += """
            <tr>
                <td>{rank}</td>
                <td>{fitness:.4f}</td>
                <td>{icir:.4f}</td>
                <td>{ir:.4f}</td>
                <td>{annual_return:.4f}</td>
                <td>{max_drawdown:.4f}</td>
                <td>{depth}</td>
                <td class="expression">{expression}</td>
            </tr>
        """.format(
            rank=factor['rank'],
            fitness=factor['fitness'],
            icir=perf.get('icir', np.nan),
            ir=perf.get('ir', np.nan),
            annual_return=perf.get('annual_return', np.nan),
            max_drawdown=perf.get('max_drawdown', np.nan),
            depth=factor['depth'],
            expression=factor['expression'][:100]  # 截断过长的表达式
        )

    html += """
        </table>
    </body>
    </html>
    """

    # 保存到文件
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html)
        print(f"报告已保存到: {output_file}")

    return html


def check_expression_validity(individual, pset=None):
    """
    检查GP表达式是否包含无意义的模式

    检查以下无效模式：
    1. ts_corr(x, x) - 自相关，结果恒为1
    2. ts_cov(x, x) - 自协方差，等价于方差
    3. div(x, x) - 自除，结果恒为1
    4. sub(x, x) - 自减，结果恒为0
    5. sign(sign(...)) - 多层sign嵌套，容易导致常数
    6. sign(abs(...)) - abs后的sign总是1（对正数）

    Args:
        individual: DEAP GP个体
        pset: DEAP原语集（可选）

    Returns:
        tuple: (is_valid, invalid_reasons)
            - is_valid: bool, True表示表达式有效
            - invalid_reasons: list, 无效原因列表
    """
    invalid_reasons = []

    # 需要检查的函数名（这些函数不应该接受相同的参数）
    forbidden_same_args_funcs = [
        'ts_corr', 'ts_cov', 'div', 'sub', 'corr', 'cov'
    ]

    # 检查过多的sign嵌套
    sign_depth = _count_nested_signs(individual)
    if sign_depth > 2:
        invalid_reasons.append(
            f"Invalid pattern: Too many nested sign() operations (depth={sign_depth}). "
            f"This often leads to constant factors."
        )

    # 遍历表达式树，查找问题模式
    i = 0
    while i < len(individual):
        node = individual[i]

        # 检查是否是函数节点
        if hasattr(node, 'name') and hasattr(node, 'arity'):
            func_name_base = node.name

            # 检查是否是需要不同参数的函数（去除后缀如 _20, _60）
            for forbidden_func in forbidden_same_args_funcs:
                if func_name_base.startswith(forbidden_func):
                    # 检查是否是二元函数
                    if node.arity == 2:
                        # 获取两个参数的位置
                        arg1_idx = i + 1
                        arg2_idx = i + 2

                        # 查找第二个参数的实际位置（需要跳过第一个参数的子树）
                        arg2_idx = _get_second_arg_index(individual, i)

                        if arg1_idx < len(individual) and arg2_idx < len(individual):
                            arg1 = individual[arg1_idx]
                            arg2 = individual[arg2_idx]

                            # 检查两个参数是否相同（对于终端节点）
                            if _are_same_nodes(arg1, arg2):
                                invalid_reasons.append(
                                    f"Invalid pattern: {func_name_base}({arg1}, {arg2}) - "
                                    f"same arguments have no information"
                                )
                    break

        i += 1

    is_valid = len(invalid_reasons) == 0
    return is_valid, invalid_reasons


def _count_nested_signs(individual):
    """
    计算表达式中最大sign嵌套深度

    Args:
        individual: GP个体

    Returns:
        int: 最大sign嵌套深度
    """
    def count_signs_recursive(node_idx, depth=0):
        """递归计算sign嵌套深度"""
        if node_idx >= len(individual):
            return depth

        node = individual[node_idx]

        # 如果是sign函数，增加深度
        if hasattr(node, 'name') and node.name == 'sign':
            current_depth = depth + 1
            # 递归检查子节点
            if hasattr(node, 'arity') and node.arity > 0:
                child_idx = node_idx + 1
                max_child_depth = count_signs_recursive(child_idx, current_depth)
                return max_child_depth
            return current_depth

        # 如果不是sign，重置深度但继续检查子节点
        if hasattr(node, 'arity') and node.arity > 0:
            max_depth = depth
            child_idx = node_idx + 1
            for _ in range(node.arity):
                child_depth = count_signs_recursive(child_idx, depth)
                max_depth = max(max_depth, child_depth)
                # 跳到下一个子节点
                child_idx = _skip_subtree(individual, child_idx)
            return max_depth

        return depth

    max_sign_depth = 0
    for i in range(len(individual)):
        depth = count_signs_recursive(i, 0)
        max_sign_depth = max(max_sign_depth, depth)

    return max_sign_depth


def _skip_subtree(individual, node_idx):
    """
    跳过一个节点及其整个子树，返回下一个兄弟节点的索引

    Args:
        individual: GP个体
        node_idx: 当前节点索引

    Returns:
        int: 下一个兄弟节点的索引
    """
    if node_idx >= len(individual):
        return node_idx + 1

    node = individual[node_idx]
    idx = node_idx + 1
    nodes_to_skip = getattr(node, 'arity', 0)

    while nodes_to_skip > 0 and idx < len(individual):
        child_node = individual[idx]
        nodes_to_skip += getattr(child_node, 'arity', 0)
        nodes_to_skip -= 1
        idx += 1

    return idx


def _get_second_arg_index(individual, func_idx):
    """
    获取二元函数第二个参数的索引位置

    Args:
        individual: GP个体
        func_idx: 函数节点的索引

    Returns:
        int: 第二个参数的索引
    """
    if func_idx >= len(individual):
        return func_idx + 1

    # 跳过第一个参数（及其子树）
    idx = func_idx + 1
    nodes_to_skip = 1  # 需要跳过的节点数

    while nodes_to_skip > 0 and idx < len(individual):
        node = individual[idx]
        if hasattr(node, 'arity'):
            nodes_to_skip += node.arity
        nodes_to_skip -= 1
        idx += 1

    return idx


def _are_same_nodes(node1, node2):
    """
    检查两个节点是否相同

    Args:
        node1: 第一个节点
        node2: 第二个节点

    Returns:
        bool: 是否相同
    """
    # 比较终端节点
    if hasattr(node1, 'value') and hasattr(node2, 'value'):
        return str(node1.value) == str(node2.value)

    # 比较函数节点
    if hasattr(node1, 'name') and hasattr(node2, 'name'):
        return node1.name == node2.name

    # 比较字符串表示
    return str(node1) == str(node2)


def export_best_factor_as_class(individual, pset, factor_name: str,
                                output_file: str = None):
    """
    将最佳因子导出为Python类文件

    Args:
        individual: DEAP GP个体
        pset: DEAP原语集
        factor_name: 因子名称
        output_file: 输出文件路径

    Returns:
        str: Python类代码
    """
    expression = str(individual)

    code = f'''"""
自动生成的因子类
由DEAP遗传算法挖掘生成
"""

from factor import factor
import numpy as np
import pandas as pd
from analysis import *


class {factor_name}(factor):
    """
    因子名称: {factor_name}
    表达式: {expression}
    深度: {individual.height}
    大小: {len(individual)}
    """

    def set_factor_property(self):
        factor_property = {{
            'factor_name': '{factor_name}',
            'data_needed': {list(set([str(node.value) for node in individual if hasattr(node, 'value') and isinstance(node.value, str)]))},
            'factor_type': 'market'
        }}
        return factor_property

    def calc_factor(self):
        """
        计算因子值
        表达式: {expression}
        """
        # TODO: 手动实现表达式逻辑
        # 这里需要根据表达式手动编写计算逻辑
        # 因为GP编译的函数难以直接序列化

        # 示例：self.factor_data = -self.data['close'].rolling(20).std()
        raise NotImplementedError("请根据表达式手动实现calc_factor方法")
'''

    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(code)
        print(f"因子类已导出到: {output_file}")

    return code


# ============================================================================
# 测试代码
# ============================================================================

if __name__ == '__main__':
    print("工具函数模块测试...")

    # 测试示例
    print("\n如果需要测试，请先运行genetic_algorithm.py生成结果")

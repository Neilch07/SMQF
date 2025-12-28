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

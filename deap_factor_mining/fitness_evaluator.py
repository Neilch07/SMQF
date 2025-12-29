"""
适应度评估器
评估GP生成的因子表达式的性能，计算综合适应度分数
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple
import traceback
import sys
import os

# 添加父目录到路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gp_factor import GPFactor
from config import FITNESS_WEIGHTS, FACTOR_CONFIG
from utils import check_expression_validity


class FitnessEvaluator:
    """
    适应度评估器类

    负责评估GP个体（因子表达式）的性能
    """

    def __init__(self, pset, factor_params=None, fitness_weights=None,
                 shared_data=None, verbose=False):
        """
        初始化评估器

        Args:
            pset: DEAP原语集
            factor_params: 因子评价参数
            fitness_weights: 适应度权重
            shared_data: 共享数据（避免重复加载）
            verbose: 是否打印详细信息
        """
        self.pset = pset
        self.factor_params = factor_params or FACTOR_CONFIG.copy()
        self.fitness_weights = fitness_weights or FITNESS_WEIGHTS.copy()
        self.shared_data = shared_data or {}
        self.verbose = verbose

        # 评估计数器
        self.eval_count = 0
        self.success_count = 0
        self.failure_count = 0

        # 缓存：避免重复评估相同的个体
        self.cache = {}

    def evaluate(self, individual) -> Tuple[float]:
        """
        评估单个个体

        Args:
            individual: DEAP GP个体

        Returns:
            tuple: 适应度值（单目标返回单元素tuple）
        """
        self.eval_count += 1

        # 检查缓存
        individual_str = str(individual)
        if individual_str in self.cache:
            return self.cache[individual_str]

        # ====================================================================
        # CRITICAL: 检查表达式有效性，拒绝无意义的模式
        # ====================================================================
        is_valid, invalid_reasons = check_expression_validity(individual, self.pset)

        if not is_valid:
            # 表达式无效，给予严重惩罚
            penalty_score = -10000.0  # 极低的适应度
            self.failure_count += 1

            if self.verbose and self.failure_count <= 5:
                print(f"[Eval {self.eval_count}] INVALID EXPRESSION: {individual_str}")
                for reason in invalid_reasons:
                    print(f"  - {reason}")

            # 缓存并返回惩罚分数
            self.cache[individual_str] = (penalty_score,)
            return (penalty_score,)

        try:
            # 创建GP因子对象
            if self.shared_data and 'data' in self.shared_data:
                # 使用共享数据
                gp_factor = GPFactor(
                    individual=individual,
                    pset=self.pset,
                    factor_name=f'gp_factor_{self.eval_count}',
                    params=self.factor_params,
                    data=self.shared_data['data']  # 传入共享的data字典
                )
                # 手动设置其他必要的属性（这些通常在load_data()中设置）
                gp_factor.returns = self.shared_data.get('returns')
                gp_factor.sectors = self.shared_data.get('sectors')
                gp_factor.universe = self.shared_data.get('universe')
                gp_factor.index_returns = self.shared_data.get('index_returns')
            else:
                # 没有共享数据，让因子自己加载
                gp_factor = GPFactor(
                    individual=individual,
                    pset=self.pset,
                    factor_name=f'gp_factor_{self.eval_count}',
                    params=self.factor_params
                )

            # 运行因子评价（关闭显示）
            gp_factor.run(turnoff_display=True)

            # ====================================================================
            # CRITICAL: 直接检查因子数据，拒绝常数因子
            # ====================================================================
            factor_data = gp_factor.get_factor()

            if factor_data is None or factor_data.empty:
                if self.verbose:
                    print(f"[Eval {self.eval_count}] REJECTED: Factor data is empty")
                self.cache[individual_str] = (-10000.0,)
                return (-10000.0,)

            # 检查是否为常数因子
            all_values = factor_data.values.flatten()
            all_values = all_values[~np.isnan(all_values)]

            if len(all_values) == 0:
                if self.verbose:
                    print(f"[Eval {self.eval_count}] REJECTED: All factor values are NaN")
                self.cache[individual_str] = (-10000.0,)
                return (-10000.0,)

            unique_count = len(np.unique(all_values))
            if unique_count <= 1:
                if self.verbose:
                    print(f"[Eval {self.eval_count}] REJECTED: Constant factor (unique_count={unique_count}, value={all_values[0] if len(all_values) > 0 else 'NaN'})")
                self.cache[individual_str] = (-10000.0,)
                return (-10000.0,)

            # 检查横截面标准差（区分度）
            cs_std = factor_data.std(axis=1).mean()
            if cs_std < 1e-6:
                if self.verbose:
                    print(f"[Eval {self.eval_count}] REJECTED: Cross-sectional std too low ({cs_std:.2e})")
                self.cache[individual_str] = (-10000.0,)
                return (-10000.0,)

            # 提取性能指标
            results = gp_factor.get_performance()

            # 计算综合适应度
            fitness_score = self._calculate_fitness(results, individual, factor_data)

            # 缓存结果
            self.cache[individual_str] = (fitness_score,)

            self.success_count += 1

            if self.verbose and self.eval_count % 10 == 0:
                print(f"[Eval {self.eval_count}] Fitness: {fitness_score:.4f} | "
                      f"ICIR: {results.get('anual_icir', np.nan):.4f} | "
                      f"IR: {results.get('Annualized IR', np.nan):.4f}")

            return (fitness_score,)

        except Exception as e:
            # 评估失败，返回极低的适应度
            self.failure_count += 1

            if self.verbose:
                print(f"[Eval {self.eval_count}] FAILED: {str(e)[:100]}")
                if self.verbose and self.failure_count <= 3:
                    traceback.print_exc()

            # 返回极低的适应度值
            self.cache[individual_str] = (-1000.0,)
            return (-1000.0,)

    def _calculate_fitness(self, results: Dict, individual, factor_data=None) -> float:
        """
        根据因子性能指标计算综合适应度分数

        用户最关心IC，所以IC的权重最高！

        Args:
            results: 因子性能结果字典
            individual: GP个体
            factor_data: 因子数据（用于额外验证）

        Returns:
            float: 适应度分数
        """
        fitness = 0.0

        # 0. 二次检查：如果有因子数据，再次验证不是常数
        if factor_data is not None:
            all_values = factor_data.values.flatten()
            all_values = all_values[~np.isnan(all_values)]
            if len(all_values) > 0 and len(np.unique(all_values)) <= 1:
                return -10000.0  # 常数因子，拒绝

        # 1. 检查换手率 - 如果换手率接近0，说明因子是常数，给予严重惩罚
        turnover = results.get('avg_turn_over', np.nan)
        if not np.isnan(turnover) and turnover < 1e-6:
            # 换手率接近0，因子几乎不变，视为常数因子
            return -10000.0  # 极低适应度，直接返回

        # 2. 平均IC - 最重要的指标！给予极高权重
        avg_ic = results.get('daily_avg_ic', np.nan)
        if not np.isnan(avg_ic) and not np.isinf(avg_ic):
            # IC的正常范围应该是 [-1, 1]，如果超出范围说明有问题
            if abs(avg_ic) > 1.0:
                # IC异常，可能是数据问题或计算错误
                return -5000.0
            # 使用绝对值IC，放大1000倍以确保它主导适应度
            fitness += self.fitness_weights['ic'] * abs(avg_ic) * 1000
        else:
            # IC缺失，给予严重惩罚
            fitness -= 100.0

        # 3. ICIR（年化IC信息比率）- 第二重要
        icir = results.get('anual_icir', np.nan)
        if not np.isnan(icir) and not np.isinf(icir):
            fitness += self.fitness_weights['icir'] * abs(icir) * 10  # 放大10倍
        else:
            # ICIR缺失，给予惩罚
            fitness -= 50.0

        # 4. IR（年化信息比率）
        ir = results.get('Annualized IR', np.nan)
        if not np.isnan(ir) and not np.isinf(ir):
            fitness += self.fitness_weights['ir'] * ir * 5  # 放大5倍
        else:
            fitness -= 20.0

        # 5. 换手率检查（应该在合理范围内）
        # 换手率太低（<0.01）可能是常数因子，太高（>0.5）则成本太高
        if not np.isnan(turnover) and not np.isinf(turnover):
            # 换手率超过0.5则惩罚
            if turnover > 0.5:
                fitness += self.fitness_weights['turnover_penalty'] * (turnover - 0.5) * 10

        # 6. 表达式复杂度奖励（在合理范围内鼓励复杂因子）
        complexity = len(individual)
        depth = individual.height
        # 奖励中等复杂度（深度3-6，大小10-30），过于简单或复杂都会被轻微惩罚
        if 3 <= depth <= 6 and 10 <= complexity <= 30:
            complexity_bonus = 0.5  # 小幅奖励
        elif depth < 3 or complexity < 5:
            complexity_bonus = -1.0  # 太简单，惩罚
        elif depth > 8 or complexity > 50:
            complexity_bonus = -0.5  # 太复杂，轻微惩罚
        else:
            complexity_bonus = 0.0
        fitness += complexity_bonus

        # 7. 最大回撤惩罚
        max_drawdown = results.get('Maximum Drawdown', np.nan)
        if not np.isnan(max_drawdown) and not np.isinf(max_drawdown):
            if max_drawdown > 0.3:  # 回撤超过30%则惩罚
                fitness -= (max_drawdown - 0.3) * 5

        # 8. 奖励高年化收益
        annual_return = results.get('Annualized Return', np.nan)
        if not np.isnan(annual_return) and not np.isinf(annual_return):
            if annual_return > 0.1:  # 年化收益超过10%则奖励
                fitness += (annual_return - 0.1) * 2

        return fitness

    def evaluate_batch(self, population) -> list:
        """
        批量评估种群

        Args:
            population: 个体列表

        Returns:
            list: 适应度值列表
        """
        return [self.evaluate(ind) for ind in population]

    def get_statistics(self) -> Dict:
        """
        获取评估统计信息

        Returns:
            dict: 统计信息
        """
        return {
            'total_evaluations': self.eval_count,
            'successful_evaluations': self.success_count,
            'failed_evaluations': self.failure_count,
            'success_rate': self.success_count / self.eval_count if self.eval_count > 0 else 0,
            'cache_size': len(self.cache)
        }

    def reset_statistics(self):
        """重置统计信息"""
        self.eval_count = 0
        self.success_count = 0
        self.failure_count = 0

    def clear_cache(self):
        """清空缓存"""
        self.cache.clear()


# ============================================================================
# 并行评估器（可选）
# ============================================================================

class ParallelFitnessEvaluator(FitnessEvaluator):
    """
    并行适应度评估器

    使用多进程加速评估
    """

    def __init__(self, pset, factor_params=None, fitness_weights=None,
                 shared_data=None, verbose=False, n_jobs=-1):
        """
        初始化并行评估器

        Args:
            n_jobs: 并行任务数，-1表示使用全部CPU核心
        """
        super().__init__(pset, factor_params, fitness_weights, shared_data, verbose)
        self.n_jobs = n_jobs

    def evaluate_batch(self, population) -> list:
        """
        并行批量评估种群

        Args:
            population: 个体列表

        Returns:
            list: 适应度值列表
        """
        from concurrent.futures import ProcessPoolExecutor, as_completed
        import multiprocessing as mp

        # 确定使用的进程数
        if self.n_jobs == -1:
            n_jobs = mp.cpu_count()
        else:
            n_jobs = min(self.n_jobs, mp.cpu_count())

        # 单线程评估（避免序列化问题）
        # 注意：DEAP的GP个体可能难以序列化，因此这里仍使用串行
        # 如果需要真正的并行，需要特殊处理
        return [self.evaluate(ind) for ind in population]


# ============================================================================
# 数据加载器：预加载数据供所有因子共享
# ============================================================================

def load_shared_data(factor_params):
    """
    预加载数据，供所有因子共享使用

    这样可以避免每个因子重复加载数据，显著提升速度

    Args:
        factor_params: 因子参数

    Returns:
        dict: 共享数据字典，包含data, returns, sectors, universe等
    """
    print("预加载共享数据...")

    # 方法1: 尝试直接从quant_lib导入
    try:
        # 添加quant_lib目录到路径
        import sys
        import os
        quant_lib_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'quant_lib'))
        if quant_lib_dir not in sys.path:
            sys.path.insert(0, quant_lib_dir)

        from factor import factor

        # 创建一个简单的临时因子类来加载数据
        class TempFactor(factor):
            def set_factor_property(self):
                return {
                    'factor_name': 'temp_factor',
                    'data_needed': ['close', 'open', 'high', 'low', 'volume', 'return'],
                    'factor_type': 'market'
                }

            def calc_factor(self):
                self.factor_data = self.data['close']

        temp_factor = TempFactor(params=factor_params)
        temp_factor.load_data()

        # 提取所有必要的数据（不仅是data字典，还包括returns等）
        shared_data = {
            'data': temp_factor.data.copy(),
            'returns': temp_factor.returns.copy() if temp_factor.returns is not None else None,
            'sectors': temp_factor.sectors.copy() if temp_factor.sectors is not None else None,
            'universe': temp_factor.universe.copy() if temp_factor.universe is not None else None,
            'index_returns': temp_factor.index_returns.copy() if temp_factor.index_returns is not None else None,
        }

        print(f"共享数据加载完成！数据字段: {list(shared_data['data'].keys())}")

        return shared_data

    except Exception as e:
        print(f"警告: 数据加载失败: {e}")
        import traceback
        traceback.print_exc()
        print("返回空数据字典，每个因子将独立加载数据")
        return {}


# ============================================================================
# 测试代码
# ============================================================================

if __name__ == '__main__':
    from deap import creator, base, tools
    from gp_primitives import create_primitive_set
    import time

    print("测试适应度评估器...")

    # 创建原语集
    pset = create_primitive_set(
        window_sizes=[20],
        data_fields=['close', 'volume'],
        enable_division=False,
        enable_high_complexity=False
    )

    # 创建GP类型
    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

    # 创建工具箱
    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # 生成测试种群
    population = toolbox.population(n=5)

    print(f"\n生成了 {len(population)} 个个体")
    for i, ind in enumerate(population):
        print(f"个体 {i+1}: {ind}")

    # 设置测试参数（使用较短的时间范围加快测试）
    test_params = FACTOR_CONFIG.copy()
    test_params['start_date'] = '2022-01-01'
    test_params['end_date'] = '2023-12-31'
    test_params['start_os_date'] = None
    test_params['display'] = False

    # 创建评估器
    print("\n创建评估器...")
    evaluator = FitnessEvaluator(
        pset=pset,
        factor_params=test_params,
        verbose=True
    )

    # 评估种群
    print("\n开始评估...")
    start_time = time.time()

    for i, ind in enumerate(population):
        fitness = evaluator.evaluate(ind)
        print(f"个体 {i+1} 适应度: {fitness[0]:.4f}")

    elapsed = time.time() - start_time
    print(f"\n评估完成！总耗时: {elapsed:.2f}秒")

    # 打印统计信息
    stats = evaluator.get_statistics()
    print("\n评估统计:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print("\n适应度评估器测试完成！")
